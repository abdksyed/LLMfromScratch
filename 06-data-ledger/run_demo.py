#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

from tdes.config import (
    ARTIFACTS,
    CRASH_AFTER,
    FORK_AT,
    SEED,
    TOTAL_BATCHES,
)
from tdes.core import (
    LOG,
    Firewall,
    build_documents,
    dump,
    freeze_tokenizer,
    run_firewall,
    validate_manifests,
    verify_tokenizer,
    write_shards,
)
from tdes.prove import (
    check_packing,
    fork_run,
    learning_ok,
    mixture_report,
    opus_counts,
    performance_report,
    replay_interval,
    write_evidence,
    write_opus_ledger,
)
from tdes.runtime import Engine, Jsonl, SimulatedCrash
from tdes.schedule import OPUS, BatchStream, compile_mixture


def _reset(path: Path):
    if not path.exists():
        return
    for p in path.rglob("*"):
        if p.is_file():
            os.chmod(p, 0o644)
    shutil.rmtree(path)


def main():
    _reset(ARTIFACTS)
    for name in ("manifests", "ledgers", "checkpoints", "shards"):
        (ARTIFACTS / name).mkdir(parents=True, exist_ok=True)
    LOG.open(ARTIFACTS / "run.log")

    tok, tmeta = freeze_tokenizer()
    verify_tokenizer(tok, tmeta)

    docs = build_documents(tok)
    manifests = write_shards(docs, tmeta["tokenizer_hash"])
    validate_manifests(manifests, tmeta["tokenizer_hash"])

    firewall = Firewall()
    run_firewall(firewall, manifests, docs)

    mix = compile_mixture()
    train_docs = [d for d in docs if d.split == "train"]
    proxy = [d for d in train_docs if d.lane == "reasoning"][:30]
    rng = np.random.default_rng(SEED)
    opus = OPUS(proxy, rng)
    stream = BatchStream(train_docs, mix, tmeta["pad_id"], rng, opus, firewall)
    docs_by_id = {d.doc_id: d for d in docs}

    engine = Engine(stream)
    engine.load_model()
    packing_report = None

    try:
        while stream.batch_id < TOTAL_BATCHES:
            if stream.batch_id == FORK_AT:
                engine.save_checkpoint("ckpt-003")
            if stream.batch_id == CRASH_AFTER:
                write_opus_ledger(opus)
                expected = stream.peek()
                engine.save_checkpoint(
                    "ckpt-006",
                    extra={
                        "expected_next_batch_id": expected.batch_id,
                        "expected_next_batch_hash": expected.hash(),
                        "expected_spans": expected.spans,
                    },
                )
                dump(
                    ARTIFACTS / "checkpoints" / "expected_next.json",
                    {
                        "batch_id": expected.batch_id,
                        "hash": expected.hash(),
                        "spans": expected.spans,
                    },
                )
                LOG.event("crash simulated")
                raise SimulatedCrash()
            batch = stream.next_batch()
            if packing_report is None:
                packing_report = check_packing(batch, docs_by_id)
                LOG.event("batches packed")
            engine.train_batch(batch)
    except SimulatedCrash:
        pass

    payload = engine.restore("ckpt-006")
    LOG.event("run resumed")
    batch = stream.next_batch()
    resume_ok = (
        batch.hash() == payload["expected_next_batch_hash"]
        and batch.batch_id == payload["expected_next_batch_id"]
    )
    dump(
        ARTIFACTS / "checkpoints" / "resume.json",
        {
            "expected_batch_id": payload["expected_next_batch_id"],
            "resumed_batch_id": batch.batch_id,
            "expected_hash": payload["expected_next_batch_hash"],
            "resumed_hash": batch.hash(),
            "matched": resume_ok,
            "expected_spans": payload.get("expected_spans"),
            "resumed_spans": batch.spans,
        },
    )
    if resume_ok:
        LOG.ok("resume_next_batch_matched")
    else:
        LOG.fail("resume_next_batch_matched")
    engine.train_batch(batch)
    while stream.batch_id < TOTAL_BATCHES:
        engine.train_batch(stream.next_batch())
    write_opus_ledger(opus, log=False)

    replay_ok, replay_matches = replay_interval(docs_by_id, tmeta["pad_id"])
    main_steps, main_seconds, main_loss_tokens = engine.steps, engine.train_seconds, engine.train_loss_tokens

    engine.consumption.close()
    engine.learning.close()
    engine.consumption = Jsonl(ARTIFACTS / "ledgers" / "fork" / "consumption.jsonl")
    engine.learning = Jsonl(ARTIFACTS / "ledgers" / "fork" / "learning.jsonl")
    engine.restore("ckpt-003")
    fork_ok, _ = fork_run(engine)
    engine.steps, engine.train_seconds, engine.train_loss_tokens = main_steps, main_seconds, main_loss_tokens
    perf = performance_report(engine)

    mix_rep = mixture_report()
    oc, opus_ok = opus_counts()
    packing_ok = packing_report is not None and not packing_report["issues"]
    eval_ok = any(b.get("split") == "eval" for b in firewall.blocked)
    tput_ok = perf["loss_bearing_tokens_per_second"] > 0 and perf["packing_utilization"] > 0.5

    results = {
        "tokenizer_integrity": {
            "result": "PASS",
            "evidence": "manifests/tokenizer.json",
            "tokenizer_hash": tmeta["tokenizer_hash"],
        },
        "evaluation_firewall": {
            "result": "PASS" if eval_ok else "FAIL",
            "evidence": "ledgers/firewall.json",
            "blocked": firewall.blocked[:8],
        },
        "packing_correctness": {
            "result": "PASS" if packing_ok else "FAIL",
            "evidence": "manifests/packing_report.json",
            "issues": [] if packing_report is None else packing_report["issues"],
        },
        "mixture_compliance": {
            "result": "PASS" if mix_rep["floors_met_every_batch"] else "FAIL",
            "evidence": "manifests/mixture.json",
            **mix_rep,
        },
        "opus_audit_trail": {
            "result": "PASS" if opus_ok else "FAIL",
            "evidence": "ledgers/opus.jsonl",
            "counts": oc,
        },
        "crash_recovery": {
            "result": "PASS" if resume_ok else "FAIL",
            "evidence": "checkpoints/resume.json",
            "expected_batch_id": payload["expected_next_batch_id"],
            "resumed_batch_id": batch.batch_id,
            "expected_hash": payload["expected_next_batch_hash"],
            "resumed_hash": batch.hash(),
        },
        "replay": {
            "result": "PASS" if replay_ok else "FAIL",
            "evidence": "ledgers/replay.json",
            "matches": [
                {
                    "batch_id": m["batch_id"],
                    "original_hash": m["original_hash"],
                    "replay_hash": m["replay_hash"],
                    "matched": m["matched"],
                }
                for m in replay_matches
            ],
        },
        "learning_trace": {
            "result": "PASS" if learning_ok() else "FAIL",
            "evidence": "ledgers/learning.jsonl",
        },
        "throughput": {
            "result": "PASS" if tput_ok else "FAIL",
            "evidence": "performance.json",
            **perf,
        },
        "fork": {
            "result": "PASS" if fork_ok else "FAIL",
            "evidence": "ledgers/fork.json",
        },
    }
    write_evidence(results)
    engine.close()
    LOG.close()


if __name__ == "__main__":
    main()

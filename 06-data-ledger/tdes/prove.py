from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .config import (
    ARTIFACTS,
    BATCH_SIZE,
    FLOORS,
    FORK_BATCHES,
    LANES,
    REPLAY_END,
    REPLAY_START,
    SEQ_LEN,
)
from .core import LOG, dump
from .schedule import PackedBatch, pack_pieces, Piece
from .runtime import Engine


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rebuild_batch(rec, docs_by_id, pad_id) -> PackedBatch:
    pieces = []
    for sp in rec["spans"]:
        doc = docs_by_id[sp["doc_id"]]
        pieces.append(Piece(doc, sp["src_start"], sp["src_end"], sp["policy"]))
    ids, attn, pos, loss, spans = pack_pieces(pieces, pad_id)
    lane_tokens = {l: 0 for l in LANES}
    lane_loss = {l: 0 for l in LANES}
    for sp in spans:
        lane_tokens[sp["lane"]] += sp["end"] - sp["start"]
        lane_loss[sp["lane"]] += sp["loss_bearing"]
    return PackedBatch(
        batch_id=rec["batch_id"],
        stage=rec["stage"],
        input_ids=ids,
        attention_mask=attn,
        position_ids=pos,
        loss_mask=loss,
        spans=spans,
        lane_tokens=lane_tokens,
        lane_loss_tokens=lane_loss,
        planned_tokens=rec.get("planned_tokens") or {},
        opus_decision_ids=rec.get("opus_decision_ids") or [],
        utilization=float(attn.sum()) / ids.size,
        loss_bearing=int(loss.sum()),
    )


def replay_interval(docs_by_id, pad_id, start=REPLAY_START, end=REPLAY_END):
    rows = load_jsonl(ARTIFACTS / "ledgers" / "consumption.jsonl")
    matches = []
    all_ok = True
    for rec in rows:
        if rec["batch_id"] < start or rec["batch_id"] >= end:
            continue
        rebuilt = rebuild_batch(rec, docs_by_id, pad_id)
        h = rebuilt.hash()
        ok = h == rec["batch_hash"]
        all_ok = all_ok and ok
        matches.append(
            {
                "batch_id": rec["batch_id"],
                "original_hash": rec["batch_hash"],
                "replay_hash": h,
                "matched": ok,
                "token_spans": rec["spans"],
            }
        )
    dump(ARTIFACTS / "ledgers" / "replay.json", {"start": start, "end": end, "matches": matches})
    LOG.event("historical stream replayed")
    if all_ok and matches:
        LOG.ok("replay_hash_matched")
    else:
        LOG.fail("replay_hash_matched")
    return all_ok and bool(matches), matches


def check_packing(batch, docs_by_id) -> dict:
    issues = []
    pad = batch.attention_mask == 0
    if not np.all(batch.loss_mask[pad] == 0):
        issues.append("pad_has_loss")
    if not np.all(batch.input_ids[pad] == batch.input_ids[pad][0] if pad.any() else True):
        pass
    for sp in batch.spans:
        doc = docs_by_id[sp["doc_id"]]
        src = np.array(doc.token_ids[sp["src_start"] : sp["src_end"]])
        packed = batch.input_ids[sp["seq"], sp["start"] : sp["end"]]
        if not np.array_equal(src, packed):
            issues.append(f"span_mismatch_{sp['doc_id']}")
        src_lm = list(doc.loss_mask[sp["src_start"] : sp["src_end"]])
        src_lm[0] = 0
        packed_lm = list(batch.loss_mask[sp["seq"], sp["start"] : sp["end"]])
        if packed_lm != src_lm:
            issues.append(f"loss_mask_{sp['doc_id']}")
        pos = list(batch.position_ids[sp["seq"], sp["start"] : sp["end"]])
        if pos != list(range(sp["end"] - sp["start"])):
            issues.append(f"position_ids_{sp['doc_id']}")
        if doc.split != "train":
            issues.append(f"eval_in_batch_{sp['doc_id']}")
    report = {
        "batch_id": batch.batch_id,
        "issues": issues,
        "utilization": batch.utilization,
        "loss_bearing": batch.loss_bearing,
        "n_spans": len(batch.spans),
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
    }
    dump(ARTIFACTS / "manifests" / "packing_report.json", report)
    return report


def floors_ok(rec) -> bool:
    real = max(1, rec["n_real_tokens"])
    for lane, floor in FLOORS.items():
        share = rec["lane_tokens"].get(lane, 0) / real
        if share + 1e-9 < floor:
            return False
    return True


def mixture_report():
    rows = load_jsonl(ARTIFACTS / "ledgers" / "consumption.jsonl")
    planned = Counter()
    actual = Counter()
    floor_pass = True
    for rec in rows:
        for l in LANES:
            planned[l] += rec.get("planned_tokens", {}).get(l, 0)
            actual[l] += rec.get("lane_tokens", {}).get(l, 0)
        floor_pass = floor_pass and floors_ok(rec)
    pt, at = sum(planned.values()) or 1, sum(actual.values()) or 1
    planned_share = {l: planned[l] / pt for l in LANES}
    actual_share = {l: actual[l] / at for l in LANES}
    return {
        "planned_tokens": dict(planned),
        "actual_tokens": dict(actual),
        "planned_share": planned_share,
        "actual_share": actual_share,
        "floors_met_every_batch": floor_pass,
        "n_batches": len(rows),
    }


def opus_counts():
    rows = load_jsonl(ARTIFACTS / "ledgers" / "opus.jsonl")
    c = Counter(r["decision"] for r in rows)
    needed = {"accept", "reject", "defer", "floor_override"}
    return dict(c), needed.issubset(c)


def fork_run(engine: Engine):
    orig = {r["batch_id"]: r["batch_hash"] for r in load_jsonl(ARTIFACTS / "ledgers" / "consumption.jsonl")}
    engine.stream.opus.T = 1.7
    engine.stream.rng = np.random.default_rng(99)
    engine.stream.opus.rng = engine.stream.rng
    for cur in engine.stream.cursors.values():
        cur.skip()
    hashes = []
    for _ in range(FORK_BATCHES):
        batch = engine.stream.next_batch()
        engine.train_batch(batch)
        hashes.append({"batch_id": batch.batch_id, "hash": batch.hash()})
    dump(
        ARTIFACTS / "ledgers" / "fork.json",
        {
            "batches": hashes,
            "parent_hashes": orig,
            "diverged": all(h["hash"] != orig.get(h["batch_id"]) for h in hashes),
        },
    )
    diverged = all(h["hash"] != orig.get(h["batch_id"]) for h in hashes)
    LOG.event("branch forked")
    return diverged, hashes


def write_opus_ledger(opus, log=True):
    path = ARTIFACTS / "ledgers" / "opus.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in opus.decisions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if log:
        LOG.event("OPUS decisions recorded")


def performance_report(engine: Engine):
    rows = load_jsonl(ARTIFACTS / "ledgers" / "consumption.jsonl")
    real = sum(r["n_real_tokens"] for r in rows)
    slots = sum(r["n_input_tokens"] for r in rows)
    loss_tok = sum(r["n_loss_bearing"] for r in rows)
    tps = engine.train_loss_tokens / engine.train_seconds if engine.train_seconds else 0.0
    report = {
        "steps": engine.steps,
        "train_seconds": engine.train_seconds,
        "packing_utilization": real / slots if slots else 0.0,
        "loss_bearing_fraction": loss_tok / real if real else 0.0,
        "loss_bearing_tokens": loss_tok,
        "loss_bearing_tokens_per_second": tps,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
    }
    dump(ARTIFACTS / "performance.json", report)
    LOG.event("performance measured")
    return report


def write_evidence(results: dict):
    dump(ARTIFACTS / "evidence.json", results)
    order = [
        ("Tokenizer integrity", "tokenizer_integrity"),
        ("Evaluation firewall", "evaluation_firewall"),
        ("Packing correctness", "packing_correctness"),
        ("Mixture compliance", "mixture_compliance"),
        ("OPUS audit trail", "opus_audit_trail"),
        ("Crash recovery", "crash_recovery"),
        ("Replay", "replay"),
        ("Learning trace", "learning_trace"),
        ("Throughput", "throughput"),
    ]
    lines = [
        "# Evidence",
        "",
        "| Requirement | Result | Evidence |",
        "| --- | --- | --- |",
    ]
    for title, key in order:
        rec = results[key]
        lines.append(f"| {title} | {rec['result']} | {rec['evidence']} |")
    (ARTIFACTS / "evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.event("audit completed")


def learning_ok():
    cons = load_jsonl(ARTIFACTS / "ledgers" / "consumption.jsonl")
    learn = load_jsonl(ARTIFACTS / "ledgers" / "learning.jsonl")
    if not cons or not learn:
        return False
    cmap = {r["batch_id"]: r for r in cons}
    for rec in learn:
        c = cmap.get(rec["batch_id"])
        if not c:
            return False
        if not rec.get("span_losses"):
            return False
        src = {sp["doc_id"] for sp in c["spans"]}
        got = {sp["doc_id"] for sp in rec["span_losses"]}
        if not got.issubset(src):
            return False
    return True

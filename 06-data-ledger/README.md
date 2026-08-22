# V5 Training Data Execution System

A small, complete data path for V5: documents become immutable tokenized shards, a frozen mixture packs them into batches, OPUS admits or blocks them, training writes consumption and learning ledgers, and a crash/resume/replay/fork loop proves the stream is reconstructable.

The model is `Qwen/Qwen3-0.6B-Base`. The corpus is `ryanmarten/OpenThoughts-1k-sample`, with a small Indic seed set so every protected lane is real.

## Run

```bash
../.venv/bin/python run_demo.py
```

This regenerates `submission_artifacts/` with the log, evidence bundle, manifests, ledgers, checkpoints, and performance report.

## Path

```
documents
  -> tokenized shards
  -> manifests
  -> mixture schedule
  -> packing
  -> batches
  -> training
  -> consumption ledger
  -> learning ledger
  -> checkpoint
  -> crash
  -> resume
  -> replay
  -> audit
```

## Design

**Shards and tokenizer.** Every document is hashed, tokenized with a frozen Qwen3 tokenizer, and written to a read-only jsonl shard. The shard file hash and tokenizer hash live in the manifest. Eval and valid shards are built the same way, then refused by the train firewall.

**Mixture and floors.** Foundation and Capability use the V5 lane shares. Each batch reserves 30% of tokens for protected floors and gives the remaining 70% to the stage mixture.

**Packing.** Concat packing for general, code, reasoning, and Indic. Agentic traces keep observations in context with a zero loss mask. Long-context items are chunks of genuinely longer documents. Position ids reset at each packed span. Padding has zero attention and zero loss.

**OPUS.** Candidates are scored by CountSketch alignment with a 30-document reasoning proxy, uniqueness, exposure, and in-batch redundancy. Boltzmann sampling yields accept or reject. A full lane is deferred. A protected floor can override a reject.

**Ledgers and checkpoints.** Consumption records batch hashes, token spans, and source content hashes. Learning records per-span loss against those same spans. A checkpoint is a ledger offset plus last-layer/optimizer/stream state.

**Crash, resume, replay, fork.** The run checkpoints after six batches, peeks the next batch hash, and crashes. Resume must emit that exact batch. Replay rebuilds batches 1–3 from shard token spans and compares hashes. A fork from batch 3 changes OPUS temperature and continues on a side ledger.

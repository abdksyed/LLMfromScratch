# V5 Training Data Execution System — Implementation Plan

Greenfield project. Self-contained: nothing in this repo outside
`06-v5-data-execution/` is used or referenced.

**Evidence discipline used throughout this document:**

- **[M] measured** — value came back from a live API or command in this session.
- **[D] derived** — arithmetic on a measured value, or on a published figure
  that is cited.
- **[E] estimated** — not yet measured. Every `[E]` has a calibration task in §22.

Never present `[E]` as `[M]` in the README or evidence bundle.

---

## 1. Scope

Build a small but complete training-data execution system proving the pipeline

```
documents → tokenized shards → manifests → mixture schedule → packing →
batches → training → consumption ledger → learning ledger → checkpoint →
crash → resume → replay → audit
```

is **correct, reproducible, auditable, and efficient**. Scale is explicitly not
the goal. The graded artifact is the data system, not model quality.

Two execution surfaces:

| Surface | Who runs it | Hardware | Purpose |
|---|---|---|---|
| `python run_demo.py` | the grader | CPU, no GPU | full pipeline, all invariants, minutes |
| `torchrun --nproc_per_node=2 run_demo.py --profile server` | you | vast.ai, 2× 24 GB | real MoE run under DDP, hours |

Same code path. Only `tdx/config.py` values differ. This matters: a grader
without a GPU must still regenerate every artifact.

---

## 2. Hardware decision

### 2.1 Does 32 GB fit a 200M MoE?

Parameter count derived from `FlameF0X/TinyMoE-200m-2x16`'s published
`config.json` **[M]** (`hidden=384`, `inter=768`, `layers=12`, `heads=8`,
`kv_heads=4`, `num_local_experts=16`, `num_experts_per_tok=2`, `vocab=32000`,
`tie_word_embeddings=false`, `rope_theta=1e6`):

| Component | Params |
|---|---:|
| attention per layer (q 147,456 + k 73,728 + v 73,728 + o 147,456) | 442,368 |
| router per layer (384×16) | 6,144 |
| 16 experts per layer (16 × 3 × 384×768) | 14,155,776 |
| norms per layer | 768 |
| **per layer** | **14,605,056** |
| × 12 layers | 175,260,672 |
| embed_tokens (32000×384) | 12,288,000 |
| lm_head, untied | 12,288,000 |
| final norm | 384 |
| **total** | **199,837,056** |

**[D]** So "200m" is accurate. Active per token = attention + router + 2 experts
+ lm_head = **38,914,560 ≈ 38.9M** (5.1× sparsity).

Training memory, AdamW, bf16 compute with fp32 master weights **[D]**:

| Buffer | Bytes |
|---|---:|
| params bf16 | 400 MB |
| params fp32 master | 799 MB |
| grads bf16 | 400 MB |
| AdamW `exp_avg` fp32 | 799 MB |
| AdamW `exp_avg_sq` fp32 | 799 MB |
| **model + optimizer** | **≈ 3.2 GB** |
| activations @ 32,768 tok/step, chunked CE, SDPA (no S×S matrix) | ≈ 7.4 GB **[D]** |
| **peak, 1× 32 GB** (`world_size=1`, all 32,768 on one device) | **≈ 11.6 GB of 32 GB** |
| **peak, 2× 24 GB** (16,384 tok/rank + DDP gradient buckets) | **≈ 8.3 GB per rank of 24 GB** |

**Answer: both are ample — 2.8× headroom on one 32 GB card, 2.9× per rank on two
24 GB cards.** Note the per-rank figure is *lower* than the single-card figure:
DDP splits activations but duplicates the 3.2 GB of weights and optimizer state.
Batch sizing is derived in §2.5. Three consequences:

1. **The long-context lane can be genuine.** S=16,384 at batch 1 peaks ≈7.9 GB
   on one device **[D]**, so real 16K sequences instead of short docs padded to
   look long. This is the one axis where per-device VRAM binds — see §2.5.
   Requires `max_position_embeddings` 1024 → 16384 and `sliding_window` disabled;
   `rope_theta=1e6` already supports the extension. Logged as a documented
   deviation from the reference config.
2. **You could train 4–6× bigger.** 32 GB supports ~1B params at this sparsity.
   `--profile server-xl` is left as a config-only knob, unused by default.
3. **VRAM is not the constraint. Throughput is.** See §22.

### 2.2 Maximizing both GPUs: batch policy

Two devices, 24 GB each. The policy is: **push `micro_batch` per rank until
throughput stops improving, then set the token budget so the step count stays
useful.** Batch size and budget are independent knobs; neither is fixed.

**Fixed cost per rank** (from §2.1): 3.2 GB params + optimizer, ~0.4 GB DDP
gradient buckets **[E]**, ~0.9 GB CUDA context + cuBLAS workspace **[E]** =
**4.5 GB**. That leaves 19.5 GB; reserving 10% against allocator fragmentation
gives **~17.5 GB practical per rank [D]**.

**Per-token activation cost [D].** Summing tensors saved for backward: attention
path ~2.3 KB, MoE path (top-2 × 768 × 3 tensors) ~9.2 KB, residuals + norms
~2.8 KB → **~14.3 KB/token/layer**, × 12 layers = **~172 KB/token**. The LM-head
region adds bf16 logits (64 KB) + fp32 CE workspace (128 KB) = ~192 KB/token
naive, or **~65 KB/token** with chunked cross-entropy. Totals: **~237 KB/token
chunked**, ~364 KB naive. Chunked CE is used throughout — it buys ~1.5× the batch
for a few lines of code.

**Ceiling: 17.5 GB ÷ 237 KB = ~77,000 tok/rank [D].** The ceiling is not the
target, though, because throughput saturates before memory does:

| `micro_batch`/rank | peak/rank **[D]** | % of 24 GB |
|---:|---:|---:|
| 8,192 | 6.4 GB | 26% |
| 16,384 | 8.2 GB | 34% |
| 32,768 | 11.9 GB | 50% |
| **49,152** | **15.6 GB** | **65%** |
| 65,536 | 19.3 GB | 80% |
| 77,000 | 21.9 GB | 91% — fragmentation risk |

Calibration (§22) times one real fwd+bwd at each size, records tok/s and peak
VRAM, and picks the knee — the largest size where tok/s is still climbing.

**Default: `micro_batch=49,152`/rank → 98,304 tokens/step global** at
`world_size=2`. 65% occupancy is deliberate: it leaves room for allocator
fragmentation and for the long-context lane, and if the sweep shows 65,536 still
climbing it is a one-line config change.

**Step count follows from the budget, and the budget is a knob.** 985M ÷ 98,304 =
**~10,020 optimizer steps** — five stages at ~2,000 steps each, enough for a real
LR schedule, stage transitions, OPUS decisions and several crash/resume/replay
cycles. If more steps are wanted at the same occupancy, raise the budget rather
than shrink the batch: §5 has **2.08B unique tokens** against a 985M budget, so
`--budget-tokens 1.97B` doubles the steps to ~20,040 with every lane exposure
still ≤ 1.0. Wall-clock roughly doubles. The elastic lanes are `reasoning`
(0.21× exposure today) and `general_web` (take more C4 shards).

**Gradient accumulation stays available, unused by default.** Global batch is
`micro_batch × accum_steps × world_size`. Only `micro_batch` affects occupancy;
`accum_steps` trades wall-clock for a larger global batch at zero memory cost.
Default `accum_steps=1`.

**Where per-device VRAM genuinely binds: long context.** A single sequence cannot
be split across ranks under DDP — that needs sequence/context parallelism, out of
scope here. So a 16,384-token sequence must fit on *one* device: ~3.7 GB
activations + 4.5 GB fixed = **~8.2 GB [D]**, comfortable. At
`micro_batch=49,152` the long-context stage runs 3 × 16K sequences per rank.

**Why two ranks make the graded invariant harder — and better.**

> Each rank must receive a **disjoint, deterministic** slice; the union across
> ranks must be gap-free; and resume must restore every rank consistently with no
> batch skipped or repeated.

That is the 150-point section (§20), and proving it across two ranks is a
materially stronger claim than proving it on one. §14 makes `rank` and
`world_size` part of the batch-derivation key, §19 stores every rank's cursor in
the data-plane checkpoint, and the demo profile asserts disjointness plus
gap-free union at a simulated `world_size=2` **on CPU** — so the invariant is
demonstrated in the grader's GPU-less run too. `world_size=1` remains the default
for `run_demo.py`: multi-GPU is additive, never required, so a distributed bug
cannot break the graded path.

---

## 3. Source set

Every source below was checked **unauthenticated** in this session. `gated=False`
plus a `200`/`206` response with correct magic bytes means the grader can fetch
it with no token and no terms click.

### 3.1 Rejected, with reasons

| Source | Verdict |
|---|---|
| `nampdn-ai/tiny-webtext` | **[M]** file list is `.gitattributes` + `README.md` only. No data. |
| `nampdn-ai/tiny-codes` | **[M]** `gated=auto`; unauth GET → **401**. Breaks "no manual intervention". |
| `nampdn-ai/tiny-textbooks` | **[M]** `gated=auto` → 401. Same defect. |
| `togethercomputer/RedPajama-Data-1T-Sample` | **[M]** API → 401. |
| `cerebras/SlimPajama-627B` | **[M]** API → 401. |
| `allenai/math_qa` | **[M]** lists no parquet files — same defect as tiny-webtext. |
| `stas/openwebtext-10k` | **[M]** only `process.txt` (2,039 B). No data. |
| `HuggingFaceFW/fineweb-edu` | ungated **[M]**, but min unit is a **2.15 GB** file **[M]**. Too coarse to slice. |
| `monology/pile-uncopyrighted` | ungated **[M]**, but 11 GB shards **[M]** and `license=other`. |
| `Skylion007/openwebtext`, `tiiuae/falcon-refinedweb` | ungated **[M]**, but my guessed file paths **404'd [M]** — unverified, so not used. |

### 3.2 Accepted

| Lane | Source | Config / file | License | Verified |
|---|---|---|---|---|
| `general_web` | `allenai/c4` | `en/c4-train.{00000..00003}-of-01024.json.gz` | odc-by | `206`, gzip magic **[M]** |
| `narrative` | `roneneldan/TinyStories` | `TinyStories-valid.txt`, `-train.txt` | cdla-sharing-1.0 | `200` **[M]** |
| `code` | `code-search-net/code_search_net` | `python/train` | other → §3.4 | `200`, `PAR1` **[M]** |
| `code` (structured) | `google-research-datasets/mbpp` | `full/train` | cc-by-4.0 | `200`, `PAR1` **[M]** |
| `reasoning` | `open-r1/OpenR1-Math-220k` | `all/default-{00000,00001}-of-00010.parquet` | apache-2.0 | ungated **[M]** |
| `agentic_tool` | `glaiveai/glaive-function-calling-v2` | `glaive-function-calling-v2.json` | apache-2.0 | ungated **[M]** |
| `long_context` | C4 + CSN + OpenR1, length-filtered | — | inherited | — |
| **EVAL / TEST** | `openai/openai_humaneval` | `test` | mit | `200`, `PAR1` **[M]** |
| **EVAL / TEST** | `google-research-datasets/mbpp` | `full/test` | cc-by-4.0 | `200`, `PAR1` **[M]** |
| **EVAL / TEST** | `openai/gsm8k` | `main/test` | mit | `200`, `PAR1` **[M]** |
| **EVAL / TEST** | `code_search_net` | `python/test` | other | `200`, `PAR1` **[M]** |
| **EVAL / TEST** | `allenai/c4` | `en/c4-validation.00000-of-00008.json.gz` | odc-by | `206` **[M]** |

Two changes worth calling out:

**C4 replaces wikitext as the general-web spine.** wikitext-103 has only 122M
tokens **[D]**, which forced 1.88× repetition against a 207M target. C4 `en` has
1024 train shards of ~319 MB each **[M]**; take as many as needed. wikitext is
dropped from the training lanes entirely — it is Wikipedia-derived and would
create intra-lane near-duplicate overlap with C4's Wikipedia mirrors, burning
dedup budget for no gain.

**`agentic_tool` is now real data.** `glaive-function-calling-v2` is 112,960
rows / 254 MB / ~63.6M tokens **[M]** of genuine system-prompt + user + function
call + function response + assistant conversations, apache-2.0. This removes the
last synthetic lane — previously it was templated from mbpp. Every scored lane is
now collected human or model-generated data with recorded provenance, which
removes the obvious "was this simulated?" attack surface at grading step 3.

### 3.3 Pinned revision SHAs

All 40 chars, resolved from the API this session **[M]**. Pinning by SHA rather
than `main` is what makes the corpus reproducible — `main` can move.

| Dataset | Revision SHA |
|---|---|
| `allenai/c4` | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` |
| `roneneldan/TinyStories` | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| `code-search-net/code_search_net` | `bd0cf261e357a3eb5c8fba490d23ec1a1cd59555` |
| `open-r1/OpenR1-Math-220k` | `e4e141ec9dea9f8326f4d347be56105859b2bd68` |
| `glaiveai/glaive-function-calling-v2` | `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac` |
| `google-research-datasets/mbpp` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` |
| `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` |
| `openai/openai_humaneval` | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` |

Fetch by explicit URL, no library magic:

```
https://huggingface.co/datasets/{repo}/resolve/{sha}/{path}
```

### 3.4 License gate — blocking

`code_search_net` reports `license: other` **[M]**. Its card states the corpus
is assembled from permissively-licensed GitHub repos, but "other" is not a
clearance. **Build task 0** (§22): read the card and `LICENSE`, record the exact
terms in `manifests/licenses.json`, and if redistribution or training use is not
clearly permitted, drop it and take the code lane from mbpp + a
permissively-licensed alternative. Do not ship an unresolved license.

---

## 4. Measured dataset inventory

Rows and uncompressed bytes from the datasets-server `/size` endpoint **[M]**.
Download bytes from the HF tree API **[M]**. Token estimates **[D]** from bytes ÷
chars-per-token, ratio stated per row. **Exact token counts come from the frozen
tokenizer at shard-build time and are what the manifests record** — these figures
size the plan, they do not appear in evidence.

| Source | Rows | Uncompressed | Download | c/tok | ~tokens |
|---|---:|---:|---:|---:|---:|
| TinyStories train | 2,119,719 | 2,008,276,893 | 1,924,281,556 | 4.2 | 478.2M |
| TinyStories valid | 21,990 | 17,512,286 | 19,447,282 | 4.2 | 4.2M |
| CSN python train | 412,178 | 1,258,692,461 | 521,785,752 | 3.5 | 359.6M |
| CSN python test | 22,176 | 68,510,710 | 28,744,352 | 3.5 | 19.6M |
| OpenR1-Math `all` | 225,129 | 10,161,169,949 | 12,649,340,809 (60 files) | 4.0 | 2,540.3M |
| glaive-fc-v2 | 112,960 | 254,485,835 | 271,190,065 | 4.0 | 63.6M |
| mbpp full/train | 374 | 176,926 | 87,223 | 3.5 | 50.6K |
| mbpp full/test | 500 | 244,167 | 115,824 | 3.5 | 69.8K |
| gsm8k main/train | 7,473 | 3,990,538 | 2,306,545 | 4.0 | 997.6K |
| gsm8k main/test | 1,319 | 719,293 | 419,088 | 4.0 | 179.8K |
| humaneval test | 164 | 194,394 | 83,920 | 3.5 | 55.5K |

**C4 sizing is derived, not measured.** datasets-server reported 2,571,635 rows /
5.83 GB for `en` **[M]**, but that is only its converted subset — C4 `en` is
~365M documents. The T5 paper's published figure is ~156B tokens over 1024
shards, so **~152M tokens per shard [D]**, at 319 MB download / ~732 MB
uncompressed per shard **[M]/[D]**. `prepare_data.py` records the true count.

### 4.1 What gets downloaded

| Source | Files taken | Download |
|---|---|---:|
| C4 `en` train | 4 shards | 1.28 GB |
| C4 `en` validation (eval) | 1 shard | ~320 MB |
| TinyStories train | 1 `.txt` | 1.92 GB |
| CSN python | train + test | 551 MB |
| OpenR1-Math `all` | 2 of 10 files | 428 MB |
| glaive-fc-v2 | 1 JSON | 271 MB |
| mbpp + gsm8k + humaneval | all | 2.9 MB |
| **total** | | **≈ 4.8 GB** |

Disk: 4.8 GB download + ~9 GB extracted + 2.0 GB shards (uint16, since
vocab 32,000 < 65,536) + ~14 GB checkpoints + ~1 GB artifacts ≈ **31 GB [D]**.
**Provision 60 GB.**

Demo profile is far smaller: TinyStories-valid (19 MB), 1 C4 shard, CSN first
5,000 rows, OpenR1 first 2,000 rows, glaive first 5,000 rows, all of
mbpp/gsm8k/humaneval → ~400 MB download, ~40 MB shards **[D]**.

---

## 5. Lane supply and the no-repeat constraint

You asked for no lane training twice on the same data. Target = exposure ≤ 1.0.

Budget: **1.0B loss-bearing tokens**, of which 1.5% (15M) is a held-out anneal
reserve → **985M active**.

| Lane | Source | Available | Target | Exposure |
|---|---|---:|---:|---:|
| `general_web` | C4 en, 4 shards | 608.0M **[D]** | 285.7M | **0.47×** |
| `narrative` | TinyStories train | 478.2M **[D]** | 216.7M | **0.45×** |
| `code` | CSN python + mbpp | 359.7M **[D]** | 285.7M | **0.79×** |
| `reasoning` | OpenR1 2 files + gsm8k | 509.1M **[D]** | 108.4M | **0.21×** |
| `agentic_tool` | glaive-fc-v2 | 63.6M **[M]** | 49.3M | **0.78×** |
| `long_context` | length-filtered from above | 60.0M **[D]** | 39.4M | **0.66×** |
| **total** | | **2,078.6M** | **985.2M** | **0.47× mean** |

**Every lane ≤ 1.0×. No token is trained on twice.** Total unique supply is
2.08B against a 985M budget — 2.1× headroom, so lane shares can be retuned
without breaching the constraint. Scaling to a 2B-token run stays under 1.0×
by taking more C4 shards and OpenR1 files.

`long_context` is **carved** from the other lanes by length filter (documents
≥4K tokens), not new supply. Its tokens are attributed to `long_context` and
deducted from the parent lane, so no double-counting. OpenR1 averages ~11.3K
tokens/row **[D]** — genuine long reasoning traces, ideal here.

### 5.1 The tension this creates, and how it is resolved

With every lane in surplus, `floor_override` and `defer_exposure_cap` never
fire — but the assignment explicitly requires demonstrating *"OPUS acceptance,
rejection, deferral and protected-floor override."* An unexercised code path
cannot be evidenced.

Resolution: **two runs of the same compiler, one flag apart.**

| Run | Flag | Demonstrates |
|---|---|---|
| A — healthy | default | all floors met from surplus, every exposure ≤1.0, zero overrides |
| B — scarcity drill | `--restrict-lane reasoning=1.2M --restrict-lane agentic_tool=0.9M` | supply-infeasible floor → `floor_override` with arithmetic, `defer_exposure_cap`, recorded deficit |

Both write `mixture_plan.json` + `opus_decisions.jsonl`; both are in the evidence
bundle. The drill is not artificial — it is exactly what happens when a source
fails its license or PII gate mid-project, and the restriction is recorded in the
manifest so a grader sees the cause. Crucially, run A proves the healthy path and
run B proves the override path, and the *same code* produces both. That is
stronger evidence than either alone.

### 5.2 Contamination risk to check — blocking

`OpenR1-Math-220k` and `gsm8k` train are both math. OpenR1 is DeepSeek-R1 traces
over public math problems, and **gsm8k test is one of my eval sets**. If OpenR1
contains gsm8k test items, training on it leaks eval.

This is not a reason to drop OpenR1 — it is the firewall's job. Build task 1
(§22) runs exact-hash + MinHash of OpenR1 against all five eval sets **before**
shard sealing, records the measured overlap rate in
`manifests/contamination_report.json`, and drops every colliding record. A
measured non-zero overlap that was caught and removed is *better* evidence than
an assumed-clean corpus.

### 5.3 Curriculum stages

| Stage | Share | Tokens | Character |
|---|---:|---:|---|
| Foundation | 20% | 200.0M | narrative + general web heavy, short seq |
| Capability | 35% | 350.0M | code ramps up, 2K–4K seq |
| Specialization | 30% | 300.0M | code + reasoning + agentic peak, 4K–8K |
| Consolidation | 10% | 100.0M | long context enters, 8K–16K |
| Quality cooldown | 3.5% | 35.0M | highest-quality slice, 16K dominant |
| Anneal reserve | 1.5% | 15.0M | **held out; never served in the main run** |
| **total** | **100%** | **1,000.0M** | |

Sequence-length buckets: 1K / 2K / 4K / 8K / 16K, mass shifting later per stage.
Per-stage lane shares are produced by the supply-aware solver (§15), not
hand-written here, and land in `manifests/mixture_plan.json`. A `[PASS]`
assertion proves the reserve was never consumed.

---

## 6. Profiles

| | `demo` (default) | `server` | `server-xl` |
|---|---|---|---|
| Target | grader, CPU | vast.ai 2× 24 GB | future |
| Params | ~5.4M **[D]** | 199,837,056 **[D]** | ~1B |
| hidden / layers / heads | 128 / 4 / 4 | 384 / 12 / 8 | — |
| experts / top-k | 8 / 2 | 16 / 2 | — |
| vocab | 8,192 | 32,000 | 32,000 |
| max seq | 512 | 16,384 | 16,384 |
| tokens/step | 4,096 | 32,768 global (16,384/rank × 2) | — |
| steps | 128 | ~30,060 **[D]** | — |
| budget | ~0.5M tok | 985M tok | — |
| runtime | minutes **[E]** | 1–3.5 h **[E]** | — |

Identical topology across profiles — GQA, RoPE, top-2 routing, `router_aux_loss`.
Only width, depth, expert count and vocab change.

---

## 7. Dependencies and the plane boundary

`requirements.txt`, pinned: `torch`, `numpy`, `pyarrow`, `zstandard`,
`tokenizers`, `gigatoken`. Nothing else. `requirements-gpu.txt` adds the CUDA
wheel index in one line.

**torch over hand-written numpy backprop.** Hand-deriving backward for GQA +
RoPE + router aux loss is a correctness liability on an assignment that grades a
data system, not autograd.

**Compute plane vs data plane — the load-bearing rule.** torch appears *only* in
`tdx/model/`. Tokens, spans, hashes, batch ids, ledgers and manifests are pure
stdlib ints and bytes. This is what makes "bit-exact replay" an honest claim: if
a float entered a hash chain, replay would fail on a different BLAS build while
the README advertised determinism. §18 states the boundary; an inline import scan
in `run_demo.py` enforces it and emits `[PASS] plane_isolation_verified`.

---

## 8. Repo layout

```
06-v5-data-execution/
  run_demo.py              # the one command
  prepare_data.py          # network stage, run once
  audit.py                 # independent re-verification, fresh process
  README.md
  requirements.txt  requirements-gpu.txt
  tdx/
    config.py              # profiles + config_hash
    spec.py                # V5 constants: lanes, stages, floors, caps
    util/                  # canonical json, sha256, seeded rng, event log
    acquire/               # pinned download, slicing, raw manifest
    corpus/                # records, lane+split assignment, MinHash clusters
    tokenizer/             # BPE train, freeze, tokenizer_hash, encode
    shards/                # immutable writer, manifests, merkle root
    firewall/              # split enforcement, blacklist, contamination probe
    mixture/               # stages, lane weights, floor solver, infeasibility
    opus/                  # utility scoring, redundancy, decision records
    packing/               # 5 policies, mask/pos/segment builder, validators
    batching/              # deterministic rank-aware stream, batch ids
    model/                 # Mixtral-topology MoE (torch, isolated here)
    ledgers/               # consumption + learning, hash chains
    checkpoint/            # two-plane checkpoints
    runtime/               # trainer, crash, resume, replay, fork, perf
    evidence/              # evidence.json + evidence.md generators
  tests/                   # ~24 pytest modules
  data/                    # gitignored: raw + shards
  vendored_sample/         # ~2 MB committed, offline fallback
  submission_artifacts/    # generated
```

---

## 9. Acquisition

`prepare_data.py` is the only network stage. `run_demo.py` runs fully offline and
verifies the merkle root before touching anything; if the slice is missing it
auto-invokes `prepare_data.py`, so the single-command requirement holds.

Bounded deterministic slice per source, recorded in the manifest:

- whole files or whole parquet row-groups, taken in **file order**, never random
  sample — the slice is a pure function of `(revision_sha, file_list, row_cap)`
- record `rows_taken`, `bytes_taken`, `sha256` of every downloaded file and of
  the extracted slice

`manifests/raw_manifest.json` is committed. It makes acquisition auditable
without storing 4.8 GB in git: re-run, re-hash, compare. That is the provenance
chain — revision SHA → file hash → slice hash → shard hash → merkle root.

**Offline fallback.** `vendored_sample/` holds a ~2 MB slice of the openly
licensed sources, committed. `run_demo.py --data-source vendored` runs the entire
demonstration with **zero network**. The one-command requirement should not
depend on HF being reachable during grading.

---

## 10. Tokenizer

**No BPE trainer is written.** The reference model `FlameF0X/TinyMoE-200m-2x16`
ships a complete tokenizer **[M]** — `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json` — and it is **BPE, 32,000 vocab, 58,980 merges,
`byte_fallback=true`, Metaspace pre-tokenizer, `LlamaTokenizer` class** **[M]**,
matching `config.json`'s `vocab_size=32000` **[M]** exactly. Training a vocabulary
from scratch would add a whole subsystem, risk a mismatch with the reference
config, and earn nothing: the rubric grades tokenizer *integrity*, not tokenizer
*authorship*.

**Base vocabulary — vendored, frozen, hashed.**

1. `prepare_data.py` fetches the three tokenizer files at a pinned revision SHA
   and commits them to `vendored_tokenizer/`.
2. `tokenizer_hash = sha256(canonical_json(tokenizer.json))`, recorded in
   `manifests/tokenizer.json` with the source repo id and revision.
3. Files are marked read-only; every later load re-hashes and aborts on mismatch.

Freezing is the graded property, and it is now *exact* rather than self-attested —
the hash is of an artifact whose provenance is a pinned upstream SHA.

**Our layer on top — added special tokens.** This is where the project's own
tokenizer work lives, and a real defect in the base config drives it:
`tokenizer_config.json` aliases `pad_token` to `</s>` while `config.json` has
`pad_token_id: null` **[M]**. Padding with `</s>` and then masking on the pad id
would silently mask genuine end-of-document tokens — a loss-mask bug that would
corrupt §13's per-lane masking and never announce itself. Six tokens are appended
through the standard added-tokens mechanism (no retraining, no merge-table
surgery):

| Token | Purpose |
|---|---|
| `<\|pad\|>` | explicit pad, distinct from `</s>` — fixes the aliasing defect |
| `<\|doc\|>` | document separator inside packed sequences (§13) |
| `<\|im_start\|>` / `<\|im_end\|>` | turn boundaries; assistant-turn loss masking |
| `<\|tool_call\|>` / `<\|tool_result\|>` | `glaive` agentic-lane structure |

Vocab goes 32,000 → 32,006, padded to **32,064** for tensor-core alignment
(multiple of 64). Cost is ~49K params across embedding + untied head — total
199,886,208, still "200m". `vocab_size` is updated to match and the change is
recorded as a documented deviation, next to `max_position_embeddings`
1024 → 16384.

**Encoding — gigatoken, behind a parity gate.** `gigatoken` **v0.10.0, MIT
licensed** **[M]** is a Rust BPE encoder that *consumes* an existing vocabulary
rather than training one, and claims ~1000× HF `tokenizers` throughput. Encoding
~31 GB of text once is worth accelerating.

It is a young single-author library, so it is not trusted blind. A parity gate
runs before the corpus is encoded: 10,000 documents sampled deterministically
across all six lanes are encoded with **both** `gigatoken` and HF `tokenizers`,
and the id sequences must match exactly. The comparison hash lands in
`manifests/tokenizer.json`. On mismatch the run continues on the HF encoder and
records `encoder=hf_fallback` — a config flag, not a code change. Note the base
tokenizer uses a Metaspace pre-tokenizer, so gigatoken's SentencePiece-family
path applies and the realised speedup will be below its headline figure **[E]**;
the gate makes correctness independent of that.

Both profiles share this one frozen tokenizer — `demo` does not get a smaller
vocabulary. A single byte-identical tokenizer across surfaces is worth more than
a slightly faster CPU demo.

`[PASS] tokenizer_hash_verified` · `[PASS] tokenizer_roundtrip_verified` ·
`[PASS] encoder_parity_verified` · `[PASS] pad_token_distinct_from_eos`

---

## 11. Shards and manifests

Shards are `uint16` little-endian token arrays + a JSON sidecar. Sealed: written
once, `chmod 0444`, never appended.

Per-shard manifest: `shard_id`, `lane`, `split`, `source_dataset`,
`revision_sha`, `license`, `provenance_kind`, `tokenizer_hash`, `token_count`,
`doc_count`, `doc_offsets`, `content_sha256`, `created_utc`, `writer_version`,
`config_hash`.

Per-record: `record_id`, `source_dataset`, `revision_sha`, `license`,
`provenance_kind`, `lane`, `split`, `token_start`, `token_end`, `text_sha256`,
`cluster_id`, `segments` (typed spans for structured records).

`corpus_manifest.json` holds a **merkle root** over sorted shard hashes. One
number pins the entire corpus; every later artifact carries it.

`[PASS] shards_sealed` · `[PASS] manifests_validated` · `[PASS] merkle_root_stable`

---

## 12. Firewall

Three layers, then a post-run sweep:

1. **Split** — eval/test records live in a separate shard tree; the loader
   refuses any shard whose manifest `split != "train"`.
2. **Hash blacklist** — `manifests/blacklist.json` holds `text_sha256` of every
   eval record; any candidate matching is rejected.
3. **Near-duplicate** — 8-shingle MinHash against eval; over threshold rejects.

**Active red-team, not a passive claim.** Four deliberate intrusions, each
blocked by a *named* layer and logged:

| Attempt | Blocked by |
|---|---|
| eval shard handed to the loader | split |
| verbatim humaneval doc in the candidate pool | hash blacklist |
| lightly-paraphrased gsm8k test item | MinHash |
| gsm8k test item relabelled `split=train` | manifest cross-check vs blacklist |

Plus §5.2's measured OpenR1×eval overlap scan, and a final sweep over the whole
consumption ledger asserting **zero** eval `record_id` in any loss-bearing batch.

`[PASS] eval_shard_blocked` · `[PASS] blacklist_hit_blocked` ·
`[PASS] contamination_probe_blocked` · `[PASS] relabel_attempt_blocked` ·
`[PASS] no_eval_tokens_in_ledger`

---

## 13. Packing policies

| Lane | Policy | Why |
|---|---|---|
| `general_web` | concat-and-chop | max utilization; `<eod>` + segment reset at doc boundary |
| `narrative` | concat-and-chop | same, accounted separately |
| `code` | best-fit, length-bucketed | keep files intact, minimize pad |
| `reasoning` | greedy, atomic | question + trace + answer must not split |
| `agentic_tool` | structure-preserving | no cross-conversation attention |
| `long_context` | pad-only | one genuine long doc per row; never fabricate length |

Row invariants, asserted in code and tested on random cases:

1. `input_ids`, `loss_mask`, `position_ids`, `segment_ids` are equal length.
2. `position_ids` restart at 0 per segment, step +1 within.
3. Attention = causal ∧ same-segment. Verified by materializing the dense mask
   and asserting zeros across every boundary — leakage is *demonstrated absent*,
   not asserted.
4. `loss_mask=0` on pad, on each segment's first token, and on all
   non-target spans.
5. Zero loss on: agentic system prompt / user turn / `<|tool_result|>` observation;
   reasoning question; mbpp task statement. Active loss on: assistant turn,
   `<|tool_call|>` name+arguments, reasoning trace, final answer, code solution.
6. `sum(loss_mask)` equals the ledger's `loss_bearing_tokens`.
7. No eval `record_id` in any segment.
8. `max(position_ids) < max_position_embeddings`.
9. Every segment maps to a real `(shard_id, token_start, token_end)`.

`[PASS] packing_invariants_verified` · `[PASS] no_cross_segment_attention` ·
`[PASS] loss_mask_policy_verified`

---

## 14. Batch identity and determinism

Two hashes per batch:

- `batch_id = sha256(canonical(recipe))` — ordered
  `(shard_id, record_id, token_start, token_end, segment_index)` spans plus
  policy, stage, seq_len, rank.
- `content_hash = sha256(input_ids ‖ loss_mask ‖ position_ids ‖ segment_ids)`
  over raw little-endian bytes.

Replay must match **both**. A recipe match alone would not prove the tensors were
rebuilt identically, and the assignment asks for "batch ids, token spans and
hashes."

**The stream is a pure function; the cursor is one integer.**

```
batch(step, rank) = f(mixture_plan_hash, merkle_root, tokenizer_hash,
                      seed, step, rank, world_size)
```

Any step is reconstructible without replaying prior steps. This single decision
is what makes crash-resume, replay, and fork-from-earlier-checkpoint provable
rather than hopeful — a stateful loader would make fork genuinely hard. Exposure
bookkeeping, the only naturally stateful part, is precomputed into a seeded
deterministic draw plan per stage, so the runtime cursor is one integer.

---

## 15. Mixture compiler, floors, infeasibility

Compile-time, before any training:

1. Read supply per lane from manifests.
2. Apply stage volumes (§5.3) and stage lane weights.
3. **Floor solver**: floors reserve 30% of every selector batch; stage mixture +
   OPUS assign the other 70%.
4. **Feasibility check** — for each lane, `target ≤ unique × exposure_cap`?
   - feasible → enforce the declared floor strictly.
   - infeasible → emit `floor_infeasible` with full arithmetic, write an
     override record (`declared`, `effective`, `deficit`, `reason`,
     `reallocated_to`), then enforce the *effective* floor strictly.

Detecting infeasibility at compile time rather than failing mid-run is the
correct engineering behaviour and it is auditable: the evidence shows declared
floor, feasible floor, deficit, and zero violations of the feasible floor.

Output `manifests/mixture_plan.json` with `mixture_plan_hash`.

`[PASS] mixture_compiled` · `[PASS] floors_satisfied` ·
`[PASS] floor_override_recorded` (run B) · `[PASS] reserve_never_consumed`

---

## 16. OPUS

Real gradients, not a stand-in:

```
utility(x) ≈ −⟨g_proxy, P · g(x)⟩
```

`g_proxy` from a Bench-Proxy slice **disjoint from eval and from train**; `P` the
live AdamW diagonal `1/(√v̂+ε)`. Cost is bounded and **disclosed**: probe slice of
the first 512 tokens, gradients over `lm_head` + last block only, 16 candidates
per window, one window per 4 steps. An approximation stated plainly beats an
approximation hidden.

Redundancy penalty from `cluster_id` overlap with already-selected candidates,
then Boltzmann sampling at temperature `T`.

Decision records, all outcomes exercised: `accept`, `reject_low_utility`,
`reject_redundant`, `reject_firewall`, `defer_exposure_cap`,
`defer_stage_mismatch`, `floor_override_accept`, `rescue_protected_floor`.
Each carries `candidate_id`, `lane`, `utility`, `cluster_id`, `exposure_before`,
`stage`, `reason`. Floors fill first; OPUS runs only on the residual; disabled
for the anneal reserve.

`[PASS] opus_decisions_recorded` · `[PASS] opus_all_outcomes_exercised` ·
`[PASS] opus_deterministic_given_seed`

---

## 17. Model

Mixtral topology, from the reference config: RMSNorm, GQA (8 q / 4 kv), RoPE
`theta=1e6`, SwiGLU experts, top-2 routing with `router_aux_loss_coef=0.02`.

MoE earns its place in the *data* system rather than being decoration: the
learning ledger records **per-lane expert routing histograms**, so the audit can
show which lanes routed to which experts, plus router entropy and load balance.
That is a genuine data-system signal — lane-to-expert specialisation — and it is
only available because the model is MoE.

`[PASS] router_load_balanced` (>1 expert used per lane) ·
`[PASS] expert_routing_by_lane_recorded`

---

## 18. Ledgers

Both append-only JSONL with a `prev_row_hash` → `row_hash` chain, so truncation
or edits are detectable.

**Consumption** (what was served), per batch: `seq`, `global_step`, `rank`,
`epoch`, `stage`, `batch_id`, `content_hash`, `policy`, `seq_len`, `rows`,
`segments[]` (`shard_id`, `record_id`, `lane`, `token_start`, `token_end`),
`tokens_total`, `tokens_non_pad`, `tokens_loss_bearing`, `by_lane{}`,
`pad_fraction`, `utilization`, `exposure_after{}`, `prev_row_hash`, `row_hash`.

**Learning** (what was learned), per step: `seq`, `global_step`, `batch_id`,
`loss_total`, `loss_lm`, `loss_router_aux`, `loss_by_lane{}`,
`loss_by_shard{}`, `token_loss_sample[]` (k sampled loss-bearing tokens with
`record_id`, `position`, `loss`, `ppl = exp(loss)`), `grad_norm`, `lr`,
`update_norm`, `expert_load[]`, `router_entropy`, `step_seconds`,
`data_seconds`, `prev_row_hash`, `row_hash`.

`loss_by_shard` is the join that makes "loss linked to source data" literal.

**Float boundary, stated not buried.** Chained hashes cover data-plane fields
only. Loss floats are recorded to full precision but **excluded from the chain**;
determinism tests assert exact equality for data-plane fields and
`abs(a-b) < 1e-4` for floats. bf16 + differing BLAS across machines makes
bit-identical floats impossible; claiming otherwise would be false. Replay
equality is proven where it can be proven.

`[PASS] ledger_chain_intact` · `[PASS] learning_trace_linked`

---

## 19. Checkpoints — two planes

A full 200M training state is ~2.8 GB **[D]**. Saving that every 100 steps over
30,060 steps would write ~840 GB. Split it:

| Plane | Contents | Size | Cadence |
|---|---|---:|---|
| data | stream cursor, ledger `seq` + head hashes, `next_batch_id_expected`, exposure counters, RNG state, all plan hashes, per-rank cursors | ~10 KB **[D]** | every 100 steps |
| model | params, optimizer, scheduler, scaler | ~2.8 GB **[D]** | every 5,000 steps + fork points |

**Both planes carry ledger offsets**, so checkpoints stay tied to ledger position
— the graded property — without the disk cost. Keep last 3 model checkpoints plus
fork points ≈ 14 GB **[D]**.

`next_batch_id_expected` is written *at save time*. On resume it must equal the
recomputed value. That single field is what turns "we resumed" into "we resumed
at exactly the right place."

Under `world_size>1`, every rank's cursor is in the data-plane checkpoint and
rank 0 writes it after a barrier, so restore is globally consistent.

`[PASS] checkpoint_saved` · `[PASS] checkpoint_ledger_offsets_match`

---

## 20. Crash, resume, replay, fork

**Crash — real, and off a checkpoint boundary.** Training runs in a child
process; at a configured step it raises and exits non-zero. Crash step is
deliberately **not** a checkpoint multiple (demo: crash 21, last ckpt 16), so ~5
committed ledger rows sit past the checkpoint. Crashing on a boundary would make
the hard case vanish.

**Resume — quarantine and compare.** The post-checkpoint tail is *moved* to
`ledgers/precrash_tail.jsonl`, not deleted — deleting it would destroy the
evidence. Then re-derive those steps and compare. Assertions:

1. stored `next_batch_id_expected` == recomputed → `resume_next_batch_matched`
2. re-derived rows are byte-identical to the quarantined tail → `resume_tail_identical`
3. final ledger `seq` is contiguous `0..N`, no gap → `resume_no_gap`
4. `batch_id` multiset has no duplicates → `resume_no_duplicate`
5. under `world_size=2`, per-rank `batch_id` sets are disjoint and their union is
   gap-free → `resume_rank_disjoint`

**Replay.** Pick a committed interval, rebuild from `(plan_hash, merkle_root,
seed, step, rank)`, compare `batch_id`, every token span, and `content_hash`.
The ledger is opened read-only and its head hash asserted unchanged.
→ `replay_hash_matched`, `replay_spans_matched`, `replay_ledger_immutable`

**Fork — both halves.** From an earlier model checkpoint:
- same seed → reproduces the parent's batches exactly (determinism proof)
- changed lane weights → diverges, and records `parent_run_id`, `fork_step`,
  `parent_ckpt_hash`, `divergence_step` (lineage proof)

→ `fork_identical_seed_reproduced`, `fork_divergent_recorded`, `fork_lineage_recorded`

---

## 21. Performance

All recomputed from ledger rows, never measured ad hoc:

- `non_pad_utilization` = Σ`tokens_non_pad` / Σ`tokens_total`
- `loss_bearing_ratio` = Σ`tokens_loss_bearing` / Σ`tokens_total`
- `useful_loss_bearing_tokens_per_second` = Σ`tokens_loss_bearing` / Σ`step_seconds`
- `data_pipeline_fraction` = Σ`data_seconds` / Σ`step_seconds`
- per-lane and per-policy utilization; MoE `expert_load` skew

A test regenerates `performance.json` from the ledgers and asserts equality, so
every throughput claim is reconstructible — the rubric refuses credit otherwise.

`[PASS] throughput_reconstructible` · `[PASS] packing_utilization_reported`

---

## 22. Build order

Two blocking gates first — both can force a redesign, so neither waits.

0. **License gate (§3.4).** Resolve `code_search_net`. Record in
   `manifests/licenses.json` or drop the source.
1. **Contamination scan (§5.2).** OpenR1 × all five eval sets, exact + MinHash.
   Record measured overlap; drop collisions.
2. **Calibration.** On a rented 2× 24 GB box: run the §2.3 preflight, then
   `nvidia-smi topo -m` to record the actual interconnect. Time one real fwd+bwd
   at `server` size on **1 rank**, then the same under
   `torchrun --nproc_per_node=2` — the delta *is* the all-reduce cost, measured
   rather than assumed. Record peak VRAM per rank. On CPU: time the `demo`
   profile. This replaces every `[E]` in §2, §6 and §21 with `[M]`. If demo
   exceeds ~10 min, shrink the demo — never the invariants.
3. util + spec + config; canonical JSON, hashing, seeded RNG, event log.
4. acquire + raw manifest + vendored sample.
5. tokenizer: train, freeze, hash (+ tests).
6. corpus records, lane/split assignment, MinHash clusters, shards, manifests,
   merkle root (+ tests).
7. firewall + four intrusion probes (+ tests).
8. packing: 5 policies, masks, dense-mask validator (+ tests).
9. mixture compiler, floor solver, infeasibility override (+ tests).
10. OPUS: utility, redundancy, decisions (+ tests).
11. rank-aware batch stream, batch ids (+ tests).
12. MoE model, AdamW, grad check (+ tests).
13. ledgers with chains (+ tests).
14. two-plane checkpoints (+ tests).
15. crash → resume → quarantine-compare (+ tests).
16. replay + fork (+ tests).
17. performance from ledgers (+ reconstruction test).
18. `audit.py` — independent re-verification in a fresh process.
19. evidence bundle + the tamper test.
20. README; `.gitignore` with `!submission_artifacts/run.log`.
21. Run A (healthy) and Run B (scarcity drill) on the server; commit artifacts.

---

## 23. `run_demo.py` phases

Preflight (deps, device capability, disk) → verify raw manifest → freeze/verify
tokenizer → seal shards + merkle → build blacklist → **4 firewall intrusions** →
compile mixture (feasibility + overrides) → validate packing → train to
deliberate crash *(child, non-zero exit)* → resume + verify + finish *(child)* →
replay interval → fork ×2 → audit → performance → pytest → emit
`evidence.json` + `evidence.md` → summary table.

Flags: `--profile {demo,server,server-xl}`, `--data-source {vendored,hub}`,
`--world-size N`, `--restrict-lane LANE=N`, `--reasoning-extra socratic`,
`--seed N`.

---

## 24. Tests (~24 pytest modules)

tokenizer freeze + roundtrip + tie-break determinism · shard immutability
(rehash, mutation detected, `chmod` enforced) · merkle stability · manifest
schema completeness · 5 packing policies on random cases · dense attention mask
has zero cross-segment mass · loss-mask policy per lane · position-id reset ·
utilization arithmetic · firewall: all four intrusions blocked · contamination
scan finds a planted near-duplicate · floor solver feasible + infeasible ·
reserve untouched · OPUS determinism, all outcomes reachable, floor-first
ordering · stream purity (`batch(step)` independent of call order) · **rank
disjointness at `world_size=2`, union gap-free** · ledger chain detects a flipped
byte · loss→shard join completeness · checkpoint offsets match ledger heads ·
resume: no gap, no duplicate, tail byte-identical · replay: ids + spans + hashes
match, ledger head unchanged · fork: same-seed reproduces, divergent records
lineage · MoE grad check vs finite differences · **compute/data plane isolation:
no torch import outside `tdx/model/`** · performance reconstruction from ledgers ·
**tamper test: mutate one ledger row → the affected requirement flips to FAIL**.

That last one is the direct answer to grading step 3. Evidence that cannot fail
is not evidence.

---

## 25. Artifacts

```
submission_artifacts/
  run.log                      # full event sequence + [PASS] markers
  evidence.json  evidence.md   # derived, never hardcoded
  manifests/   raw_manifest, licenses, contamination_report, tokenizer,
               shard_*, corpus_manifest (merkle), blacklist,
               mixture_plan_runA, mixture_plan_runB
  ledgers/     consumption.jsonl, learning.jsonl, opus_decisions.jsonl,
               precrash_tail.jsonl, events.jsonl
  checkpoints/ data_plane/*.json, model_plane/*.pt, fork_lineage.json
  performance.json
  audit/       audit_report.json
```

`evidence.json` rows: `id`, `requirement`, `result`, `checks[]`,
`evidence_paths[]`, `computed_values{}`, `notes`. `evidence.md` renders the
required table. `bundle_sha256` plus `audit.py` re-derivation prove nothing was
hand-written.

---

## 26. Rubric mapping

| Area | Pts | Where |
|---|---:|---|
| End-to-end execution | 150 | §23, one command, both profiles |
| Shards, manifests, tokenizer integrity | 100 | §10, §11 |
| Packing, masks, batch correctness | 150 | §13, §14 |
| Mixture, floors, OPUS | 150 | §15, §16, runs A+B |
| Consumption + learning ledgers | 150 | §18 |
| Checkpoint, crash, resume, replay, fork | 150 | §19, §20 |
| Eval + validation firewall | 50 | §12, five real benchmarks |
| Throughput + packing efficiency | 50 | §21 |
| Tests, evidence, docs | 50 | §24, §25 |

---

## 27. Risks

| Risk | Mitigation |
|---|---|
| `code_search_net` license unclear | **Blocking gate §3.4**; drop if not clear |
| OpenR1 contains gsm8k test items | **Blocking scan §5.2**; measured + removed |
| C4 tokens/shard is `[D]` not `[M]` | `prepare_data.py` records truth; take more shards if short |
| `sm_120` toolchain mismatch | preflight asserts capability, prints cu128 command |
| Grader has no GPU | demo profile is CPU-only and the default |
| Grader has no network | `--data-source vendored`, committed |
| Grader has no `pip` | README documents `uv`; preflight prints both |
| Dataset gated/renamed later | pinned SHAs + vendored fallback |
| Float drift across machines | data plane float-free; floats within tolerance (§18) |
| MoE router collapse | aux loss + `expert_load` recorded; test asserts >1 expert |
| Distributed bugs eat schedule | `world_size=1` default; multi-GPU is additive, tested on CPU |
| Server run exceeds budget | take fewer C4 shards; lane shares are config |

---

## 28. Decisions taken

1. **Rent 2× 24 GB on vast.ai; data plane rank-aware from day one** (§2.2).
   Renting removes the capital argument that favoured one big card, learning
   DDP/NCCL is a stated goal, and rank-disjoint gap-free batching is a
   materially stronger claim on the graded invariant. `world_size=1` stays the
   default, so the grader's CPU path is untouched. Image: `cuda-12.8.1-auto`
   preferred over the linked `cuda-13.2.1-auto` (§2.3).
2. **Every lane exposure ≤ 1.0 — no token trained twice** (§5). 2.08B unique
   supply against a 985M budget.
3. **C4 `en` replaces wikitext** as general web (§3.2); wikitext dropped to avoid
   intra-lane Wikipedia overlap.
4. **`glaive-function-calling-v2` for agentic** — removes the last synthetic
   lane; all six lanes are now real data.
5. **`OpenR1-Math-220k` for reasoning** — 509M tokens, also feeds long context.
6. **Scarcity is a drill, not a default** (§5.1). Run A healthy, Run B
   `--restrict-lane`, same code, both in evidence.
7. **200M MoE is the server path; ~5.4M MoE is the demo path.** Identical
   topology.
8. **Two-plane checkpoints** (§19) — ledger-tied without 840 GB of writes.
9. **torch confined to `tdx/model/`** (§7), enforced by test.
10. **Generated artifacts committed**, with `!submission_artifacts/run.log`
    negation since a bare `*.log` would silently drop a required deliverable.

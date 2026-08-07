# Plan

The hypothesis is testable: a dense 40B decoder-only model trained with the finalized V5 tokenizer, 150 tokens per parameter, a 64K native context, and 6.00T loss-bearing V5 tokens will beat the matched general-web baseline on coding, repository repair, agentic tool use, reasoning, and the ten selected Indic languages while retaining general knowledge. Run the full model only if the proxy and data gates pass.

## Budget and benchmarks

Every loss-bearing token belongs to one of the six data categories. The remaining 90B tokens are held for the anneal reserve.

| Data category | Share | Tokens | Evaluation and primary signal |
| --- | ---: | ---: | --- |
| General web and knowledge | 31.5% | 1.890T | MMLU, ARC-Challenge, held-out web loss; normalized accuracy and perplexity |
| Source code and documentation | 22.0% | 1.320T | LiveCodeBench, HumanEval-style tests, repository QA; pass@1 and execution correctness |
| Agentic and tool use | 12.0% | 720B | SWE-bench Verified, Terminal-Bench, ToolBench, WebArena, BrowserComp; task success, valid calls, and argument correctness |
| Reasoning and STEM | 14.0% | 840B | AIME, MATH, GSM8K, Frontier Math where allowed; exact or verifier success under a fixed test budget |
| Genuine long context | 5.0% | 300B | RULER, MRCR, needle retrieval, repository long-context QA; length-stratified accuracy and recall |
| Indic multilingual | 14.0% | 840B | IndicXTREME subsets, per-language perplexity, worst-language score; macro, worst-language, and script-specific loss |
| Anneal reserve | 1.5% | 90B | Held out from normal loading and OPUS until the final anneal |
| **Total** | **100.0%** | **6.000T** | **5.91T active plus 90B reserve** |

Use this weighted score to judge the mixture:

`20% general + 25% code + 25% agentic/tool + 15% reasoning + 5% long-context + 10% Indic`

For code and agentic results, use execution or verifier checks. Judge-model scores are supplementary. Pin benchmark versions and harnesses. Remove benchmark items, close paraphrases, and known public solutions before any proxy or full-run count.

## Supply and Indic data

Measure each source after retokenization, filtering, and deduplication. The available source volumes are:

- General web: DCLM-Pool has 240T raw Common Crawl tokens; DCLM-baseline has 3.8T; FineWeb has 15T in the paper and more than 18.5T on its current Hub card; FineWeb-Edu has 1.3T in the original release and 5.4T in the score-2 release; Dolma has 3T. Pin one release for each source.
- Code: The Stack v2 has 900B+ unique tokens across 619 languages. The 1.320T code target also needs GitHub issues and pull requests, notebooks, documentation, and permissively licensed repositories.
- Agentic and tool use: ToolBench has 16,464 APIs and about 126K instruction/query examples. The current cleaning pilot has only 53,569,839 output tokens, about 0.007% of the 720B target. Treat it as a process fixture, not as inventory. Collect, replay, and verify more agent data before using this category's full share.
- Reasoning and long context: measure clean-token supply after admission. Source names alone do not justify a target volume.

The 840B Indic target has four tiers:

| Tier | Reported stock | Target | Share | Exposure before cross-source deduplication |
| --- | ---: | ---: | ---: | ---: |
| Verified native | 56.7460B | 222.6B | 26.5% | 3.92x |
| Unverified native | 22.2790B | 63B | 7.5% | 2.83x |
| Translated/derived | 172.7128B | 252B | 30.0% | 1.46x |
| Synthetic native/generated, excluding translation | 488.02B | 302.4B | 36.0% | 0.62x |
| **Selected-ten upper bound** | **739.7578B** | **840B** | **100.0%** | **Repetition required** |

Even before cross-source deduplication, the selected-ten inventory is below the 840B target. Verified native, unverified native, and translated/derived data must be repeated within the default four-exposure cap. Cover the remaining shortfall only with new independently verified collection or generation, or with a proxy-tested reallocation. Never report repeated tokens as unique stock. Keep synthetic and translated data separate from native human text in the manifest and loss report.

The language allocation is:

| Language | Share | Tokens |
| --- | ---: | ---: |
| Hindi | 25.0% | 210.0B |
| Bengali | 18.0% | 151.2B |
| Tamil | 15.0% | 126.0B |
| Marathi | 10.0% | 84.0B |
| Gujarati | 8.0% | 67.2B |
| Malayalam | 7.0% | 58.8B |
| Kannada | 6.0% | 50.4B |
| Punjabi | 5.0% | 42.0B |
| Telugu | 4.0% | 33.6B |
| Odia | 2.0% | 16.8B |

These are the ten selected languages. Their shares are based on how much data is available for each language.

## Training controls

Assign every example to one data category. Keep exact normalized hashes and 8-shingle MinHash clusters. The default exposure cap for an exact or near-duplicate cluster is 4.0. The manifest records unique clean V5 tokens, planned loss-bearing tokens, cluster IDs, and cumulative exposure.

| Stage | Share | Volume | General | Code | Agentic | Reasoning | Long | Indic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foundation | 20.0% | 1.200T | 44.5% | 18.0% | 8.0% | 12.0% | 3.0% | 14.5% |
| Capability | 35.0% | 2.100T | 35.0% | 22.0% | 10.0% | 15.0% | 3.0% | 15.0% |
| Specialization | 30.0% | 1.800T | 24.0% | 25.0% | 15.0% | 16.0% | 5.0% | 15.0% |
| Consolidation | 10.0% | 600B | 26.45% | 23.6% | 16.8% | 9.9% | 13.25% | 10.0% |
| Quality cooldown | 3.5% | 210B | 14.43% | 24.0% | 20.57% | 16.0% | 15.0% | 10.0% |
| Anneal reserve | 1.5% | 90B | held out | held out | held out | held out | held out | held out |

Difficulty bands use target continuation or reasoning length. They do not require long answers when a short answer is correct.

| Band | Length | Example |
| --- | ---: | --- |
| Low | 32-128 tokens | Compute `84 / 7` and give `12`; complete one missing function; make one API call. |
| Medium | 128-512 tokens | Solve a two-equation word problem; repair one file and run its test; make two dependent API calls. |
| High | 512-2,048 tokens | Give an AIME derivation; patch several files and run tests; make four to eight dependent calls. |
| Ultra | 2,048-16,384+ tokens | Produce a proof; complete a SWE-bench repair loop; finish a long-horizon terminal or browser task. |

### Sequence length

This schedule controls how much token mass uses each context length at each stage. For example, Foundation uses 80% of its tokens in 4K examples and 20% in 8K examples. These must be real long examples, not short examples padded with filler.

| Stage | 4K | 8K | 16K | 32K | 64K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foundation | 80% | 20% | 0% | 0% | 0% |
| Capability | 60% | 30% | 10% | 0% | 0% |
| Specialization | 20% | 45% | 25% | 10% | 0% |
| Consolidation | 0% | 15% | 35% | 35% | 15% |
| Quality cooldown | 0% | 0% | 10% | 35% | 55% |
| Anneal reserve | 0% | 0% | 0% | 30% | 70% |

Track loss-bearing tokens separately from all context tokens, including zero-loss tool observations.

Protected floors apply to every selector batch:

| Data category | Minimum |
| --- | ---: |
| General web and knowledge | 1.0% |
| Source code and documentation | 10.0% |
| Agentic and tool use | 5.0% |
| Reasoning and STEM | 4.0% |
| Genuine long context | 2.0% |
| Indic multilingual | 8.0% |

These floors use 30% of each batch. The scheduler assigns the remaining 70% by stage mixture and OPUS. Run static license, language, PII, secret, and contamination filters first. OPUS scores the residual pool with optimizer-induced utility, the AdamW state, preconditioning, a redundancy penalty, and Boltzmann sampling. Use a 30M-token Bench-Proxy that is disjoint from evaluation data, decontaminated with exact, normalized, and MinHash checks, and split into a held-out slice. Fill floors first, log utility and cluster statistics, and disable OPUS for the reserve. Do not rank by raw loss alone. Use `utility(x) ~= - <proxy_gradient, optimizer_preconditioner * gradient(x)>`.

For agent traces, keep raw observations in context but give them a zero loss mask. Train on the user task, authorization scope, assistant plan, tool name and JSON arguments, interpretation, and final action. Each trace needs a fixture or repository hash, schema version, command allowlist, exit status, tests, and replay result. Never tokenize secrets, credentials, private logs, or personal data.

## Reserve and data gate

Hold back 90B for annealing: 30B verified Indic native, 20B verified code and mathematics, 20B executable agent/tool traces, 10B genuine long-context data, and 10B general web and knowledge. Release it only after Quality cooldown validation; use 32K/64K examples, linear learning-rate decay to zero, and final-checkpoint averaging. No unverified or unchecked synthetic data enters the reserve.

A source enters the full manifest only when every admitted record has source ID, license/provenance, retrieval date, V5 token count, content hash, data category, tier, language, and split assignment; exact normalized deduplication is complete; MinHash clusters and owners are recorded; train/evaluation overlap is below 0.1%; and no private key, credential, access token, or unredacted private log remains. PII redaction is sampled manually in every language and source family. Agent traces must replay with valid schemas and expected outcomes. Generated Indic data must pass language ID, repetition, and factuality checks, with an independent audit averaging at least 4/5 for grammar, faithfulness, factuality, and readability. Every tier and protected data category must meet its post-dedup target within four exposures or have an approved collection or generation plan. Each source must appear in the 1B proxy before contributing a material share to the 3B or 40B run.

The full run is reviewed only after this data-gating threshold is met. A clean but undersized source is a supply failure for its planned share, not a quality success.

## Proxy decision

Run the same V5 tokenizer, architecture family, optimizer schedule, sequence curriculum, manifest schema, cleaning gates, and OPUS implementation at both scales:

| Run | Budget | Seeds |
| --- | --- | ---: |
| 1B proxy | 1B model, 100-150B tokens | 2 |
| 3B proxy | 3B model, 300-450B tokens | 3 |

Evaluate four arms: the matched general-web baseline; an Indic-low ablation that replaces the 14% Indic target with the baseline mixture; a code/agent-low ablation that moves 10 percentage points from code plus agentic data to general web; and the target V5 mixture. All arms use the same total loss-bearing tokens, stage volumes, and static quality gates. Run a random-residual selector control for the target arm to separate mixture value from OPUS value. Record the weighted score, every data category's primary metric, held-out web loss, throughput, context-length loss, Indic loss by language, and the full tool-execution check set.

Promote the target only if all gates pass:

- Weighted score improves at least 2% over baseline at 1B and 4% at 3B.
- The paired-bootstrap 95% confidence interval lower bound clears each threshold.
- General held-out loss regression is no worse than 1% at 1B or 0.5% at 3B.
- The Spearman correlation between 1B and 3B arm rankings is at least 0.80.
- Train/evaluation overlap remains below 0.1% under exact and MinHash checks.
- Every replayed agentic/tool trace has valid calls, correct schemas, and the expected test or verifier result.
- No protected floor is violated in any batch.

Any failed gate rejects the mixture. The next experiment changes one variable at a time: data-category share, Indic tier ratio, OPUS temperature, or cluster penalty. No full run proceeds on arithmetic, a point estimate, or an unverified supply assumption.

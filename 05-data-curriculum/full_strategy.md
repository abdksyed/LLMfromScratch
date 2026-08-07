# V5 mixture and curriculum strategy

This is the training plan for a dense 40B V5 model. It covers base pretraining and
midtraining. Instruction tuning, RL, preference optimization, and retrieval at
deployment are outside this budget.

All token counts use the finalized V5 tokenizer. The 6.00T budget covers
loss-bearing tokens. We track context tokens with a zero loss mask separately
because they still use memory and FLOPs.

## 1. Hypothesis and fixed assumptions

We expect a 40B dense model trained for **6.00T loss-bearing V5 tokens** to
outperform a general-web baseline on coding, repository repair, agentic tool use,
reasoning, and the ten selected Indic languages while retaining general knowledge.

We reject that hypothesis if the target mixture fails the proxy gates in Section
12. Passing the arithmetic check is not enough to approve the full run.


| Item                         | V5 decision                                              | Reason for fixing it now                                                                                 |
| ---------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Model                        | 40B dense decoder-only Transformer                       | The model being planned.                                                                                 |
| Loss-bearing budget          | 6.00T tokens                                             | 150 tokens/parameter, including a measurable cooldown.                                                   |
| Tokens per parameter         | 150                                                      | A planning prior, not a law. Proxy runs test whether extra domain tokens pay for their opportunity cost. |
| Normal training budget       | 5.91T tokens                                             | 98.5% of the total; consumed before the held-out reserve.                                                |
| Anneal reserve               | 90B tokens                                               | 1.5% of the total; never exposed to OPUS during the main run.                                            |
| Native context target        | 64K tokens                                               | Supports repository and multi-step tool contexts without making every sequence 256K or 1M.               |
| Tokenizer                    | The finalized V5 tokenizer                               | Every stock estimate is provisional until retokenized with this tokenizer.                               |
| Primary optimizer assumption | AdamW-like optimizer with the production optimizer state | OPUS scores must use the optimizer-induced update geometry rather than raw loss alone.                   |


For accounting, each example belongs to one lane. A 64K repository example assigned
to the long-context lane is counted there, not again in the code lane. A source may
be repeated across epochs when the exposure cap permits it, but repeated tokens are
not counted as new unique stock.

### Why this mixture is the hypothesis

Coding and agentic behavior are the main product differentiators. Code-only training
would leave the model without the language and factual knowledge needed to
understand real issues. General web and knowledge gets 31.5%, and code gets 22%.
Agentic and tool-use data share one 12% lane because tool calls are part of the
agentic path. Reasoning, long context, and Indic stay separate so we can measure
their effects.

## 2. Benchmark-to-lane contract

We set the mixture from the evaluations backward. Remove benchmark data, close
paraphrases, and known public solution text from training before making any proxy or
full-run count. Pin benchmark versions and harnesses in a manifest; a benchmark
name alone does not define a reproducible metric.


| Lane                               | Capability claim                                                        | Evaluation family                                                                                 | Primary signal                                                          |
| ---------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| General web and knowledge          | broad language modeling, factual recall, common sense                   | MMLU, ARC-Challenge, held-out web loss                                                            | normalized average accuracy plus held-out perplexity                    |
| Source code and code documentation | completion, code understanding, API and documentation grounding         | LiveCodeBench, HumanEval-style held-out tests, repository-level code QA                           | pass@1 and execution-verified correctness                               |
| Agentic and tool use               | issue decomposition, shell control, iterative repair, API/browser calls | SWE-bench Verified, Terminal-Bench, ToolBench, WebArena, BrowserComp                              | resolved-task rate, task success, valid-call rate, argument correctness |
| Reasoning and STEM                 | mathematical and scientific multi-step reasoning                        | AIME, MATH, GSM8K, Frontier Math where licensing and release status allow                         | exact answer or verifier success, with a fixed test-time budget         |
| Genuine long-context documents     | retrieval and reasoning over long source context                        | RULER, MRCR, needle retrieval, repository-level long-context QA                                   | length-stratified accuracy and retrieval recall                         |
| Indic multilingual                 | native language modeling across ten selected languages                  | IndicXTREME on available selected-language subsets, per-language perplexity, worst-language score | macro score, worst-language score, and script-specific loss             |


The normalized proxy score used for promotion is:

```text
20% general + 25% code + 25% agentic/tool + 15% reasoning
+ 5% long-context + 10% Indic
```

For agentic/tool-use evaluation, also check execution and schemas. A judge-model
score alone is not enough.

## 3. Aggregate token budget

These shares use the full 6.00T budget. The anneal reserve has its own lane so a
normal data loader cannot spend it by accident.


| Lane                               | Share of 6.00T | Loss-bearing tokens |
| ---------------------------------- | -------------- | ------------------- |
| General web and knowledge          | 31.5%          | 1.890T              |
| Source code and code documentation | 22.0%          | 1.320T              |
| Agentic and tool use               | 12.0%          | 720B                |
| Reasoning and STEM                 | 14.0%          | 840B                |
| Genuine long-context documents     | 5.0%           | 300B                |
| Indic multilingual                 | 14.0%          | 840B                |
| Anneal reserve                     | 1.5%           | 90B                 |
| **Total**                          | **100.0%**     | **6.000T**          |


The accounting separates source code from agentic/tool-use traces. A Python file
in The Stack v2 is code; a terminal transcript that edits and tests a Python
repository is agentic data. API documentation and API/browser trajectories belong
to the same agentic/tool-use lane. A multi-page repository context counts as
long-context only when its accounting label is long-context and it is not counted
again in code or agentic data.

### Lane source map


| Lane                 | Candidate source families                                                                                                                                                                | V5 treatment                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| General              | FineWeb, FineWeb-Edu, DCLM-baseline, Dolma, licensed encyclopedic and reference text                                                                                                     | Filter, deduplicate, decontaminate, then reserve educational and factual subsets for later stages.                                         |
| Code                 | The Stack v2, GitHub pull requests and issues, notebooks, code documentation, permissively licensed repositories                                                                         | Separate source files, docs, tests, and generated files; enforce license, opt-out, malware, secret, and repository-level split checks.     |
| Agentic and tool use | verified SWE-bench and Terminal-Bench trajectories, executable shell/file-edit traces, ToolBench/API-bank-style calls, WebArena/BrowserComp trajectories, high-quality in-house rollouts | Keep environment observations in context but mask raw observations from the loss. Require replay, schema validation, or test verification. |
| Reasoning            | ProofPile-style mathematical text, science and textbook material, verified solutions, theorem/proof data, carefully generated problems                                                   | Hold out all public benchmark questions and use symbolic, executable, or expert verification where possible.                               |
| Long context         | genuinely long books, legal/technical documents, repositories, issue threads, and tool sessions                                                                                          | Assign each sequence to this lane before mixing. Do not create length by padding a short example.                                          |
| Indic                | Sangraha, IndicCorp v2, BhashaKritika, licensed native web/books/news/reference text, verified translations, audited generation                                                          | Keep the ten languages and the native, unverified, translated, and synthetic tiers distinct in the manifest and loss report.               |




## 4. Data sources and available volume

These are headline sizes for the candidate datasets. Before they become training
tokens, the V5 pipeline retokenizes, deduplicates, and filters them.


| Source        | Available volume                                                          | How V5 uses it                                                                                        |
| ------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| DCLM-Pool     | 240T Common Crawl tokens                                                  | A raw web reservoir. It is not 240T clean V5 tokens.                                                  |
| DCLM-baseline | 3.8T high-quality tokens                                                  | A filtered general-web source; the DCLM paper also reports a 4.1T code/math combination.              |
| FineWeb       | 15T tokens in the paper; more than 18.5T on the current Hub card          | Pin one release and count it once.                                                                    |
| FineWeb-Edu   | 1.3T tokens in the original release; 5.4T in the separate score-2 release | Use for educational and reasoning-heavy general text after deduplication.                             |
| Dolma         | 3T tokens                                                                 | Broad English web, academic, code, books, social, and encyclopedic data.                              |
| The Stack v2  | 900B+ unique code tokens across 619 programming languages                 | Main code source after license, opt-out, generated-file, secret, and repository-level filtering.      |
| ToolBench     | 16,464 APIs and about 126K instruction/query examples                     | Serialize calls and retokenize them with V5; use together with WebArena and BrowserComp trajectories. |
| Sangraha      | 251B tokens across 22 languages                                           | V5 keeps the ten-language high-volume subset below.                                                   |
| IndicCorp v2  | 20.9B tokens                                                              | Adds native Indic web data and overlaps with Sangraha, so deduplicate across sources.                 |
| BhashaKritika | 540B synthetic tokens across ten languages                                | Supplies the same ten-language set selected for V5; use only filtered and audited data.               |


The Stack v2 report describes repository deduplication, license processing, PII
redaction, malicious-code filtering, and a 4K-to-16K context progression. V5 applies
the same controls, but it does not treat a headline dataset size as clean V5 stock.

### Reported Indic inventory before cross-source deduplication

This inventory covers only the ten selected languages. It remains an upper bound
because Sangraha, IndicCorp v2, and BhashaKritika can contain the same source
material or translations of the same material.


| Indic tier                                        | Reported stock | Components in the inventory                                                                | Target draw               | Target / reported stock          |
| ------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------ | ------------------------- | -------------------------------- |
| Verified native                                   | 56.7460B       | Selected ten from Sangraha verified 44.1500B + selected ten from IndicCorp v2 12.5960B     | 222.6B                    | 3.92x                            |
| Unverified native                                 | 22.2790B       | Selected ten from Sangraha unverified                                                      | 63B                       | 2.83x                            |
| Translated/derived                                | 172.7128B      | Selected ten from Sangraha synthetic 117.4528B + filtered BhashaKritika translation 55.26B | 252B                      | 1.46x                            |
| Synthetic native/generated, excluding translation | 488.02B        | Filtered BhashaKritika non-translation data                                                | 302.4B                    | 0.62x                            |
| **Selected-ten upper bound**                      | **739.7578B**  | **not cross-source deduplicated**                                                          | **840B total Indic draw** | **declared repetition required** |


The selected-ten upper bound is smaller than the 840B target. We set the verified
native target to 222.6B to stay below the four-exposure cap, down from the initial
231B draft. The remaining volume comes from audited synthetic-native data.
Cross-source deduplication may reduce the available stock further, so the manifest
lists all repeats and generated tokens explicitly.

### Indic tier split

The exact split of the 840B Indic lane is:


| Tier                       | Share of Indic lane | Tokens   |
| -------------------------- | ------------------- | -------- |
| Verified native            | 26.5%               | 222.6B   |
| Unverified native          | 7.5%                | 63B      |
| Translated/derived         | 30.0%               | 252B     |
| Synthetic native/generated | 36.0%               | 302.4B   |
| **Total**                  | **100.0%**          | **840B** |


V5 uses the ten languages covered by BhashaKritika and the high-volume portions of
Sangraha and IndicCorp v2. The training target excludes Urdu, Nepali, Assamese,
Sanskrit, Sindhi, Kashmiri, Konkani, Maithili, Manipuri, Bodo, Dogri, and Santali.
The shares below reflect data availability, not population.


| Language                      | Share of Indic lane | Planned tokens |
| ----------------------------- | ------------------- | -------------- |
| Hindi                         | 25.0%               | 210.0B         |
| Bengali                       | 18.0%               | 151.2B         |
| Tamil                         | 15.0%               | 126.0B         |
| Marathi                       | 10.0%               | 84.0B          |
| Gujarati                      | 8.0%                | 67.2B          |
| Malayalam                     | 7.0%                | 58.8B          |
| Kannada                       | 6.0%                | 50.4B          |
| Punjabi                       | 5.0%                | 42.0B          |
| Telugu                        | 4.0%                | 33.6B          |
| Odia (Oriya in BhashaKritika) | 2.0%                | 16.8B          |
| **Total**                     | **100.0%**          | **840.0B**     |


Synthetic native data is not treated as equivalent to native human text. Every
generated item carries source document, generator, prompt, model version,
generation seed, language-ID result, quality filters, and verification status.
Translation is a separate tier even when the output reads fluently because it has
different cultural, lexical, and contamination risks.

## 5. Repetition and source accounting

For each source and tier, the manifest records:

```text
unique_clean_v5_tokens
planned_loss_bearing_tokens
planned_exposure = planned_loss_bearing_tokens / unique_clean_v5_tokens
```

The default maximum exposure for an exact or near-duplicate cluster is **4.0**.
The loader may revisit a source at different stages, but it must log the cluster ID
and cumulative exposure. A proxy can justify a lower or higher cap for a specific
source family. No source starts above four exposures by assumption.

The selected-ten stock implies these approximate exposures:

- Verified native text needs roughly 3.92 exposures before cross-source
deduplication.
- Unverified native text needs roughly 2.83 exposures.
- Translated/derived text needs roughly 1.46 exposures.
- Synthetic non-translation stock is sampled at roughly 0.62 of its reported
volume, subject to quality and language balancing.

Run the supply gate again after exact and MinHash deduplication. If a tier cannot
cover its target at no more than four exposures, V5 either generates a new,
independently verified pool or changes the tier allocation in the proxy. The
manifest never reports repeated tokens as unique data.

Repository-level deduplication is mandatory for code. A fork, vendored copy,
generated lockfile, or mirrored documentation page should inherit a cluster ID.
For web and Indic sources, retain both exact normalized text hashes and 8-shingle
MinHash clusters. This lets evaluators distinguish removing a duplicate from
downweighting a related document.

## 6. Protected floors and OPUS selection

Run static license, language, PII, secret, and contamination filters before the
dynamic selector. OPUS may then choose from the residual pool, but every selector
batch must retain these minimum fractions:


| Lane                               | Protected minimum of every selector batch |
| ---------------------------------- | ----------------------------------------- |
| General web and knowledge          | 1.0%                                      |
| Source code and code documentation | 10.0%                                     |
| Agentic and tool use               | 5.0%                                      |
| Reasoning and STEM                 | 4.0%                                      |
| Genuine long-context documents     | 2.0%                                      |
| Indic multilingual                 | 8.0%                                      |


Together the floors account for 30%; they are minimums, not the aggregate target.
The scheduler fills the remaining 70% according to the stage mixture and selector.
Keep the Indic and agentic/tool floors because an English-heavy loss filter can mark
both types of data as atypical. The code floor stops a high-loss general-web stream
from starving the core capability.

### OPUS-style residual selector

This selector follows OPUS, Optimizer-induced Projected Utility Selection, rather
than simply keeping the highest-loss samples.

1. Construct a **30M-token Bench-Proxy** from in-distribution training material
  that is aligned with the benchmarks but contains no public evaluation item or
   close paraphrase. Fix its lane labels and target weights for the run.
2. At checkpoints, compute a stable proxy descent direction with a small ghost
  model or ghost forward/backward pass. CountSketch projections avoid materializing
   a full gradient for every candidate.
3. Score each candidate by the approximate improvement it would provide in the
  optimizer-induced update space. One useful sign convention is:

Include the AdamW state, momentum, and preconditioning in the score. Raw
perplexity and raw gradient similarity are auxiliary diagnostics only.
4. Apply an in-step redundancy penalty and Boltzmann sampling so that a single
   high-scoring cluster does not collapse the stream's diversity.
5. Fill protected floors first, then run OPUS only over the residual pool. Log
   candidate counts, selected counts, utility distributions, lane shares, source
   clusters, and cumulative exposures.

Do not run the selector on the anneal reserve. If OPUS cannot find a candidate for
a protected floor, stop the run and repair the manifest. The loader must not
substitute general web text and continue.

## 7. Curriculum and stage arithmetic

The stage percentages below use the full 6.00T budget. Lane percentages within a
row sum to 100% of that stage. Hold the reserve until the final row.


| Stage            | Share of full budget | Volume | General  | Code     | Agentic/tool | Reasoning | Long     | Indic    |
| ---------------- | -------------------- | ------ | -------- | -------- | ------------ | --------- | -------- | -------- |
| Foundation       | 20.0%                | 1.200T | 44.5     | 18.0     | 8.0          | 12.0      | 3.0      | 14.5     |
| Capability       | 35.0%                | 2.100T | 35.0     | 22.0     | 10.0         | 15.0      | 3.0      | 15.0     |
| Specialization   | 30.0%                | 1.800T | 24.0     | 25.0     | 15.0         | 16.0      | 5.0      | 15.0     |
| Consolidation    | 10.0%                | 600B   | 26.45    | 23.6     | 16.8         | 9.9       | 13.25    | 10.0     |
| Quality cooldown | 3.5%                 | 210B   | 14.43    | 24.0     | 20.57        | 16.0      | 15.0     | 10.0     |
| Anneal reserve   | 1.5%                 | 90B    | held out | held out | held out     | held out  | held out | held out |


The five active rows total 98.5%, and the reserve is 1.5%. The table shows stage
percentages to two decimal places. Using those displayed values, the weighted lane
reconciliation rounds to:

```text
general 31.5%, code 22.0%, agentic/tool 12.0%, reasoning 14.0%,
long 5.0%, Indic 14.0% = 98.5% active budget
```

Adding the reserve gives the aggregate table in Section 3.

### Stage intent

**Foundation, 1.2T.** Establish language, tokenization, factual coverage, and basic
code syntax. Most examples are low or medium difficulty. General web is large
enough to avoid a brittle specialist model. Code and Indic data appear from the
first stage instead of being introduced late.

**Capability, 2.1T.** Shift from broad text toward educational web,
code/documentation, basic tool schemas, and verified solutions. Increase medium and
high difficulty gradually. Long-context examples remain limited, but start to
include genuine 8K and 16K documents.

**Specialization, 1.8T.** Focus on repository repair, multi-step reasoning, tool
chains, and Indic generation. Agentic/tool data rises to 15% of this stage and code
to 25%, while general web remains 24% so the model keeps factual grounding.

**Consolidation, 600B.** Revisit difficult and long examples, resolve contradictory
patterns, and improve general-web grounding and agentic/tool trace quality. The
13.25% long-context share provides the first sustained 32K/64K exposure.

**Quality cooldown, 210B.** Use the highest-quality verified material from all
lanes, with high and ultra difficulty dominant. This stage remains part of the
active 5.91T and is separate from the held-out 90B reserve.

## 8. Difficulty and reasoning-length bands

Difficulty bands describe the expected continuation length or explicit reasoning
trace. They do not require every answer to be verbose; a short correct answer stays
short. The bands overlap so a stage transition does not create a sudden gradient or
length discontinuity.


| Band   | Target generated length | Concrete examples                                                                                                                                         |
| ------ | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Low    | 32-128 tokens           | Compute `84 / 7` and give `12`; complete one missing function; make one API call with a known schema.                                                     |
| Medium | 128-512 tokens          | Solve a two-equation word problem; repair a single-file bug and explain the test; make two dependent API calls.                                           |
| High   | 512-2,048 tokens        | Give an AIME-style derivation; patch several files and run tests; perform four to eight dependent tool calls.                                             |
| Ultra  | 2,048-16,384+ tokens    | Produce a proof or Frontier Math-style derivation; execute a SWE-bench repository repair loop; plan and complete a long-horizon terminal or browser task. |


The stage proportions overlap deliberately:


| Stage            | Low | Medium | High | Ultra |
| ---------------- | --- | ------ | ---- | ----- |
| Foundation       | 70% | 25%    | 5%   | 0%    |
| Capability       | 45% | 35%    | 17%  | 3%    |
| Specialization   | 20% | 35%    | 30%  | 15%   |
| Consolidation    | 10% | 25%    | 35%  | 30%   |
| Quality cooldown | 5%  | 15%    | 30%  | 50%   |


Apply these proportions to each lane where the band is meaningful. A general factual
paragraph is not forced into an ultra reasoning trace, and an agent trajectory
normally has a longer target than a code completion.

## 9. Sequence-length curriculum

The sequence-length schedule describes actual examples, not padding. Short
documents may be packed to a shorter boundary; padding a 200-token example with
filler does not make it a long-context example.


| Stage            | Sequence-length mixture by token mass |
| ---------------- | ------------------------------------- |
| Foundation       | 4K dominant: 80% 4K, 20% 8K           |
| Capability       | 60% 4K, 30% 8K, 10% 16K               |
| Specialization   | 20% 4K, 45% 8K, 25% 16K, 10% 32K      |
| Consolidation    | 15% 8K, 35% 16K, 35% 32K, 15% 64K     |
| Quality cooldown | 10% 16K, 35% 32K, 55% 64K             |
| Anneal reserve   | 30% 32K, 70% 64K                      |


Long examples come from books, legal and technical documents, whole repository
contexts, issue threads, and complete tool sessions. Split a sequence that is too
long for the current stage only at a document-aware boundary, and retain its
document ID for deduplication. The length schedule draws on the 4K-to-16K
progression reported for StarCoder2 and on the long-context data-engineering
literature. The 64K target is a V5 design choice, not a claim that a 64K model is
free to train.

For every stage, the run records both:

```text
loss_bearing_v5_tokens
all_context_tokens_including_zero_loss_observations
```

This keeps masked tool output from making the compute budget look smaller than it
is.

## 10. Agent trajectory serialization and loss mask

Serialize agentic data with stable role and tool-schema delimiters. Train on
explicit written rationale or action traces that are licensed for training. This
does not mean extracting private hidden chain-of-thought from users or models.


| Segment                                              | Kept in context | Loss mask |
| ---------------------------------------------------- | --------------- | --------- |
| User issue, task, and authorization scope            | yes             | 1         |
| Assistant written plan or reasoning trace            | yes             | 1         |
| Tool-call name and JSON arguments                    | yes             | 1         |
| Raw tool result or file/environment observation      | yes             | 0         |
| Shell logs, progress output, and returned error text | yes             | 0         |
| Assistant interpretation after observing the result  | yes             | 1         |
| Final answer, patch summary, and next action         | yes             | 1         |


Keep raw observations in context because conditioning on them is the capability we
want. They do not count as loss-bearing tokens, but they use sequence length,
memory, and compute. This matters especially for the 720B agentic/tool target.

Each agent trace must have an execution record: repository commit or fixture hash,
tool schema version, command allowlist, exit status, tests run, and whether the
final patch solves the task. Use observation dropout during training so the model
does not depend on one exact log shape. Never pass secrets, access tokens, private
logs, personal data, or unredacted credentials to the tokenizer.

## 11. Anneal reserve

Keep the 90B reserve out of all normal manifests. Release it only after the quality
cooldown passes its validation checkpoint.


| Reserve component                                          | Tokens  | Admission rule                                                                       |
| ---------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------ |
| Verified Indic native or independently verified Indic data | 30B     | Language and factuality checks pass; no unverified tier.                             |
| Verified code and mathematics                              | 20B     | Executable tests, symbolic checks, or expert verification pass.                      |
| Executable agent and tool traces                           | 20B     | Replay succeeds, tool schemas validate, and expected execution results are recorded. |
| Genuine long-context documents and traces                  | 10B     | 32K/64K complete contexts; no padding-only examples.                                 |
| General web and knowledge                                  | 10B     | Source attribution, privacy redaction, and contamination checks pass.                |
| **Total**                                                  | **90B** | **No unverified or unchecked synthetic data.**                                       |


The reserve uses 32K/64K examples, a linear learning-rate decay to zero, and
checkpoint averaging over the final validated checkpoints. Do not resample with
OPUS inside the reserve. Reject a candidate that is high-loss but unverified. Data
quality matters most late in training, when behavior can become permanent.

## 12. Proxy experiment design

Promote the mixture only after matched small runs.

### Runs


| Run      | Model and token budget          | Seeds | Purpose                                                                  |
| -------- | ------------------------------- | ----- | ------------------------------------------------------------------------ |
| 1B proxy | 1B dense model, 100-150B tokens | 2     | Cheap ranking of lane shares, tier ratios, and curriculum shape.         |
| 3B proxy | 3B dense model, 300-450B tokens | 3     | Check whether the 1B ranking survives a larger model and longer context. |


Both proxies use the V5 tokenizer, the same architecture family, optimizer
schedule, sequence curriculum, data manifest schema, cleaning gates, and OPUS
implementation as the intended run. A 1B and 3B run with different tokenizers or
stage definitions is not a scaling test.

### Arms

1. **Matched general-web baseline:** general web remains the dominant source and all
  arms receive the same total loss-bearing tokens, stage volumes, and static
   quality gates.
2. **Indic-low ablation:** replace the 14% Indic target with the baseline mixture,
  while retaining the same tokenizer and language evaluation.
3. **Code/agent-low ablation:** reduce code plus agentic data by 10 percentage
  points and assign those tokens to general web, holding everything else fixed.
4. **Target V5 mixture:** the tables above, with protected floors and
  OPUS residual selection.

For the target arm, also run a selector control with random residual sampling. This
separates the value of the mixture from the value of dynamic selection.

### Measurements

For each arm, record the weighted score from Section 2, each lane's primary
metric, held-out web loss, training throughput, context-length loss, language-wise
Indic loss, and the full data-quality and tool-execution check set. Evaluate code
and agent benchmarks with contamination-controlled versions and execution, not
just judge-model scores.

The OPUS Bench-Proxy is 30M tokens and is disjoint from all evaluation data. Build
it by retrieving benchmark-aligned examples from the training corpus, then run
exact, normalized, and MinHash decontamination against the evaluation sets. Give
the proxy its own held-out slice so selector overfitting is visible.

### Promotion and rejection gates

Promote the target arm only if every condition below holds:

- At 1B, the weighted target score improves by at least 2% over the matched
general-web baseline.
- At 3B, the weighted target score improves by at least 4%.
- The paired-bootstrap 95% confidence interval lower bound clears the corresponding improvement threshold; a point estimate is not enough.
- General held-out loss regression is no worse than 1% at 1B and 0.5% at 3B.
- The Spearman correlation between 1B and 3B rankings of mixture arms is at least
0.80.
- Train/evaluation overlap is below 0.1% under the pinned exact and MinHash
contamination tests.
- Every replayed agentic/tool trace has valid calls, correct schemas, and the
expected test or verifier outcome.
- No protected floor is violated in any batch; a violation is a run failure, not
an average that can be smoothed away.

If an arm fails, change one quantity at a time in the next experiment: first a lane
share, then an Indic tier ratio, then an OPUS temperature or cluster penalty. Do not
change several quantities at once and call the result a curriculum finding.

## 13. Cleaning backlog and data-gating threshold

The current repository cleaning pilot shows that the pipeline runs; it does not
show that the full agentic/tool target exists. Its report records 11,463 input
records, 1,212 output records, and 53,569,839 output tokens after verifier,
rollout, language, heuristic, PII, decontamination, and MinHash stages. That is
only about 0.007% of the planned 720B agentic/tool lane, so treat it as a process
fixture, not as an inventory to extrapolate.

The cumulative cleaning work is prioritized toward the starved slots:

1. Generate and replay executable agentic terminal/SWE trajectories, with separate repository and tool-session IDs.
2. Expand verified-native data for the ten selected Indic languages and audit the existing Sangraha/IndicCorp overlap.
3. Collect genuine long documents and repository contexts, then assign them exclusively to the long-context lane.
4. Add code documentation, tests, pull requests, and issue discussions after license and repository-level deduplication.
5. Only then fill remaining short general and synthetic pools.



### Admission threshold for a full-scale source

A source family enters the full manifest only after every check below has a recorded
result:

- 100% of admitted records have source ID, license/provenance, retrieval date,
V5-tokenizer count, content hash, lane, tier, language, and split assignment.
- Exact normalized deduplication is complete; MinHash clusters are retained and
cross-source duplicates are removed or assigned one owner. Treat the current
cleaning pipeline's 8-shingle, 128-permutation, approximately 0.67 similarity
setting as a calibration starting point, not a universal setting.
- Public benchmark overlap is below 0.1% after exact, normalized, and near-duplicate
checks. Suspected matches are quarantined rather than counted as clean.
- No confirmed private key, credential, access token, or unredacted private log
remains. PII redaction is sampled manually in every language and source family.
- Agent traces meet the replay/verifier threshold; the reserve requires a perfect
replay result. Generated Indic samples pass language ID, repetition, and
factuality checks and have an independent audit sample with mean quality at
least 4/5 on grammar, faithfulness, factuality, and readability.
- PII, private keys, credentials, access tokens, and private logs are handled as
data-hygiene exclusions; these checks do not create a separate training lane.
- Every Indic tier and every protected lane can meet its target at no more than
four cumulative exposures after deduplication, or has an explicit approved
generation/collection plan.
- The source is represented in the 1B proxy before it can contribute material
share to the 3B or 40B run.

Apply the gate cumulatively. A clean source that lacks enough unique tokens is not a
quality failure, but it does invalidate the supply assumption for its planned
share. The schedule must expose that failure early.

## 14. Model-report audit and what is transferable

Model reports provide useful evidence about training practice, but their token
counts cannot be compared directly. MoE reports distinguish total from active
parameters, model cards use different tokenizers, and some reports include
continued pretraining or post-training in a headline number. That is why V5 uses
its own manifest instead of copying a competitor's number.


| Report or release       | Publicly reported fact                                                                                                                     | V5 implication                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| OLMo 3                  | Open report describing staged pretraining, midtraining, and annealing                                                                      | Supports explicit stage transitions and open manifests; the exact mixture is not copied.                                    |
| Tulu 3                  | Post-training recipe and evaluations, not a base-pretraining token budget                                                                  | Useful for downstream evaluation design, excluded from the 6.00T base budget.                                               |
| Llama 3                 | More than 15T pretraining tokens are reported for the family                                                                               | Confirms that large dense models can benefit from long training, but its private mixture is not a source inventory.         |
| Qwen2.5                 | Reports up to about 18T pretraining tokens for the family                                                                                  | A scale reference only; tokenizer and post-training boundaries differ.                                                      |
| Qwen3                   | Reports about 36T pretraining tokens                                                                                                       | Reinforces the value of broad, high-quality, long-running training, not a direct 40B target.                                |
| DeepSeek-V3             | Reports 671B total, 37B active, and 14.8T training tokens                                                                                  | Shows why active and total MoE parameters must not be compared with dense 40B arithmetic.                                   |
| DeepSeek-R1             | Primarily a reasoning post-training and RL result                                                                                          | Evidence for verifiable reasoning, not a base-corpus token count.                                                           |
| Gemma 3 and Phi-4       | Strong public recipes with incomplete directly comparable mixture disclosure                                                               | Use their evaluations and filtering lessons, not inferred token totals.                                                     |
| StarCoder2/The Stack v2 | 900B+ unique code tokens, 619 languages; StarCoder2 training used 3.3-4.3T tokens and 4K then 16K context                                  | Direct support for repository deduplication, code lane sizing, and gradual context expansion.                               |
| DeepSeek-V4             | Pro is 1.6T total/49B active; Flash is 284B total/13B active; the shared release card says more than 32T pretraining tokens and 1M context | A current MoE and long-context reference. Do not turn a shared `>32T` disclosure into a false Pro-versus-Flash token split. |
| Gemma 4 31B             | About 30.7B dense parameters; no pretraining-token total disclosed                                                                         | Architecture and 256K context are reference points, not a token target.                                                     |
| Gemma 4 26B A4B         | About 25.2B total and 3.8B active parameters in the model card; no pretraining-token total disclosed                                       | Keep total versus active parameters explicit; do not infer data volume from the name.                                       |
| Mistral Small 4         | 119B total, 6.5B activated, 256K context; no training-token total disclosed                                                                | Another MoE/context reference with no transferable data budget.                                                             |
| Inkling Small           | 276B total, 12B active; no Small-specific token total disclosed                                                                            | Do not transfer a parent Inkling 45T figure to Inkling Small.                                                               |
| Qwen3.6 27B and 35B-A3B | Public cards report 27B dense and 35B total/3B active, respectively, with 262K native context; no reliable training-token total disclosed  | Useful evidence for long-context and agent evaluation, not for V5 token accounting.                                         |


V5 uses 64K, below the 256K and 1M examples in this audit. This lets the proxy test
long-context data quality before the full run pays the cost of making every
late-stage sequence large.

## 15. Risks and explicit abort conditions

**Indic repetition risk.** The verified-native tier cannot provide 222.6B unique
tokens. V5 therefore explicitly allows repeated exposures plus new collection or
generation, with a four-exposure cap and a per-language worst-case metric. If the
worst-language score falls while the macro score rises, the macro score does not
pass the gate.

**Synthetic collapse.** BhashaKritika-scale synthetic data may help in an Indic
setting with limited data, but it can also repeat styles, facts, and generator
quirks.
Require generator IDs, semantic cluster caps, teacher disagreement, source
grounding, and human audit. Synthetic data cannot enter the reserve merely because
its perplexity is low.

**OPUS starvation.** Apply protected floors before the selector and measure them
per batch. A high general-web utility score cannot spend Indic or agentic tokens.

**Long-context illusion.** Padding short examples or repeating one document does
not establish retrieval. Promotion requires length-stratified RULER/MRCR and
repository QA improvements on unseen documents.

**Agent trace contamination and replay errors.** Raw tool output is masked from
loss and secrets are removed, but environment observations still condition the
model. Replay checks must use versioned fixtures and validate tool schemas, command
outputs, exit statuses, and test results. Reject any trace with an invalid call
sequence or incorrect result, regardless of benchmark score.

**Supply shortfall.** If a lane cannot satisfy its post-dedup target at four
exposures, the target mixture is not ready. Use a proxy-tested reallocation or a
documented new source. Do not silently count the same records again.

## 16. References

Paper links use arXiv IDs. Dataset links point to the release pages checked for the
inventory audit. Release descriptions can change, so the V5 manifest also stores a
retrieval date and commit or revision ID.

### Data, selection, and training practice

- [OLMo 3, arXiv:2512.13961](https://arxiv.org/abs/2512.13961)
- [Tulu 3, arXiv:2411.15124](https://arxiv.org/abs/2411.15124)
- [DCLM, arXiv:2406.11794](https://arxiv.org/abs/2406.11794)
- [FineWeb and FineWeb-Edu, arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
- [Dolma, arXiv:2402.00159](https://arxiv.org/abs/2402.00159)
- [StarCoder2 and The Stack v2, arXiv:2402.19173](https://arxiv.org/abs/2402.19173)
- [OPUS, arXiv:2602.05400](https://arxiv.org/abs/2602.05400)
- [FineWeb release](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [FineWeb-Edu release](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Dolma release](https://huggingface.co/datasets/allenai/dolma)
- [The Stack v2 release](https://huggingface.co/datasets/bigcode/the-stack-v2)



### Code, tools, reasoning, and long context

- [ToolLLM and ToolBench, arXiv:2307.16789](https://arxiv.org/abs/2307.16789)
- [SWE-bench, arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- [LiveCodeBench, arXiv:2403.07974](https://arxiv.org/abs/2403.07974)
- [Long-context data engineering, arXiv:2402.10171](https://arxiv.org/abs/2402.10171)



### Indic data

- [Sangraha and IndicLLMSuite, arXiv:2403.06350](https://arxiv.org/abs/2403.06350)
- [Sangraha release](https://huggingface.co/datasets/ai4bharat/sangraha)
- [IndicCorp v2, arXiv:2212.05409](https://arxiv.org/abs/2212.05409)
- [BhashaKritika, arXiv:2511.10338](https://arxiv.org/abs/2511.10338)
- [BhashaKritika release](https://huggingface.co/datasets/krutrim-ai-labs/BhashaKritika)



### Model reports and release cards

- [Llama 3, arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- [Phi-4, arXiv:2412.08905](https://arxiv.org/abs/2412.08905)
- [Qwen2.5, arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
- [Qwen3, arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- [DeepSeek-V3, arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1, arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- [DeepSeek-V4, arXiv:2606.19348](https://arxiv.org/abs/2606.19348)
- [DeepSeek-V4 Pro model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [DeepSeek-V4 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [Gemma 3, arXiv:2503.19786](https://arxiv.org/abs/2503.19786)
- [Gemma 4, arXiv:2607.02770](https://arxiv.org/abs/2607.02770)
- [Gemma 4 31B model card](https://huggingface.co/google/gemma-4-31B)
- [Mistral Small 4 model card](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)
- [Inkling Small model card](https://huggingface.co/thinkingmachines/Inkling-Small)
- [Qwen3.6 27B model card](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen3.6 35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

**Access note:** We checked the model-card and Hub inventory observations on
2026-08-03. When a release card gives only a family-wide or lower-bound token
count, this specification keeps that uncertainty visible instead of inventing a
per-model number.
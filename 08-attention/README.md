# Chronology of Attention Mechanisms (2014–2026)

A chronological index of 47 attention mechanisms from early recurrent alignment to modern frontier architectures. Each entry summarizes what the mechanism does and the primary engineering trade-off it introduces.

---

### 1. Bahdanau Additive Attention (Sep 2014)
* **Paper**: [Bahdanau et al., arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
* **What it does**: Dynamically aligns decoder steps to encoder hidden states using a learned feed-forward network instead of compressing the entire input into a single fixed vector.
* **Trade-off**: Evaluates sequentially during recurrent unrolling, preventing parallel training across the sequence length.

### 2. Luong Multiplicative Attention (Aug 2015)
* **Paper**: [Luong et al., arXiv:1508.04025](https://arxiv.org/abs/1508.04025)
* **What it does**: Replaces additive feed-forward alignment with faster bilinear dot-product operations and introduces local windowed attention.
* **Trade-off**: Still bounded by sequential recurrent unrolling and lacks variance scaling for high-dimensional hidden states.

### 3. Scaled Dot-Product & Multi-Head Attention (Jun 2017)
* **Paper**: [Vaswani et al., arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
* **What it does**: Eliminates recurrence completely by evaluating all pairwise token interactions in parallel across multiple independent representation subspaces.
* **Trade-off**: Incurs quadratic compute and memory cost during prefill and requires caching all past key-values during autoregressive generation.

### 4. Absolute Sinusoidal & Learned Positional Encodings (Jun 2017)
* **Paper**: [Vaswani et al., arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
* **What it does**: Adds fixed sinusoidal frequencies or learned coordinate vectors directly to token input embeddings to give permutation-invariant attention sequence order awareness.
* **Trade-off**: Additive coordinates contaminate semantic token embeddings and fail to extrapolate cleanly beyond the training context length.

### 5. Shaw Relative Positional Representations (Mar 2018)
* **Paper**: [Shaw et al., arXiv:1803.02155](https://arxiv.org/abs/1803.02155)
* **What it does**: Injects learned relative distance vectors into attention logits and value aggregations based on clipped token offsets.
* **Trade-off**: Adds tensor broadcasting overhead during logit computation and cannot distinguish distances beyond the clipping boundary.

### 6. Sparse Transformer (Apr 2019)
* **Paper**: [Child et al., arXiv:1904.10509](https://arxiv.org/abs/1904.10509)
* **What it does**: Factorizes the full attention matrix into alternating local neighborhood windows and strided periodic steps to cut prefill compute to sub-quadratic scaling.
* **Trade-off**: Uses fixed geometric patterns that cannot dynamically adapt to data, requiring custom block-sparse GPU kernels to achieve real speedups.

### 7. T5 Relative Position Bias (Oct 2019)
* **Paper**: [Raffel et al., arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
* **What it does**: Adds learned scalar bias terms directly to attention logits with logarithmic distance binning for long offsets.
* **Trade-off**: Does not compress the key-value cache or reduce quadratic prefill compute, and cannot extrapolate past its maximum distance bucket.

### 8. Multi-Query Attention / MQA (Nov 2019)
* **Paper**: [Noam Shazeer, arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
* **What it does**: Forces all query heads to share a single key head and single value head, drastically reducing memory bandwidth reads during autoregressive decoding.
* **Trade-off**: Reduces model capacity and degrades multi-head associative retrieval performance on complex reasoning tasks.

### 9. Reformer (Jan 2020)
* **Paper**: [Kitaev et al., arXiv:2001.04451](https://arxiv.org/abs/2001.04451)
* **What it does**: Uses Locality-Sensitive Hashing to group similar queries and keys into buckets and combines this with reversible layers to avoid storing backprop activations.
* **Trade-off**: Hashing is stochastic and produces irregular bucket sizes that lead to low GPU Tensor Core utilization.

### 10. Routing Transformer (Mar 2020)
* **Paper**: [Roy et al., arXiv:2003.05997](https://arxiv.org/abs/2003.05997)
* **What it does**: Clusters queries and keys dynamically on a unit hypersphere using spherical k-means to compute attention only within matching clusters.
* **Trade-off**: Online centroid updates introduce non-differentiable routing barriers and GPU synchronization stalls.

### 11. Longformer (Apr 2020)
* **Paper**: [Beltagy et al., arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
* **What it does**: Combines local sliding windows, dilated gaps, and task-specific global tokens to process long documents with linear complexity.
* **Trade-off**: Requires manually designating which tokens receive global attention, and non-global tokens lack direct single-layer cross-attention over long distances.

### 12. Linformer (Jun 2020)
* **Paper**: [Wang et al., arXiv:2006.04768](https://arxiv.org/abs/2006.04768)
* **What it does**: Projects the sequence length dimension of keys and values down to a fixed low rank using linear projection matrices.
* **Trade-off**: Binds the model to a predetermined maximum sequence length and cannot be applied straightforwardly to causal autoregressive generation.

### 13. Linear Attention via Kernel Feature Maps (Jun 2020)
* **Paper**: [Katharopoulos et al., arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
* **What it does**: Decomposes softmax normalization into kernel feature maps, allowing matrix multiplications to be reordered from quadratic to linear time with constant-memory recurrent decoding.
* **Trade-off**: Ungated additive state updates cause memory saturation and fail to match the sharp retrieval focus of exponential softmax attention.

### 14. BigBird (Jul 2020)
* **Paper**: [Zaheer et al., arXiv:2007.14062](https://arxiv.org/abs/2007.14062)
* **What it does**: Formulates sparse attention as a graph of local windows, random connections, and global tokens with theoretical proofs of universal sequence approximation.
* **Trade-off**: Random memory gathering is hardware-unfriendly on GPUs and complex to apply to causal autoregressive generation.

### 15. Performer / FAVOR+ (Sep 2020)
* **Paper**: [Choromanski et al., arXiv:2009.14794](https://arxiv.org/abs/2009.14794)
* **What it does**: Approximates exact softmax attention unbiasedly using Positive Orthogonal Random Features without modifying the model architecture.
* **Trade-off**: Requires hundreds of random projection dimensions to keep variance low, increasing intermediate tensor memory.

### 16. Nyströmformer (Feb 2021)
* **Paper**: [Xiong et al., arXiv:2102.03902](https://arxiv.org/abs/2102.03902)
* **What it does**: Approximates the full attention matrix in linear time by selecting landmark tokens and computing an iterative Moore-Penrose pseudoinverse.
* **Trade-off**: Segment-mean pooling blurs isolated fine-grained tokens, and pseudoinverse iterations add kernel execution overhead.

### 17. Delta Rule Linear Attention (Feb 2021)
* **Paper**: [Schlag et al., arXiv:2102.11174](https://arxiv.org/abs/2102.11174)
* **What it does**: Equips linear recurrent attention with an error-correcting delta rule to selectively overwrite and update stale memory associations.
* **Trade-off**: Sequential matrix subtraction updates could not initially be parallelized across sequence length during GPU training.

### 18. Rotary Position Embedding / RoPE (Apr 2021)
* **Paper**: [Su et al., arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
* **What it does**: Rotates query and key vectors in 2D slices by position-dependent angles, naturally encoding relative distance through vector inner products.
* **Trade-off**: Rotary angles rotate into unseen out-of-distribution phases when evaluated beyond the training context length.

### 19. Top-k Attention (Jun 2021)
* **Paper**: [Gupta et al., arXiv:2106.06899](https://arxiv.org/abs/2106.06899)
* **What it does**: Masks attention logits outside the top-k highest scoring positions to eliminate tail noise and reduce value tensor read bandwidth.
* **Trade-off**: Finding top-k logits across raw uncompressed tokens still requires calculating all pairwise dot products during prefill.

### 20. ALiBi: Attention with Linear Biases (Aug 2021)
* **Paper**: [Press et al., arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
* **What it does**: Adds non-parametric linear distance penalties with geometric head slopes directly to attention logits, enabling zero-shot context length extrapolation.
* **Trade-off**: Fixed linear decay imposes a monotonic recency bias that hurts performance on long-range associative lookups.

### 21. KERPLE (May 2022)
* **Paper**: [Chi et al., arXiv:2205.09921](https://arxiv.org/abs/2205.09921)
* **What it does**: Generalizes relative position bias using conditionally positive definite kernels to provide flexible, learnable logarithmic or power-law distance decay.
* **Trade-off**: Adds learnable kernel parameters that require careful tuning, while prefill compute remains quadratic.

### 22. FlashAttention (May 2022)
* **Paper**: [Dao et al., arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
* **What it does**: Reorganizes exact softmax attention into on-chip SRAM block tiles and uses online softmax rescaling to eliminate intermediate GPU memory read/write bottlenecks.
* **Trade-off**: Accelerates execution speed via memory IO efficiency rather than reducing the quadratic algorithmic FLOP bill.

### 23. Grouped-Query Attention / GQA (May 2023)
* **Paper**: [Ainslie et al., arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
* **What it does**: Partitions query heads into groups that share key-value heads, delivering 4x to 8x KV cache compression while preserving near-MHA reasoning quality.
* **Trade-off**: Reduces distinct key-value subspace representations compared to full Multi-Head Attention.

### 24. NoPE: No Positional Embeddings (May 2023)
* **Paper**: [Kazemnejad et al., arXiv:2305.19466](https://arxiv.org/abs/2305.19466)
* **What it does**: Omits all explicit positional encodings in causal decoder-only models, demonstrating that causal triangular masking alone allows models to infer position.
* **Trade-off**: Converges more slowly during pretraining compared to RoPE and cannot be used in bidirectional models.

### 25. Position Interpolation / PI (Jun 2023)
* **Paper**: [Chen et al., arXiv:2306.15595](https://arxiv.org/abs/2306.15595)
* **What it does**: Linearly downscales position indices to map long sequence coordinates into a pretrained model's original RoPE frequency bounds with minimal fine-tuning.
* **Trade-off**: Uniform scaling compresses high frequencies, blurring distinctions between immediately adjacent tokens.

### 26. NTK-Aware RoPE Scaling (Jun 2023)
* **Paper**: [bloc97, LocalLLaMA Release](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
* **What it does**: Scales RoPE base frequencies non-linearly to interpolate low frequencies while preserving high-frequency local resolution for zero-shot context extension.
* **Trade-off**: An empirical heuristic that loses precision when extending context windows beyond 8x without fine-tuning.

### 27. FlashAttention-2 (Jul 2023)
* **Paper**: [Tri Dao, arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
* **What it does**: Improves FlashAttention by parallelizing across the sequence length, cutting redundant non-matmul operations, and optimizing warpgroup layouts up to 73% of peak GPU utilization.
* **Trade-off**: Requires low-level warp assembly tuning for specific GPU microarchitectures.

### 28. YaRN (Aug 2023)
* **Paper**: [Peng et al., arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
* **What it does**: Divides RoPE dimensions into three frequency bands and adds logit temperature scaling to prevent attention distribution flattening over long contexts.
* **Trade-off**: Introduces multiple empirical hyperparameters that must be tuned for specific model scales.

### 29. Attention Sinks / StreamingLLM (Sep 2023)
* **Paper**: [Xiao et al., arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
* **What it does**: Retains the initial 4 tokens in the KV cache alongside a rolling local window to prevent softmax entropy collapse during infinite-length streaming.
* **Trade-off**: Permanently discards tokens evicted from the rolling window, preventing retrieval of distant semantic details.

### 30. Mistral SWA + GQA (Oct 2023)
* **Paper**: [Jiang et al., arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
* **What it does**: Combines 4x Grouped-Query Attention with a 4,096-token sliding window and rolling ring buffer cache for efficient long-context generation.
* **Trade-off**: Eliminates single-layer direct attention across distances larger than the window size.

### 31. Gated Linear Attention / GLA (Dec 2023)
* **Paper**: [Yang et al., arXiv:2312.06635](https://arxiv.org/abs/2312.06635)
* **What it does**: Adds data-dependent decay gates to linear recurrent states and computes intra-chunk operations via parallel GPU Tensor Core matrix multiplications.
* **Trade-off**: The fixed-size recurrent state has finite capacity and cannot perform exact single-step needle retrieval across unstructured text.

### 32. LongRoPE (Feb 2024)
* **Paper**: [Ding et al., arXiv:2402.13753](https://arxiv.org/abs/2402.13753)
* **What it does**: Uses an evolutionary search algorithm to discover non-uniform position- and dimension-dependent RoPE scaling factors, extending context to 2 million tokens.
* **Trade-off**: Requires running an evolutionary search stage prior to fine-tuning, and long decodes still require massive KV cache memory.

### 33. Infini-attention (Apr 2024)
* **Paper**: [Munkhdalai & Faruqui, arXiv:2404.07143](https://arxiv.org/abs/2404.07143)
* **What it does**: Combines local masked multi-head attention with a long-term compressive linear recurrent memory inside a single attention block.
* **Trade-off**: Compressive memory lossily encodes distant historical tokens, losing fine-grained exact retrieval fidelity.

### 34. Multi-Head Latent Attention / MLA (May 2024)
* **Paper**: [DeepSeek-AI, arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
* **What it does**: Compresses key-value projections into a low-rank latent vector with decoupled rotary keys, cutting KV cache memory footprint by 93.3% while keeping full query head expressiveness.
* **Trade-off**: Requires pretraining low-rank projection matrices from scratch and adds matrix projection compute overhead during prefill.

### 35. Cross-Layer Attention / CLA (May 2024)
* **Paper**: [Brandon et al., arXiv:2405.12981](https://arxiv.org/abs/2405.12981)
* **What it does**: Shares key-value cache representations across adjacent layers, cutting the total model KV cache memory footprint by an additional 2x.
* **Trade-off**: Slightly limits layer-specific representation specialization in upper layers.

### 36. Contextual Position Encoding / CoPE (May 2024)
* **Paper**: [Golovneva et al., arXiv:2405.18719](https://arxiv.org/abs/2405.18719)
* **What it does**: Counts dynamic contextual boundaries (such as words or sentences) via attention weights rather than advancing positions by raw token indices.
* **Trade-off**: Dynamic gate accumulation adds kernel complexity during training and requires custom kernels for fast autoregressive decoding.

### 37. DeltaNet (Jun 2024)
* **Paper**: [Yang et al., arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
* **What it does**: Parallelizes the error-correcting delta rule across the sequence length using chunkwise Householder matrix factorizations.
* **Trade-off**: Requires complex custom GPU kernels to evaluate chunkwise Householder factorizations on hardware.

### 38. FlashAttention-3 (Jul 2024)
* **Paper**: [Shah et al., arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
* **What it does**: Exploits NVIDIA Hopper Tensor Memory Accelerators (TMA), warpgroup asynchronous scheduling, and FP8 low-precision matrix multiplications up to 1.2 PFLOPs/s.
* **Trade-off**: Tied to Hopper and newer GPU architectures with hardware TMA support.

### 39. Multimodal RoPE / M-RoPE (Sep 2024)
* **Paper**: [Wang et al., arXiv:2409.12191](https://arxiv.org/abs/2409.12191)
* **What it does**: Decomposes rotary channel frequencies across temporal, height, and width coordinates to preserve 3D spatial and temporal geometry in vision-language models.
* **Trade-off**: Splits head frequency bandwidth across spatial dimensions and requires tracking 3D coordinate tuples in the KV cache.

### 40. Gated DeltaNet (Dec 2024)
* **Paper**: [Yang et al., arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
* **What it does**: Unifies input-dependent exponential retention decay with error-correcting delta writes for high-capacity linear-time sequence modeling.
* **Trade-off**: Increases per-step arithmetic complexity over basic gated linear attention.

### 41. DeepSeek Native Sparse Attention / NSA (Feb 2025)
* **Paper**: [Yuan et al., arXiv:2502.11089](https://arxiv.org/abs/2502.11089)
* **What it does**: Builds a hardware-aligned sparse hierarchy combining coarse compressed tokens, top-k selected 64-token blocks, and a local sliding window directly on Tensor Cores.
* **Trade-off**: Tri-branch routing and kernel dispatch add significant architectural and implementation complexity.

### 42. DeepSeek Sparse Attention / DSA (Sep 2025)
* **Paper**: [DeepSeek-AI, arXiv:2512.02556](https://arxiv.org/abs/2512.02556)
* **What it does**: Uses a lightweight FP8 Lightning Indexer network to select the top-k latent KV cache entries before running full attention operations.
* **Trade-off**: If the indexer misses a critical token, downstream attention heads cannot recover the missing information.

### 43. Kimi Delta Attention / KDA (Oct 2025)
* **Paper**: [Moonshot AI, arXiv:2510.26692](https://arxiv.org/abs/2510.26692)
* **What it does**: Introduces fine-grained channel-wise gating into the delta rule update and interleaves 3 KDA layers with 1 full MLA layer in a 3:1 hybrid stack with Attention Residuals.
* **Trade-off**: Channel-wise gating increases parameter counts and requires specialized chunkwise GPU kernels.

### 44. DroPE (Dec 2025)
* **Paper**: [Gelberg et al., arXiv:2512.12167](https://arxiv.org/abs/2512.12167)
* **What it does**: Pretrains with RoPE, then zeroes out positional embeddings during a short recalibration stage to achieve zero-shot length extrapolation without frequency scaling hacks.
* **Trade-off**: Requires a dedicated post-pretraining recalibration phase.

### 45. GLM-5 IndexCache DSA (Feb 2026)
* **Paper**: [GLM-5 Team, arXiv:2602.15763](https://arxiv.org/abs/2602.15763)
* **What it does**: Reuses top-k routing indices from the Lightning Indexer across 4 consecutive sparse attention layers, slashing indexer compute overhead by 75%.
* **Trade-off**: Constrains consecutive layers to identical sparsity subsets, slightly reducing cross-layer routing diversity.

### 46. Qwen 3.5 / 3.8 Interleaved Hybrid (Feb 2026)
* **Paper**: [Alibaba Qwen Team, Technical Report](https://qwen.ai/blog?id=qwen3.5)
* **What it does**: Interleaves 3 Gated DeltaNet linear recurrence layers with 1 full GQA attention layer across the entire depth to cut decode latency and memory by ~70%.
* **Trade-off**: Full attention layers still incur quadratic memory scaling for their 25% allocation in the layer stack.

### 47. DeepSeek-V4: Compressed Sparse Attention & HCA (Jun 2026)
* **Paper**: [DeepSeek-AI, arXiv:2606.19348](https://arxiv.org/abs/2606.19348)
* **What it does**: Bounds active attention to a fixed 1,152 tokens regardless of context length by pairing overlapping-window CSA with heavily compressed attention and hyper-connections.
* **Trade-off**: Heavily compressed branches permanently discard fine-grained local details outside the local sliding window.

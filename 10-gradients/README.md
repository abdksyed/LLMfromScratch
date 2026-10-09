# Gradients, Tensors, and Loss Diagnostics

Comprehensive empirical inspection of gradient dynamics, tensor dimensional mappings, autograd verification, micro-batch accumulation arithmetic, Model FLOPs Utilization (MFU), and floating-point precision on `Qwen/Qwen3.5-0.8B`.

---

## 1. Model Architecture and Tensor Dimensional Breakdown

Inspected model: `Qwen/Qwen3.5-0.8B`
* **Architecture**: Hybrid Attention Transformer (24 layers total: 18 Linear Attention + 6 Full Attention, full attention interval = 4)
* **Hidden size**: 1024
* **Intermediate size**: 3584 (SwiGLU)
* **Vocabulary size**: 248,320
* **Tied weights**: Yes (`embed_tokens.weight` shares memory with `lm_head.weight`)

### Tensor Dimensions and Physical Meaning

| Component | Tensor Name | Shape | Dimension Meaning |
| :--- | :--- | :--- | :--- |
| **Embeddings** | `model.embed_tokens.weight` | `(248320, 1024)` | Dim 0 (248320): Unique vocabulary token IDs.<br>Dim 1 (1024): Hidden representation vector per token. |
| **Output Head** | `lm_head.weight` | `(248320, 1024)` | Dim 0 (248320): Vocab projection score per token.<br>Dim 1 (1024): Input hidden dimension. Tied to embed tokens. |
| **Linear Attention** | `layer.0.linear_attn.dt_bias` | `(16,)` | Dim 0 (16): Delta-rule timescale bias parameter across state-space channels. |
| **Linear Attention** | `layer.0.linear_attn.A` | `(16,)` | Dim 0 (16): State decay transition coefficients. |
| **Linear Attention** | `layer.0.linear_attn.in_proj.weight` | `(4096, 1024)` | Dim 0 (4096): Combined input projection (Q, K, V, and gate expansions).<br>Dim 1 (1024): Layer input hidden dimension. |
| **Linear Attention** | `layer.0.linear_attn.out_proj.weight` | `(1024, 2048)` | Dim 0 (1024): Layer output projection back to residual stream.<br>Dim 1 (2048): Linear attention channel output dimension. |
| **Full Attention (GQA)** | `layer.3.self_attn.q_proj.weight` | `(4096, 1024)` | Dim 0 (4096): Query projection dimension (16 query heads × 256 head dim).<br>Dim 1 (1024): Residual hidden size. |
| **Full Attention (GQA)** | `layer.3.self_attn.k_proj.weight` | `(1024, 1024)` | Dim 0 (1024): Key projection dimension (4 KV heads × 256 head dim for Grouped Query Attention).<br>Dim 1 (1024): Residual hidden size. |
| **Full Attention (GQA)** | `layer.3.self_attn.v_proj.weight` | `(1024, 1024)` | Dim 0 (1024): Value projection dimension (4 KV heads × 256 head dim).<br>Dim 1 (1024): Residual hidden size. |
| **Full Attention (GQA)** | `layer.3.self_attn.o_proj.weight` | `(1024, 4096)` | Dim 0 (1024): Projected output to residual stream.<br>Dim 1 (4096): Concatenated multi-head attention outputs (16 × 256). |
| **SwiGLU MLP** | `layer.i.mlp.gate_proj.weight` | `(3584, 1024)` | Dim 0 (3584): Intermediate dimension for SiLU gating branch.<br>Dim 1 (1024): Residual hidden size. |
| **SwiGLU MLP** | `layer.i.mlp.up_proj.weight` | `(3584, 1024)` | Dim 0 (3584): Intermediate dimension for linear value branch.<br>Dim 1 (1024): Residual hidden size. |
| **SwiGLU MLP** | `layer.i.mlp.down_proj.weight` | `(1024, 3584)` | Dim 0 (1024): Projection back to residual stream.<br>Dim 1 (3584): SwiGLU elementwise product dimension. |
| **Normalization** | `language_model.norm.weight` | `(1024,)` | Dim 0 (1024): Channel-wise RMSNorm scale factor before final projection. |
| **Logits** | `outputs.logits` | `(batch, seq, 248320)` | Batch index, token sequence position, and unnormalized log probability per vocabulary item. |

---

## 2. Gradient Verification by Hand

To verify the analytical derivative produced by PyTorch autograd, a single parameter in Layer 0 was nudged by $\epsilon = \pm 0.1$, evaluating the change in loss via first-order Taylor expansion:

$$L(w + \epsilon) \approx L(w) + \epsilon \cdot \frac{\partial L}{\partial w}$$

### Experimental Protocol and Precision
Evaluating this in bfloat16 causes significant discretization error because bfloat16 allocates only 7 bits to the mantissa. Converting the model to float32 provides the 23 mantissa bits necessary to resolve variations down to $10^{-7}$.

* **Base loss $L(w)$**: `2.6759240627`
* **Target parameter initial value**: `-0.77734375`
* **Autograd gradient $\frac{\partial L}{\partial w}$**: `-0.0034833194`

### Numerical Results

| Perturbation ($\epsilon$) | First-Order Expected Loss | Empirical Loss with Nudge | Absolute Difference | Agreement |
| :--- | :--- | :--- | :--- | :--- |
| **$\epsilon = -0.1$** | `2.6762723947` | `2.6762807369` | $8.34 \times 10^{-6}$ | **5 decimal places** |
| **$\epsilon = +0.1$** | `2.6755757308` | `2.6755843163` | $8.59 \times 10^{-6}$ | **5 decimal places** |

The residual discrepancy of $8 \times 10^{-6}$ represents the expected second-order curvature term $\frac{1}{2}\epsilon^2 \nabla^2 L$, confirming that `loss.backward()` computes the exact analytical gradient of the computational graph.

---

## 3. Broken Gradient Accumulation: The Average-of-Averages Fallacy

When micro-batches vary in valid sequence length, taking the arithmetic mean of micro-batch losses breaks mathematical equivalence with standard batching.

### Mathematical Formulation

Given two micro-batches with token counts $N_1$ and $N_2$ and sequence losses $L_1$ and $L_2$:
* **Correct token-weighted loss**:
  $$L_{\text{true}} = \frac{\sum_{i=1}^{N_1} \ell_{1,i} + \sum_{j=1}^{N_2} \ell_{2,j}}{N_1 + N_2} = \frac{N_1 L_1 + N_2 L_2}{N_1 + N_2}$$
* **Broken average-of-averages loss**:
  $$L_{\text{broken}} = \frac{L_1 + L_2}{2}$$

### Empirical Demonstration

A test was conducted with two micro-batches of unequal length:
* **Micro-batch 1**: 4 random noisy token IDs ($N_1 = 4$, initial loss $L_1 = 18.7500$)
* **Micro-batch 2**: 39 structured token IDs ($N_2 = 39$, initial loss $L_2 = 2.6875$)

```text
======================================================================
STEP 0 STARTING VALUES:
Short noisy batch loss:      18.7500
Long real batch loss:         2.6875
Broken accumulation loss:    10.7188  <-- (18.75 + 2.6875) / 2
Correct accumulation loss:    4.1817  <-- (4*18.75 + 39*2.6875) / 43
======================================================================
```

### Why This Breaks Training
1. **Disproportionate weighting**: In the broken setup, every token in the short batch is weighted as $\frac{1}{2 \times 4} = 0.125$, while every token in the long batch is weighted as $\frac{1}{2 \times 39} \approx 0.0128$. Tokens in the noisy micro-batch exert **nearly $10\times$ more influence** on parameter updates.
2. **Optimization trajectory**: Over 10 steps of optimization, the broken accumulation loss hovers above 6.0 while correct token-weighted accumulation converges smoothly below 3.0.
3. **Engineering rule**: Accumulate raw unnormalized loss sums or scale micro-batch losses by $\frac{N_{\text{micro}}}{N_{\text{total}}}$ prior to backward passes.

---

## 4. Gradient Norm as a Leading Indicator of Instability

Gradient norm ($\|\mathbf{g}\|_2 = \sqrt{\sum_i g_i^2}$) monitors the local slope of the loss landscape. In an optimization run on sentence tokens, logging the global gradient norm revealed an instance where the gradient norm signaled destabilization before the loss increased.

```text
Grad norm moved before loss at Step 11:
  Step 10: Loss = 0.0011 | Grad Norm = 0.1807
  Step 11: Loss = 0.0010 | Grad Norm = 0.2334  (+29.2% spike in gradient norm)
  Step 12: Loss = 0.0013 | Grad Norm = 0.3496  (+30.0% increase in loss)
```

### Operational Takeaway
At Step 11, the loss slightly declined from `0.0011` to `0.0010`. However, the gradient norm surged from `0.1807` to `0.2334`. By Step 12, the loss had increased to `0.0013`, and the gradient norm climbed further to `0.3496`. The scalar loss measures the position at the current evaluation point; the gradient norm reflects the steepness of the terrain being entered. Monitoring gradient norm provides an early warning mechanism for optimizer divergence and learning rate schedule failure.

---

## 5. Model FLOPs Utilization (MFU) Analysis

MFU measures the fraction of hardware peak theoretical floating-point performance sustained during real training steps:

$$\text{MFU} = \frac{\text{Theoretical FLOPs per Step}}{\text{Step Time (seconds)} \times \text{Peak Device TFLOP/s}}$$

For a decoder model with $P$ non-embedding active parameters, sequence length $L$, and batch size $B$, the compute per step is modeled as:
$$\text{FLOPs} \approx 6 \times B \times L \times P$$

### Measured Benchmark on Apple Silicon / Metal

* **Batch size**: 1
* **Sequence length**: 2,048 tokens
* **Tokens per step**: 2,047
* **Measured step time**: 278.93 ms
* **Achieved throughput**: 37.56 TFLOP/s
* **Achieved MFU**: **24.26%**

### Analysis of the Distance to 40% Target
Achieving the industry standard 40-50% MFU target on this architecture is constrained by three factors:
1. **Low Tensor Core occupancy at Batch Size 1**: Matrix multiplications with $M=1$ or small token blocks operate in a memory bandwidth bound regime. Modern systolic arrays require larger batch sizes ($B \ge 8$ or sequences packed to 4,096+ tokens) to hide memory latency behind arithmetic compute.
2. **Linear attention kernel fallback**: Without specialized fused kernels (`causal-conv1d` and chunked delta-rule CUDA kernels), the 18 linear attention layers fall back to standard PyTorch tensor operations. These operations trigger intermediate global memory writes and kernel launch overheads.
3. **Synchronization and graph compilation**: Eager-mode execution incurs CPU-GPU dispatch overhead. Compiling the model via `torch.compile(mode="reduce-overhead")` and using fused kernel implementations are required to bridge the remaining 16% gap to the 40% benchmark.

---

## 6. Floating-Point Representations: 0.1 in FP32, BF16, and FP8 E4M3

Decimal $0.1$ has an infinite repeating binary expansion:
$$0.1_{10} = 0.00011001100110011\dots_2 = 1.100110011\dots_2 \times 2^{-4}$$

The table below contrasts how standard deep learning formats truncate this value:

```text
======================================================================
BIT PATTERNS FOR DECIMAL: 0.1
======================================================================
1. FP32 (Single Precision)
   Raw Bits:     0 | 01111011 | 10011001100110011001101
                 S   Exp (8b)  Mantissa (23b)
   Hex:          0x3DCCCCCD
   Stored Value: 0.10000000149011612
   Error:        0.00000000149011611 (~1.49e-9)

2. BF16 (Brain Floating Point)
   Raw Bits:     0 | 01111011 | 1001101
                 S   Exp (8b)  Mantissa (7b)
   Hex:          0x3DCD
   Stored Value: 0.10009765625000000
   Error:        0.00009765624999999 (~9.77e-5)

3. FP8 E4M3 (NVIDIA / OCP Format)
   Raw Bits:     0 | 0011 | 101
                 S  Exp(4b) Mant(3b)
   Hex:          0x1D
   Stored Value: 0.10156250000000000
   Error:        0.00156249999999999 (~1.56e-3)
======================================================================
```

### Comparative Structure

| Precision | Total Bits | Exponent Bits | Mantissa Bits | Exponent Bias | Relative Truncation Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 | 8 | 23 | 127 | $\approx 1.5 \times 10^{-8}$ |
| **BF16** | 16 | 8 | 7 | 127 | $\approx 9.8 \times 10^{-4}$ |
| **FP8 (E4M3)** | 8 | 4 | 3 | 7 | $\approx 1.6 \times 10^{-2}$ |

### Selection for Training: Mixed Precision BF16 / FP32

**Selection: Mixed-precision BF16 for activations and forward/backward passes, paired with FP32 master weights and optimizer accumulators.**

#### Technical Rationale:
1. **Why not FP8 for master weights**: FP8 E4M3 has only 3 mantissa bits, creating a 1.56% discretization error on a number as simple as 0.1. A typical parameter update $\Delta w = \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}}} \approx 10^{-4}$ completely underflows relative to weight magnitudes of $\sim 10^{-1}$, zeroing out updates. FP8 is viable for GEMM tensor core inputs with dynamic per-tensor scaling, but unusable for master weights and unscaled gradient accumulators.
2. **Why BF16 over FP16**: Both are 16-bit formats, but standard FP16 allocates 5 bits to the exponent (dynamic range $\sim 10^{-5}$ to $6.5 \times 10^4$). Large transformers frequently experience gradient underflows in FP16 unless managed by fragile loss scaling. BF16 retains the full 8 exponent bits of FP32 (range $\sim 10^{-38}$ to $10^{38}$), completely eliminating gradient underflow while delivering double the arithmetic throughput and half the memory footprint of FP32.

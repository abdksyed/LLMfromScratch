# Adam Optimizer Mechanics, Dynamics, and Scaling Laws

Empirical analysis and reproduction of the Adam optimizer, bias correction horizons, layer-wise update ratios, learning rate schedules (Cosine vs. WSD), and width-dependent hyperparameter scaling.

---

## 1. Reproducing Adam by Hand

Adam tracks exponentially decaying moving averages of past gradients (first moment $m_t$) and squared gradients (second uncentered moment $v_t$).

### Mathematical Formulation
Given hyperparameter values $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, and learning rate $\eta = 10^{-3}$:

1. **Biased first moment**:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2. **Biased second moment**:
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
3. **Bias-corrected first moment**:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
4. **Bias-corrected second moment**:
   $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
5. **Parameter update**:
   $$\Delta w_t = \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}, \quad w_t = w_{t-1} - \Delta w_t$$

### Step-by-Step Validation Against PyTorch
Starting with an initial weight $w_0 = 1.5$ and applying five sequential gradients $g \in [0.1, -0.2, 0.05, 0.3, -0.15]$, manual math was compared directly against `torch.optim.Adam` internal state tensors:

| Step | Gradient ($g$) | $m$ (Manual / PyTorch) | $v$ (Manual / PyTorch) | $\hat{m}$ | $\hat{v}$ | Weight ($w_t$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | $+0.10$ | `0.010000` / `0.010000` | `0.00001000` / `0.00001000` | `0.100000` | `0.010000` | `1.499000` |
| **2** | $-0.20$ | `-0.011000` / `-0.011000` | `0.00004999` / `0.00004999` | `-0.057895` | `0.025008` | `1.499366` |
| **3** | $+0.05$ | `-0.004900` / `-0.004900` | `0.00005244` / `0.00005244` | `-0.018081` | `0.017498` | `1.499503` |
| **4** | $+0.30$ | `0.025590` / `0.025590` | `0.00014239` / `0.00014239` | `0.074411` | `0.035650` | `1.499109` |
| **5** | $-0.15$ | `0.008031` / `0.008031` | `0.00016475` / `0.00016475` | `0.019611` | `0.033015` | `1.499001` |

Manual and PyTorch values matched with absolute differences below $10^{-7}$ across all parameters.

---

## 2. Bias Correction Dynamics and Convergence Horizon

Because $m_0 = 0$ and $v_0 = 0$, both running moments are biased toward zero at early iterations. Disabling bias correction yields the uncorrected update step:

$$\Delta w_t^{\text{no\_bc}} = \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$$

The ratio of uncorrected step to corrected step simplifies to:

$$\frac{\Delta w_t^{\text{no\_bc}}}{\Delta w_t^{\text{bc}}} \approx \frac{1 - \beta_1^t}{\sqrt{1 - \beta_2^t}}$$

### Early-Step Dynamics (First 20 Steps)
* **Step 1 explosion**: At $t=1$, $m_1 = (1 - 0.9)g_1 = 0.1 g_1$ and $v_1 = (1 - 0.999)g_1^2 = 0.001 g_1^2$.
  $$\frac{m_1}{\sqrt{v_1}} = \frac{0.1}{\sqrt{0.001}} = \sqrt{10} \approx 3.162$$
  The uncorrected step is **$3.16\times$ larger** than the intended step size $\eta$.
* **Steps 5 to 20 amplification**: While $(1 - \beta_1^t)$ rapidly approaches $1.0$, the denominator $\sqrt{1 - \beta_2^t}$ remains tiny because $\beta_2 = 0.999$. At step 10, the uncorrected step is **$6.53\times$ larger** than the corrected step. Uncorrected weights traverse over $5\times$ more distance in the first 20 iterations.

### When Does Bias Correction Stop Mattering?
Running a 5,000-step simulation tracking relative difference $\frac{|\Delta w^{\text{no\_bc}} - \Delta w^{\text{bc}}|}{|\Delta w^{\text{bc}}|}$ reveals:

```text
Total steps simulated: 5000
Difference < 5% at step:  2375
Difference < 1% at step:  3925
```

* **First moment ($\beta_1 = 0.9$)**: Reaches $> 0.99$ by step $t \approx \frac{\ln(0.01)}{\ln(0.9)} \approx 44$ steps.
* **Second moment ($\beta_2 = 0.999$)**: Has an effective memory horizon of $\frac{1}{1 - \beta_2} = 1,000$ steps.
  * For a 5% difference ($\frac{1}{\sqrt{1 - 0.999^t}} \le 1.05$), analytical calculation requires $t \ge 2,374$ steps.
  * For a 1% difference ($\frac{1}{\sqrt{1 - 0.999^t}} \le 1.01$), analytical calculation requires $t \ge 3,924$ steps.
* **Conclusion**: Bias correction does not stop mattering in the first 20 steps. It requires **2,400 to 4,000 steps** before the second-moment estimate aligns within 1-5% of true gradient variance.

---

## 3. Layer-Wise Update-to-Weight Ratio and Warmup Dynamics

The update-to-weight ratio evaluates how aggressively optimizer steps displace model weights relative to their Euclidean magnitude:

$$\text{Ratio} = \frac{\|\Delta W\|_2}{\|W\|_2} = \frac{\|W_t - W_{t-1}\|_2}{\|W_t\|_2}$$

In stable neural network pre-training, healthy update-to-weight ratios sit in the range of $\sim 10^{-4}$ to $10^{-3}$.

### Empirical Tracking Under Warmup
Using a multi-layer model trained with a 30-step linear warmup schedule over 60 total steps:

```text
Configured warmup steps: 30
Layer Name                                    | Warmup Ends At Step  | Peak Ratio  
----------------------------------------------------------------------------------
fc1.weight                                    | Step 31              | 0.002514
fc1.bias                                      | Step 31              | 0.006424
fc2.weight                                    | Step 31              | 0.007873
fc2.bias                                      | Step 31              | 0.013277
----------------------------------------------------------------------------------
Overall warmup stops altering the ratio at: Step 31
```

### Transition Mechanics
1. **Steps 1 to 30**: The learning rate increases linearly ($\eta_t = \eta_{\max} \cdot \frac{t}{30}$). The update norm $\|\Delta W\|_2$ scales in direct proportion, driving the ratio upward.
2. **Step 31**: The learning rate reaches peak and stops growing. The rate of change $\frac{d}{dt}\left(\frac{\|\Delta W\|_2}{\|W\|_2}\right)$ transitions from positive slope to zero or negative, marking the exact boundary where warmup ceases altering the update magnitude.

---

## 4. Cosine vs. WSD Training with Early Stopping at Step 200 of 300

Two identical models initialized from the exact same random state were trained with a planned budget of 300 steps. Both runs were terminated early at step 200.

### Schedule Formulations
* **Cosine**: Ramps linearly over 30 steps, then smoothly decays following a half-period cosine curve until step 300.
* **Warmup-Stable-Decay (WSD)**: Ramps linearly over 30 steps, maintains 100% peak learning rate during the stable phase (steps 31 to 240), and reserves decay for the final 20% of steps (steps 241 to 300).

### Results at Step 200

| Model | Status at Step 200 | Current LR | Final Loss at Step 200 |
| :--- | :--- | :--- | :--- |
| **Cosine** | 67% through planned decay | $0.00302$ (decayed to $\approx 30\%$ of peak) | **`0.070429`** |
| **WSD** | Inside Stable phase | $0.01000$ (100% peak LR) | **`0.047456`** |

### Which Model to Keep?

#### Decision Rule:
* **For immediate, zero-shot deployment without further computation**:
  Keep the **Cosine model** if it achieved lower loss on task data due to learning rate cooling. When stopped at step 200, the Cosine model has already spent 170 steps cooling down into a sharp local basin.
* **For continuous pre-training, domain adaptation, or branching (the core value of WSD)**:
  Keep the **WSD model**.
  1. **The Cosine checkpoint lock-in**: Stopping early under Cosine leaves a model trained with an annealed, near-zero learning rate. Resuming training from an annealed Cosine checkpoint requires re-warming the learning rate, which causes catastrophic loss spikes and degrades final convergence.
  2. **The WSD flexibility**: The WSD checkpoint at step 200 is in the high-learning-rate stable phase. It can continue training for 500, 1,000, or 10,000 more steps without penalty. Alternatively, branching a short 20-step decay from step 200 drops loss rapidly below the Cosine baseline.

---

## 5. Learning Rate Sweep Across Widths (256, 512, 1,024) and Width 4,096 Extrapolation

Under Standard Parameterization (SP), networks of varying widths ($d_{\text{model}} \in [256, 512, 1024]$) were swept across learning rates from $10^{-4}$ to $10^{-1}$ on identical data.

### Empirical Minima

| Width | Optimal LR ($\eta^*$) | Log Scale Position | Scaling from Previous |
| :--- | :--- | :--- | :--- |
| **256** | $1.0 \times 10^{-2}$ ($0.0100$) | $10^{-2.0}$ | — |
| **512** | $3.0 \times 10^{-3}$ ($0.0030$) | $10^{-2.5}$ | $\approx \mathbf{3.3\times}$ decrease ($-0.5$ decade) |
| **1,024** | $1.0 \times 10^{-3}$ ($0.0010$) | $10^{-3.0}$ | $\approx \mathbf{3.0\times}$ decrease ($-0.5$ decade) |

### Value at Width 4,096
Extrapolating two doublings ($1024 \to 2048 \to 4096$) along the empirical trend of $-0.5$ decade per octave:
1. **Width 2,048 ($2\times$)**: $1.0 \times 10^{-3} \div 3.16 \approx \mathbf{3.16 \times 10^{-4}}$ ($10^{-3.5}$)
2. **Width 4,096 ($4\times$)**: $3.16 \times 10^{-4} \div 3.16 \approx \mathbf{1.0 \times 10^{-4}}$ ($10^{-4.0}$)

*(Under standard $1/d$ scaling: $\eta^*(4096) = 0.0010 \times \frac{1024}{4096} = \mathbf{2.5 \times 10^{-4}}$).*

### Confidence Assessment: Low to Moderate
1. **Standard Parameterization lacks scale invariance**: Unlike Maximal Update Parameterization ($\mu$P), standard PyTorch does not ensure invariant optimal learning rates.
2. **Leftward divergence cliff**: In the empirical curves, Width 1,024 diverges completely at $10^{-2}$, where Width 256 achieved its minimum. At width 4,096, the stability boundary moves even further left. Setting an aggressive learning rate results in instant numerical overflow.

---

## 6. The Tuning Principle: Tune Both Sides Before Accepting a Comparison

> *"Tune both sides before accepting a comparison. Almost every optimizer claim that failed to replicate was a well tuned method measured against a badly tuned one."*

### Failure Mode in the Literature
In published optimizer benchmarks, novel optimizers often appear superior only because their hyperparameters were heavily grid-searched, while baseline AdamW was run with default $\eta = 10^{-3}$. When baseline AdamW is swept over learning rates and beta schedules with equal compute budget, reported performance gaps frequently vanish.

### Application to Width 4,096
Evaluating width 4,096 at a single unverified learning rate like $1.0 \times 10^{-3}$ will diverge, falsely suggesting that wide networks perform worse. Evaluating width 4,096 only at the extrapolated point estimate of $1.0 \times 10^{-4}$ risks measuring an uncalibrated run against finely tuned 256/512 baselines.

**Mandatory protocol**: Always tune width 4,096 across a targeted 3-point grid around the extrapolation:
$$\eta \in \{3.0 \times 10^{-5},\; 1.0 \times 10^{-4},\; 3.0 \times 10^{-4}\}$$

Only comparisons between fully tuned minima reflect true architectural differences.

# 6. Hyperparameters

Every variational setting in [config.yaml](../../config.yaml), what it does, how the current value was chosen, and how to change it. This is the file to read before touching the config.

## 6.1 The settings at a glance

| Key | Current value | Applies to | One-line meaning |
|---|---|---|---|
| `model.name` | `variational_partial_unet` | | which model; anything starting with `variational` uses the variational path |
| `model.variational_decoder_levels` | 1 | `variational_partial_unet` only | how many decoder blocks, from the output end, are variational |
| `model.initial_rho` | -6.5 | all variational models | starting weight noise, σ = softplus(ρ) ≈ 1.5 × 10⁻³ |
| `model.prior_standard_deviation` | 1.0 | all variational models | width p of the N(0, p²) prior |
| `training.kl_weight` | 10⁻⁶ | all variational models | β, the weight on the KL term |
| `training.rho_learning_rate` | 0.01 | all variational models | learning rate for the ρ parameters only |

**The current values are tuned for `variational_partial_unet`.** `initial_rho` and `prior_standard_deviation` are shared keys. Switching `model.name` to `variational_unet` without changing them gives that model a setting it was never tested with. For `variational_unet` use about `initial_rho: -8.0` and `prior_standard_deviation: 0.1` ([section 6.8](#68-settings-for-the-other-two-models)).

## 6.2 Why these settings are so sensitive

Three scale facts drive everything. They are easy to miss because none of them is visible in the code.

**Fact 1: the KL is proportional to the number of variational weights.** Each weight contributes roughly 4.5 to 6 nats. The table shows the KL per training image (KL / N, the `kl` value in the progress bar) at initialisation:

| Model and setting | Variational weights n | KL / N | × β = KL term in the loss |
|---|---|---|---|
| `variational_unet`, ρ = -5, p = 1, β = 0.1 | 1,925,025 | 13,135 | 1,314 |
| `variational_unet`, ρ = -8, p = 0.1, β = 10⁻⁶ | 1,925,025 | 15,207 | 0.015 |
| `variational_partial_unet`, 1 level, ρ = -6.5, p = 1, β = 10⁻⁶ | 27,745 | 252 | 0.00025 |
| `variational_simple_cnn`, ρ = -5, p = 1, β = 0.1 | 289 | 1.97 | 0.20 |

For comparison, the reconstruction loss of a trained UNet is about 0.0008.

**Fact 2: the reconstruction loss is a per-pixel mean.** A properly scaled Bayesian objective would sum over the 102,400 pixels of each image. Using the mean shrinks the data term by that factor, so the KL weight must shrink with it. The β that corresponds to Bayes' rule is about 1.6 × 10⁻⁸ ([01_theory.md, section 1.4](01_theory.md#why-β--1-is-not-the-correct-bayesian-value-here)).

**Fact 3: the weights are small, and smaller in wide layers.** Initial means have a standard deviation of 1/√(3 · fan_in): 0.19 in the first layer, 0.012 in the bottleneck ([02_layers.md, section 2.4](02_layers.md#initialisation-of-the-means)). A σ that sounds tiny can be half the size of the weights.

## 6.3 `kl_weight`

**What it is.** β in

$$
\text{loss} = \text{reconstruction} + \beta \cdot \frac{\operatorname{KL}}{N}
$$

**What it does.** It sets how strongly the KL pushes every σ upward (and every mean toward zero) against the reconstruction term, which wants every σ small. It is the knob that sets *how much* uncertainty the model ends up with.

### The equilibrium, and what `kl_weight` costs

This derivation is an approximation, but it gives the right orders of magnitude and a useful rule.

For small noise, the expected reconstruction loss under sampled weights is the loss at the means plus a penalty that grows with σ²:

$$
\mathbb{E}\left[L(\mu + \sigma\varepsilon)\right] \approx L(\mu) + \frac{1}{2}\sum_i h_i \sigma_i^2
$$

Here h_i is the second derivative of the loss with respect to weight i: how sensitive the loss is to that weight. This assumes small σ and ignores interactions between weights.

The KL term contributes, per weight, β/N times (−ln σ + σ²/(2p²) + terms without σ). Setting the derivative of the total with respect to σ_i to zero:

$$
h_i \sigma_i - \frac{\beta}{N}\left(\frac{1}{\sigma_i} - \frac{\sigma_i}{p^2}\right) = 0
\quad\Longrightarrow\quad
\sigma_i^2 = \frac{\beta / N}{h_i + \beta / (N p^2)} \approx \frac{\beta}{N h_i}
$$

Three consequences:

1. **Sensitive weights get a small σ, unimportant weights get a large one.** σ_i is proportional to 1/√h_i. This is what makes the learned noise informative.
2. **σ scales with √β.** Multiplying `kl_weight` by 4 doubles every σ and, roughly, the epistemic std.
3. **The cost in loss is the same for every weight.** Substituting back, each weight contributes ½ h_i σ_i² = β / (2N). Summed over the n variational weights:

$$
\text{extra reconstruction loss} \approx \frac{n \beta}{2N}
$$

This last formula is the practical rule. It does not depend on the sensitivities at all.

### Applying the rule

For `variational_partial_unet` with one level: n = 27,745, N = 660, β = 10⁻⁶.

$$
\frac{27{,}745 \times 10^{-6}}{2 \times 660} = 2.1 \times 10^{-5}
$$

That is 2.5% of a reconstruction loss of 0.00083. The value 10⁻⁶ was chosen to land at "a couple of percent".

What the same β would cost with more variational weights:

| Model | n | Predicted extra loss at β = 10⁻⁶ | As a share of 0.00083 |
|---|---|---|---|
| partial, 0 levels | 33 | 2.5 × 10⁻⁸ | 0.003% |
| partial, 1 level | 27,745 | 2.1 × 10⁻⁵ | 2.5% |
| partial, 2 levels | 138,465 | 1.0 × 10⁻⁴ | 13% |
| partial, 3 levels | 581,089 | 4.4 × 10⁻⁴ | 53% |
| `variational_unet` | 1,925,025 | 1.5 × 10⁻³ | 176% |

**So `kl_weight` must be rescaled whenever the number of variational weights changes.** To keep a target cost c:

$$
\beta = \frac{2 N c}{n}
$$

For a 2% cost on the fully variational UNet this gives β ≈ 1.1 × 10⁻⁸, which happens to be close to the Bayes-rule value of 1.6 × 10⁻⁸.

### How well the prediction held

The rule predicts the cost of *sampling*: how much worse the loss is with sampled weights than with the mean weights. In the three archived runs of the recommended model that cost was 0.7% to 1.1% ([07_experiments.md, section 7.3](07_experiments.md#73-numbers-for-the-variational-runs)). The prediction was 2.5%. The same order of magnitude, two to four times too pessimistic. Likely reasons: σ had not reached equilibrium after 30 epochs, and the formula ignores that the means also adapt to make the network robust to the noise.

The predicted typical σ, using the measured total sensitivity of about 6 for these layers ([section 6.7](#67-measured-sensitivity-to-weight-noise)), is √(β / (N · 6 / n)) ≈ 2.6 × 10⁻³. The observed medians were 2.3 to 2.4 × 10⁻³.

The rule says nothing about the other cost of a variational model: how much worse the *mean-weight* network is than a plain UNet trained without noise. That cost was larger and less predictable, between 1% and 14% ([07_experiments.md, section 7.6](07_experiments.md#76-finding-3-a-small-variational-head-costs-the-least-accuracy)).

### Why 0.1 failed

With β = 0.1 and n = 1.9 million, the predicted equilibrium cost is 146, against a loss of 0.001. The equilibrium is unreachable: the model would have to destroy itself to satisfy the KL. In the 10 epochs that run lasted, Adam's bounded step size kept it from getting far: the KL fell only 8%, and every σ rose from 6.7 to 10.2 × 10⁻³. That is why the result was clearly worse and not useless. A longer run with those settings would have kept raising σ.

### Values to try

| `kl_weight` | Expected effect on `variational_partial_unet` (1 level) |
|---|---|
| 0 | no pull toward the prior; σ only shrinks, slowly; uncertainty decays toward nothing |
| 10⁻⁷ | σ about 3 times smaller than now; cost near zero |
| 10⁻⁶ (current) | predicted cost of sampling 2.5%, measured 0.7% to 1.1% |
| 3 × 10⁻⁶ | σ about 1.7 times larger; predicted cost 7.5% |
| 10⁻⁵ | σ about 3 times larger; predicted cost 25% |

Only the current value has been run. The others are predictions from the formula.

## 6.4 `initial_rho`

**What it is.** The value every ρ starts at. All weights begin with the same σ = softplus(ρ).

| ρ | σ |
|---|---|
| -5 | 6.7 × 10⁻³ |
| -6 | 2.5 × 10⁻³ |
| -6.5 | 1.5 × 10⁻³ |
| -8 | 3.4 × 10⁻⁴ |
| -10 | 4.5 × 10⁻⁵ |

**Why it matters so much.** ρ moves slowly. With the shared learning rate it effectively does not move at all in 30 epochs, so the initial value *is* the final value ([04_training.md, section 4.5](04_training.md#45-a-separate-learning-rate-for-ρ)). Even with the separate ρ learning rate, the 5th percentile of σ ended at 1.3 to 1.4 × 10⁻³ from a start of 1.5 × 10⁻³, and the smallest σ of all at about 0.7 × 10⁻³. Weights that should have a much smaller σ had only got part of the way. In a 30-epoch run the initial value still shapes the result.

**How -6.5 was chosen for the partial model.** From the measurement in [section 6.7](#67-measured-sensitivity-to-weight-noise): uniform noise of σ = 0.001 on the last decoder block and output layer raises the loss by 0.3%, and σ = 0.003 raises it by 3.2%. A start of 1.5 × 10⁻³ sits between, at the low-cost end. It is also about 4% to 6% of the size of the weights in those layers (std 0.024 to 0.034).

The choice was originally made from an earlier run of the same measurement on a different trained UNet, which was not saved and gave 0.5% and 7.7%. The archived measurement leads to the same choice.

**The rule of thumb.** σ should start at a few percent of the standard deviation of the weights in the *widest* variational layer, because one value serves all layers. That is why the right value differs so much between models:

| Model | Widest variational layer | Weight std there | Suitable ρ | σ as a share |
|---|---|---|---|---|
| `variational_partial_unet`, 1 level | 64 → 32 | 0.024 | -6.5 | 6% |
| `variational_unet` | 256 → 256 | 0.012 | -8 | 3% |
| either, with ρ = -5 | | | | 28% / 56% |

## 6.5 `rho_learning_rate`

**What it is.** A learning rate used only for parameters whose names end in `_rho`. All other parameters use `training.learning_rate`.

**Why 0.01.** Ten times the shared rate gives ρ room to move by several units over a 30-epoch run, enough to change σ by an order of magnitude where the gradient consistently says so. It was the first value tried. No other value has been tested.

**The evidence.** Two pairs of runs, identical within each pair except for this setting ([07_experiments.md, section 7.7](07_experiments.md#77-finding-4-what-the-separate-ρ-learning-rate-does)):

- With the separate rate, σ spread out over a factor of about 10 between its smallest and largest value. Without it, every σ stayed within 19% of where it started.
- The test loss was about 4% lower with the separate rate, in both pairs. Two identical seeded runs already differ by about 3%, so this is suggestive, not established.
- The uncertainty numbers were not better with it.

**What is not established.** Why a model whose σ is on average *larger* gets a *better* loss. The natural explanation is the one from [section 6.3](#the-equilibrium-and-what-kl_weight-costs): a learned σ puts the noise on weights that do not matter and keeps it off those that do, while a uniform σ of the same average size hits the sensitive weights too. That explanation fits the numbers but was not tested directly.

**To turn it off**, remove the key or set it to `null`. Everything then behaves as it did before the feature was added.

## 6.6 `prior_standard_deviation` and `variational_decoder_levels`

### `prior_standard_deviation`

**What it is.** The width p of the zero-mean normal prior on every variational weight.

**What it affects**, in order of importance:

1. **Weight decay on the means.** The KL pulls each mean toward zero with strength β / (N p²). At the current settings that is 1.5 × 10⁻⁹, which is nothing. With p = 0.1 it is 100 times larger and still negligible. This only becomes relevant if β is large.
2. **The ceiling on σ.** From the equilibrium formula, a weight the loss does not depend on at all (h = 0) ends at σ = p. A wide prior lets irrelevant weights become very noisy. A narrow one caps them.
3. **The size of the logged KL.** Changing p from 1 to 0.1 subtracts ln 10 ≈ 2.3 nats per weight. It changes the number in the progress bar and nothing about the optimisation of σ, because that term has no gradient.

**Why 1.0 now.** With a small β the prior width barely matters, and the wide value keeps the KL acting on σ only, without also shrinking the means. This was a reasoned choice, not a tested one. The fully variational run that used 0.1 was not compared with the same run at 1.0.

The value 1.0 is far wider than any weight in the network (the largest initial std is 0.19). A prior matched to the weights would be closer to 0.1. Whether that would be better here is open.

### `variational_decoder_levels`

**What it is.** How many decoder blocks, counted from the output, are variational. See [03_models.md, section 3.4](03_models.md#the-variational_decoder_levels-setting).

**Why 1.** It is the smallest setting that makes a whole block variational, which gives spatially structured uncertainty (3 × 3 kernels) instead of a per-pixel rescaling (the 1 × 1 output conv alone). The measurement below shows 2 levels cost about twice as much as 1 at σ = 0.003, so 2 is a reasonable thing to try, provided `kl_weight` is lowered by the ratio of n (about 5 times) to keep the cost fixed. A reason to try it: with one level the uncertainty did not react to unfamiliar inputs at all ([07_experiments.md, section 7.8](07_experiments.md#78-finding-5-the-head-only-uncertainty-does-not-react-to-unfamiliar-inputs)).

## 6.7 Measured sensitivity to weight noise

Before the values for the partial model were chosen, the effect of weight noise was measured directly.

**Method.** [scripts/weight_noise_sensitivity.py](../../scripts/weight_noise_sensitivity.py) loads a trained plain UNet into a `VariationalPartialUNet` by copying each `weight` into the matching `weight_mean`. Every σ in the variational layers is set to one fixed value, and the validation loss is evaluated with sampled weights (75 images, five passes with a fresh weight sample per batch). The epistemic std is the mean over the first validation batch from 30 samples.

The table is for the checkpoint of `unet_seed42`, whose validation loss without noise is 0.000852. The result is stored in [runs/unet_seed42/weight_noise_sensitivity.json](runs/unet_seed42/weight_noise_sensitivity.json).

| Variational decoder levels | Variational weights | σ | Loss increase | Mean epistemic std |
|---|---|---|---|---|
| 0 | 33 | 0.0003 | +0.0% | 0.0004 |
| 0 | 33 | 0.001 | +0.2% | 0.0015 |
| 0 | 33 | 0.003 | +2.4% | 0.0030 |
| 0 | 33 | 0.01 | +31.6% | 0.0118 |
| 1 | 27,745 | 0.0003 | +0.0% | 0.0005 |
| 1 | 27,745 | 0.001 | +0.3% | 0.0018 |
| 1 | 27,745 | 0.003 | +3.2% | 0.0048 |
| 1 | 27,745 | 0.01 | +54.0% | 0.0168 |
| 2 | 138,465 | 0.0003 | +0.1% | 0.0004 |
| 2 | 138,465 | 0.001 | +0.4% | 0.0018 |
| 2 | 138,465 | 0.003 | +6.0% | 0.0057 |
| 2 | 138,465 | 0.01 | +33.4% | 0.0198 |

Level 0 is the output convolution alone. Level 1 adds the last decoder block, and is the recommended setting.

At σ = 0.0003 the change in loss is at the limit of what the measurement resolves. At σ = 0.01 the loss depends strongly on the individual weight sample, so those rows are rough: the increase for level 2 comes out smaller than for level 1, which cannot be a real effect.

**What it shows:**

- **The cost grows roughly with σ².** From 0.001 to 0.003, a factor of 3 in σ, the loss increase grows by a factor of about 10. That is the quadratic penalty assumed in [section 6.3](#the-equilibrium-and-what-kl_weight-costs). Fitting ½ Σh σ² to the level 1 rows gives a total sensitivity Σh of about 5 to 9.
- **The epistemic std grows roughly in proportion to σ.**
- **There is a usable window.** Between 0.001 and 0.003 the std is about 0.002 to 0.005 for a cost of 0.3% to 3%. Below it the std vanishes. Above it accuracy collapses.
- **The output convolution alone accounts for most of the cost.** Its 33 weights cost 2.4% at σ = 0.003, against 3.2% for all 27,745 weights of level 1.

**Caveat.** This measures a network trained *without* noise, with the same σ on every weight. It says which range of σ is usable. It does not predict the accuracy of a variational model trained in that range, which also depends on how training with noise goes ([07_experiments.md, section 7.6](07_experiments.md#76-finding-3-a-small-variational-head-costs-the-least-accuracy)).

## 6.8 Settings for the other two models

**`variational_unet`.** Tested once with `initial_rho: -8.0`, `prior_standard_deviation: 0.1`, `kl_weight: 0.000001`, no `rho_learning_rate` (the run `full_tuned`): test loss 11% to 15% above the plain UNet, with no visible uncertainty. By the cost rule in [section 6.3](#applying-the-rule), β = 10⁻⁶ is still about 100 times too large for 1.9 million weights. A value near 10⁻⁸ together with `rho_learning_rate: 0.01` is the untested combination most likely to improve it.

**`variational_simple_cnn`.** No run is recorded. With only 289 variational weights, the cost rule gives 2% at β ≈ 7.6 × 10⁻⁵ (taking a loss of 0.00083 as the reference, which this smaller model will not reach). Its variational layer has fan_in 288, so weight std 0.034 and a suitable `initial_rho` of about -6.5 to -7.

## 6.9 Tuning guide

| Goal | Change | Expect |
|---|---|---|
| More visible uncertainty | raise `kl_weight` (3 × 10⁻⁶, then 10⁻⁵) | std up by √ of the factor; loss cost up in proportion |
| More visible uncertainty, more spatial structure | `variational_decoder_levels: 2` and divide `kl_weight` by 5 | more of the decoder contributes to the spread |
| Best possible accuracy | lower `kl_weight` to 10⁻⁷ | closer to the plain UNet, smaller std |
| σ should adapt more | more epochs, or a higher `rho_learning_rate` | wider spread of σ values; untested above 0.01 |
| A proper Bayesian weighting | `kl_weight` ≈ 2s²/P ≈ 1.6 × 10⁻⁸ | very small σ and std; this is what the data alone support |
| Training set size changes | keep β / N fixed, or recompute from the cost rule | the KL per image depends on N |
| Total loss is in the hundreds or thousands | `kl_weight` is orders of magnitude too large | see [section 6.3](#why-01-failed) |

When comparing against a non-variational model, keep `seed`, `epochs`, `learning_rate`, `batch_size`, `normalization` and the augmentation settings identical. Two runs of the same model with different seeds differed by up to 10% here, and two runs with the same seed by about 3%, so a single comparison says little. The comparisons in [07_experiments.md](07_experiments.md) were made at equal seeds where seeded runs exist.

# CPU hyperparameter search for `variational_unet`

This folder holds a hyperparameter search for `variational_unet` run on a 4-core CPU, alongside the GPU sweep in `docs/variational/08_variational_unet_sweep.md` (branch `variational-unet-sweep`). 135 finished trials in five stages. Every file here is a raw record; this README states what they show.

## Result

**Recommended setting:** the GPU sweep's `relu_bs8` with `initial_rho: -7.0` instead of -8.0. The full config is [recommended_variational_unet.yaml](recommended_variational_unet.yaml):

```yaml
model:    {name: variational_unet, activation: relu, normalization: none,
           initial_rho: -7.0, prior_standard_deviation: 0.1}
training: {batch_size: 8, learning_rate: 0.001, scheduler: none, loss: mse,
           kl_weight: 0.00000001, rho_learning_rate: 0.01}
```

With the default `config.yaml` and the notebook runner:

```bash
python scripts/run_notebook.py --set model.name=variational_unet --set model.activation=relu \
  --set model.initial_rho=-7.0 --set model.prior_standard_deviation=0.1 \
  --set training.batch_size=8 --set training.scheduler=none --set training.epochs=40
```

What the evidence supports (stage 5, three fresh seeds, the highest-fidelity runs):

1. **The tuned variational UNet is as accurate as a plain UNet trained the same way.** Six configurations, variational and deterministic, all within 0.11 dB, less than the seed-to-seed spread.
2. **`initial_rho` -7 gives about 3 times the epistemic std of -8 (1.1 × 10⁻³ against 3.9 × 10⁻⁴) at the same accuracy.** -6.5 costs 0.33 dB (stage 4). The usable range ends between -7 and -6.5.
3. **Nothing tried improves on `relu_bs8`'s accuracy.** Every change in stage 4 was neutral or worse on paired seeds.
4. **All runs were still improving at the end.** Longer training is the most likely further gain, as the GPU sweep also found.

## How the search was run

The machine has no GPU: one full-size epoch takes about 8 minutes. So each trial trains on random square crops for a fixed number of steps ([scripts/hpo/proxy_train.py](../../scripts/hpo/proxy_train.py)) and is **always validated on the full 320 × 320 validation images** with the notebook's deterministic noise (seed 42). The runner uses the repository's own model factory, optimiser, KL scaling (`recon + kl_weight · KL / 660`), clamp and PSNR, so every value means what it means in `config.yaml`. Four single-threaded trials run in parallel ([scripts/hpo/sweep.py](../../scripts/hpo/sweep.py)).

Each trial also records the sampled-weight validation loss (`sampling_cost`), the epistemic std over 8 validation images from 8 samples (`std_over_error`, `corr_std_error`) and quantiles of the learned σ.

**CPU runs repeat exactly.** A seeded trial run twice gives identical digits (stage 1b `eps_rho-10` and stage 2 trial 0), unlike the GPU runs in `docs/variational/07_experiments.md`. The remaining noise is between seeds, about 0.1 to 0.3 dB, so every decision below compares configurations on the same seeds.

| Stage | Files | Trials | Proxy budget | Purpose |
|---|---|---|---|---|
| 1 | `stage1*` | 8 of 24 (stopped) | 96 px, 600 steps × 16 | one-at-a-time screen |
| 1b | `stage1b*` | 12 | same | the Adam eps fix (below) |
| 2 | `stage2*`, [stage2_optuna.py](../../scripts/hpo/stage2_optuna.py) | 48 | 9,600 images | multi-objective Optuna: val PSNR and std/error |
| 3 | `stage3*` | 12 of 20 | same | seeded check of the stage-2 winners |
| 4 | `stage4*` | 37 | 96 px, 1,200 steps × 8 | changes on `relu_bs8`, paired seeds 100-102 |
| 5 | `stage5*` | 18 | 128 px, 1,600 steps × 8 | final confirmation, fresh seeds 200-202 |

`*_plan.json` is what each stage ran, `*.jsonl` one line per finished trial, `*.log` the sweep output. `python scripts/hpo/summarize.py results/hpo/stage4.jsonl --sort` prints a table.

## Findings

### Why σ does not move: Adam's eps

The GPU sweep and the earlier `full_tuned` run both found that σ ends at its starting value. The reason is Adam's `eps` (10⁻⁸). The gradient on each ρ is about β/N · (1/σ) · sigmoid(ρ) from the KL, about 10⁻¹⁴ at β = 10⁻⁸, plus a reconstruction part of order 10⁻⁸. Adam divides by √v + eps, so with gradients below eps its step shrinks from `lr` to roughly `lr · g / eps`: ρ is effectively frozen whatever `rho_learning_rate` says.

`proxy_train.py` has a `rho_adam_eps` option that gives the ρ group its own eps (10⁻¹⁶ was used). With it σ does learn: from 3.4 × 10⁻⁴ it grew to 6.7 × 10⁻³ in 600 steps (stage 1b) and to 3.7 × 10⁻² in 1,200 steps without a schedule (stage 4).

**But it does not help.** On paired seeds against `relu_bs8` (stage 4):

| Eps fix with | Δ val PSNR | std / error |
|---|---|---|
| `kl_weight` 10⁻⁸ (3 seeds) | -0.43 dB | 8% (against 7%) |
| `kl_weight` 10⁻⁹ | -0.22 dB | 6% |
| `kl_weight` 10⁻¹⁰ | -0.29 dB | 15%, correlation 0.31 |
| `rho_learning_rate` 10⁻³ | -0.17 dB | 5% |

Once σ can move, the KL term actually acts, σ grows toward the prior on unimportant weights, and accuracy drops without a matching gain in uncertainty. So the frozen σ is a bug in what the code claims to do, but harmless for accuracy: `initial_rho` is then simply the noise-level setting. The option was not moved into `utils/model_configs.py`.

### What `relu_bs8` beats (stage 4, paired seeds)

| Change to `relu_bs8` | Δ val PSNR |
|---|---|
| plain `unet`, same settings (3 seeds) | -0.03 |
| `initial_rho` -7 | -0.02 (std +46%) |
| residual blocks (`scripts/hpo/variants.py`) | +0.12 (-0.04 and +0.27) |
| local reparameterisation | +0.01 (2 × slower, less uncertainty) |
| cosine schedule | -0.08 |
| leaky ReLU | -0.12 |
| warm-up 5% + KL warm-up 30% | -0.12 |
| L1 loss | -0.27 |
| `initial_rho` -6.5 | -0.33 |
| global residual (predict the noise) | -0.48 |
| group norm | -0.49 |

Stages 2 and 3 had preferred leaky ReLU, group norm, L1 and warm-up. That came from a cosine schedule shared by all stage-2 trials and from lucky seeds: the stage-2 "balanced" winner scored 30.01 dB at its own seed and 29.67 dB averaged over three new ones. Stage 4 tested each element on `relu_bs8` alone and agrees with the GPU sweep that they do not help.

### Final confirmation (stage 5)

128 px crops, 1,600 steps at batch 8, seeds 200-202 that no earlier decision used. Validation PSNR in dB.

| Configuration | 200 | 201 | 202 | Mean | sd | Epistemic std | Correlation std vs error |
|---|---|---|---|---|---|---|---|
| residual blocks, ρ -7 | 31.00 | 30.86 | 30.69 | 30.85 | 0.16 | 1.3 × 10⁻³ | 0.47 |
| `residual_unet` (deterministic) | 30.67 | 30.94 | 30.91 | 30.84 | 0.15 | | |
| **`relu_bs8`, ρ -7** | 30.80 | 31.03 | 30.52 | **30.78** | 0.26 | **1.1 × 10⁻³** | **0.58** |
| `relu_bs8` | 30.71 | 30.89 | 30.70 | 30.77 | 0.11 | 3.9 × 10⁻⁴ | 0.62 |
| residual blocks, ρ -8 | 31.01 | 30.82 | 30.39 | 30.74 | 0.32 | 4.9 × 10⁻⁴ | 0.49 |
| `unet` | 30.81 | 30.71 | 30.69 | 30.74 | 0.06 | | |

Residual blocks were not confirmed: at ρ -8 they gained nothing, and the ρ -7 lead is inside the noise, with a worse std-error correlation. `relu_bs8` with ρ -7 is recommended because it is the simplest setting with the extra uncertainty.

## Caveats

- These are proxy runs on crops. Absolute PSNRs are lower than full-size 40-epoch runs (30.8 against 31.5 dB in the GPU sweep), and differences below about 0.2 dB are not resolved. The ranking agrees with the GPU sweep wherever both tested the same thing (batch 8, L1, group norm, ρ -6, cosine).
- The test set was not used by any stage.
- Stage 1 and stage 3 were cut short by stops and container restarts; their missing trials were replaced by later stages.

## Also changed

`models/variational.py` converts `initial_rho` to float, so `initial_rho: -8` written without a decimal point no longer fails with "Only Tensors of floating point and complex dtype can require gradients".

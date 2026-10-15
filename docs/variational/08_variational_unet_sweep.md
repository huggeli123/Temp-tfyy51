# 8. Hyperparameter sweep for `variational_unet`

The fully variational UNet had only been run twice before ([07_experiments.md](07_experiments.md#72-the-runs)), and [section 6.8](06_hyperparameters.md#68-settings-for-the-other-two-models) listed a fair retest as open. This file records a systematic search for the setting that gives `variational_unet` the highest **validation PSNR**: about 90 training runs in three stages.

## 8.1 Summary

1. **The tuned settings beat the previous setting by about 0.4 dB.** Over three seeds and 40 epochs, the best tuned configurations average 31.50 to 31.53 dB validation PSNR, against 31.11 dB for the setting of the earlier run `full_tuned`. Every seed of the three best configurations (lowest 31.42 dB) is above every seed of the old setting (highest 31.13 dB), so this is not run-to-run noise. In test loss it is about 8% lower (0.000733 against 0.000797).
2. **Batch size 8 instead of 16 is the change that matters most.** It was the largest single improvement in the screen (+0.71 dB), and every good trial of the Bayesian search used it.
3. **Once the batch size is 8, the other variational settings matter very little.** Learning rates from 4 × 10⁻⁴ to 1.5 × 10⁻³, `initial_rho` from -11 to -7, `kl_weight` from 10⁻¹⁰ to 3 × 10⁻⁸ and priors from 0.03 to 0.17 all ended within 0.4 dB of each other in 15-epoch trials, which is about the noise between two runs.
4. **GELU and ReLU are a tie,** and **a cosine learning-rate schedule did not help** (one of its three seeds dropped to 30.81 dB).
5. **The learned σ barely moves from its starting value** in any run: it ends 0% to 24% above `softplus(initial_rho)`. So `initial_rho` decides how much weight noise the trained model has, and validation PSNR cannot tell the settings apart. The configuration with the best mean PSNR (`initial_rho` -10) ends up with almost no weight noise and therefore almost no uncertainty. **For a model that is meant to be variational, use `initial_rho` around -8 (or -7.2), not -10**: it gives the same PSNR with 8 to 16 times more weight noise. See [section 8.6](#86-what-the-tuning-did-to-the-uncertainty).

Recommended setting for `variational_unet` (the `relu_bs8` configuration below):

<!-- docs-check: illustrative -->
```yaml
model:
  name: variational_unet
  activation: relu
  initial_rho: -8.0
  prior_standard_deviation: 0.1
training:
  epochs: 40
  batch_size: 8
  learning_rate: 0.001
  scheduler: none
  kl_weight: 0.00000001   # 1e-8
  rho_learning_rate: 0.01
```

All other keys as in [config.yaml](../../config.yaml) (channels 32, depth 3, kernel size 3, no normalization, MSE loss, Adam, augmentation on, noise 0.3).

## 8.2 How the search was done

Three stages, each run with a script that trains in-process with the same functions the notebook uses (`create_dataloaders`, `create_model`, `get_optimizer`, `train_model_variational`, `evaluate_variational`):

| Stage | Script | Trials | Epochs | Purpose |
|---|---|---|---|---|
| 1. Screen | [scripts/sweep_variational_unet.py](../../scripts/sweep_variational_unet.py) | 34 | 15 | change one setting at a time from a starting point, to see which settings matter at all |
| 2. Bayesian search | [scripts/optuna_variational_unet.py](../../scripts/optuna_variational_unet.py) | 42 started: 22 completed, 16 pruned, 4 interrupted | 15 | search the settings that mattered jointly |
| 3. Confirmation | [scripts/sweep_variational_unet.py](../../scripts/sweep_variational_unet.py) | 14 | 40 | train the best candidates longer, with seeds 42, 43 and 44 |

**The quantity optimised** is the validation PSNR of the best-validation-loss checkpoint, using the posterior-mean network (`sample=False`), the same checkpoint the notebook reports test numbers for. The validation noise is deterministic, so all trials are scored on the same noisy images.

**The starting point** of stage 1 followed the advice of [section 6.8](06_hyperparameters.md#68-settings-for-the-other-two-models): `initial_rho` -8, `prior_standard_deviation` 0.1, `kl_weight` 10⁻⁸, `rho_learning_rate` 0.01, everything else as in `config.yaml` (batch size 16, ReLU, learning rate 10⁻³).

**Bayesian optimisation.** Stage 2 used [Optuna](https://optuna.org) with its TPE sampler. TPE fits one probability model to the settings of the good trials and one to the bad trials, and proposes new settings where the ratio between the two is largest. Over time it samples more and more from the region that worked. A pruner stops a trial early if it is clearly failing, which saves GPU time. Optuna is not in `requirements.txt` (that list is kept identical to `environment.yml`); install it with `pip install optuna` to rerun the search.

**Note on comparability with section 7.** Since commit `dd0e84e` the evaluation clamps predictions to [0, 1]. This raises the PSNR of every model, so the numbers here cannot be compared with the older runs in [07_experiments.md](07_experiments.md). The `before` configuration in stage 3 is the old `full_tuned` setting rerun under the new evaluation, and is the fair comparison.

## 8.3 Stage 1: one change at a time

Each row changes one setting from the starting point (`base`). 15 epochs, seed 42, validation PSNR in dB.

| Change | Val PSNR | | Change | Val PSNR |
|---|---|---|---|---|
| **batch size 8** | **31.01** | | prior p = 0.03 | 30.15 |
| activation GELU | 30.73 | | prior p = 0.3 | 30.09 |
| `initial_rho` -10 | 30.67 | | activation leaky ReLU | 30.04 |
| `kl_weight` 10⁻⁷ | 30.46 | | no augmentation | 29.98 |
| `initial_rho` -9 | 30.45 | | group norm | 29.97 |
| shared ρ learning rate | 30.45 | | depth 4 | 29.96 |
| `kl_weight` 10⁻⁹ | 30.41 | | instance norm | 29.85 |
| `initial_rho` -7 | 30.40 | | learning rate 3 × 10⁻³ | 29.84 |
| ρ learning rate 0.03 | 30.36 | | cosine schedule | 29.76 |
| **base** | **30.30** | | depth 2 | 29.65 |
| `kl_weight` 10⁻⁶ | 30.29 | | batch norm | 29.58 |
| prior p = 1.0 | 30.25 | | `initial_rho` -6 | 29.56 |
| `kl_weight` 10⁻¹⁰ | 30.23 | | L1 loss | 29.23 |
| smooth L1 loss | 30.21 | | activation SiLU | 29.14 |
| AdamW, weight decay 10⁻⁴ | 30.19 | | learning rate 2 × 10⁻³ | 28.52 |
| learning rate 3 × 10⁻⁴ | 30.18 | | kernel size 5 | 26.69 |
| `kl_weight` 0 | 30.17 | | | |
| 48 channels | 30.16 | | | |

A 64-channel trial was started and abandoned: the model did not fit in the 8 GB of the GPU and ran many times slower. A batch size 4 trial (in the first, abandoned Optuna study) reached 31.07 dB, no better than batch size 8, at six times the training time, and was dropped from the search.

**How to read this table.** A single 15-epoch run is noisy. GELU alone gave 30.73 dB and `initial_rho` -10 alone 30.67 dB, but the two together (Optuna trial 2) gave 29.91 dB, and the validation curves still rise at epoch 15. Only the large effects are reliable here:

- **Clearly better:** batch size 8.
- **Clearly worse:** kernel size 5, learning rate 2 × 10⁻³ or more, `initial_rho` -6 (too much starting noise), L1 loss, SiLU, batch norm, depth 2.
- **Not distinguishable from base:** everything between about 30.0 and 30.5 dB, including every `kl_weight` from 0 to 10⁻⁶. The KL term is far too small to matter at these values: with `kl_weight` 0 the result is the same as with 10⁻⁶.
- **The cosine row is misleading:** over 15 epochs the schedule brings the learning rate to zero while the model is still improving. It was retested at 40 epochs in stage 3.

## 8.4 Stage 2: Bayesian search with Optuna

**Search space**, chosen from stage 1: learning rate 2 × 10⁻⁴ to 1.5 × 10⁻³ (log scale), batch size {8, 16}, `initial_rho` -12 to -7, `prior_standard_deviation` 0.03 to 1 (log), `kl_weight` 10⁻¹⁰ to 10⁻⁶ (log), `rho_learning_rate` 10⁻³ to 3 × 10⁻² (log), activation {ReLU, GELU}, scheduler {none, cosine}. Architecture was fixed at 32 channels, depth 3, kernel size 3, because stage 1 showed no gain from changing it.

**Pruning had to be adjusted twice.** A median pruner (stop a trial whose best PSNR so far is below the median of earlier trials at the same epoch) stopped 13 trials in a row, some of them level with the eventual best trial at the same epoch: at epoch 6 the run-to-run noise (about 0.3 dB) is larger than the differences it decided on. A 25th-percentile pruner at epoch 8 had the same problem, because only strong trials had ever reached epoch 8. The final pruner only stops a trial whose best PSNR is below 29.3 dB after 8 epochs, which catches failures only. The trials of all three phases are in the same study.

**Result.** The best 19 completed trials:

| Trial | Val PSNR | Batch | Activation | Learning rate | `initial_rho` | `kl_weight` | Prior | ρ learning rate |
|---|---|---|---|---|---|---|---|---|
| 1 | 31.21 | 8 | GELU | 1.0 × 10⁻³ | -10.0 | 1 × 10⁻⁸ | 0.10 | 0.010 |
| 38 | 31.09 | 8 | ReLU | 6.9 × 10⁻⁴ | -7.2 | 8 × 10⁻¹⁰ | 0.17 | 0.009 |
| 36 | 31.09 | 8 | ReLU | 8.4 × 10⁻⁴ | -7.6 | 4 × 10⁻⁹ | 0.11 | 0.006 |
| 29 | 31.08 | 8 | ReLU | 6.3 × 10⁻⁴ | -7.8 | 1 × 10⁻⁸ | 0.06 | 0.003 |
| 24 | 31.08 | 8 | ReLU | 1.4 × 10⁻³ | -8.4 | 1 × 10⁻⁹ | 0.08 | 0.003 |
| 0 | 31.07 | 8 | ReLU | 1.0 × 10⁻³ | -8.0 | 1 × 10⁻⁸ | 0.10 | 0.010 |
| 25 | 31.04 | 8 | ReLU | 1.3 × 10⁻³ | -7.5 | 4 × 10⁻⁹ | 0.12 | 0.003 |
| 32 | 31.03 | 8 | ReLU | 8.9 × 10⁻⁴ | -7.6 | 3 × 10⁻⁹ | 0.05 | 0.002 |
| 35 | 31.00 | 8 | ReLU | 5.1 × 10⁻⁴ | -8.7 | 4 × 10⁻⁹ | 0.04 | 0.003 |
| 34 | 31.00 | 8 | ReLU | 8.2 × 10⁻⁴ | -7.6 | 1 × 10⁻⁸ | 0.05 | 0.007 |
| 27 | 31.00 | 8 | ReLU | 1.3 × 10⁻³ | -8.5 | 6 × 10⁻⁹ | 0.06 | 0.003 |
| 28 | 30.95 | 8 | ReLU | 1.5 × 10⁻³ | -7.1 | 2 × 10⁻¹⁰ | 0.16 | 0.005 |
| 37 | 30.94 | 8 | ReLU | 8.9 × 10⁻⁴ | -7.2 | 1 × 10⁻⁸ | 0.13 | 0.007 |
| 3 | 30.94 | 8 | ReLU | 4.3 × 10⁻⁴ | -9.0 | 4 × 10⁻¹⁰ | 0.05 | 0.001 |
| 26 | 30.93 | 8 | ReLU | 1.2 × 10⁻³ | -8.2 | 3 × 10⁻⁹ | 0.06 | 0.010 |
| 30 | 30.91 | 8 | ReLU | 5.0 × 10⁻⁴ | -8.1 | 2 × 10⁻⁸ | 0.06 | 0.002 |
| 33 | 30.89 | 8 | ReLU | 1.0 × 10⁻³ | -7.1 | 1 × 10⁻⁹ | 0.08 | 0.006 |
| 4 | 30.89 | 8 | ReLU | 1.4 × 10⁻³ | -11.1 | 2 × 10⁻⁹ | 0.06 | 0.006 |
| 31 | 30.84 | 8 | ReLU | 4.0 × 10⁻⁴ | -7.6 | 3 × 10⁻⁸ | 0.03 | 0.002 |

The two completed trials with batch size 16 reached 29.91 and 29.64 dB. Trials 0 to 2 were the hand-picked starting points.

**What it shows.** The search settled on batch size 8, ReLU and no schedule, and then found a broad plateau: 19 quite different settings all within 31.21 to 30.84 dB, a spread no larger than the noise between two runs of the same setting. There is no sharp optimum to find in these variational settings.

The parameter importances Optuna reports ([sweep/optuna_param_importance.json](sweep/optuna_param_importance.json)) rank `prior_standard_deviation` first and batch size last. Do not read that as "batch size does not matter": importance is computed over the completed trials only, and almost all of those used batch size 8, so the data contains little variation in batch size to explain anything with. The ranking among the other settings describes differences within the plateau, which are inside the noise.

GELU was proposed only a few times after the starting points. Some of those trials were stopped by the over-eager first pruner, so the search did not test GELU fairly. Stage 3 does.

## 8.5 Stage 3: confirmation over three seeds

Five configurations, 40 epochs, seeds 42, 43 and 44:

| Name | What it is |
|---|---|
| `before` | the earlier `full_tuned` setting: batch 16, ReLU, `initial_rho` -8, prior 0.1, `kl_weight` 10⁻⁶, shared ρ learning rate |
| `best_gelu` | Optuna trial 1: batch 8, GELU, `initial_rho` -10, prior 0.1, `kl_weight` 10⁻⁸, ρ learning rate 0.01, learning rate 10⁻³ |
| `best_gelu_cosine` | `best_gelu` with a cosine schedule |
| `relu_bs8` | Optuna trial 0: as `best_gelu` but ReLU and `initial_rho` -8 |
| `optuna_relu` | Optuna trial 38: batch 8, ReLU, `initial_rho` -7.2, prior 0.17, `kl_weight` 8 × 10⁻¹⁰, ρ learning rate 0.009, learning rate 6.9 × 10⁻⁴ |

Validation PSNR (dB) of the best checkpoint, then test PSNR and test loss of the same checkpoint:

| Configuration | Seed 42 | Seed 43 | Seed 44 | **Mean val** | Mean test PSNR | Mean test loss |
|---|---|---|---|---|---|---|
| `best_gelu` | 31.56 | 31.61 | 31.42 | **31.53** | 31.59 | 0.000733 |
| `relu_bs8` | 31.45 | 31.51 | 31.55 | **31.50** | 31.59 | 0.000733 |
| `optuna_relu` | 31.54 | 31.43 | not run | **31.49** | 31.56 | 0.000736 |
| `best_gelu_cosine` | 31.55 | 31.52 | 30.81 | **31.29** | 31.36 | 0.000771 |
| `before` | 31.11 | 31.13 | 31.09 | **31.11** | 31.18 | 0.000797 |

The `optuna_relu` seed 44 run was interrupted and not repeated. Records of all runs, including the per-epoch histories, are in [sweep/stage3_confirm/](sweep/stage3_confirm/).

**What is established:**

- All three batch-size-8 configurations without a schedule beat `before` at every seed, by 0.3 to 0.5 dB. The improvement is real.
- `best_gelu`, `relu_bs8` and `optuna_relu` cannot be told apart: their means are 0.04 dB apart, while each one's seeds spread over 0.1 to 0.2 dB.
- The cosine schedule is not better. Two seeds matched the others, and one fell to 30.81 dB.

**What is not established:**

- That GELU is better than ReLU. The difference of 0.03 dB is far inside the noise.
- That 40 epochs is enough. The best checkpoint was at epoch 32 to 40, most often in the last five epochs, so the models were still improving. Longer training is the most likely further gain.
- How the tuned `variational_unet` compares with a plain `unet` trained the same way (batch 8, 40 epochs). Those reference runs were prepared but not completed. The older `unet` runs in section 7 used batch 16, 30 epochs and the unclamped evaluation, so they are not a fair comparison.

## 8.6 What the tuning did to the uncertainty

[scripts/variational_summary.py](../../scripts/variational_summary.py) was run on the seed-42 checkpoints of three configurations. The learned σ was also read from all 14 checkpoints. (`variational_summary.py` computes the uncertainty on the first validation batch, which has 16 images at batch size 16 and 8 at batch size 8.)

| Seed 42 | `initial_rho` | Starting σ | Learned σ (median) | Test PSNR | Mean epistemic std | Correlation std vs error |
|---|---|---|---|---|---|---|
| `before` | -8 | 3.35 × 10⁻⁴ | 4.16 × 10⁻⁴ | 31.19 | 0.00084 | 0.64 |
| `relu_bs8` | -8 | 3.35 × 10⁻⁴ | 3.51 × 10⁻⁴ | 31.54 | 0.00051 | 0.63 |
| `best_gelu` | -10 | 4.54 × 10⁻⁵ | 4.75 × 10⁻⁵ | 31.60 | 0.00006 | 0.54 |

Over all 14 confirmation checkpoints the learned σ ended between 1.00 and 1.24 times its starting value: 1.24 for `before`, 1.03 to 1.05 for the others. With `kl_weight` at 10⁻⁸ or below, the KL term barely pushes σ up, while the reconstruction loss pushes it down only slowly. Over 40 epochs the starting value is what remains. In practice **`initial_rho` sets the amount of weight noise**, and validation PSNR is nearly blind to it in this range:

- `best_gelu` (`initial_rho` -10) has the highest mean PSNR, but its weight noise is so small that the epistemic std is 0.00006, about 8 times less than `relu_bs8`. Sampled-weight and mean-weight test PSNR agree to the fourth decimal (31.5994 and 31.5993). It behaves as a deterministic UNet.
- `relu_bs8` (`initial_rho` -8) has the same PSNR within noise and keeps a visible uncertainty. Its std is 2.7 times larger at input noise 0.5 than at the training level 0.3 (0.00104 against 0.00039), and it correlates 0.63 with the error.
- `optuna_relu` (`initial_rho` -7.2) keeps σ ≈ 7.5 × 10⁻⁴, 16 times `best_gelu`'s, at the same PSNR (31.49 dB mean).

So a search that only maximises PSNR drifts toward removing the weight noise, which defeats the purpose of a variational model, while gaining nothing measurable. This is why the recommendation in [section 8.1](#81-summary) is `relu_bs8` and not the configuration with the highest mean. If more uncertainty is wanted, `initial_rho` between -8 and -7 is the setting to raise. -6 was clearly worse in stage 1 (29.56 dB against 30.30), so the useful range ends somewhere between -7 and -6.

## 8.7 Files and how to rerun

| What | Where |
|---|---|
| Screening and confirmation runner | [scripts/sweep_variational_unet.py](../../scripts/sweep_variational_unet.py) `trials.json --stage NAME [--keep-checkpoints]` |
| Bayesian search | [scripts/optuna_variational_unet.py](../../scripts/optuna_variational_unet.py) `--trials 40 --epochs 15` (needs `pip install optuna`) |
| Ranked table of the runner's trials | [scripts/summarize_sweep.py](../../scripts/summarize_sweep.py) |
| One line per finished trial, all stages | [sweep/all_trials.jsonl](sweep/all_trials.jsonl) |
| All Optuna trials with their settings | [sweep/optuna_trials.json](sweep/optuna_trials.json) |
| Optuna parameter importance | [sweep/optuna_param_importance.json](sweep/optuna_param_importance.json) |
| Config, history and uncertainty summary of each confirmation run | [sweep/stage3_confirm/](sweep/stage3_confirm/) |

The full outputs, including checkpoints and the Optuna database, are in `results/variational_unet_sweep/` on the machine the sweep ran on (not in git). The runs were made with Python 3.14, PyTorch 2.11 and an RTX 3060 Ti. One 40-epoch run takes about 7.5 minutes.

## 8.8 Open questions

1. **Plain `unet` at batch 8 and 40 epochs, three seeds.** Without it, it is unknown whether the tuned variational UNet now matches the deterministic one or whether the plain UNet improved by the same amount. The runner already supports it: a trial with `model.name: unet` trains a plain model with `train_model`.
2. **Longer training (60 to 80 epochs).** The best checkpoints were still at the end of training.
3. **A wider `initial_rho` range at batch 8,** for example -7.5, -7 and -6.5 over three seeds, to find how much weight noise can be added before PSNR drops.

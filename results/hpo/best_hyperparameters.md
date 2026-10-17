# Best hyperparameters for `variational_unet`

This file combines two independent searches and states the best setting each hyperparameter has evidence for.

| | GPU sweep | CPU search |
|---|---|---|
| Where | `docs/variational/08_variational_unet_sweep.md` and `docs/variational/sweep/` on branch `variational-unet-sweep` | [README.md](README.md) and the `stage*.jsonl` records in this folder |
| Hardware | RTX 3060 Ti | 4-core CPU, no GPU |
| Training | full 320 × 320 images, 15 epochs (screen and Optuna), 40 epochs (confirmation) | random 96 or 128 px crops, 600 to 1,600 steps, always validated on the full-size images |
| Runs | about 90: 34 screen, 42 Optuna, 14 confirmation | 135 finished: 5 stages, the last two on paired seeds |
| Strength | real training length and image size | many paired-seed comparisons, plain UNet references, uncertainty measured in every trial |

The two searches were run separately and with different methods. Where both tested a setting, they point the same way every time, which is the main reason to trust the result.

## The recommended setting

```yaml
model:
  name: variational_unet
  channels: 32
  depth: 3
  kernel_size: 3
  activation: relu
  normalization: none
  initial_rho: -7.0
  prior_standard_deviation: 0.1
augmentation:
  enabled: true            # flip p = 0.5, rotation 10 degrees
training:
  epochs: 40               # at least; 60-80 is likely better, untested
  batch_size: 8
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0
  scheduler: none
  loss: mse
  kl_weight: 0.00000001    # 1e-8
  rho_learning_rate: 0.01
```

The same file with every other key filled in from `config.yaml` is [recommended_variational_unet.yaml](recommended_variational_unet.yaml).

This is the GPU sweep's recommended `relu_bs8` with one change: `initial_rho` -7.0 instead of -8.0. That change gives about three times the uncertainty at the same accuracy (see [initial_rho](#initial_rho)).

**Expected result.** The GPU sweep's `relu_bs8` (with ρ -8) reached 31.50 dB mean validation PSNR and 31.59 dB test PSNR over three seeds at 40 epochs, against 31.11 / 31.18 dB for the earlier `full_tuned` setting. The CPU search found ρ -7 level with ρ -8 in accuracy, so the same numbers are expected. **The ρ -7 version has not been run at full size yet.**

**How it compares with a plain UNet.** The CPU search trained a plain `unet` with the same settings on the same seeds, in two stages: it ended 0.03 dB below `relu_bs8` both times (stage 4: 3 seeds, stage 5: 3 fresh seeds), so it is equally accurate. The tuned variational model costs nothing in accuracy, but it does not beat the plain one either. The GPU sweep left this comparison open.

## Hyperparameter by hyperparameter

**How to read the numbers.**
- **GPU Δ** is a single 15-epoch run at seed 42 against the GPU screen's base (30.30 dB: batch 16, ReLU, ρ -8). Two GPU runs of the same setting can differ by about 0.3 dB, so only differences larger than that count.
- **CPU Δ** is the paired mean over 2 or 3 seeds against `relu_bs8` (stage 4, 30.27 dB), unless the cell says otherwise.
- "–" means that search did not test the setting.

### Settings with a clear answer

| Setting | Best value | GPU evidence | CPU evidence | Verdict |
|---|---|---|---|---|
| `batch_size` | **8** | 8: +0.71 dB, the largest single gain; confirmed +0.4 dB over 3 seeds at 40 epochs. 4: no better, 6 × slower | every top trial of the stage-2 Optuna search used 8 | **agreed** |
| `learning_rate` | **1e-3** | 3e-4: -0.12; 2e-3: -1.78; 3e-3: -0.46. Optuna plateau 4e-4 to 1.5e-3 | Optuna top trials 6e-4 to 1.8e-3; one trial at 2.9e-3 collapsed (16.4 dB) | **agreed**: 5e-4 to 1.5e-3 all work, above 2e-3 fails |
| `scheduler` | **none** | cosine -0.54 at 15 epochs; at 40 epochs one of three seeds fell to 30.81 dB | cosine -0.08 | **agreed** |
| `activation` | **relu** (GELU equal) | GELU +0.43 alone, but tied with ReLU over 3 seeds at 40 epochs (0.03 dB apart); leaky ReLU -0.26; SiLU -1.16 | leaky ReLU -0.12; SiLU never in the top stage-2 trials | **agreed**: ReLU or GELU |
| `normalization` | **none** | group -0.33; instance -0.45; batch -0.72 | group -0.49; with batch norm, sampled weights were often far worse than the mean weights (up to +105% loss) | **agreed** |
| `loss` | **mse** | L1 -1.08; smooth L1 -0.09 | L1 -0.27 | **agreed** |
| `optimizer` / `weight_decay` | **adam**, 0 | AdamW with weight decay 1e-4: -0.11 | – | GPU only, no gain |
| `augmentation` | **on** | off: -0.32 | – | GPU only |
| `kernel_size` | **3** | 5: -3.61 | – | GPU only, clearly |
| `depth` | **3** | 2: -0.65; 4: -0.35 | – | GPU only |
| `channels` | **32** | 48: -0.14; 64 did not fit in 8 GB | – | GPU only, no gain |
| `epochs` | **40 or more** | best checkpoints at epoch 32 to 40, mostly the last five | every validation curve still rising at the end of the run | **agreed**: train longer |

### `initial_rho`

The setting that decides how variational the model is. Both searches found that the learned σ stays at its starting value (see [Adam eps](#why-σ-stays-where-it-starts-adams-eps)), so `initial_rho` fixes the amount of weight noise.

| ρ | Starting σ | GPU (15 epochs, 1 seed) | CPU stage 4 (paired) | CPU stage 5 (3 fresh seeds) |
|---|---|---|---|---|
| -10 | 4.5 × 10⁻⁵ | +0.37; best mean over 3 seeds at 40 epochs (31.53) but epistemic std only 6 × 10⁻⁵ | 30.09, level with the plain UNet; std 1% of the error (stage 1b) | – |
| -9 | 1.2 × 10⁻⁴ | +0.15 | – | – |
| -8 | 3.4 × 10⁻⁴ | base; 31.50 over 3 seeds at 40 epochs, std 5.1 × 10⁻⁴ | reference | 30.77 dB, std 3.9 × 10⁻⁴ |
| **-7** | **9.1 × 10⁻⁴** | +0.10; Optuna's ρ -7.2 setting reached 31.49 over 2 seeds at 40 epochs | **-0.02**, std +46% | **30.78 dB, std 1.1 × 10⁻³ (× 2.8)** |
| -6.5 | 1.5 × 10⁻³ | – | -0.33 | – |
| -6 | 2.5 × 10⁻³ | -0.74 | -0.17 against ρ -8 (stage 1b, one seed, eps fix on) | – |

**Best value: -7.** It is the most weight noise that costs no accuracy in either search.
- -10 is as accurate, but it is a deterministic UNet in practice. The GPU sweep measured its sampled-weight and mean-weight test PSNR equal to four decimals.
- The usable range ends between -7 and -6.5.
- At -7 the uncertainty map still follows the error (correlation 0.58, against 0.62 at -8), and drawing sampled weights costs 0.2% in loss.

### `kl_weight`, `prior_standard_deviation`, `rho_learning_rate`

| Setting | Best value | GPU evidence | CPU evidence |
|---|---|---|---|
| `kl_weight` | **1e-8** | 0 to 1e-6 all within ±0.16 dB, which is noise; Optuna plateau 2e-10 to 3e-8 | same flat result with Adam's default eps; with the eps fix, every value tried lost accuracy (1e-8: -0.43, 1e-9: -0.22, 1e-10: -0.29) |
| `prior_standard_deviation` | **0.1** | 0.03: -0.15; 0.3: -0.21; 1.0: -0.05 | 1.0 equal to 0.1 (stage 1b) |
| `rho_learning_rate` | **0.01** (or unset) | shared: +0.15; 0.03: +0.06 (noise) | irrelevant with the default eps; with the eps fix, 0.1 destroyed the model (16.4 dB) and 1e-3 cost -0.17 |

**All three are effectively inactive at the recommended setting**, because σ cannot move (next section). That is why the GPU sweep found them flat. Keep the documented values. Changing them has no measurable effect unless Adam's eps for the ρ group is changed as well.

### Why σ stays where it starts: Adam's eps

This was found by the CPU search only. It explains the GPU sweep's observation that "the learned σ barely moves from its starting value".

1. **The gradient on ρ is tiny.** At β = 1e-8, its KL part is about β/N · (1/σ) · sigmoid(ρ) ≈ 10⁻¹⁴, and its reconstruction part is of order 10⁻⁸.
2. **Adam's eps swamps it.** Adam divides each step by √v + eps, with eps = 10⁻⁸. With a gradient this far below eps, the step shrinks from `lr` to about `lr · g / eps`. ρ is frozen, whatever `rho_learning_rate` is set to.
3. **A smaller eps unfreezes it.** Giving the ρ group its own eps of 10⁻¹⁶ (`rho_adam_eps` in `scripts/hpo/proxy_train.py`) lets σ learn: it grew 20 to 100 times in a few hundred steps.
4. **But the unfrozen σ does not pay off.** In every combination tried, it cost 0.2 to 0.4 dB without more useful uncertainty: σ grows on weights that do not matter.

**Conclusion: leave Adam's eps at its default.** `initial_rho` is then the one control for weight noise, as recommended above. The fix is not in the training code.

### Architecture and training-method variants

| Variant | GPU | CPU | Verdict |
|---|---|---|---|
| residual blocks (variational version of `residual_unet`) | – | stage 4: +0.12 (-0.04 and +0.27); stage 5: ρ -8 +0.00, ρ -7 +0.11, both inside the noise, and the uncertainty follows the error less well (0.47) | not established, not recommended |
| global residual (predict the noise) | – | -0.48 | worse |
| local reparameterisation trick | – | +0.01, 2 × slower, a third of the uncertainty | no gain |
| learning-rate warm-up 5% + KL warm-up 30% | – | -0.12 | no gain |
| `variational_partial_unet` instead | earlier runs in `07_experiments.md`: 1% to 14% worse than the plain UNet | – | the fully variational model is now as accurate, so the partial model loses its main argument |

## Where the searches disagreed, and why

Only once.

**The disagreement.** CPU stages 2 and 3 favoured leaky ReLU, group norm, L1 loss and warm-up; the GPU screen ranked all of them worse.

**The cause.** All stage-2 trials shared a cosine schedule, and the stage-2 winners had lucky seeds. The "balanced" winner scored 30.01 dB at its own seed and 29.67 dB averaged over three new ones.

**The resolution.** CPU stage 4 tested each element separately on `relu_bs8`, on paired seeds, and found every one neutral or worse. That matches the GPU sweep. The GPU result stands.

**The lesson.** A single run, or a search that picks its best trial by score, overstates by about 0.3 dB in this problem. Every recommendation in this file rests on settings compared over several seeds.

## Comparability caveats

- **Absolute PSNR differs between the searches** (CPU about 30.8 dB, GPU about 31.5 dB), because the CPU trials train on crops for a fraction of an epoch-equivalent budget. Only the differences within each search are compared here.
- **The CPU trials repeat exactly on the same seed. The GPU trials do not:** two identical runs ended 2.8% apart in test loss, as recorded in `07_experiments.md`. GPU single-run differences below about 0.3 dB are noise.
- **Neither search chose anything on the test set.** The GPU confirmation reports test PSNR for information only.

## What to run next

To make this recommendation final, on the GPU with the notebook runner:

1. **`relu_bs8` with ρ -7.0, three seeds, 40 epochs.** This confirms at full size that it matches ρ -8 (31.50 dB):

   ```bash
   python scripts/run_notebook.py --set model.name=variational_unet --set model.activation=relu \
     --set model.initial_rho=-7.0 --set model.prior_standard_deviation=0.1 \
     --set training.batch_size=8 --set training.scheduler=none --set training.epochs=40 \
     --set training.kl_weight=0.00000001 --set training.rho_learning_rate=0.01 --set seed=42
   ```

   Repeat with `seed=43` and `seed=44`.
2. **The same at 80 epochs, one seed.** Both searches saw the models still improving at the end.
3. **Plain `unet` at batch 8 and 40 epochs, three seeds.** This repeats the CPU finding that the two are equally accurate, at full size.

# 7. Experiments and findings

Every run the conclusions rest on, with exact numbers, what each showed, and what remains open.

Every number in the tables of this file is read from a record in [runs/](runs/). `python scripts/check_variational_docs.py` fails if a table and its record disagree.

## 7.1 Common setup

All runs share these settings:

| Setting | Value |
|---|---|
| Data | 660 training, 75 validation, 75 test images, 320 × 320, greyscale |
| Noise | Gaussian, std 0.3 for training, validation and test |
| Augmentation | horizontal flip (probability 0.5), rotation ±10° |
| Architecture | `channels: 32`, `depth: 3`, `kernel_size: 3`, ReLU, no normalisation |
| Optimiser | Adam, learning rate 0.001, no weight decay, batch size 16 |
| Loss | MSE |

All runs were made with the notebook, on Python 3.14.0 with PyTorch 2.11.0+cu128 on an NVIDIA GeForce RTX 3060 Ti. That is a pip environment (`requirements.txt`), not the Conda environment in `environment.yml`.

The three unseeded runs were made by hand in the notebook. The seven seeded runs were made by executing the same notebook from the command line with [scripts/run_notebook.py](../../scripts/run_notebook.py), which changes the seed, the model or the ρ learning rate for one run without editing `config.yaml`. The `config.yaml` archived with each run records the settings that run actually used.

Test loss and test PSNR are always from the best-validation checkpoint. For variational models they use the posterior-mean network.

**Seeding.** The three runs whose names contain no seed were made before the notebook called `set_seed`. Their initial weights, batch order and training noise were random. The other seven are seeded. Runs with the same seed start from the same weights and see the same batches and the same training noise.

**Seeded runs still do not repeat exactly.** `partial_learned_seed42_repeat` is `partial_learned_seed42` run a second time, with nothing changed. After the first epoch the two training losses agree to five digits, so the seed does its job. From the second epoch on they drift apart, because arithmetic on the GPU is not deterministic, and after 30 epochs their test losses differ by 2.8%.

**How much runs differ.** This sets the limit on what can be concluded:

- The same configuration and seed, run twice: 2.8% apart.
- The plain UNet, three runs: test loss from 0.000807 to 0.000834, a spread of 3%.
- The recommended variational model, seed 42 against seed 43: 7% to 10% apart.

So a difference of a few percent between two single runs is not evidence of anything, even at the same seed. Only differences that are large, or that repeat across several runs, mean something.

## 7.2 The runs

| Run | Model | Seed | `initial_rho` | Prior p | `kl_weight` | ρ learning rate | Epochs | Test loss | Test PSNR |
|---|---|---|---|---|---|---|---|---|---|
| `full_default_10ep` | `variational_unet` | none | -5.0 | 1.0 | 0.1 | shared | 10 | 0.001562 | 28.15 dB |
| `full_tuned` | `variational_unet` | none | -8.0 | 0.1 | 1e-06 | shared | 30 | 0.000924 | 30.50 dB |
| `unet_unseeded` | `unet` | none | | | | | 30 | 0.000831 | 30.99 dB |
| `unet_seed42` | `unet` | 42 | | | | | 30 | 0.000834 | 30.98 dB |
| `unet_seed43` | `unet` | 43 | | | | | 30 | 0.000807 | 31.12 dB |
| `partial_learned_seed42` | `variational_partial_unet` | 42 | -6.5 | 1.0 | 1e-06 | 0.01 | 30 | 0.000841 | 30.93 dB |
| `partial_learned_seed42_repeat` | `variational_partial_unet` | 42 | -6.5 | 1.0 | 1e-06 | 0.01 | 30 | 0.000865 | 30.81 dB |
| `partial_learned_seed43` | `variational_partial_unet` | 43 | -6.5 | 1.0 | 1e-06 | 0.01 | 30 | 0.000922 | 30.51 dB |
| `partial_sharedlr_seed42` | `variational_partial_unet` | 42 | -6.5 | 1.0 | 1e-06 | shared | 30 | 0.000875 | 30.76 dB |
| `partial_sharedlr_seed43` | `variational_partial_unet` | 43 | -6.5 | 1.0 | 1e-06 | shared | 30 | 0.000963 | 30.33 dB |

"Shared" means the ρ parameters used the ordinary learning rate of 0.001. Both `variational_partial_unet` configurations use `variational_decoder_levels: 1`.

Each name is a folder in [runs/](runs/) holding the run's `config.yaml`, its `results.json` with the full training history, and for variational models a `variational_summary.json`. `partial_learned_seed42` is the configuration currently in `config.yaml`, and `partial_learned_seed42_repeat` is a second run of exactly that.

**Earlier trial runs that are not archived.** Before these runs, the two `variational_partial_unet` configurations were each trained once from a script, not from the notebook, and nothing was saved. They gave test losses of 0.000829 (separate ρ rate) and 0.001017 (shared rate). An earlier version of this file drew two conclusions from that single pair: that the model "matched" the plain UNet, and that the separate ρ learning rate made a 23% difference. Neither held up in the seeded runs above. See sections 7.6 and 7.7.

## 7.3 Numbers for the variational runs

These three tables come from `variational_summary.json`, written by [scripts/variational_summary.py](../../scripts/variational_summary.py) from each run's best checkpoint.

**Learned σ, in units of 10⁻³, and the effect of sampling.** The last two columns compare the test loss with sampled weights (average of 5 passes) with the posterior-mean test loss in section 7.2.

| Run | Smallest σ | 5th percentile | Median | 95th percentile | Largest σ | Test loss, sampled weights | Cost of sampling |
|---|---|---|---|---|---|---|---|
| `full_default_10ep` | 7.22 | 10.20 | 10.20 | 10.20 | 10.23 | 0.001853 | +18.6% |
| `full_tuned` | 0.31 | 0.39 | 0.40 | 0.40 | 0.41 | 0.000925 | +0.1% |
| `partial_learned_seed42` | 0.72 | 1.36 | 2.37 | 7.42 | 7.56 | 0.000847 | +0.7% |
| `partial_learned_seed42_repeat` | 0.74 | 1.37 | 2.36 | 7.84 | 8.28 | 0.000872 | +0.9% |
| `partial_learned_seed43` | 0.67 | 1.32 | 2.31 | 5.64 | 7.79 | 0.000932 | +1.1% |
| `partial_sharedlr_seed42` | 1.35 | 1.49 | 1.56 | 1.74 | 1.75 | 0.000886 | +1.2% |
| `partial_sharedlr_seed43` | 1.36 | 1.49 | 1.59 | 1.77 | 1.78 | 0.000967 | +0.5% |

Starting values: 6.72 for `full_default_10ep`, 0.34 for `full_tuned`, 1.50 for the five partial runs.

**Uncertainty on the first validation batch**, 16 images, 30 stochastic forward passes.

| Run | Mean epistemic std | Mean absolute error | Std as a share of the error | Correlation of std with error |
|---|---|---|---|---|
| `full_default_10ep` | 0.01360 | 0.01797 | 76% | 0.64 |
| `full_tuned` | 0.00056 | 0.01164 | 5% | 0.63 |
| `partial_learned_seed42` | 0.00206 | 0.00923 | 22% | 0.57 |
| `partial_learned_seed42_repeat` | 0.00201 | 0.00895 | 23% | 0.57 |
| `partial_learned_seed43` | 0.00205 | 0.00962 | 21% | 0.57 |
| `partial_sharedlr_seed42` | 0.00259 | 0.01047 | 25% | 0.55 |
| `partial_sharedlr_seed43` | 0.00249 | 0.01215 | 21% | 0.51 |

**Mean epistemic std at four input noise levels.** The same 16 clean validation images, corrupted at each level. Every model was trained at 0.3 only.

| Run | Noise 0.0 | Noise 0.1 | Noise 0.3 (training) | Noise 0.5 | Std at 0.5 relative to 0.3 | PSNR at 0.3 | PSNR at 0.5 |
|---|---|---|---|---|---|---|---|
| `full_default_10ep` | 0.01196 | 0.01081 | 0.01203 | 0.02283 | × 1.90 | 29.34 dB | 17.56 dB |
| `full_tuned` | 0.00068 | 0.00056 | 0.00064 | 0.00173 | × 2.71 | 31.61 dB | 18.23 dB |
| `partial_learned_seed42` | 0.00146 | 0.00173 | 0.00197 | 0.00238 | × 1.21 | 32.09 dB | 18.85 dB |
| `partial_learned_seed42_repeat` | 0.00147 | 0.00188 | 0.00199 | 0.00237 | × 1.19 | 31.99 dB | 18.75 dB |
| `partial_learned_seed43` | 0.00175 | 0.00174 | 0.00233 | 0.00217 | × 0.93 | 31.74 dB | 19.08 dB |
| `partial_sharedlr_seed42` | 0.00178 | 0.00224 | 0.00243 | 0.00263 | × 1.08 | 31.97 dB | 19.10 dB |
| `partial_sharedlr_seed43` | 0.00223 | 0.00216 | 0.00227 | 0.00235 | × 1.03 | 31.25 dB | 18.55 dB |

## 7.4 Finding 1: the natural defaults break the fully variational UNet

`full_default_10ep` used the settings that look reasonable at first sight: every layer variational, `kl_weight: 0.1`, `initial_rho: -5`, a standard-normal prior.

What its training log shows:

| Epoch | `train_loss` (reconstruction) | `train_kl` (KL / N) | `train_total_loss` |
|---|---|---|---|
| 1 | 0.00578 | 13,077 | 1,307.67 |
| 5 | 0.00164 | 12,588 | 1,258.85 |
| 10 | 0.00159 | 11,979 | 1,197.92 |

- **The KL term is about a million times the reconstruction loss.** The optimiser is, numerically, minimising the KL.
- **The KL falls by a constant amount each epoch**, about 122. Spread over 1.9 million weights that is 0.042 nats per weight per epoch, which is what results if every ρ rises by one learning rate (0.001) on each of the 42 steps.
- **The σ table confirms it exactly.** After 420 steps, 90% of all weights have the same σ of 10.20 × 10⁻³. That is ρ = -5 raised by 0.42. Adam pushed every σ upward at its maximum rate for the whole run. The reconstruction term had no say.
- **The validation loss does not settle.** It moves between 0.0016 and 0.0022 with no steady trend.
- **Sampling is expensive.** With sampled weights the test loss is 18.6% higher than with the mean weights.

This run lasted 10 epochs, so it cannot be compared with the 30-epoch runs directly. The three plain UNet runs had a validation loss of 0.00114 to 0.00134 after their tenth epoch. This run had 0.00161: 1.2 to 1.4 times higher.

The causes, in order of importance, are the three scale facts in [06_hyperparameters.md, section 6.2](06_hyperparameters.md#62-why-these-settings-are-so-sensitive): a KL proportional to 1.9 million weights, a per-pixel-mean reconstruction loss, and a single σ that is 56% of the weight scale in the widest layer.

The one thing this run has is a large uncertainty: an epistemic std that is 76% of the error.

## 7.5 Finding 2: tuned, it is closer, but its uncertainty is tiny

`full_tuned` fixed the scales: σ 20 times smaller, `kl_weight` 100,000 times smaller, a narrower prior.

- **Test loss 0.000924.** That is 11% to 15% above the three plain UNet runs. This is one unseeded run. The gap is larger than the UNet's own spread of 3%, but about the same size as the difference between two seeds of the partial model, so it is not established that this model is worse than the partial one.
- **σ hardly moved.** It went from 0.34 to 0.40 × 10⁻³, the same for nearly every weight. The "posterior" is the initial value.
- **The epistemic std is 5% of the error.** On the shared colour scale of the uncertainty plot that is black. This is why the plot "showed nothing".
- **Sampling costs nothing** (+0.1%), for the same reason: the noise is too small to matter.

So with σ fixed at its initial value, the fully variational UNet offered a choice between a large uncertainty with a clearly worse loss (`full_default_10ep`) and a nearly unaffected loss with an uncertainty too small to see (`full_tuned`).

By the cost rule in [06_hyperparameters.md](06_hyperparameters.md#applying-the-rule), `kl_weight: 1e-06` is still about 100 times too large for this many weights, and this run did not use the separate ρ learning rate. It is therefore not the best this model can do.

## 7.6 Finding 3: a small variational head costs the least accuracy

The three `partial_learned` runs make only the last decoder block and the output layer variational, and give ρ its own learning rate.

Compared with the plain UNet at the same seed:

| Seed | Plain UNet | Partial, learned σ | Difference |
|---|---|---|---|
| 42 | 0.000834 | 0.000841 | +0.9% |
| 42, second run | 0.000834 | 0.000865 | +3.8% |
| 43 | 0.000807 | 0.000922 | +14.3% |

What this supports:

- **The plain UNet was better every time.** Each of the three partial runs has a higher test loss than each of the three plain UNet runs. On average it is 6% higher.
- **The size of the gap is not pinned down.** It was negligible in one run and clear in another. Three runs are too few to say more than "between 1% and 14%".
- **The partial model varies more between runs** (0.000841 to 0.000922, 10%) than the plain UNet (3%). Whether that is a property of the model or chance cannot be told from three runs each.
- **Sampling costs about 1%** (+0.7% to +1.1%). Predictions with sampled weights are almost as good as with the mean weights.
- **σ was learned.** From a common start of 1.50 × 10⁻³, the values spread over a factor of 4 to 6 between the 5th and 95th percentile, in all three runs. Most weights ended with a larger σ than they started with.
- **σ had not settled.** In `partial_learned_seed42` the 95th percentile is within 2% of the largest value: a large group of weights was still rising at the same rate when training stopped. Thirty epochs are not enough for the equilibrium described in [06_hyperparameters.md](06_hyperparameters.md#the-equilibrium-and-what-kl_weight-costs).
- **The uncertainty numbers are the most repeatable result here.** The epistemic std is 21% to 23% of the error, and its correlation with the error is 0.57 in all three runs.

What it does not support: that this model is as accurate as the plain UNet. An earlier version of this file said so on the strength of one unseeded trial run.

## 7.7 Finding 4: what the separate ρ learning rate does

`partial_sharedlr_seed42` and `partial_sharedlr_seed43` are the same model and seeds as the runs above, without `rho_learning_rate`. Each pair starts from identical weights and sees identical batches.

| Seed | Learned σ (ρ rate 0.01) | Shared rate | Difference |
|---|---|---|---|
| 42 | 0.000841 | 0.000875 | +4.1% |
| 43 | 0.000922 | 0.000963 | +4.5% |

- **Without the separate rate, σ did not learn.** Every σ ended between 1.35 and 1.78 × 10⁻³, within 19% of the starting value. With it, σ ranged from 0.67 to 8.28 × 10⁻³. This is the one clear effect of the setting.
- **The test loss was about 4% higher without it, in both pairs.** That is suggestive, not established. Against the second run of the seed-42 configuration the difference shrinks to 1.2%, and two identical runs already differ by 2.8% ([section 7.1](#71-common-setup)).
- **The uncertainty was not improved by it.** Without the separate rate the epistemic std was slightly larger (0.0025 to 0.0026 against 0.0020 to 0.0021) and the correlation with the error slightly lower (0.51 to 0.55 against 0.57).
- **The cost of sampling shows no pattern** (+1.2% and +0.5% without, +0.7% to +1.1% with).

So the separate ρ learning rate gives a σ that actually depends on the data, and possibly a small improvement in accuracy. It does not give a better uncertainty estimate by any measure used here.

The explanation offered in [06_hyperparameters.md](06_hyperparameters.md#the-equilibrium-and-what-kl_weight-costs), that a learned σ moves the noise onto weights that do not matter, is consistent with this but was not tested directly.

The unarchived trial pair mentioned in section 7.2 showed a 23% difference. The seeded runs show that most of that was the difference between two unrelated random runs.

## 7.8 Finding 5: the head-only uncertainty does not react to unfamiliar inputs

The third table in section 7.3 raises the input noise from the training level of 0.3 to 0.5. Every model fails there: PSNR drops from between 29 and 32 dB to between 17.6 and 19.1 dB. An uncertainty estimate that reflects "this input is unlike my training data" should rise sharply.

- **In the five partial runs it did not.** The mean epistemic std changed by a factor of 0.93 to 1.21. In one run it fell.
- **In the two fully variational runs it rose** by a factor of 1.90 and 2.71.
- **At noise 0.0 and 0.1 nothing rose either.** These levels are also outside the training data, but the models handle them about as well as the training level, so no warning is called for. The std was lower than at the training level in all five partial runs, and at most 6% higher in the fully variational ones.

This fits the structure of the models. In `variational_partial_unet` the features are computed by deterministic layers, and only the last three convolutions are sampled. The spread can only express doubt about how fixed features map to pixel values. When the features themselves are wrong because the input is unfamiliar, nothing in the model varies. In `variational_unet` every layer is sampled, so a disturbed input can change how much the samples disagree.

Even there, the rise is small in absolute terms. `full_tuned` reaches a std of 0.0017 at a noise level where the root-mean-square error of its prediction is above 0.1. No model here gives a usable warning.

**The correlation with the error says less than it seems.** It is between 0.51 and 0.64 for all seven variational runs. That includes the four runs where σ is the same for every weight and nothing about it was learned. A correlation of that size therefore comes from the structure of the images, not from a learned posterior: both the error and the effect of weight noise are larger at edges and fine detail than in flat background.

## 7.9 What the numbers say together

| Model | Test loss against the plain UNet | Epistemic std as a share of the error | Std rises at noise 0.5 |
|---|---|---|---|
| Fully variational, natural defaults (10 epochs) | 1.2 to 1.4 times higher (validation, epoch 10) | 76% | yes, × 1.9 |
| Fully variational, tuned | 11% to 15% higher (one run) | 5% | yes, × 2.7 |
| Partial, uniform σ | 5% and 19% higher (same seed) | 21% to 25% | no |
| Partial, learned σ | 1% to 14% higher (same seed, three runs) | 21% to 23% | no |

The conclusion the evidence supports:

> For in-distribution CT denoising with this UNet, the plain model was the most accurate in every comparison. Making every layer variational with the natural defaults was clearly worse. Tuned, it came within about 11% to 15%, but its weight noise was then too small to see. Making only the last decoder block variational and learning the per-weight standard deviations cost between 1% and 14% in three seeded comparisons, 6% on average, and produced a per-pixel spread about a fifth the size of the error. That spread correlates with the error, but so does the spread of every other variational model tried, and unlike the fully variational models it did not grow when the input noise was raised beyond the training level.

In short: `variational_partial_unet` with a learned σ is the cheapest variational setting found, and it is what `config.yaml` selects. Its uncertainty map shows where the image has structure the model finds hard. It is not a measure of confidence and not a detector of unfamiliar inputs.

## 7.10 Open questions

In rough order of value for the report:

1. **More seeds.** Five or more each of `unet` and the partial model with learned σ, to turn "between 1% and 14%" into a number with a spread. One command per run: `python scripts/run_notebook.py --set seed=44`.
2. **Deterministic training.** Seeded runs drift apart on the GPU. Setting `torch.backends.cudnn.deterministic = True` in `set_seed` would probably make a seeded run repeat exactly, at some cost in speed. That would make paired comparisons, such as the one for the ρ learning rate, far sharper. It has not been tried.
3. **A fair retest of `variational_unet`**: `kl_weight` near 1e-08 with `rho_learning_rate: 0.01`. It is the only model whose uncertainty reacted to unfamiliar input, and it has not been run with a learned σ. *Done: see [08_variational_unet_sweep.md](08_variational_unet_sweep.md).*
4. **`variational_decoder_levels: 2` or `3`**, with `kl_weight` reduced in proportion to the number of variational weights. Does the std start to react to the noise level as more of the network is sampled?
5. **A `kl_weight` sweep** (1e-07, 1e-06, 1e-05) on the partial model, to check the predicted trade-off between std and loss.
6. **Longer training.** σ had not settled after 30 epochs.
7. **Why the partial model varies more between seeds** than the plain UNet, if it does.
8. **A deep ensemble of the plain UNet** as an uncertainty baseline. Three such runs already exist.
9. **Calibration**: how often the truth falls within one or two std of the prediction.
10. **Where the std is high.** Whether it follows edges, fine structures, or image intensity.

## 7.11 Regenerating these numbers

To repeat a run:

1. Run the notebook with the settings of the run. Either copy the values from the run's archived `config.yaml` into the repository's `config.yaml` and run `experiments/00_baseline.ipynb` from the top after restarting the kernel, or leave `config.yaml` alone and pass the differences on the command line:

   ```bash
   python scripts/run_notebook.py --set seed=43 --set training.rho_learning_rate=null
   ```

   Both write a new folder under `results/00_baseline/`.
2. For a variational model, run `python scripts/variational_summary.py results/00_baseline/<run>`.
3. To update the record, run `python scripts/archive_run.py results/00_baseline/<run> <name>`, then `python scripts/check_variational_docs.py`. The check lists every table entry that no longer matches.

The sensitivity table in [06_hyperparameters.md, section 6.7](06_hyperparameters.md#67-measured-sensitivity-to-weight-noise) is regenerated with `python scripts/weight_noise_sensitivity.py results/00_baseline/<unet run>`. It used the checkpoint of `unet_seed42`.

What to expect from a repeat:

- **A repeat does not give the same digits, even on the same machine.** The seeded run that was repeated ended 2.8% away from the first ([section 7.1](#71-common-setup)). Expect a rerun to land within a few percent of the archived number, not on it.
- **The conclusions above are stated loosely enough to survive that.** A claim that depends on a difference of a few percent is marked as not established.
- **Checkpoints and plots are not archived.** `results/` is not in git. Step 2 regenerates them.

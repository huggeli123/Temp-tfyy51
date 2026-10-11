# Variational (Bayes-by-backprop) code: documentation

This folder explains every variational part of the project: the theory, the layers, the three variational models, the training and inference code, the hyperparameters, and what the experiments showed.

The base models (`linear`, `nonlinear`, `simple_cnn`, `residual_cnn`, `unet`) and the ordinary training loop (`train_model`, `train_one_epoch`, `evaluate`) are not covered here, except where the variational code reuses them.

## Reading order

| File | What it covers |
|---|---|
| [01_theory.md](01_theory.md) | What a variational layer is, the objective, the reparameterisation trick, the closed-form KL, and the gradients that drive everything else |
| [02_layers.md](02_layers.md) | `models/variational.py` line by line: the helpers and the three layer classes, plus `sanity_check.py` |
| [03_models.md](03_models.md) | `variational_simple_cnn`, `variational_unet`, `variational_partial_unet`: architecture, parameter counts, how the KL is collected |
| [04_training.md](04_training.md) | The variational loops in `train.py`, the loss, the history keys, the separate learning rate for ρ, and the notebook flow |
| [05_uncertainty.md](05_uncertainty.md) | `predict_with_uncertainty`, the uncertainty plot, how to read the numbers, and what they do not mean |
| [06_hyperparameters.md](06_hyperparameters.md) | Every variational config key, how its value was chosen, and a tuning guide |
| [07_experiments.md](07_experiments.md) | All runs so far with exact numbers, what they showed, and open questions |

If you only read one file, read [06_hyperparameters.md](06_hyperparameters.md). The single most important fact in this folder is there: the variational settings are extremely scale-sensitive, and the defaults that look natural (`kl_weight: 0.1`, `initial_rho: -5`, every layer variational) make the model clearly worse than a plain UNet. If you read two, add [07_experiments.md](07_experiments.md) for what the models did and did not achieve.

## Where the code lives

| Piece | File | Main names |
|---|---|---|
| Layers and helpers | [models/variational.py](../../models/variational.py) | `positive_scale`, `sample_gaussian`, `diagonal_gaussian_kl`, `VariationalConv2d`, `VariationalConvTranspose2d`, `VariationalLinear` |
| Model: last conv variational | [models/variational_simple_cnn.py](../../models/variational_simple_cnn.py) | `VariationalSimpleCNN` |
| Model: every conv variational | [models/variational_unet.py](../../models/variational_unet.py) | `VariationalDoubleConv`, `VariationalUNet` |
| Model: only the decoder end variational | [models/variational_partial_unet.py](../../models/variational_partial_unet.py) | `VariationalPartialUNet` |
| Factory | [models/\_\_init\_\_.py](../../models/__init__.py) | `create_model` branches `variational_*` |
| Training and inference | [train.py](../../train.py) | `train_one_epoch_variational`, `evaluate_variational`, `train_model_variational`, `predict_with_uncertainty` |
| Optimizer groups | [utils/model_configs.py](../../utils/model_configs.py) | `get_parameter_groups`, `get_optimizer(..., rho_learning_rate=)` |
| Plots | [utils/plot_utils.py](../../utils/plot_utils.py) | `plot_uncertainty_examples`, the `sample` detection in `save_denoising_examples` |
| Checks of the code | [sanity_check.py](../../sanity_check.py) | `run_checks`, `run_model_checks` |
| Check of these docs | [scripts/check_variational_docs.py](../../scripts/check_variational_docs.py) | see [Keeping these docs in sync](#keeping-these-docs-in-sync) |
| Analysis of a trained run | [scripts/variational_summary.py](../../scripts/variational_summary.py), [scripts/weight_noise_sensitivity.py](../../scripts/weight_noise_sensitivity.py) | write the `.json` files the numbers in these docs come from |
| Archived run records | [runs/](runs/), filled by [scripts/archive_run.py](../../scripts/archive_run.py) | `config.yaml`, `results.json` and summaries of every run cited here |
| Orchestration | [experiments/00_baseline.ipynb](../../experiments/00_baseline.ipynb) | the `is_variational` branches |
| Running the notebook from the command line | [scripts/run_notebook.py](../../scripts/run_notebook.py) | `--set key=value` overrides one config value for one run |
| Settings | [config.yaml](../../config.yaml) | `model.prior_standard_deviation`, `model.initial_rho`, `model.variational_decoder_levels`, `training.kl_weight`, `training.rho_learning_rate` |
| Original specification | [variational_layers_pytorch_launch_guide.md](../../variational_layers_pytorch_launch_guide.md) | the guide the layers were written from |

## Notation used in all files

| Symbol | Meaning | In the code |
|---|---|---|
| w | one weight (or bias) of a layer | the sampled `weight` / `bias` tensors |
| μ | posterior mean of that weight | `weight_mean`, `bias_mean` |
| ρ | unconstrained parameter that sets the std | `weight_rho`, `bias_rho` |
| σ | posterior standard deviation, σ = softplus(ρ) | `positive_scale(rho)` |
| ε | standard normal noise | `torch.randn_like(mean)` |
| p | standard deviation of the prior N(0, p²) | `prior_standard_deviation` |
| KL | KL divergence from posterior to prior, summed over all variational weights | the second return value of `forward` |
| β | weight on the KL term | `training.kl_weight` |
| N | number of training images (660) | `dataset_size` |
| n | number of variational weights in the model | number of elements in all `*_rho` tensors |
| P | pixels per image (320 × 320 = 102,400) | not in the code; used in the derivations |
| h | sensitivity of the loss to one weight (second derivative) | not in the code; measured in [06_hyperparameters.md](06_hyperparameters.md) |

## Quick start

To train the recommended variational model, `config.yaml` needs:

```yaml
model:
  name: variational_partial_unet
  prior_standard_deviation: 1.0
  initial_rho: -6.5
  variational_decoder_levels: 1

training:
  kl_weight: 0.000001
  rho_learning_rate: 0.01
```

Then restart the notebook kernel and run `experiments/00_baseline.ipynb` top to bottom. Any model whose name starts with `variational` automatically takes the variational branches of the notebook.

To check the layers and models in isolation, run `python sanity_check.py` from the repository root.

## Keeping these docs in sync

These files quote code and state numbers. Both can go stale without anyone noticing, so there is a script that compares them with their sources:

```bash
python scripts/check_variational_docs.py
```

Run it after changing the variational code, the notebook, or the archived runs. It reports each mismatch and exits with an error if there is one. It checks that:

1. **Links work.** Every relative link and heading anchor resolves.
2. **Code excerpts are real.** Every Python block in these files still exists in the source. Comments, docstrings and layout are ignored, and a line with only `...` marks an omission. Usage examples that are not source code carry a `<!-- docs-check: illustrative -->` marker and are skipped.
3. **Numbers from the code are right.** Parameter counts, `create_model` defaults, weight scales, the ρ to σ table, the size of the KL at initialisation, and the dataset size are recomputed and compared with the tables.
4. **Numbers from experiments are right.** Test losses, learned σ, and uncertainty numbers are read from [runs/](runs/) and compared with the tables in 05, 06 and 07.

What it cannot check is the reasoning. If a result changes, the tables will be flagged, but the sentences that interpret them have to be reread by a person.

### Where the experiment numbers come from

`results/` is not in git, so the small text records of every run cited here are copied into [runs/](runs/): the `config.yaml` snapshot, `results.json`, and for variational models a `variational_summary.json`. To add a run:

```bash
python scripts/run_notebook.py --set seed=44                         # or run the notebook by hand
python scripts/variational_summary.py results/00_baseline/<run>      # variational models only
python scripts/archive_run.py results/00_baseline/<run> <name>
```

Checkpoints and plots are not archived. They are regenerated by running the notebook with the archived `config.yaml`.

## One-paragraph summary

A variational layer stores a mean and a standard deviation for every weight instead of a single number. Each forward pass during training draws a random weight from that distribution, and a KL term in the loss pulls the distribution toward a prior. At test time the means give one deterministic prediction, and repeating the forward pass with fresh samples gives a spread that is read as epistemic (model) uncertainty. In this project the plain UNet was the most accurate model in every comparison. Making every layer variational cost 11% to 15% in test loss once tuned, with a spread too small to see. Making only the last decoder block and the output layer variational, and letting the standard deviations be learned with their own learning rate, cost between 1% and 14% in three seeded comparisons and gave a per-pixel spread about a fifth the size of the error. That spread marks the image regions the model finds hard. It did not grow when the input noise was raised beyond the training level, so it is not a detector of unfamiliar inputs. The evidence is in [07_experiments.md](07_experiments.md).

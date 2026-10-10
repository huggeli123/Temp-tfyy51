# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TFYY51 course project (Linköping University): CT image denoising with PyTorch. Clean grayscale CT images are loaded from `ct_data/{train,val,test}` and Gaussian noise is added on the fly; models learn noisy -> clean.

## Setup and running

- Environment: `conda env create -f environment.yml`, then `conda activate tfyy51` (Python 3.10, PyTorch, torchvision, numpy, matplotlib, pyyaml, tqdm, jupyterlab). Note: `dataset.py` also imports PIL (comes in via torchvision).
- Data: `ct_data.zip` is committed; the extracted `ct_data/` folder, `results/` and `outputs/` are gitignored. Unzip `ct_data.zip` in the repo root before running (660 train / 75 val / 75 test images).
- There is no CLI entry point, test suite, or linter. The pipeline is driven from notebooks in `experiments/` (e.g. `00_baseline.ipynb`), which are run in VS Code/Jupyter with the `tfyy51` kernel. `python sanity_check.py` runs shape/sampling/KL/gradient checks on the variational layers.
- Restart the notebook kernel after editing `train.py`, `dataset.py`, `models/` or `utils/`; stale imports are a known pitfall.

## Architecture

The notebook is the orchestrator; there is no top-level `main`. A run flows like this:

1. `utils.misc.load_config` reads `config.yaml` (the main file edited per experiment), and `set_seed` seeds it.
2. `utils.dataloader.create_dataloaders(config)` builds `CTDenoisingDataset` loaders for train/val/test.
3. `models.create_model(config["model"])` is a factory over `linear`, `nonlinear`, `simple_cnn`, `residual_cnn`, `unet`. `linear` ignores activation/normalization; `linear` and `nonlinear` ignore parts of `depth`/`kernel_size` (see comments in `config.yaml`). Shared building blocks (activation/normalization selection) live in `models/layers.py`.
4. `utils.model_configs.get_loss` / `get_optimizer` map config strings (`mse|l1|smooth_l1`, `adam|adamw|sgd`) to torch objects. To add an option, extend these if/elif chains.
5. `train.train_model` runs epochs of `train_one_epoch` + `evaluate`, tracks PSNR (`utils.metrics.calculate_psnr`), saves `best_model.pt` by lowest **val loss**, and returns a history dict.
6. `utils.experiment.create_experiment` makes `results/<experiment_name>/<timestamp>/` and snapshots the config there; `save_history` writes `results.json`. Plot helpers are in `utils/plot_utils.py` and `utils/print_utils.py`.

## Things to know

- Noise is generated in `CTDenoisingDataset.__getitem__` (Gaussian, then clamped to [0, 1]); `noise.train_val_std` and `noise.test_std` are separate config keys. `deterministic_noise=True` seeds noise with `seed + index` so val/test noise is reproducible; training noise is fresh each load.
- Augmentation (horizontal flip, rotation) applies only to the `train` split and only when `augmentation.enabled`. Clean and noisy images are derived from the same augmented clean image, so targets stay aligned.
- Adding a model: create the class in `models/`, export it in `models/__init__.py`, and add a branch in `create_model`; models take `channels, depth, activation, normalization, num_groups, kernel_size`.
- `output.directory` in `config.yaml` (`outputs`) is the default for `train_model`'s `output_dir`; the README describes notebook results going under `results/`.

## Variational (Bayes-by-backprop) layers

Implementation follows `variational_layers_pytorch_launch_guide.md`.

- `models/variational.py` has `VariationalConv2d` / `VariationalLinear`. Both store `weight_mean` and `weight_rho` (sigma = softplus(rho)) and return `(output, kl)` from `forward(x, sample=True)`; `sample=False` uses the means.
- Variational models return `(prediction, kl)`, unlike all other models, so they need the separate loops in `train.py` (`train_model_variational`, `evaluate_variational`) which compute `recon + kl_weight * kl / dataset_size`. The notebook switches on `is_variational` (model name starts with `variational`). `plot_utils.save_denoising_examples` detects a `sample` argument on `forward` and shows the posterior mean.
- Variational models: `variational_simple_cnn` (only the output conv is variational), `variational_unet` (every conv variational; needs a very small `initial_rho`, about -8, and a tiny `kl_weight`, or it is clearly worse than `unet`) and `variational_partial_unet` (deterministic UNet except the last `variational_decoder_levels` decoder blocks and the output conv). They read optional `prior_standard_deviation` and `initial_rho` from the model config.
- `training.rho_learning_rate` (optional) gives the `*_rho` parameters their own learning rate via `get_optimizer`; without it sigma stays near `initial_rho` in a short run. `kl_weight` is relative to a per-pixel mean MSE, so values around 1e-8..1e-5 are the useful range, not 0.1.
- In the variational history, `train_loss` and `val_loss` are both reconstruction-only; the objective actually minimised is in `train_total_loss`, the KL in `train_kl`.

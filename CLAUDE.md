# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TFYY51 course project (Linköping University): CT image denoising with PyTorch. Clean grayscale CT images are loaded from `ct_data/{train,val,test}` and Gaussian noise is added on the fly; models learn noisy -> clean.

## Setup and running

- Environment: `conda env create -f environment.yml`, then `conda activate tfyy51` (Python 3.10, PyTorch, torchvision, numpy, matplotlib, pyyaml, tqdm, jupyterlab). Note: `dataset.py` also imports PIL (comes in via torchvision). `environment.yml` is the course's file for the Linux student computers; leave it as it is. The pip/venv alternative for other machines is `requirements.txt` (see README); keep the two package lists the same.
- Machine-specific state stays out of git: `.vscode/settings.json` is ignored, and notebooks are committed without outputs or kernel metadata through the `nbstripout` filter in `.gitattributes` (one-time `nbstripout --install` per clone, see README).
- Data: `ct_data.zip` is committed; the extracted `ct_data/` folder, `results/` and `outputs/` are gitignored. Unzip `ct_data.zip` in the repo root before running (660 train / 75 val / 75 test images).
- There is no test suite or linter. The pipeline is driven from notebooks in `experiments/` (e.g. `00_baseline.ipynb`), which are run in VS Code/Jupyter with the `tfyy51` kernel. `python scripts/run_notebook.py [--set key=value ...]` executes that notebook unchanged from the command line; each `--set` overrides one config value for that run without editing `config.yaml`. `python sanity_check.py` runs shape/sampling/KL/gradient checks on the variational layers and models.
- Restart the notebook kernel after editing `train.py`, `dataset.py`, `models/` or `utils/`; stale imports are a known pitfall.

## Architecture

The notebook is the orchestrator; there is no top-level `main`. A run flows like this:

1. `utils.misc.load_config` reads `config.yaml` (the main file edited per experiment), and `set_seed` seeds it.
2. `utils.dataloader.create_dataloaders(config)` builds `CTDenoisingDataset` loaders for train/val/test.
3. `models.create_model(config["model"])` is a factory over `linear`, `nonlinear`, `simple_cnn`, `residual_cnn`, `unet`, `residual_unet` and the three variational models (see below). `linear` ignores activation/normalization; `linear` and `nonlinear` ignore parts of `depth`/`kernel_size` (see comments in `config.yaml`). Shared building blocks (activation/normalization selection) live in `models/layers.py`.
4. `utils.model_configs.get_loss` / `get_optimizer` / `get_scheduler` map config strings (`mse|l1|smooth_l1`, `adam|adamw|sgd`, `training.scheduler`: `none|cosine`) to torch objects; `get_scheduler` returns `None` for `none`, and `cosine` is `CosineAnnealingLR` with `T_max=epochs`. To add an option, extend these if/elif chains.
5. `train.train_model` runs epochs of `train_one_epoch` + `evaluate`, tracks PSNR (`utils.metrics.calculate_psnr`), saves `best_model.pt` by lowest **val loss**, and returns a history dict (also `epoch_time` and `learning_rates`). An optional `scheduler` is stepped once per epoch, after the learning rate for that epoch is recorded.
6. `utils.experiment.create_experiment` makes `results/<experiment_name>/<timestamp>/` and snapshots the config there; `save_history` writes `results.json`. Plot helpers are in `utils/plot_utils.py` and `utils/print_utils.py`.

## Things to know

- Noise is generated in `CTDenoisingDataset.__getitem__` (Gaussian, then clamped to [0, 1]); `noise.train_val_std` and `noise.test_std` are separate config keys. `deterministic_noise=True` seeds noise with `seed + index` so val/test noise is reproducible; training noise is fresh each load.
- `evaluate` and `evaluate_variational` clamp predictions to [0, 1] before computing loss and PSNR; the training loops do not. So train and val loss are not computed quite the same way, and runs archived in `docs/variational/runs/` from before commit `dd0e84e` were evaluated without the clamp.
- Augmentation (horizontal flip, rotation) applies only to the `train` split and only when `augmentation.enabled`. Clean and noisy images are derived from the same augmented clean image, so targets stay aligned.
- Adding a model: create the class in `models/`, export it in `models/__init__.py`, and add a branch in `create_model`; models take `channels, depth, activation, normalization, num_groups, kernel_size`.
- `output.directory` in `config.yaml` (`outputs`) is the default for `train_model`'s `output_dir`; the README describes notebook results going under `results/`.

## Variational (Bayes-by-backprop) layers

Implementation follows `variational_layers_pytorch_launch_guide.md`. In-depth documentation (theory, layers, models, training, uncertainty, hyperparameters, experiment results) is in `docs/variational/`; start with its `README.md`.

- After changing variational code, the notebook, or the archived runs, run `python scripts/check_variational_docs.py`. It fails if a code excerpt in the docs no longer exists in the source, if a number that follows from the code (parameter counts, defaults, weight scales) differs from the docs, or if an experiment number differs from the records in `docs/variational/runs/`.
- `results/` is gitignored, so a run the docs rely on is archived with `python scripts/archive_run.py <run_dir> <name>` (copies `config.yaml` and the `.json` records to `docs/variational/runs/<name>/`). `scripts/variational_summary.py <run_dir>` adds the learned sigma, sampled-weight loss and uncertainty numbers of a variational run; `scripts/weight_noise_sensitivity.py <unet run_dir>` measures how much weight noise a trained UNet tolerates.
- Runs are noisy: the same seeded config run twice ended 2.8% apart in test loss (GPU arithmetic is not deterministic), and different seeds up to 10% apart. Do not draw a conclusion from a difference of a few percent between two runs; `docs/variational/07_experiments.md` records what is and is not established.
- Derivations and tuned values live only in `docs/variational/06_hyperparameters.md`. Keep `config.yaml` comments to what a key does and point there, so the same number is not maintained twice.

- `models/variational.py` has `VariationalConv2d` / `VariationalLinear`. Both store `weight_mean` and `weight_rho` (sigma = softplus(rho)) and return `(output, kl)` from `forward(x, sample=True)`; `sample=False` uses the means.
- Variational models return `(prediction, kl)`, unlike all other models, so they need the separate loops in `train.py` (`train_model_variational`, `evaluate_variational`) which compute `recon + kl_weight * kl / dataset_size`. The notebook switches on `is_variational` (model name starts with `variational`). `plot_utils.save_denoising_examples` detects a `sample` argument on `forward` and shows the posterior mean.
- Variational models: `variational_simple_cnn` (only the output conv is variational), `variational_unet` (every conv variational; needs a very small `initial_rho`, about -8, and a tiny `kl_weight`, or it is clearly worse than `unet`) and `variational_partial_unet` (deterministic UNet except the last `variational_decoder_levels` decoder blocks and the output conv). They read optional `prior_standard_deviation` and `initial_rho` from the model config.
- `training.rho_learning_rate` (optional) gives the `*_rho` parameters their own learning rate via `get_optimizer`; without it sigma stays near `initial_rho` in a short run. `kl_weight` is relative to a per-pixel mean MSE, so values around 1e-8..1e-5 are the useful range, not 0.1.
- In the variational history, `train_loss` and `val_loss` are both reconstruction-only; the objective actually minimised is in `train_total_loss`, the KL in `train_kl`.

# TFYY51 – CT Image Denoising with Deep Learning

This repository contains material for the TFYY51 project course (Ingenjörsprojektet) at Linköping University. The task is **CT image denoising**.


## Environment Setup

This repository uses a Conda environment defined in:

```text
environment.yml
```

The environment contains the Python packages needed to run the project notebooks and training code.


### 2. Create the environment

Open a terminal in the root of the repository, where `environment.yml` is located.

Run:

```bash
conda env create -f environment.yml
```

This creates the project environment.

The environment only needs to be created once.

### 3. Activate the environment

Activate it with:

```bash
conda activate tfyy51
```

When the environment is active, the terminal should show something similar to:

```text
(tfyy51) user@computer:~/path/to/tfyy51$
```

Each time you open a new terminal, activate the environment again using:

```bash
conda activate tfyy51
```

### 4. Select the environment in VS Code / Jupyter

When opening a Jupyter notebook in VS Code:

1. Open the notebook.
2. Click the kernel selector in the top-right corner.
3. Select the Python environment named `tfyy51`.

Make sure the notebook is using the same environment that was created from `environment.yml`.

### 5. Check that the environment is active

You can list the available Conda environments with:

```bash
conda info --envs
```

The active environment is marked with `*`.

For example:

```text
# conda environments:
#
base                     /home/user/miniconda3
tfyy51                *  /home/user/miniconda3/envs/tfyy51
```

You can also check which Python executable is being used:

```bash
which python
```

It should point to the `tfyy51` Conda environment.

---


## Repository structure

```text
TFYY51-master/
│
├── ct_data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── experiments/
│   └── 01_baseline.ipynb
│
├── models/
│   ├── __init__.py
│   ├── layers.py
│   ├── linear.py
│   ├── nonlinear.py
│   ├── simple_cnn.py
│   ├── residual_cnn.py
│   └── unet.py
│
├── results/
│   └── ...
│
├── utils/
│   └── ...
│
├── config.yaml
├── dataset.py
├── train.py
├── sanity_check.py
└── README.md
```

### `ct_data/`

Contains the CT images used in the project.

The dataset is divided into:

| Split | Number of images |
|---|---:|
| Training | 660 |
| Validation | 75 |
| Test | 75 |

The images stored in these folders are clean images. Noisy images are generated during loading according to the settings in `config.yaml`.

### `experiments/`

Contains the Jupyter notebooks used to run experiments.

`00_baseline.ipynb` provides a starting point for running the complete training and evaluation pipeline.

You are encouraged to create additional notebooks when performing new experiments.

For example:

```text
experiments/
├── 01_baseline.ipynb
├── compare_models.ipynb
├── noise_experiment.ipynb
└── ...
```

Each notebook creates a corresponding experiment folder inside `results/`. The experiment name is specified in the notebook.

For example:

```python
experiment_name = "first_experiment_with_unet"
```

will create results under:

```text
results/
└── first_experiment_with_unet/
    └── <run>/
```

Different runs are stored separately so previous results are not overwritten.

A result folder may contain files such as:

```text
results/
└── first_experiment_with_unet/
    └── <run>/
        ├── config.yaml
        ├── best_model.pt
        ├── results.json
        ├── loss_curve.png
        ├── psnr_curve.png
        ├── denoising_examples.png
        └── test_examples.png
```

### `models/`

Contains the neural network architectures available for the project.

Currently available models are:

```text
linear
nonlinear
simple_cnn
residual_cnn
unet
```

### `utils/`

Contains utility functions used by the notebooks and training code, such as plotting, metrics, experiment handling, and configuration summaries.

### `dataset.py`

Contains the PyTorch dataset used to load CT images and prepare the model inputs and targets.

### `train.py`

Contains the functions used for training, validation, and evaluation.


---

## Configuration

The main experiment settings are controlled through:

```text
config.yaml
```

**This is the main file that should be changed when trying a new experiment.**

For example, the configuration controls:

- model architecture,
- number of channels,
- network depth,
- kernel size,
- activation function,
- normalization,
- optimizer,
- loss function,
- learning rate,
- weight decay,
- batch size,
- number of epochs,
- training and test noise levels,
- data augmentation,
- random seed.

An example configuration is:

```yaml
seed: 42

data:
  root: ct_data
  image_size: null

noise:
  train_val_std: 0.10
  test_std: 0.10

model:
  name: unet
  channels: 32
  depth: 3
  kernel_size: 3
  activation: relu
  normalization: none
  num_groups: 4

training:
  epochs: 10
  batch_size: 16
  num_workers: 0
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0
  loss: mse

augmentation:
  enabled: false
  horizontal_flip: true
  horizontal_flip_probability: 0.5
  rotation_degrees: 5

output:
  save_best_model: true
  save_example_images: true
```

When testing a new setting, modify `config.yaml` and run the notebook again.

When comparing different settings, it is recommended to create a new experiment/notebook and give the experiment a descriptive name so the results are stored separately.

For example:

```text
experiments/
├── 01_baseline.ipynb
├── compare_activations.ipynb
├── compare_models.ipynb
└── different_noise_levels.ipynb
```

with corresponding result folders:

```text
results/
├── baseline/
├── compare_activations/
├── compare_models/
└── different_noise_levels/
```

The configuration used for each run is also saved together with the results, making it possible to inspect the settings used for an earlier experiment.

---

## Jupyter Notebook Troubleshooting

When working with Jupyter notebooks, Python modules can sometimes remain loaded in memory even after files such as `train.py`, `dataset.py`, or files inside `models/` have been changed.

If the notebook starts behaving unexpectedly, imports appear outdated, or results do not seem to reflect recent code changes, the safest solution is usually to restart the notebook kernel.

In VS Code:

1. Click **Restart** at the top of the notebook.
2. Click **Clear All Outputs**.
3. Run the notebook again from the beginning using **Run All**.

This resets the Python environment and ensures that the latest versions of all scripts and models are loaded.

It is especially recommended to restart the kernel after modifying:

- `train.py`
- `dataset.py`
- files inside `models/`
- files inside `utils/`

Restarting the kernel is also useful if a previous training run crashed or if the notebook is in an unclear state.

If you still face issues talk to the supervisor.
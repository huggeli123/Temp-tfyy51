# 4. Training and evaluation

This file covers the variational half of [train.py](../../train.py), the optimizer groups in [utils/model_configs.py](../../utils/model_configs.py), and how the notebook ties them together. Prediction with uncertainty has its own file: [05_uncertainty.md](05_uncertainty.md).

## 4.1 Why there is a second set of loops

The ordinary loops (`train_one_epoch`, `evaluate`, `train_model`) call `model(noisy_images)` and expect a tensor back. A variational model returns `(prediction, kl)` and needs the KL added to the loss. Rather than add branches to the ordinary loops, the variational versions are separate functions with the same structure:

| Ordinary | Variational |
|---|---|
| `train_one_epoch` | `train_one_epoch_variational` |
| `evaluate` | `evaluate_variational` |
| `train_model` | `train_model_variational` |
| (none) | `predict_with_uncertainty` |

## 4.2 `train_one_epoch_variational`

The core of one training step:

```python
optimizer.zero_grad()

# Stochastic forward pass: weights are sampled
predictions, kl = model(noisy_images, sample=True)

reconstruction = criterion(predictions, clean_images)

scaled_kl = kl / dataset_size

loss = reconstruction + kl_weight * scaled_kl

loss.backward()
optimizer.step()
```

- **`sample=True`.** Each minibatch uses one fresh draw of the weights.
- **`reconstruction`** is the configured criterion. With `mse` it is the mean squared error over all pixels of all images in the batch.
- **`scaled_kl = kl / dataset_size`.** `dataset_size` is the number of *training images*, 660. The reasoning is in [01_theory.md, section 1.4](01_theory.md#14-the-objective).
- **`loss`** is what is minimised. Its gradient reaches the means through both terms, and each ρ through both terms.

The function accumulates four running sums, each weighted by the batch size so the last (smaller) batch is counted correctly, and returns their averages:

| Key | What it is |
|---|---|
| `loss` | reconstruction + β × KL / N, the quantity minimised |
| `reconstruction` | the reconstruction loss alone, under sampled weights |
| `kl` | KL / N. Note: *not* multiplied by β |
| `psnr` | PSNR of the sampled predictions against the clean images |

The progress bar shows the same four per batch.

**Reading the `kl` value.** Because it is KL / N without β, it is large even when the KL term in the loss is small. For `variational_partial_unet` with the current config it is about 252, and the KL's contribution to the loss is 252 × 10⁻⁶ ≈ 0.00025. To get the contribution, multiply the shown `kl` by `kl_weight`.

## 4.3 `evaluate_variational`

The same computation inside `torch.no_grad()`, with no optimizer step, and with one extra argument:

```python
predictions, kl = model(noisy_images, sample=sample)
```

- **`sample=False`** (the default) evaluates the posterior-mean network. The result is deterministic.
- **`sample=True`** draws one weight sample per batch. The result changes from call to call.

It returns the same four keys as the training function.

`dataset_size` is still the size of the *training* set, even when evaluating on validation or test data. It only affects the `loss` and `kl` keys, which exist for completeness. The numbers that are actually reported come from `reconstruction` and `psnr`, which do not depend on it.

## 4.4 `train_model_variational`

The full loop. For each epoch it runs one training epoch and one validation pass, records the metrics, and saves the best model.

The validation half of each epoch shows the two choices that matter, `sample=False` and which number becomes `val_loss`:

```python
val_metrics = evaluate_variational(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    device=device,
    dataset_size=dataset_size,
    kl_weight=kl_weight,
    sample=False,
    description=f"Epoch {epoch + 1}/{epochs} - Val",
)

val_loss = val_metrics["reconstruction"]
```

`dataset_size` is set once, before the loop, to `len(train_loader.dataset)`.

### The history it returns

| Key | Contents | Weights used |
|---|---|---|
| `train_loss` | reconstruction loss per epoch | sampled |
| `val_loss` | reconstruction loss per epoch | posterior mean |
| `train_psnr` | PSNR per epoch | sampled |
| `val_psnr` | PSNR per epoch | posterior mean |
| `train_total_loss` | reconstruction + β × KL / N per epoch | sampled |
| `train_kl` | KL / N per epoch | (does not depend on sampling) |
| `epoch_time` | wall-clock seconds per epoch, training + validation | |

Three design decisions are built into this:

**`train_loss` and `val_loss` are both reconstruction-only.** The first four keys have the same names and meaning as in the ordinary `train_model`. That keeps `save_history`, `plot_training_history` and `plot_psnr_history` working unchanged, and makes the curves directly comparable with a non-variational run. The loss that was actually minimised is in `train_total_loss`.

**Validation uses the posterior mean.** So the validation curve is deterministic for a given set of weights, and does not jump around because of sampling.

**The best model is chosen by validation *reconstruction* loss.** `best_model.pt` is saved whenever `val_loss` improves. The KL plays no part in the choice. If it did, a run with a large KL would simply pick the epoch where the KL happened to be smallest, which says nothing about denoising quality.

A consequence of the first two: within one run, `train_loss` is measured with weight noise and `val_loss` without. If σ is large, training loss can sit *above* validation loss for that reason alone. With the small σ of the current config the effect is about 1%.

### A consistency check on the logged numbers

For the first `variational_unet` run (`kl_weight: 0.1`), epoch 1 logged `train_kl` = 13,076.67 and `train_total_loss` = 1,307.67. Indeed 0.1 × 13,076.67 = 1,307.67, with the reconstruction loss of 0.006 invisible next to it. A total loss in the thousands is the signature of a KL weight that is far too large.

### The checkpoint

`best_model.pt` is the model's `state_dict`: all means, all ρ values, and the weights of any deterministic layers. Loading it back requires building the model with the same config, including `variational_decoder_levels`. The notebook reloads it after training, so all later cells use the best epoch, not the last one.

## 4.5 A separate learning rate for ρ

File: [utils/model_configs.py](../../utils/model_configs.py). Config key: `training.rho_learning_rate`.

### The problem

Adam moves each parameter by at most roughly one learning rate per step. With 660 images and batch size 16 there are 42 steps per epoch, so 1,260 steps in 30 epochs. At the shared learning rate of 0.001, ρ can move by about 1.3 in total, and only if every single step pushed the same way. In reality the gradient on ρ is mostly noise with a small consistent part ([01_theory.md, section 1.7](01_theory.md#17-the-gradients-and-what-they-imply)), so it moves far less.

This was measured on `variational_partial_unet` over 30 epochs, with every weight starting at the same σ. At the shared rate every weight ended within 19% of where it started: the "posterior" is then just the number typed into `initial_rho`. With the separate rate the values spread out over a factor of about 10, which is what a learned, per-weight σ should look like. The numbers are in [07_experiments.md, section 7.7](07_experiments.md#77-finding-4-what-the-separate-ρ-learning-rate-does).

### The implementation

```python
def get_parameter_groups(model, learning_rate, weight_decay, rho_learning_rate=None):

    if rho_learning_rate is None:
        return model.parameters()

    rho_parameters = []
    other_parameters = []

    for name, parameter in model.named_parameters():
        if name.endswith("_rho"):
            rho_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    groups = [{"params": other_parameters, "lr": learning_rate, "weight_decay": weight_decay}]

    if rho_parameters:
        groups.append({"params": rho_parameters, "lr": rho_learning_rate, "weight_decay": 0.0})

    return groups
```

- **Without `rho_learning_rate`** the function returns `model.parameters()`, exactly what the optimizer received before this feature existed. Non-variational models and old configs behave identically.
- **With it**, parameters are split by name. Anything ending in `_rho` goes in the second group.
- **Weight decay is always zero for ρ.** Decay pulls toward ρ = 0, which is σ ≈ 0.69, a meaningless target.
- **A non-variational model with `rho_learning_rate` set** simply gets one group, because no parameter name matches.

`get_optimizer` passes the result to Adam, AdamW or SGD. PyTorch optimizers accept either an iterable of parameters or a list of group dictionaries, and a per-group `lr` overrides the default.

### Checking that it is active

The notebook must pass the key through:

```python
optimizer = get_optimizer(
    optimizer_name=config["training"]["optimizer"],
    model=model,
    learning_rate=config["training"]["learning_rate"],
    weight_decay=config["training"]["weight_decay"],
    rho_learning_rate=config["training"].get("rho_learning_rate"),
)
```

If that last argument is missing, nothing fails. Training silently uses the shared rate, and σ stays where `initial_rho` put it. To confirm the setting took effect, run this after creating the optimizer:

<!-- docs-check: illustrative -->
```python
print([group["lr"] for group in optimizer.param_groups])   # expected: [0.001, 0.01]
```

A single entry means the separate rate is not being applied. `python scripts/check_variational_docs.py` also fails if the notebook loses that argument, because the excerpt above is compared with the notebook.

## 4.6 The notebook flow

[experiments/00_baseline.ipynb](../../experiments/00_baseline.ipynb) seeds the run right after loading the config:

```python
config = load_config("config.yaml")

# Seed Python, NumPy and PyTorch so a run can be repeated
set_seed(config["seed"])
```

This seeds Python, NumPy and PyTorch, so the initial weights, the order of the batches, the training noise and the sampled weights are the same every time the notebook is run with the same config. That does not make a run repeat exactly. Arithmetic on the GPU is not deterministic, and the small differences grow during training: a seeded run that was repeated agreed to five digits after one epoch and ended 2.8% apart in test loss after thirty. Runs made before this line was added were not seeded; see [07_experiments.md](07_experiments.md#71-common-setup).

The notebook then decides once which path to take:

```python
is_variational = config["model"]["name"].startswith("variational")
```

and branches on it with `if is_variational:` in four cells:

| Cell | Variational branch | Otherwise |
|---|---|---|
| Training | `train_model_variational(..., kl_weight=config["training"]["kl_weight"])` | `train_model(...)` |
| Test evaluation | `evaluate_variational(..., sample=False)`, reporting `reconstruction` and `psnr` | `evaluate(...)` |
| Uncertainty | 30 stochastic passes on one validation batch, a five-panel plot, and summary numbers | prints that uncertainty is unavailable |
| Noise levels | predictive mean and mean std at noise levels 0, 0.1, the training level, and 0.5 | skipped |

One more place treats variational models differently, but it is not a branch in the notebook. The two cells that call `save_denoising_examples` are the same for every model. The function itself, in [utils/plot_utils.py](../../utils/plot_utils.py), checks whether `forward` has a `sample` argument, and if so shows the posterior-mean prediction.

`save_history` is the same for both. For a variational run, `results.json` additionally contains `train_total_loss` and `train_kl` in the history.

So the reported **test loss and test PSNR of a variational model come from the posterior-mean network**, the same single forward pass an ordinary model would do. That makes them a fair comparison with the plain UNet.

## 4.7 Practical points

- **Restart the kernel** after editing `train.py`, `models/` or `utils/`. The notebook imports them once.
- **If the notebook is open in VS Code while its file is changed on disk**, VS Code may later save its older in-memory copy over the change. This has already happened once to the `rho_learning_rate` line. If a setting seems to have no effect, check the cell source, or run `python scripts/check_variational_docs.py`.
- **Notebook outputs are not committed.** With the one-time setup in the main [README](../../README.md), git stores the notebook without outputs and without the name of the local kernel. The file on disk keeps its outputs.
- **Runtime** is essentially unchanged by the variational layers. Thirty epochs take about six minutes for the plain UNet and for both variational UNets on the GPU used here. The exact total of each run is in its `results.json` under `summary.total_training_time_seconds`.
- **`kl_weight` is read with `config["training"]["kl_weight"]`**, not `.get`. A config without that key raises a `KeyError` for variational models.

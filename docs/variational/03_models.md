# 3. The models

There are three variational models. They differ in one thing only: *which layers* are variational.

| Config name | Class | Variational layers | Variational weights (n) | Total trainable parameters |
|---|---|---|---|---|
| `variational_simple_cnn` | `VariationalSimpleCNN` | the output conv | 289 | 10,146 |
| `variational_unet` | `VariationalUNet` | every conv, every transposed conv, the output conv | 1,925,025 | 3,850,050 |
| `variational_partial_unet` | `VariationalPartialUNet` | the last decoder block and the output conv (default) | 27,745 | 1,952,770 |

All counts are for the default `channels: 32`, `depth: 3`, `kernel_size: 3`. For comparison, the plain `unet` has 1,925,025 parameters and `simple_cnn` has 9,857.

"Variational weights" counts each weight once. "Total trainable parameters" counts a mean and a ρ for each of them, plus the ordinary weights of the deterministic layers.

## 3.1 What all three have in common

**The forward signature.**

<!-- docs-check: illustrative -->
```python
prediction, kl = model(x, sample=True)
```

- `prediction` has the same shape as the input, `(batch, 1, height, width)`.
- `kl` is the sum of the KL of every variational layer in the model.
- `sample` is passed down to every variational layer.

**How the notebook recognises them.** The notebook sets `is_variational = config["model"]["name"].startswith("variational")`. A new variational model must have a name that starts with `variational`, or it will be sent to the ordinary training loop and crash on the tuple it returns.

**How plots recognise them.** `save_denoising_examples` in [utils/plot_utils.py](../../utils/plot_utils.py) checks whether `forward` has a parameter called `sample`. If so it calls `model(x, sample=False)[0]`, showing the posterior-mean prediction and dropping the KL.

**The two extra constructor arguments.** `prior_standard_deviation` and `initial_rho` are passed to every variational layer in the model. One value of each is used for the whole model. There is no per-layer setting.

**The factory.** `create_model` in [models/\_\_init\_\_.py](../../models/__init__.py) reads these from the `model` section of the config with `config.get(...)`, so they are optional:

| Model | Default `prior_standard_deviation` | Default `initial_rho` | Extra key |
|---|---|---|---|
| `variational_simple_cnn` | 1.0 | -5.0 | none |
| `variational_unet` | 1.0 | -5.0 | none |
| `variational_partial_unet` | 1.0 | -6.0 | `variational_decoder_levels` (default 1) |

The defaults for `variational_unet` are not good settings for it. See [06_hyperparameters.md](06_hyperparameters.md).

## 3.2 `variational_simple_cnn`

File: [models/variational_simple_cnn.py](../../models/variational_simple_cnn.py).

This is `SimpleCNN` with only the last convolution replaced.

```
input (1 channel)
  │
  ├─ Conv2d 1 → 32, norm, activation          deterministic
  ├─ Conv2d 32 → 32, norm, activation         deterministic   (depth - 2 of these)
  │
  └─ VariationalConv2d 32 → 1                 variational
        │
     prediction, kl
```

In the code the deterministic layers are collected into `self.features` (an `nn.Sequential`) and the last layer is `self.output_layer`:

```python
def forward(self, x, sample=True):
    hidden = self.features(x)
    prediction, kl_divergence = self.output_layer(hidden, sample=sample)
    return prediction, kl_divergence
```

- `depth` must be at least 2: one feature layer plus the output layer.
- The output convolution uses the configured `kernel_size` (3 × 3 by default), so it has 32 × 9 weights + 1 bias = 289 variational weights.
- This was the first variational model, and the simplest possible test of the layers: the "Bayesian last layer" idea. With only 289 variational weights the KL is small: about 1,300 nats at initialisation, or 2 per training image. With `kl_weight: 0.1` the KL term is then about 0.2. That is still far above a reconstruction loss of a few thousandths, but nearly four orders of magnitude below the same term for the fully variational UNet. No run of this model is recorded in [07_experiments.md](07_experiments.md), so how much that setting costs it has not been measured.

## 3.3 `variational_unet`

File: [models/variational_unet.py](../../models/variational_unet.py).

The same architecture as `UNet` in [models/unet.py](../../models/unet.py), with *every* convolution replaced by its variational counterpart.

### The building block: `VariationalDoubleConv`

The ordinary UNet uses `DoubleConv`: two rounds of (convolution, normalisation, activation) in one `nn.Sequential`. That cannot be used here, because `nn.Sequential` passes a single tensor from layer to layer and a variational convolution returns two things. `VariationalDoubleConv` therefore spells the block out:

```python
def forward(self, x, sample=True):

    x, kl_1 = self.conv1(x, sample=sample)
    x = self.post1(x)                 # normalisation + activation

    x, kl_2 = self.conv2(x, sample=sample)
    x = self.post2(x)

    return x, kl_1 + kl_2
```

`post1` and `post2` are small `nn.Sequential`s holding the normalisation and activation chosen in the config. They are ordinary deterministic layers.

### The network

For `channels: 32`, `depth: 3` and a 320 × 320 input:

```
input 1 × 320 × 320
  │
  ├─ encoder 0:  1 → 32      @ 320 ───────────────────────────────┐ skip
  │   max-pool                                                    │
  ├─ encoder 1:  32 → 64     @ 160 ─────────────────────┐ skip    │
  │   max-pool                                          │         │
  ├─ encoder 2:  64 → 128    @ 80 ──────────┐ skip      │         │
  │   max-pool                              │           │         │
  ├─ bottleneck: 128 → 256   @ 40           │           │         │
  │                                         │           │         │
  ├─ upconv 0:   256 → 128   @ 80           │           │         │
  ├─ decoder 0:  (128 + 128) → 128  ◄───────┘           │         │
  ├─ upconv 1:   128 → 64    @ 160                      │         │
  ├─ decoder 1:  (64 + 64) → 64     ◄───────────────────┘         │
  ├─ upconv 2:   64 → 32     @ 320                                │
  ├─ decoder 2:  (32 + 32) → 32     ◄─────────────────────────────┘
  │
  └─ output conv 1 × 1:  32 → 1
        │
     prediction, kl
```

Each encoder, the bottleneck, and each decoder is a `VariationalDoubleConv`. Each upconv is a `VariationalConvTranspose2d` with kernel 2 and stride 2. The output layer is a `VariationalConv2d` with a 1 × 1 kernel. Max-pooling has no weights.

If the image size is not divisible by 2^depth, the upsampled tensor can be one pixel smaller than the skip connection. The forward pass then resizes it with bilinear interpolation before concatenating, exactly as the ordinary UNet does.

### Collecting the KL

The forward pass keeps a running total:

```python
total_kl = 0.0

for encoder, pool in zip(self.encoders, self.pools):
    x, kl = encoder(x, sample=sample)
    total_kl = total_kl + kl
    ...
```

and does the same for the bottleneck, every upconv, every decoder, and the output layer. `total_kl` starts as the Python number `0.0` and becomes a tensor at the first addition.

### Where the weights are

| Layer | Weight shape | Weights | Std of initial means, 1/√(3 · fan_in) |
|---|---|---|---|
| `encoders.0.conv1` | (32, 1, 3, 3) | 288 | 0.192 |
| `encoders.0.conv2` | (32, 32, 3, 3) | 9,216 | 0.034 |
| `encoders.1.conv1` | (64, 32, 3, 3) | 18,432 | 0.034 |
| `encoders.1.conv2` | (64, 64, 3, 3) | 36,864 | 0.024 |
| `encoders.2.conv1` | (128, 64, 3, 3) | 73,728 | 0.024 |
| `encoders.2.conv2` | (128, 128, 3, 3) | 147,456 | 0.017 |
| `bottleneck.conv1` | (256, 128, 3, 3) | 294,912 | 0.017 |
| `bottleneck.conv2` | (256, 256, 3, 3) | 589,824 | 0.012 |
| `upconvs.0` | (256, 128, 2, 2) | 131,072 | 0.026 |
| `upconvs.1` | (128, 64, 2, 2) | 32,768 | 0.036 |
| `upconvs.2` | (64, 32, 2, 2) | 8,192 | 0.051 |
| `decoders.0.conv1` | (128, 256, 3, 3) | 294,912 | 0.012 |
| `decoders.0.conv2` | (128, 128, 3, 3) | 147,456 | 0.017 |
| `decoders.1.conv1` | (64, 128, 3, 3) | 73,728 | 0.017 |
| `decoders.1.conv2` | (64, 64, 3, 3) | 36,864 | 0.024 |
| `decoders.2.conv1` | (32, 64, 3, 3) | 18,432 | 0.024 |
| `decoders.2.conv2` | (32, 32, 3, 3) | 9,216 | 0.034 |
| `output_layer` | (1, 32, 1, 1) | 32 | 0.102 |

Two things stand out:

- **Almost all the weights are in the middle of the network.** The bottleneck and the first decoder block hold about 1.3 million of the 1.9 million weights.
- **Those are also the layers with the smallest weights.** A single `initial_rho` for the whole model is therefore far noisier, relative to the weights, in exactly the layers that hold most of the parameters.

### Why this model is hard to make work

Three problems compound:

1. **The KL is huge.** About 1.9 million weights at 4.5 to 5 nats each gives a KL of roughly 8.7 million nats at initialisation, or 13,000 per training image. With `kl_weight: 0.1` the KL term is about 1,300 while the reconstruction loss is about 0.002.
2. **Noise compounds through depth.** The signal passes through about 18 noisy layers. Each adds its own perturbation.
3. **One σ for all layers.** See the table above.

With careful settings it trained to within 11% to 15% of the plain UNet in the one run made, but at that point σ is so small that the uncertainty maps show nothing. It is, however, the only model whose uncertainty reacted to an unfamiliar noise level. Details in [07_experiments.md](07_experiments.md).

## 3.4 `variational_partial_unet`

File: [models/variational_partial_unet.py](../../models/variational_partial_unet.py).

The response to the problems above: keep the UNet deterministic except for the layers closest to the output.

### What is variational

With the default `variational_decoder_levels: 1`:

```
  ├─ encoder 0, 1, 2         deterministic  (DoubleConv)
  ├─ bottleneck              deterministic  (DoubleConv)
  ├─ upconv 0, 1, 2          deterministic  (nn.ConvTranspose2d)
  ├─ decoder 0               deterministic  (DoubleConv)
  ├─ decoder 1               deterministic  (DoubleConv)
  ├─ decoder 2               VARIATIONAL    (VariationalDoubleConv)
  └─ output conv 1 × 1       VARIATIONAL    (VariationalConv2d)
```

The variational parameters are:

| Layer | Weight shape | Weights + biases |
|---|---|---|
| `decoders.2.conv1` | (32, 64, 3, 3) | 18,432 + 32 |
| `decoders.2.conv2` | (32, 32, 3, 3) | 9,216 + 32 |
| `output_layer` | (1, 32, 1, 1) | 32 + 1 |
| **Total** | | **27,745** |

That is 1.4% of the network's weights.

### The `variational_decoder_levels` setting

It counts decoder blocks from the output end. The output convolution is always variational.

| Value | Variational layers | Variational weights (n) | KL at initialisation, per training image |
|---|---|---|---|
| 0 | output conv only | 33 | 0.3 |
| 1 (default) | + decoder 2 | 27,745 | 252 |
| 2 | + decoder 1 | 138,465 | 1,259 |
| 3 (= depth) | + decoder 0, the whole decoder | 581,089 | 5,283 |

The KL column is for `initial_rho: -6.5`, `prior_standard_deviation: 1.0`, divided by N = 660. It is what the training progress bar shows as `kl`.

The constructor rejects values below 0 or above `depth`.

### How the choice is made in the code

The decoder blocks are built in a loop from coarsest to finest. `index` 0 is the coarsest block:

```python
for index, level in enumerate(reversed(range(depth))):
    ...
    is_variational = index >= depth - variational_decoder_levels

    if is_variational:
        decoder = VariationalDoubleConv(skip_channels * 2, skip_channels, **block_args, **variational_args)
    else:
        decoder = DoubleConv(skip_channels * 2, skip_channels, **block_args)
```

With `depth = 3` and one variational level, `index >= 2` is true only for the last block.

Both kinds of block live in the same `nn.ModuleList`, so the forward pass has to treat them differently:

```python
if isinstance(decoder, VariationalDoubleConv):
    x, kl = decoder(x, sample=sample)
    total_kl = total_kl + kl
else:
    x = decoder(x)
```

The deterministic blocks are the same `DoubleConv` class the plain UNet uses, imported from `models/unet.py`. The upsampling layers are ordinary `nn.ConvTranspose2d` in every configuration, including `variational_decoder_levels: 3`.

### Why this placement

- **Noise does not compound.** Sampled weights only affect the last three convolutions. Everything before them is computed once, deterministically.
- **The affected layers are narrow.** Their weights are comparatively large (std 0.024 to 0.10), so a given σ is a smaller relative perturbation than in the bottleneck.
- **The KL is 70 times smaller** than for the fully variational model, simply because there are 70 times fewer variational weights.
- **The cost of the noise itself is low.** On a trained UNet, noise of σ = 0.001 on these layers raises the loss by 0.3% ([06_hyperparameters.md](06_hyperparameters.md#67-measured-sensitivity-to-weight-noise)).

The deterministic layers have no prior and no KL. They are ordinary point estimates. Strictly, the model is a deterministic feature extractor followed by a small Bayesian head. The uncertainty it reports comes only from the head, so it cannot express doubt about the features themselves.

That limitation showed up in the experiments. When the input noise was raised beyond the training level and the model failed, its uncertainty did not rise, while that of the fully variational model did ([07_experiments.md, section 7.8](07_experiments.md#78-finding-5-the-head-only-uncertainty-does-not-react-to-unfamiliar-inputs)). And trained models of this kind were not as accurate as the plain UNet: between 1% and 14% worse in three seeded comparisons.

### State-dict layout

Because the two kinds of block name their parameters differently, a checkpoint mixes both styles:

```
decoders.0.block.0.weight          deterministic block (nn.Sequential indices)
decoders.0.block.3.weight
decoders.2.conv1.weight_mean       variational block
decoders.2.conv1.weight_rho
decoders.2.conv2.weight_mean
output_layer.weight_mean
```

A checkpoint is only valid for the `variational_decoder_levels` it was trained with.

### A possible optimisation, not implemented

`predict_with_uncertainty` runs the whole network once per sample. For this model, everything up to the input of the last decoder block is identical in every pass. Caching that tensor and re-running only the variational head would make Monte Carlo prediction many times faster. The current code does not do this, because the same function serves all three models.

## 3.5 Adding another variational model

1. Create the class in `models/`. Its `forward` must accept `sample=True` and return `(prediction, kl)`.
2. Add the KL of every variational layer to a running total. Do not put variational layers inside an `nn.Sequential`.
3. Export the class in `models/__init__.py` and add a branch in `create_model`. The name must start with `variational`.
4. Keep the `_rho` suffix on standard-deviation parameters so `rho_learning_rate` finds them.

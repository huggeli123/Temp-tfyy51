# 2. The layers: `models/variational.py`

This file walks through [models/variational.py](../../models/variational.py) and the checks in [sanity_check.py](../../sanity_check.py). The mathematics behind each piece is in [01_theory.md](01_theory.md).

## 2.1 Overview

The file has two parts:

- Three **helper functions** shared by all layers: `positive_scale`, `sample_gaussian`, `diagonal_gaussian_kl`.
- Three **layer classes**, each a drop-in counterpart of a PyTorch layer:

| Variational layer | Counterpart | Used by |
|---|---|---|
| `VariationalConv2d` | `nn.Conv2d` | all three variational models |
| `VariationalConvTranspose2d` | `nn.ConvTranspose2d` | `variational_unet` (the upsampling steps) |
| `VariationalLinear` | `nn.Linear` | no model yet; available and tested |

All three classes follow the same contract.

## 2.2 The contract every layer follows

<!-- docs-check: illustrative -->
```python
output, kl = layer(x, sample=True)
```

- **Two return values.** An ordinary layer returns a tensor. A variational layer returns `(output, kl_divergence)`. Any model that contains one must collect the KL and return it too. This is why variational models need their own training loop.
- **`sample=True`** draws fresh weights for this call. Two calls with the same input give different outputs.
- **`sample=False`** uses the means. Two calls give identical outputs.
- **`kl_divergence`** is a scalar tensor: the KL of this layer's weights and biases from the prior, summed over all elements. It depends only on the parameters, not on the input or on the sampled noise. It is computed on every call, including with `sample=False`.
- **Four parameters instead of two.** `weight_mean`, `weight_rho`, `bias_mean`, `bias_rho`. The `_rho` tensors have the same shapes as the means. A variational layer therefore has exactly twice as many trainable parameters as its counterpart.

The parameter names matter in one other place: `get_parameter_groups` in [utils/model_configs.py](../../utils/model_configs.py) finds the standard-deviation parameters by the suffix `_rho`. A new variational layer must keep that naming to get the separate learning rate.

## 2.3 The helpers

### `positive_scale(rho)`

```python
def positive_scale(rho):
    return functional.softplus(rho)
```

Turns the unconstrained ρ into a strictly positive standard deviation σ = ln(1 + e^ρ). See [01_theory.md, section 1.2](01_theory.md#12-keeping-σ-positive-the-ρ-parameter) for why the logarithmic scale matters, and [06_hyperparameters.md, section 6.4](06_hyperparameters.md#64-initial_rho) for a table of values.

`functional.softplus` is numerically safe for large positive inputs (PyTorch switches to the identity above a threshold). For very negative inputs σ becomes tiny but stays positive in floating point for every value used here.

### `sample_gaussian(mean, rho, sample)`

```python
def sample_gaussian(mean, rho, sample):
    if not sample:
        return mean
    return mean + positive_scale(rho) * torch.randn_like(mean)
```

The reparameterisation trick. `torch.randn_like(mean)` creates noise with the same shape, dtype and device as the mean, so the function works unchanged on CPU and GPU.

The noise tensor has the shape of the *weights*, not of the activations. One perturbed weight tensor is used for the whole batch.

### `diagonal_gaussian_kl(mean, rho, prior_standard_deviation)`

The closed-form KL between N(μ, σ²) and N(0, p²), summed over the tensor. The formula and how to read it are in [01_theory.md, section 1.5](01_theory.md#15-the-closed-form-kl).

Two implementation notes:

- `math.log(prior_standard_deviation)` is a plain Python float, because the prior is a fixed number, not a tensor.
- `torch.log(standard_deviation)` is the only place that could produce infinity, if σ underflowed to zero. With softplus and the ρ values in use this does not happen.

## 2.4 `VariationalConv2d`

### Constructor

```python
def __init__(
    self,
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    prior_standard_deviation=1.0,
    initial_rho=-5.0,
):
```

The first eight arguments after `self` mean the same as in `nn.Conv2d`. Not supported: `padding_mode` (only zero padding) and string padding such as `"same"`.

What the constructor does:

1. **Validates `groups`.** Both channel counts must be divisible by it, as in PyTorch.
2. **Normalises the size arguments** with `_pair`, so `kernel_size=3` and `kernel_size=(3, 3)` behave the same.
3. **Creates the weight parameters** with the shape an ordinary convolution would have:

   ```python
   weight_shape = (out_channels, in_channels // groups, kernel_height, kernel_width)
   self.weight_mean = nn.Parameter(torch.empty(weight_shape))
   self.weight_rho  = nn.Parameter(torch.full(weight_shape, initial_rho))
   ```

   Every ρ starts at the same value `initial_rho`, so every weight starts with the same σ.
4. **Creates the bias parameters** the same way, or registers them as `None` when `bias=False`.
5. **Calls `reset_parameters()`** to initialise the means.

### Initialisation of the means

```python
def reset_parameters(self):
    nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))

    if self.bias_mean is not None:
        fan_in = (
            (self.in_channels // self.groups)
            * self.kernel_size[0]
            * self.kernel_size[1]
        )
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mean, -bound, bound)
```

This is deliberately identical to what `nn.Conv2d` does, so a variational layer starts from the same distribution of means as the layer it replaces.

`kaiming_uniform_` with `a = sqrt(5)` draws weights uniformly from ±1/√fan_in, where fan_in = input channels × kernel height × kernel width. The standard deviation of that distribution is

$$
\operatorname{std}(\mu) = \frac{1}{\sqrt{3 \cdot \text{fan\_in}}}
$$

This number is central to choosing `initial_rho`. The noise σ is the same for every layer, but the weights themselves get *smaller* as layers get wider:

| Layer (channels = 32) | fan_in | std of initial means | σ at ρ = -5 as a share of that |
|---|---|---|---|
| First conv, 1 → 32 | 9 | 0.19 | 3.5% |
| 32 → 32 | 288 | 0.034 | 20% |
| 64 → 64 | 576 | 0.024 | 28% |
| 128 → 128 | 1,152 | 0.017 | 39% |
| 256 → 256 (bottleneck) | 2,304 | 0.012 | 56% |
| Output conv 1×1, 32 → 1 | 32 | 0.10 | 6.6% |

A noise level that is negligible in the first layer is more than half the size of the weights in the bottleneck. This is the main reason the fully variational UNet with `initial_rho: -5` performed badly ([07_experiments.md](07_experiments.md)).

### Forward pass

```python
def forward(self, input_tensor, sample=True):

    weight = sample_gaussian(self.weight_mean, self.weight_rho, sample)

    bias = None
    if self.bias_mean is not None:
        bias = sample_gaussian(self.bias_mean, self.bias_rho, sample)

    output = functional.conv2d(
        input_tensor, weight, bias,
        self.stride, self.padding, self.dilation, self.groups,
    )

    kl_divergence = diagonal_gaussian_kl(
        self.weight_mean, self.weight_rho, self.prior_standard_deviation,
    )

    if self.bias_mean is not None:
        kl_divergence = kl_divergence + diagonal_gaussian_kl(
            self.bias_mean, self.bias_rho, self.prior_standard_deviation,
        )

    return output, kl_divergence
```

Step by step:

1. Draw one weight tensor (or take the means).
2. Draw one bias vector the same way. The bias is variational too, with its own μ and ρ.
3. Run an ordinary convolution with those weights. `functional.conv2d` is the functional form of `nn.Conv2d`: it takes the weights as arguments instead of owning them.
4. Compute the KL for the weights and add the KL for the bias.
5. Return both.

The layer does not look at `self.training`. Whether weights are sampled is controlled only by the `sample` argument. Calling `model.eval()` does not turn sampling off.

## 2.5 `VariationalConvTranspose2d`

A transposed convolution is the upsampling step in the UNet decoder (kernel 2, stride 2 doubles the height and width). This class mirrors `nn.ConvTranspose2d` in the same way.

The differences from `VariationalConv2d`:

- **The weight shape has the channels the other way round**, as in PyTorch:

  ```python
  weight_shape = (in_channels, out_channels, kernel_height, kernel_width)
  ```

- **`fan_in` for the bias uses `out_channels`.** PyTorch computes fan_in from dimension 1 of the weight tensor, which for a transposed convolution is the output channel count. The code reproduces that so the initial means match `nn.ConvTranspose2d` exactly:

  ```python
  fan_in = (
      self.out_channels
      * self.kernel_size[0]
      * self.kernel_size[1]
  )
  ```

- **It takes `output_padding`** and does not take `dilation` or `groups`.
- **The forward pass calls `functional.conv_transpose2d`.**

Everything else, including the KL computation and the `sample` flag, is identical.

## 2.6 `VariationalLinear`

The counterpart of `nn.Linear`, with weight shape `(out_features, in_features)` and the forward pass `functional.linear(input_tensor, weight, bias)`.

It acts on the last dimension of its input, like `nn.Linear`, so inputs of shape `(batch, ..., in_features)` work.

No model in the project uses it at the moment. It exists because the launch guide specified it, and it is covered by the sanity checks.

## 2.7 Things that are easy to get wrong

- **Forgetting the KL.** A model that calls a variational layer and drops the second return value still trains, but it is then just a network with weight noise and no prior. Every model in the project adds each layer's KL to a running total.
- **Expecting `model.eval()` to make predictions deterministic.** It does not. Pass `sample=False`.
- **State dicts are not interchangeable with ordinary models.** A variational layer saves `weight_mean` and `weight_rho`, not `weight`. A checkpoint from `unet` cannot be loaded directly into `variational_unet`. Copying `weight` into `weight_mean` by hand does work, and the calibration measurement in [06_hyperparameters.md](06_hyperparameters.md#67-measured-sensitivity-to-weight-noise) did exactly that.
- **Weight decay on ρ.** Weight decay pulls parameters toward zero. For ρ, zero means σ = ln 2 ≈ 0.69, which is enormous. `get_parameter_groups` sets weight decay to zero for the ρ group when a separate ρ learning rate is used. Without `rho_learning_rate`, a non-zero `training.weight_decay` *would* be applied to ρ. The config uses `weight_decay: 0.0`, so this does not currently bite.

## 2.8 The checks: `sanity_check.py`

Run `python sanity_check.py` from the repository root.

**Layer checks.** It builds one `VariationalConv2d`, one `VariationalConvTranspose2d` and one `VariationalLinear`, each next to the ordinary PyTorch layer of the same shape, and verifies seven things for each:

| # | Check | Why it matters |
|---|---|---|
| 1 | `sample=False` twice gives identical outputs | The posterior-mean prediction must be deterministic |
| 2 | `sample=True` twice gives different outputs | Sampling must actually happen |
| 3 | Output shape and weight shape match the ordinary layer | The layer is a true drop-in replacement |
| 4 | σ is finite and strictly positive | The softplus mapping works |
| 5 | The KL is a scalar, finite, and not negative | The closed-form KL is implemented correctly |
| 6 | Gradients reach `weight_mean`, `weight_rho`, `bias_mean`, `bias_rho` | The reparameterisation trick lets all four be trained |
| 7 | With ρ = -20, the sampled output equals the mean output | As σ goes to zero the layer becomes an ordinary one |

**Model checks.** It then builds each variational model through `create_model` (`variational_simple_cnn`, `variational_unet`, and `variational_partial_unet` with 0, 1 and 3 variational decoder levels) and verifies four things for each, on an input of 37 × 41 pixels:

| # | Check | Why it matters |
|---|---|---|
| 1 | The prediction has the shape of the input | The size is not divisible by 8, so the resizing before each skip connection is exercised |
| 2 | The KL is a finite, positive scalar | Every variational layer's KL is collected |
| 3 | `sample=False` twice gives identical outputs | The posterior-mean prediction is deterministic for the whole model |
| 4 | `sample=True` twice gives different outputs | The `sample` flag reaches the variational layers |

The script exits with status 1 on the first failure.

**What it does not cover:** training. Nothing checks that a loss decreases or that the optimizer groups are right. A second script, `python scripts/check_variational_docs.py`, checks that this documentation still matches the code; see the [README](README.md#keeping-these-docs-in-sync).

# Variational Linear and Convolutional Layers in PyTorch

> Launch point for implementing Bayes-by-Backprop layers in an existing CT denoising model. The goal is to understand and expose the mechanism, not hide it behind an automatic model converter.

## 1. Core idea

A deterministic layer learns one weight tensor `weight`. A mean-field variational layer instead learns

\[
q_\phi(W)=\mathcal{N}(W;\mu,\operatorname{diag}(\sigma^2)),
\qquad \sigma=\operatorname{softplus}(\rho).
\]

Each stochastic forward pass samples

\[
W=\mu+\sigma\odot\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
\]

The training objective combines the normal task loss with a KL penalty:

\[
\mathcal{L}=\mathcal{L}_{\text{task}}+\beta\,D_{\mathrm{KL}}\!\left(q_\phi(W)\Vert p(W)\right).
\]

For CT denoising, `task_loss` can initially remain the deterministic model's reconstruction loss. Predictive uncertainty is estimated from repeated stochastic forward passes.

## 2. Minimal API contract

Both variational layers below follow the same interface.

```python
output, kl_divergence = layer(input_tensor, sample=True)
```

- `input_tensor`: ordinary PyTorch tensor.
- `sample=True`: sample weights from the approximate posterior.
- `sample=False`: use posterior means for a deterministic forward pass.
- `output`: same shape contract as the corresponding `nn.Linear` or `nn.Conv2d`.
- `kl_divergence`: scalar tensor for this layer's KL contribution.

Model-level code must sum the KL terms from all variational layers. Do not silently store KL as mutable layer state.

## 3. Shared helpers

```python
import math
import torch
from torch import nn
from torch.nn import functional as functional


def positive_scale(rho: torch.Tensor) -> torch.Tensor:
    return functional.softplus(rho)


def sample_gaussian(mean: torch.Tensor, rho: torch.Tensor, sample: bool) -> torch.Tensor:
    if not sample:
        return mean
    return mean + positive_scale(rho) * torch.randn_like(mean)


def diagonal_gaussian_kl(mean: torch.Tensor, rho: torch.Tensor, prior_standard_deviation: float) -> torch.Tensor:
    standard_deviation = positive_scale(rho)
    prior_variance = prior_standard_deviation ** 2
    return 0.5 * torch.sum((standard_deviation.square() + mean.square()) / prior_variance - 1.0 + 2.0 * (math.log(prior_standard_deviation) - torch.log(standard_deviation)))
```

This uses a zero-mean isotropic Gaussian prior. More elaborate priors can come later; changing prior and posterior families simultaneously makes debugging unnecessarily queer in the bad way.

## 4. Variational linear layer

```python
class VariationalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, prior_standard_deviation: float = 1.0, initial_rho: float = -5.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_standard_deviation = prior_standard_deviation
        self.weight_mean = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), initial_rho))
        if bias:
            self.bias_mean = nn.Parameter(torch.empty(out_features))
            self.bias_rho = nn.Parameter(torch.full((out_features,), initial_rho))
        else:
            self.register_parameter("bias_mean", None)
            self.register_parameter("bias_rho", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        if self.bias_mean is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias_mean, -bound, bound)

    def forward(self, input_tensor: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        weight = sample_gaussian(self.weight_mean, self.weight_rho, sample)
        bias = None if self.bias_mean is None else sample_gaussian(self.bias_mean, self.bias_rho, sample)
        output = functional.linear(input_tensor, weight, bias)
        kl_divergence = diagonal_gaussian_kl(self.weight_mean, self.weight_rho, self.prior_standard_deviation)
        if self.bias_mean is not None:
            kl_divergence = kl_divergence + diagonal_gaussian_kl(self.bias_mean, self.bias_rho, self.prior_standard_deviation)
        return output, kl_divergence
```

Expected shapes:

- input: `[..., in_features]`
- output: `[..., out_features]`
- weight parameters: `[out_features, in_features]`

## 5. Variational 2D convolution

The statistical mechanism is identical. Only the parameter shape and deterministic operation change.

```python
class VariationalConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, dilation: int | tuple[int, int] = 1, groups: int = 1, bias: bool = True, prior_standard_deviation: float = 1.0, initial_rho: float = -5.0):
        super().__init__()
        kernel_height, kernel_width = nn.modules.utils._pair(kernel_size)
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_height, kernel_width)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = nn.modules.utils._pair(padding)
        self.dilation = nn.modules.utils._pair(dilation)
        self.groups = groups
        self.prior_standard_deviation = prior_standard_deviation
        weight_shape = (out_channels, in_channels // groups, kernel_height, kernel_width)
        self.weight_mean = nn.Parameter(torch.empty(weight_shape))
        self.weight_rho = nn.Parameter(torch.full(weight_shape, initial_rho))
        if bias:
            self.bias_mean = nn.Parameter(torch.empty(out_channels))
            self.bias_rho = nn.Parameter(torch.full((out_channels,), initial_rho))
        else:
            self.register_parameter("bias_mean", None)
            self.register_parameter("bias_rho", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        if self.bias_mean is not None:
            fan_in = (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mean, -bound, bound)

    def forward(self, input_tensor: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        weight = sample_gaussian(self.weight_mean, self.weight_rho, sample)
        bias = None if self.bias_mean is None else sample_gaussian(self.bias_mean, self.bias_rho, sample)
        output = functional.conv2d(input_tensor, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        kl_divergence = diagonal_gaussian_kl(self.weight_mean, self.weight_rho, self.prior_standard_deviation)
        if self.bias_mean is not None:
            kl_divergence = kl_divergence + diagonal_gaussian_kl(self.bias_mean, self.bias_rho, self.prior_standard_deviation)
        return output, kl_divergence
```

Expected shapes:

- input: `[batch, in_channels, height, width]`
- output: ordinary `conv2d` output shape
- weight parameters: `[out_channels, in_channels / groups, kernel_height, kernel_width]`

## 6. Integrating a layer into a denoiser

Start with one or a few strategically chosen convolutions, not the whole model.

```python
class VariationalDenoisingBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.first_convolution = nn.Conv2d(channels, channels, 3, padding=1)
        self.second_convolution = VariationalConv2d(channels, channels, 3, padding=1)

    def forward(self, input_tensor: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_tensor = functional.relu(self.first_convolution(input_tensor))
        output_tensor, kl_divergence = self.second_convolution(hidden_tensor, sample=sample)
        return output_tensor, kl_divergence
```

For a larger model, return `(prediction, total_kl)` from the model's `forward`, or use a small explicit result structure. Preserve this convention consistently.

## 7. Training API

```python
def training_step(model: nn.Module, noisy_image: torch.Tensor, clean_image: torch.Tensor, dataset_size: int, kl_weight: float = 1.0) -> dict[str, torch.Tensor]:
    prediction, total_kl = model(noisy_image, sample=True)
    reconstruction_loss = functional.mse_loss(prediction, clean_image)
    scaled_kl = total_kl / dataset_size
    total_loss = reconstruction_loss + kl_weight * scaled_kl
    return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_divergence": scaled_kl}
```

`dataset_size` scaling is one valid convention when the reconstruction term is averaged per minibatch. The important requirement is consistency between the reduction used by the data term and the KL scaling. Record the convention in experiments.

A KL warm-up can be introduced as

\[
\beta_t=\min(1,t/T_{\mathrm{warmup}}),
\]

but first verify that the unmodified objective and gradients are implemented correctly. Warm-up is not a substitute for diagnosis.

## 8. Predictive mean and epistemic variance

```python
@torch.no_grad()
def predict_with_uncertainty(model: nn.Module, noisy_image: torch.Tensor, number_of_samples: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = torch.stack([model(noisy_image, sample=True)[0] for _ in range(number_of_samples)])
    predictive_mean = predictions.mean(dim=0)
    epistemic_variance = predictions.var(dim=0, unbiased=False)
    return predictive_mean, epistemic_variance
```

This variance measures variation caused by posterior weight samples. It is not automatically calibrated, and it is not the same as aleatoric observation noise.

## 9. Minimum validation checklist

1. With `sample=False`, two forward passes must be identical.
2. With `sample=True`, two forward passes should usually differ.
3. Output shapes must match equivalent `nn.Linear` and `nn.Conv2d` layers.
4. `positive_scale(rho)` must remain finite and strictly positive.
5. KL must be finite, scalar, and non-negative up to numerical tolerance.
6. Gradients must reach both `*_mean` and `*_rho` parameters.
7. Very negative `rho` should make sampled outputs approach mean-weight outputs.
8. A deterministic baseline must remain available for comparison.
9. Report reconstruction quality and uncertainty behaviour under clean, noisier, and shifted inputs.
10. Do not interpret high predictive variance as proof that uncertainty is correct.

## 10. Suggested first milestone

- Implement and test `VariationalLinear` on a tiny regression problem.
- Generalize the same parameterization to `VariationalConv2d`.
- Replace one convolution near the denoiser bottleneck or output path.
- Compare deterministic and stochastic predictions.
- Add KL scaling only after shape, sampling, and gradient tests pass.
- Expand to more variational layers only if the partial model behaves sensibly.

## 11. Reference implementations

- [PyTorch-BayesianCNN](https://github.com/kumar-shridhar/PyTorch-BayesianCNN): explicit `BBBLinear`, `BBBConv2d`, and local-reparameterization variants.
- [Basic Bayesian Neural Network with PyTorch](https://ai4fusion-wmschool.github.io/summer2025/Lecture17_BasicBNN.html): small linear-layer walkthrough.
- [TorchBayesian](https://torchbayesian.github.io/): modern API reference and comparison point, not the primary teaching abstraction.

## 12. Scope boundary

This document implements weight-space mean-field variational inference with a Gaussian prior and posterior. It does not yet implement local reparameterization, Flipout, structured posteriors, aleatoric output distributions, calibration, or full Bayesian inference. Those are extensions after the minimal version is tested.

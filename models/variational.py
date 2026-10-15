import math

import torch
from torch import nn
from torch.nn import functional
from torch.nn.modules.utils import _pair


# ============================================================
# Shared helpers for mean-field Gaussian variational layers
# ============================================================

def positive_scale(rho):
    """
    Map the unconstrained parameter rho to a standard deviation
    sigma = softplus(rho) > 0.
    """

    return functional.softplus(rho)


def sample_gaussian(mean, rho, sample):
    """
    Reparameterised sample  W = mu + sigma * eps,  eps ~ N(0, I).

    With sample=False the posterior mean is returned, which gives
    a deterministic forward pass.
    """

    if not sample:
        return mean

    return mean + positive_scale(rho) * torch.randn_like(mean)


def diagonal_gaussian_kl(mean, rho, prior_standard_deviation):
    """
    KL( N(mean, sigma^2) || N(0, prior_std^2) ), summed over all
    elements of the tensor. Returns a scalar tensor.
    """

    standard_deviation = positive_scale(rho)
    prior_variance = prior_standard_deviation ** 2

    return 0.5 * torch.sum(
        (standard_deviation.square() + mean.square()) / prior_variance
        - 1.0
        + 2.0 * (
            math.log(prior_standard_deviation)
            - torch.log(standard_deviation)
        )
    )


# ============================================================
# Variational 2D convolution
# ============================================================

class VariationalConv2d(nn.Module):
    """
    Conv2d whose weights (and bias) have a learned diagonal Gaussian
    posterior. forward() returns (output, kl_divergence).
    """

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
        super().__init__()

        # An integer from the YAML (-8 instead of -8.0) would make an
        # integer tensor below, which cannot require gradients
        initial_rho = float(initial_rho)

        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(
                "in_channels and out_channels must be divisible by groups"
            )

        kernel_height, kernel_width = _pair(kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_height, kernel_width)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.prior_standard_deviation = prior_standard_deviation

        # Same shape as the weight of an ordinary nn.Conv2d
        weight_shape = (
            out_channels,
            in_channels // groups,
            kernel_height,
            kernel_width,
        )

        self.weight_mean = nn.Parameter(torch.empty(weight_shape))
        self.weight_rho = nn.Parameter(
            torch.full(weight_shape, initial_rho)
        )

        if bias:
            self.bias_mean = nn.Parameter(torch.empty(out_channels))
            self.bias_rho = nn.Parameter(
                torch.full((out_channels,), initial_rho)
            )
        else:
            self.register_parameter("bias_mean", None)
            self.register_parameter("bias_rho", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Means start exactly like nn.Conv2d's default initialisation
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))

        if self.bias_mean is not None:
            fan_in = (
                (self.in_channels // self.groups)
                * self.kernel_size[0]
                * self.kernel_size[1]
            )
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mean, -bound, bound)

    def forward(self, input_tensor, sample=True):

        weight = sample_gaussian(
            self.weight_mean, self.weight_rho, sample
        )

        bias = None
        if self.bias_mean is not None:
            bias = sample_gaussian(
                self.bias_mean, self.bias_rho, sample
            )

        output = functional.conv2d(
            input_tensor,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        kl_divergence = diagonal_gaussian_kl(
            self.weight_mean,
            self.weight_rho,
            self.prior_standard_deviation,
        )

        if self.bias_mean is not None:
            kl_divergence = kl_divergence + diagonal_gaussian_kl(
                self.bias_mean,
                self.bias_rho,
                self.prior_standard_deviation,
            )

        return output, kl_divergence


# ============================================================
# Variational 2D transposed convolution
# ============================================================

class VariationalConvTranspose2d(nn.Module):
    """
    ConvTranspose2d whose weights (and bias) have a learned diagonal
    Gaussian posterior. forward() returns (output, kl_divergence).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
        prior_standard_deviation=1.0,
        initial_rho=-5.0,
    ):
        super().__init__()

        # An integer from the YAML (-8 instead of -8.0) would make an
        # integer tensor below, which cannot require gradients
        initial_rho = float(initial_rho)

        kernel_height, kernel_width = _pair(kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_height, kernel_width)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.prior_standard_deviation = prior_standard_deviation

        # Same shape as the weight of an ordinary nn.ConvTranspose2d
        weight_shape = (
            in_channels,
            out_channels,
            kernel_height,
            kernel_width,
        )

        self.weight_mean = nn.Parameter(torch.empty(weight_shape))
        self.weight_rho = nn.Parameter(
            torch.full(weight_shape, initial_rho)
        )

        if bias:
            self.bias_mean = nn.Parameter(torch.empty(out_channels))
            self.bias_rho = nn.Parameter(
                torch.full((out_channels,), initial_rho)
            )
        else:
            self.register_parameter("bias_mean", None)
            self.register_parameter("bias_rho", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Means start exactly like nn.ConvTranspose2d's default
        # initialisation (fan_in is taken from dim 1 of the weight)
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))

        if self.bias_mean is not None:
            fan_in = (
                self.out_channels
                * self.kernel_size[0]
                * self.kernel_size[1]
            )
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mean, -bound, bound)

    def forward(self, input_tensor, sample=True):

        weight = sample_gaussian(
            self.weight_mean, self.weight_rho, sample
        )

        bias = None
        if self.bias_mean is not None:
            bias = sample_gaussian(
                self.bias_mean, self.bias_rho, sample
            )

        output = functional.conv_transpose2d(
            input_tensor,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
        )

        kl_divergence = diagonal_gaussian_kl(
            self.weight_mean,
            self.weight_rho,
            self.prior_standard_deviation,
        )

        if self.bias_mean is not None:
            kl_divergence = kl_divergence + diagonal_gaussian_kl(
                self.bias_mean,
                self.bias_rho,
                self.prior_standard_deviation,
            )

        return output, kl_divergence


# ============================================================
# Variational linear layer
# ============================================================

class VariationalLinear(nn.Module):
    """
    Linear layer whose weights (and bias) have a learned diagonal
    Gaussian posterior. forward() returns (output, kl_divergence).
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        prior_standard_deviation=1.0,
        initial_rho=-5.0,
    ):
        super().__init__()

        # An integer from the YAML (-8 instead of -8.0) would make an
        # integer tensor below, which cannot require gradients
        initial_rho = float(initial_rho)

        self.in_features = in_features
        self.out_features = out_features
        self.prior_standard_deviation = prior_standard_deviation

        # Same shape as the weight of an ordinary nn.Linear
        weight_shape = (out_features, in_features)

        self.weight_mean = nn.Parameter(torch.empty(weight_shape))
        self.weight_rho = nn.Parameter(
            torch.full(weight_shape, initial_rho)
        )

        if bias:
            self.bias_mean = nn.Parameter(torch.empty(out_features))
            self.bias_rho = nn.Parameter(
                torch.full((out_features,), initial_rho)
            )
        else:
            self.register_parameter("bias_mean", None)
            self.register_parameter("bias_rho", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Means start exactly like nn.Linear's default initialisation
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))

        if self.bias_mean is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias_mean, -bound, bound)

    def forward(self, input_tensor, sample=True):

        weight = sample_gaussian(
            self.weight_mean, self.weight_rho, sample
        )

        bias = None
        if self.bias_mean is not None:
            bias = sample_gaussian(
                self.bias_mean, self.bias_rho, sample
            )

        output = functional.linear(input_tensor, weight, bias)

        kl_divergence = diagonal_gaussian_kl(
            self.weight_mean,
            self.weight_rho,
            self.prior_standard_deviation,
        )

        if self.bias_mean is not None:
            kl_divergence = kl_divergence + diagonal_gaussian_kl(
                self.bias_mean,
                self.bias_rho,
                self.prior_standard_deviation,
            )

        return output, kl_divergence

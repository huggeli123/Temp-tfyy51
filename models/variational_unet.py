import torch

from torch import nn
import torch.nn.functional as F

from .layers import (
    get_activation,
    get_normalization,
)
from .variational import (
    VariationalConv2d,
    VariationalConvTranspose2d,
)


class VariationalDoubleConv(nn.Module):
    """
    Two (variational conv -> normalization -> activation) blocks.

    forward() returns (output, kl_divergence).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation="relu",
        normalization="none",
        num_groups=4,
        prior_standard_deviation=1.0,
        initial_rho=-5.0,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd, for example 3, 5, or 7"
            )

        padding = kernel_size // 2

        variational_args = {
            "prior_standard_deviation": prior_standard_deviation,
            "initial_rho": initial_rho,
        }

        self.conv1 = VariationalConv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            **variational_args,
        )
        self.post1 = nn.Sequential(
            get_normalization(normalization, out_channels, num_groups),
            get_activation(activation),
        )

        self.conv2 = VariationalConv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            **variational_args,
        )
        self.post2 = nn.Sequential(
            get_normalization(normalization, out_channels, num_groups),
            get_activation(activation),
        )

    def forward(self, x, sample=True):

        x, kl_1 = self.conv1(x, sample=sample)
        x = self.post1(x)

        x, kl_2 = self.conv2(x, sample=sample)
        x = self.post2(x)

        return x, kl_1 + kl_2


class VariationalUNet(nn.Module):
    """
    U-Net where every convolution (including the transposed
    convolutions and the output layer) has a Gaussian weight posterior.

    forward() returns (prediction, kl_divergence), where the KL is
    summed over all variational layers.
    """

    def __init__(
        self,
        channels=32,
        depth=3,
        kernel_size=3,
        activation="relu",
        normalization="none",
        num_groups=4,
        prior_standard_deviation=1.0,
        initial_rho=-5.0,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError(
                "depth must be at least 1"
            )

        self.depth = depth

        block_args = {
            "kernel_size": kernel_size,
            "activation": activation,
            "normalization": normalization,
            "num_groups": num_groups,
            "prior_standard_deviation": prior_standard_deviation,
            "initial_rho": initial_rho,
        }

        # ==================================================
        # Encoder
        # ==================================================

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_channels = 1

        for level in range(depth):

            out_channels = channels * (2 ** level)

            self.encoders.append(
                VariationalDoubleConv(
                    in_channels, out_channels, **block_args
                )
            )
            self.pools.append(nn.MaxPool2d(2))

            in_channels = out_channels

        # ==================================================
        # Bottleneck
        # ==================================================

        bottleneck_channels = channels * (2 ** depth)

        self.bottleneck = VariationalDoubleConv(
            in_channels, bottleneck_channels, **block_args
        )

        # ==================================================
        # Decoder
        # ==================================================

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        current_channels = bottleneck_channels

        for level in reversed(range(depth)):

            skip_channels = channels * (2 ** level)

            self.upconvs.append(
                VariationalConvTranspose2d(
                    current_channels,
                    skip_channels,
                    kernel_size=2,
                    stride=2,
                    prior_standard_deviation=prior_standard_deviation,
                    initial_rho=initial_rho,
                )
            )

            self.decoders.append(
                VariationalDoubleConv(
                    skip_channels * 2, skip_channels, **block_args
                )
            )

            current_channels = skip_channels

        # ==================================================
        # Output
        # ==================================================

        self.output_layer = VariationalConv2d(
            channels,
            1,
            kernel_size=1,
            prior_standard_deviation=prior_standard_deviation,
            initial_rho=initial_rho,
        )

    def forward(self, x, sample=True):

        skip_connections = []
        total_kl = 0.0

        # Encoder
        for encoder, pool in zip(self.encoders, self.pools):

            x, kl = encoder(x, sample=sample)
            total_kl = total_kl + kl

            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x, kl = self.bottleneck(x, sample=sample)
        total_kl = total_kl + kl

        # Decoder
        skip_connections = skip_connections[::-1]

        for upconv, decoder, skip in zip(
            self.upconvs,
            self.decoders,
            skip_connections,
        ):

            x, kl = upconv(x, sample=sample)
            total_kl = total_kl + kl

            # Handle image sizes not divisible by powers of 2
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x = torch.cat([skip, x], dim=1)

            x, kl = decoder(x, sample=sample)
            total_kl = total_kl + kl

        # Output
        prediction, kl = self.output_layer(x, sample=sample)
        total_kl = total_kl + kl

        return prediction, total_kl

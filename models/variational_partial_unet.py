import torch

from torch import nn
import torch.nn.functional as F

from .unet import DoubleConv
from .variational import VariationalConv2d
from .variational_unet import VariationalDoubleConv


class VariationalPartialUNet(nn.Module):
    """
    U-Net where only the last part of the decoder is variational.

    The encoder, bottleneck, upsampling layers and the first
    (depth - variational_decoder_levels) decoder blocks are ordinary
    deterministic layers (point estimates, no KL). The last
    variational_decoder_levels decoder blocks and the 1x1 output
    convolution have a Gaussian weight posterior.

    Weight noise in a few late, narrow layers does not compound through
    the network, so it costs far less accuracy than making every layer
    stochastic, and the posterior std can be learned (see
    training.rho_learning_rate).

    variational_decoder_levels = 0 makes only the output conv
    variational; 1 (default) also makes the finest decoder block
    variational; depth makes the whole decoder variational.

    forward() returns (prediction, kl_divergence), where the KL is
    summed over the variational layers only.
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
        initial_rho=-6.0,
        variational_decoder_levels=1,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError(
                "depth must be at least 1"
            )

        if not 0 <= variational_decoder_levels <= depth:
            raise ValueError(
                "variational_decoder_levels must be between 0 and depth"
            )

        self.depth = depth
        self.variational_decoder_levels = variational_decoder_levels

        block_args = {
            "kernel_size": kernel_size,
            "activation": activation,
            "normalization": normalization,
            "num_groups": num_groups,
        }

        variational_args = {
            "prior_standard_deviation": prior_standard_deviation,
            "initial_rho": initial_rho,
        }

        # ==================================================
        # Encoder (deterministic)
        # ==================================================

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_channels = 1

        for level in range(depth):

            out_channels = channels * (2 ** level)

            self.encoders.append(
                DoubleConv(in_channels, out_channels, **block_args)
            )
            self.pools.append(nn.MaxPool2d(2))

            in_channels = out_channels

        # ==================================================
        # Bottleneck (deterministic)
        # ==================================================

        bottleneck_channels = channels * (2 ** depth)

        self.bottleneck = DoubleConv(
            in_channels, bottleneck_channels, **block_args
        )

        # ==================================================
        # Decoder: the last variational_decoder_levels blocks
        # are variational, the rest deterministic
        # ==================================================

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        current_channels = bottleneck_channels

        for index, level in enumerate(reversed(range(depth))):

            skip_channels = channels * (2 ** level)

            self.upconvs.append(
                nn.ConvTranspose2d(
                    current_channels,
                    skip_channels,
                    kernel_size=2,
                    stride=2,
                )
            )

            is_variational = index >= depth - variational_decoder_levels

            if is_variational:
                decoder = VariationalDoubleConv(
                    skip_channels * 2,
                    skip_channels,
                    **block_args,
                    **variational_args,
                )
            else:
                decoder = DoubleConv(
                    skip_channels * 2,
                    skip_channels,
                    **block_args,
                )

            self.decoders.append(decoder)

            current_channels = skip_channels

        # ==================================================
        # Output (variational)
        # ==================================================

        self.output_layer = VariationalConv2d(
            channels,
            1,
            kernel_size=1,
            **variational_args,
        )

    def forward(self, x, sample=True):

        skip_connections = []

        for encoder, pool in zip(self.encoders, self.pools):

            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        total_kl = 0.0

        for upconv, decoder, skip in zip(
            self.upconvs,
            self.decoders,
            skip_connections,
        ):

            x = upconv(x)

            # Handle image sizes not divisible by powers of 2
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x = torch.cat([skip, x], dim=1)

            if isinstance(decoder, VariationalDoubleConv):
                x, kl = decoder(x, sample=sample)
                total_kl = total_kl + kl
            else:
                x = decoder(x)

        prediction, kl = self.output_layer(x, sample=sample)
        total_kl = total_kl + kl

        return prediction, total_kl

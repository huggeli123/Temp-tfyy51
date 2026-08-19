import torch

from torch import nn
import torch.nn.functional as F

from .layers import (
    get_activation,
    get_normalization,
)


class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation="relu",
        normalization="none",
        num_groups=4,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd, for example 3, 5, or 7"
            )

        padding = kernel_size // 2

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),

            get_normalization(
                normalization,
                out_channels,
                num_groups,
            ),

            get_activation(activation),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),

            get_normalization(
                normalization,
                out_channels,
                num_groups,
            ),

            get_activation(activation),
        )

    def forward(self, x):

        return self.block(x)


class UNet(nn.Module):
    """
    U-Net for image denoising.

    Parameters
    ----------
    channels : int
        Number of channels at the first encoder level.

    depth : int
        Number of encoder/downsampling levels.
    """

    def __init__(
        self,
        channels=32,
        depth=3,
        kernel_size=3,
        activation="relu",
        normalization="none",
        num_groups=4,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError(
                "depth must be at least 1"
            )

        self.depth = depth

        # ==================================================
        # Encoder
        # ==================================================

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_channels = 1

        for level in range(depth):

            out_channels = channels * (2 ** level)

            self.encoders.append(
                DoubleConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                    normalization=normalization,
                    num_groups=num_groups,
                )
            )

            self.pools.append(
                nn.MaxPool2d(2)
            )

            in_channels = out_channels

        # ==================================================
        # Bottleneck
        # ==================================================

        # After the final encoder:
        # depth=3, channels=32
        #
        # encoder outputs:
        # 32 -> 64 -> 128
        #
        # therefore bottleneck:
        # 128 -> 256

        bottleneck_channels = (
            channels * (2 ** depth)
        )

        self.bottleneck = DoubleConv(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_size,
            activation=activation,
            normalization=normalization,
            num_groups=num_groups,
        )

        # ==================================================
        # Decoder
        # ==================================================

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        current_channels = bottleneck_channels

        for level in reversed(range(depth)):

            skip_channels = (
                channels * (2 ** level)
            )

            # Example first decoder step:
            # 256 -> 128
            self.upconvs.append(
                nn.ConvTranspose2d(
                    in_channels=current_channels,
                    out_channels=skip_channels,
                    kernel_size=2,
                    stride=2,
                )
            )

            # Concatenation gives:
            # 128 decoder channels
            # +
            # 128 skip channels
            # =
            # 256 channels
            self.decoders.append(
                DoubleConv(
                    in_channels=skip_channels * 2,
                    out_channels=skip_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                    normalization=normalization,
                    num_groups=num_groups,
                )
            )

            current_channels = skip_channels

        # ==================================================
        # Output
        # ==================================================

        self.output_layer = nn.Conv2d(
            in_channels=channels,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x):

        skip_connections = []

        # ==================================================
        # Encoder
        # ==================================================

        for encoder, pool in zip(
            self.encoders,
            self.pools,
        ):

            x = encoder(x)

            skip_connections.append(x)

            x = pool(x)

        # ==================================================
        # Bottleneck
        # ==================================================

        x = self.bottleneck(x)

        # ==================================================
        # Decoder
        # ==================================================

        skip_connections = skip_connections[::-1]

        for upconv, decoder, skip in zip(
            self.upconvs,
            self.decoders,
            skip_connections,
        ):

            x = upconv(x)

            # Handle image sizes that are not perfectly
            # divisible by powers of 2.
            if x.shape[-2:] != skip.shape[-2:]:

                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x = torch.cat(
                [skip, x],
                dim=1,
            )

            x = decoder(x)

        # ==================================================
        # Output
        # ==================================================

        return self.output_layer(x)
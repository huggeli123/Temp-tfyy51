from torch import nn

from .layers import (
    get_activation,
    get_normalization,
)
from .unet import UNet


class ResidualDoubleConv(nn.Module):
    """
    Two convolutions with a shortcut around them:

        output = activation(convs(x) + shortcut(x))

    The shortcut is the identity when the channel count is unchanged,
    otherwise a 1x1 convolution that matches the channels.
    """

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
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()

        else:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

        self.activation = get_activation(activation)

    def forward(self, x):

        return self.activation(
            self.block(x) + self.shortcut(x)
        )


class ResidualUNet(UNet):
    """
    U-Net with residual blocks at every level that predicts a
    correction to the noisy input, as ResidualCNN does:

        output = x + unet(x)

    Takes the same parameters as UNet.
    """

    block_class = ResidualDoubleConv

    def forward(self, x):

        correction = super().forward(x)

        return x + correction

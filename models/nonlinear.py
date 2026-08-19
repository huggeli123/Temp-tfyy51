from torch import nn

from .layers import (
    get_activation,
    get_normalization,
)


class NonLinearModel(nn.Module):

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

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd, for example 3, 5, or 7"
            )

        padding = kernel_size // 2

        self.network = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
            ),

            get_normalization(
                normalization,
                channels,
                num_groups,
            ),

            get_activation(
                activation
            ),

            nn.Conv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )

    def forward(self, x):

        return self.network(x)
from torch import nn

from .layers import (
    get_activation,
    get_normalization,
)


class ResidualCNN(nn.Module):

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

        if depth < 2:
            raise ValueError("depth must be at least 2")

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd, for example 3, 5, or 7"
            )

        padding = kernel_size // 2

        layers = []

        # First layer
        layers.append(
            nn.Conv2d(
                in_channels=1,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        )

        layers.append(
            get_normalization(
                normalization,
                channels,
                num_groups,
            )
        )

        layers.append(
            get_activation(activation)
        )

        # Hidden layers
        for _ in range(depth - 2):

            layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

            layers.append(
                get_normalization(
                    normalization,
                    channels,
                    num_groups,
                )
            )

            layers.append(
                get_activation(activation)
            )

        # Predict residual/correction
        layers.append(
            nn.Conv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding,
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        correction = self.network(x)

        return x + correction
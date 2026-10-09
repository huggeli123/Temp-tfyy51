from torch import nn

from .layers import (
    get_activation,
    get_normalization,
)
from .variational import VariationalConv2d


class VariationalSimpleCNN(nn.Module):
    """
    SimpleCNN where only the last (output) convolution is variational.

    forward() returns (prediction, kl_divergence).
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

        if depth < 2:
            raise ValueError("depth must be at least 2")

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd, for example 3, 5, or 7"
            )

        padding = kernel_size // 2

        layers = []

        # First layer (deterministic)
        layers.append(
            nn.Conv2d(
                in_channels=1,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        )
        layers.append(get_normalization(normalization, channels, num_groups))
        layers.append(get_activation(activation))

        # Hidden layers (deterministic)
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
                get_normalization(normalization, channels, num_groups)
            )
            layers.append(get_activation(activation))

        self.features = nn.Sequential(*layers)

        # Output layer (variational)
        self.output_layer = VariationalConv2d(
            in_channels=channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            prior_standard_deviation=prior_standard_deviation,
            initial_rho=initial_rho,
        )

    def forward(self, x, sample=True):

        hidden = self.features(x)

        prediction, kl_divergence = self.output_layer(
            hidden,
            sample=sample,
        )

        return prediction, kl_divergence

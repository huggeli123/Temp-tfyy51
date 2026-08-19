from torch import nn


class LinearModel(nn.Module):

    def __init__(
        self,
        channels=32,
        depth=3,
        kernel_size=3,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd, for example 3, 5, or 7"
            )

        padding = kernel_size // 2

        self.layer = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x):

        return self.layer(x)
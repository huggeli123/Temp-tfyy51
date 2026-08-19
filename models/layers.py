from torch import nn


def get_activation(name):
    """
    Create an activation function.
    """

    name = name.lower()

    if name == "relu":
        return nn.ReLU()

    elif name == "leaky_relu":
        return nn.LeakyReLU(
            negative_slope=0.1
        )

    elif name == "gelu":
        return nn.GELU()

    elif name == "elu":
        return nn.ELU()

    elif name == "silu":
        return nn.SiLU()

    elif name == "tanh":
        return nn.Tanh()

    else:
        raise ValueError(
            f"Unknown activation: {name}"
        )


def get_normalization(
    name,
    num_channels,
    num_groups=4,
):
    """
    Create a normalization layer.
    """

    name = name.lower()

    if name == "none":
        return nn.Identity()

    elif name == "batch":
        return nn.BatchNorm2d(
            num_channels
        )

    elif name == "instance":
        return nn.InstanceNorm2d(
            num_channels,
            affine=True,
        )

    elif name == "group":

        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be "
                f"divisible by num_groups ({num_groups})"
            )

        return nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
        )

    else:
        raise ValueError(
            f"Unknown normalization: {name}"
        )
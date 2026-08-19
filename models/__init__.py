from .linear import LinearModel
from .nonlinear import NonLinearModel
from .simple_cnn import SimpleCNN
from .residual_cnn import ResidualCNN
from .unet import UNet


def create_model(config):

    model_name = config["name"].lower()

    channels = config["channels"]
    depth = config["depth"]
    kernel_size = config.get(
        "kernel_size",
        3,
    )

    # ------------------------------------------------------
    # Linear baseline
    # ------------------------------------------------------

    if model_name == "linear":

        return LinearModel(
            channels=channels,
            depth=depth,
            kernel_size=kernel_size
        )

    # ------------------------------------------------------
    # Models using nonlinearities
    # ------------------------------------------------------

    activation = config.get(
        "activation",
        "relu",
    )

    normalization = config.get(
        "normalization",
        "none",
    )

    num_groups = config.get(
        "num_groups",
        4,
    )


    model_args = {
        "channels": channels,
        "depth": depth,
        "activation": activation,
        "normalization": normalization,
        "num_groups": num_groups,
        "kernel_size": kernel_size
    }

    if model_name == "nonlinear":

        return NonLinearModel(
            **model_args
        )

    elif model_name == "simple_cnn":

        return SimpleCNN(
            **model_args
        )

    elif model_name == "residual_cnn":

        return ResidualCNN(
            **model_args
        )

    elif model_name == "unet":

        return UNet(
            **model_args
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )
import torch

# ============================================================
# Loss
# ============================================================

def get_loss(loss_name):
    """
    Create the loss function specified in the config.
    """

    loss_name = loss_name.lower()

    if loss_name == "mse":
        return torch.nn.MSELoss()

    elif loss_name == "l1":
        return torch.nn.L1Loss()

    elif loss_name == "smooth_l1":
        return torch.nn.SmoothL1Loss()

    else:
        raise ValueError(
            f"Unknown loss function: {loss_name}"
        )
    

# ============================================================
# Optimizer
# ============================================================

def get_optimizer(
    optimizer_name,
    model,
    learning_rate,
    weight_decay,
):
    """
    Create the optimizer specified in the config.
    """

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "adamw":

        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "sgd":

        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )
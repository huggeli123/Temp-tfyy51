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

def get_parameter_groups(
    model,
    learning_rate,
    weight_decay,
    rho_learning_rate=None,
):
    """
    Optimizer parameter groups.

    By default all parameters share one group. If rho_learning_rate is
    given, the variational std parameters (names ending in "_rho") get
    their own learning rate and no weight decay: rho only moves a few
    units over a short run at the shared learning rate, so the
    posterior std would otherwise stay at its initial value.
    """

    if rho_learning_rate is None:
        return model.parameters()

    rho_parameters = []
    other_parameters = []

    for name, parameter in model.named_parameters():
        if name.endswith("_rho"):
            rho_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    groups = [
        {
            "params": other_parameters,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
    ]

    if rho_parameters:
        groups.append(
            {
                "params": rho_parameters,
                "lr": rho_learning_rate,
                "weight_decay": 0.0,
            }
        )

    return groups


def get_optimizer(
    optimizer_name,
    model,
    learning_rate,
    weight_decay,
    rho_learning_rate=None,
):
    """
    Create the optimizer specified in the config.

    rho_learning_rate : float or None
        Separate learning rate for variational *_rho parameters
        (None = same as learning_rate).
    """

    optimizer_name = optimizer_name.lower()

    parameters = get_parameter_groups(
        model,
        learning_rate,
        weight_decay,
        rho_learning_rate,
    )

    if optimizer_name == "adam":

        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "adamw":

        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "sgd":

        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )

# ===========================================================
# Scheduler
# ===========================================================

def get_scheduler(scheduler_name, optimizer, epochs):
    if (scheduler_name == "none"):
        return None
    elif (scheduler_name == "cosine"):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            )
    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}"
        )
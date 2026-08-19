import torch

def calculate_psnr(predictions, targets):
    """
    Calculate the average PSNR for a batch of images.

    Assumes pixel values are in [0, 1].
    """

    # MSE for each image separately
    mse = torch.mean(
        (predictions - targets) ** 2,
        dim=(1, 2, 3),
    )

    # Avoid division by zero
    mse = torch.clamp(
        mse,
        min=1e-10,
    )

    psnr = 10 * torch.log10(
        1.0 / mse
    )

    return psnr.mean().item()
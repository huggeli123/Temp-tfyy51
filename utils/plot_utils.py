import inspect
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path

def plot_training_history(
    history,
    save_path=None,
):
    """
    Plot training and validation loss.
    """

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def save_denoising_examples(
    model,
    dataloader,
    device,
    save_path,
    num_images=4,
):
    """
    Save example denoising results.

    Each row contains:
        noisy input | model prediction | clean target
    """

    model.eval()

    # Get one batch
    noisy_images, clean_images = next(iter(dataloader))

    noisy_images = noisy_images.to(device)
    clean_images = clean_images.to(device)

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        if "sample" in inspect.signature(model.forward).parameters:
            # Variational model: show the posterior-mean network
            # (deterministic) and drop the returned KL term
            predictions = model(noisy_images, sample=False)[0]

        else:
            predictions = model(noisy_images)

    # Move everything back to CPU for plotting
    noisy_images = noisy_images.cpu()
    clean_images = clean_images.cpu()
    predictions = predictions.cpu()

    # Only show as many images as are available
    num_images = min(
        num_images,
        noisy_images.shape[0],
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        num_images,
        3,
        figsize=(10, 3 * num_images),
    )

    # matplotlib behaves differently when there is only one row
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_images):

        noisy = noisy_images[i, 0].numpy()
        prediction = predictions[i, 0].numpy()
        clean = clean_images[i, 0].numpy()

        # Clamp only for visualization
        prediction = np.clip(
            prediction,
            0.0,
            1.0,
        )

        # Noisy input
        axes[i, 0].imshow(
            noisy,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[i, 0].set_title("Noisy input")
        axes[i, 0].axis("off")

        # Model prediction
        axes[i, 1].imshow(
            prediction,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[i, 1].set_title("Model prediction")
        axes[i, 1].axis("off")

        # Clean target
        axes[i, 2].imshow(
            clean,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[i, 2].set_title("Clean target")
        axes[i, 2].axis("off")

    plt.tight_layout()

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()



def plot_psnr_history(
    history,
    save_path=None,
):
    """
    Plot training and validation PSNR.
    """

    epochs = range(
        1,
        len(history["train_psnr"]) + 1,
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        epochs,
        history["train_psnr"],
        label="Training PSNR",
    )

    plt.plot(
        epochs,
        history["val_psnr"],
        label="Validation PSNR",
    )

    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Training and validation PSNR")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_noisy_clean_examples(
    noisy_images,
    clean_images,
    noise_std,
    num_images=2,
):
    """
    Plot noisy inputs and their corresponding clean targets.

    Parameters
    ----------
    noisy_images : torch.Tensor
        Batch of noisy images with shape [B, C, H, W].

    clean_images : torch.Tensor
        Batch of clean images with shape [B, C, H, W].

    noise_std : float
        Standard deviation of the Gaussian noise.

    num_images : int
        Number of image pairs to display.
    """

    num_images = min(
        num_images,
        noisy_images.shape[0],
    )

    fig, axes = plt.subplots(
        num_images,
        2,
        figsize=(8, 3 * num_images),
    )

    # Needed when num_images = 1
    if num_images == 1:
        axes = axes[None, :]

    for i in range(num_images):

        noisy = noisy_images[i, 0].cpu().numpy()
        clean = clean_images[i, 0].cpu().numpy()

        # Noisy image
        axes[i, 0].imshow(
            noisy,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[i, 0].set_title("Noisy input")
        axes[i, 0].axis("off")

        # Clean image
        axes[i, 1].imshow(
            clean,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[i, 1].set_title("Clean target")
        axes[i, 1].axis("off")

    fig.suptitle(
        f"Gaussian noise level: σ = {noise_std:.2f}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()


def plot_augmentation_examples(
    dataset,
    index=0,
    num_examples=4,
):
    """
    Visualize different random augmentations of the same image.
    """

    if not dataset.augmentations_enabled:

        print(
            "Augmentation is disabled in the current configuration."
        )

        return

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original = dataset.get_clean_image(
        index,
        apply_augmentation=False,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        num_examples + 1,
        figsize=(3 * (num_examples + 1), 3),
    )

    # Original image
    axes[0].imshow(
        original[0].cpu().numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title("Original")
    axes[0].axis("off")

    # Different random augmentations
    for i in range(num_examples):

        augmented = dataset.get_clean_image(
            index,
            apply_augmentation=True,
        )

        axes[i + 1].imshow(
            augmented[0].cpu().numpy(),
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[i + 1].set_title(
            f"Augmentation {i + 1}"
        )

        axes[i + 1].axis("off")

    fig.suptitle(
        "Training data augmentations",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()



def plot_uncertainty_examples(
    noisy_images,
    clean_images,
    predictive_mean,
    epistemic_variance,
    save_path=None,
    num_images=4,
):
    """
    Plot predictive mean and epistemic uncertainty.

    Each row contains:
        noisy input | predictive mean | clean target |
        absolute error | epistemic std

    The error and std maps are shown on a shared colour scale per
    row (clipped at the 99.5th percentile), so they can be compared
    directly. The std only reflects
    variation between posterior weight samples; it is not
    calibrated.
    """

    noisy_images = noisy_images.detach().cpu()
    clean_images = clean_images.detach().cpu()
    predictive_mean = predictive_mean.detach().cpu()
    epistemic_std = epistemic_variance.detach().cpu().sqrt()

    num_images = min(
        num_images,
        noisy_images.shape[0],
    )

    fig, axes = plt.subplots(
        num_images,
        5,
        figsize=(17, 3.2 * num_images),
    )

    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_images):

        noisy = noisy_images[i, 0].numpy()
        clean = clean_images[i, 0].numpy()
        mean = np.clip(predictive_mean[i, 0].numpy(), 0.0, 1.0)
        error = np.abs(mean - clean)
        std = epistemic_std[i, 0].numpy()

        # Shared scale for the error and std maps of this row.
        # A high percentile is used so a few outlier pixels do not
        # make the rest of the map look black.
        map_max = max(
            np.percentile(np.concatenate([error.ravel(), std.ravel()]), 99.5),
            1e-8,
        )

        panels = [
            (noisy, "Noisy input", "gray", 1.0),
            (mean, "Predictive mean", "gray", 1.0),
            (clean, "Clean target", "gray", 1.0),
            (error, "|mean - clean|", "magma", map_max),
            (std, "Epistemic std", "magma", map_max),
        ]

        for j, (image, title, cmap, vmax) in enumerate(panels):

            handle = axes[i, j].imshow(
                image,
                cmap=cmap,
                vmin=0,
                vmax=vmax,
            )

            axes[i, j].set_title(title)
            axes[i, j].axis("off")

            if cmap == "magma" and j == 4:
                fig.colorbar(
                    handle,
                    ax=axes[i, 3:5],
                    fraction=0.025,
                    pad=0.02,
                )

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()

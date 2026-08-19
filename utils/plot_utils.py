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
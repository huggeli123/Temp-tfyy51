from pathlib import Path
import torch
from tqdm.auto import tqdm
from utils.metrics import calculate_psnr










# ============================================================
# One training epoch
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    description="Training",
):
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_psnr = 0.0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=description,
        leave=False,
    )

    for noisy_images, clean_images in progress_bar:

        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(noisy_images)

        # Loss
        loss = criterion(
            predictions,
            clean_images,
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        batch_size = noisy_images.size(0)

        psnr = calculate_psnr(
            predictions.detach(),
            clean_images,
        )

        total_loss += loss.item() * batch_size
        total_psnr += psnr * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.6f}",
            psnr=f"{psnr:.2f} dB",
        )

    average_loss = total_loss / total_samples
    average_psnr = total_psnr / total_samples

    return average_loss, average_psnr


# ============================================================
# Validation / evaluation
# ============================================================

def evaluate(
    model,
    dataloader,
    criterion,
    device,
    description="Evaluation",
):
    """
    Evaluate the model without updating its parameters.
    """

    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=description,
        leave=False,
    )

    with torch.no_grad():

        for noisy_images, clean_images in progress_bar:

            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            predictions = model(noisy_images)

            loss = criterion(
                predictions,
                clean_images,
            )

            batch_size = noisy_images.size(0)

            psnr = calculate_psnr(
                predictions,
                clean_images,
            )

            total_loss += loss.item() * batch_size
            total_psnr += psnr * batch_size
            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{loss.item():.6f}",
                psnr=f"{psnr:.2f} dB",
            )

    average_loss = total_loss / total_samples
    average_psnr = total_psnr / total_samples

    return average_loss, average_psnr

# ============================================================
# Full training loop
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    output_dir="outputs",
    save_best_model=True,
):
    """
    Train the model and evaluate on the validation set
    after every epoch.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_losses = []

    train_psnrs = []
    val_psnrs = []

    best_val_loss = float("inf")

    # Overall epoch progress bar
    epoch_bar = tqdm(
        range(epochs),
        desc="Training",
    )

    for epoch in epoch_bar:

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss, train_psnr = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            description=f"Epoch {epoch + 1}/{epochs} - Train",
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_psnr = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            description=f"Epoch {epoch + 1}/{epochs} - Val",
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_psnrs.append(train_psnr)
        val_psnrs.append(val_psnr)

        # ----------------------------------------------------
        # Update overall progress bar
        # ----------------------------------------------------

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.6f}",
            val_loss=f"{val_loss:.6f}",
            val_psnr=f"{val_psnr:.2f} dB",
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if save_best_model and val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pt",
            )

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_psnr": train_psnrs,
        "val_psnr": val_psnrs,
    }

    return history













import time
from pathlib import Path
import torch
from tqdm.auto import tqdm
from utils.metrics import calculate_psnr


def _timestamp(device):
    """
    Wall-clock time in seconds. Waits for queued CUDA work first so
    that GPU epochs are not under-measured.
    """

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)

    return time.perf_counter()









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
            predictions = torch.clamp(
                predictions,
                0.0,
                1.0,
            )

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
    scheduler=None
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
    epoch_times = []
    learning_rates = []

    best_val_loss = float("inf")

    # Overall epoch progress bar
    epoch_bar = tqdm(
        range(epochs),
        desc="Training",
    )

    for epoch in epoch_bar:

        epoch_start = _timestamp(device)

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

        epoch_times.append(_timestamp(device) - epoch_start)
        learning_rates.append(optimizer.param_groups[0]["lr"])
        if (scheduler != None):
            scheduler.step()

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
        "learning_rates": learning_rates,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_psnr": train_psnrs,
        "val_psnr": val_psnrs,
        "epoch_time": epoch_times,
    }

    return history


# ============================================================
# Variational models (Bayes by backprop)
#
# The model's forward(x, sample=...) returns (prediction, kl).
# Objective per minibatch, with the reconstruction loss averaged
# over the batch:
#
#     loss = reconstruction + kl_weight * kl / dataset_size
# ============================================================

def train_one_epoch_variational(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    dataset_size,
    kl_weight=1.0,
    description="Training",
):
    """
    Train a variational model for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
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

        optimizer.zero_grad()

        # Stochastic forward pass: weights are sampled
        predictions, kl = model(noisy_images, sample=True)

        reconstruction = criterion(
            predictions,
            clean_images,
        )

        scaled_kl = kl / dataset_size

        loss = reconstruction + kl_weight * scaled_kl

        loss.backward()
        optimizer.step()

        batch_size = noisy_images.size(0)

        psnr = calculate_psnr(
            predictions.detach(),
            clean_images,
        )

        total_loss += loss.item() * batch_size
        total_reconstruction += reconstruction.item() * batch_size
        total_kl += scaled_kl.item() * batch_size
        total_psnr += psnr * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.6f}",
            recon=f"{reconstruction.item():.6f}",
            kl=f"{scaled_kl.item():.6f}",
            psnr=f"{psnr:.2f} dB",
        )

    return {
        "loss": total_loss / total_samples,
        "reconstruction": total_reconstruction / total_samples,
        "kl": total_kl / total_samples,
        "psnr": total_psnr / total_samples,
    }


def evaluate_variational(
    model,
    dataloader,
    criterion,
    device,
    dataset_size,
    kl_weight=1.0,
    sample=False,
    description="Evaluation",
):
    """
    Evaluate a variational model.

    sample=False uses the posterior means (deterministic);
    sample=True draws one weight sample per batch.
    """

    model.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
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

            predictions, kl = model(noisy_images, sample=sample)
            predictions = torch.clamp(
                predictions,
                0.0,
                1.0,
            )

            reconstruction = criterion(
                predictions,
                clean_images,
            )

            scaled_kl = kl / dataset_size

            loss = reconstruction + kl_weight * scaled_kl

            batch_size = noisy_images.size(0)

            psnr = calculate_psnr(
                predictions,
                clean_images,
            )

            total_loss += loss.item() * batch_size
            total_reconstruction += reconstruction.item() * batch_size
            total_kl += scaled_kl.item() * batch_size
            total_psnr += psnr * batch_size
            total_samples += batch_size

            progress_bar.set_postfix(
                recon=f"{reconstruction.item():.6f}",
                psnr=f"{psnr:.2f} dB",
            )

    return {
        "loss": total_loss / total_samples,
        "reconstruction": total_reconstruction / total_samples,
        "kl": total_kl / total_samples,
        "psnr": total_psnr / total_samples,
    }


def train_model_variational(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    kl_weight=1.0,
    output_dir="outputs",
    save_best_model=True,
    scheduler = None,
):
    """
    Variational counterpart of train_model.

    The history uses the same keys as train_model (train_loss,
    val_loss, train_psnr, val_psnr) so save_history and the plot
    helpers keep working. train_loss and val_loss are both the
    reconstruction loss only, so the curves are comparable and
    best-model selection is not affected by the KL term. val_loss
    uses the posterior means.
    Extra keys: train_total_loss (reconstruction + weighted KL, the
    quantity actually minimised) and train_kl (KL / dataset_size).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_size = len(train_loader.dataset)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_psnr": [],
        "val_psnr": [],
        "train_total_loss": [],
        "train_kl": [],
        "epoch_time": [],
        "learning_rates": [],
    }

    best_val_loss = float("inf")

    epoch_bar = tqdm(
        range(epochs),
        desc="Training",
    )

    for epoch in epoch_bar:

        epoch_start = _timestamp(device)

        train_metrics = train_one_epoch_variational(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            dataset_size=dataset_size,
            kl_weight=kl_weight,
            description=f"Epoch {epoch + 1}/{epochs} - Train",
        )

        val_metrics = evaluate_variational(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            dataset_size=dataset_size,
            kl_weight=kl_weight,
            sample=False,
            description=f"Epoch {epoch + 1}/{epochs} - Val",
        )

        val_loss = val_metrics["reconstruction"]

        history["epoch_time"].append(_timestamp(device) - epoch_start)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        if scheduler != None:
            scheduler.step()
        history["train_loss"].append(train_metrics["reconstruction"])
        history["train_total_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_loss)
        history["train_psnr"].append(train_metrics["psnr"])
        history["val_psnr"].append(val_metrics["psnr"])
        history["train_kl"].append(train_metrics["kl"])

        epoch_bar.set_postfix(
            train_recon=f"{train_metrics['reconstruction']:.6f}",
            kl=f"{train_metrics['kl']:.6f}",
            val_loss=f"{val_loss:.6f}",
            val_psnr=f"{val_metrics['psnr']:.2f} dB",
        )

        if save_best_model and val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pt",
            )

    return history


@torch.no_grad()
def predict_with_uncertainty(
    model,
    noisy_images,
    number_of_samples=20,
):
    """
    Predictive mean and epistemic variance from repeated stochastic
    forward passes.

    The variance only reflects variation between posterior weight
    samples; it is not calibrated and is not aleatoric noise.
    """

    was_training = model.training
    model.eval()

    predictions = torch.stack([
        model(noisy_images, sample=True)[0]
        for _ in range(number_of_samples)
    ])

    model.train(was_training)

    predictive_mean = predictions.mean(dim=0)
    epistemic_variance = predictions.var(dim=0, unbiased=False)

    return predictive_mean, epistemic_variance













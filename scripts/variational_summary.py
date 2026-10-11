"""
Summarise a trained variational run.

    python scripts/variational_summary.py results/00_baseline/<run>

The run folder must contain config.yaml and best_model.pt, as written by
the notebook. The script writes variational_summary.json into the same
folder and prints it.

It reports what results.json does not contain:

    - the learned posterior standard deviations (sigma)
    - the test loss with sampled weights, next to the posterior-mean one
    - the uncertainty numbers for the first validation batch
    - the behaviour at other noise levels

The numbers in docs/variational/ come from these files.
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models import create_model
from models.variational import positive_scale
from train import evaluate_variational, predict_with_uncertainty
from utils.dataloader import create_dataloaders
from utils.metrics import calculate_psnr
from utils.misc import load_config, set_seed
from utils.model_configs import get_loss


# ============================================================
# Posterior standard deviations
# ============================================================

def posterior_std_summary(model):
    """
    Distribution of sigma = softplus(rho) over all variational weights.
    """

    sigma = torch.cat([
        positive_scale(parameter.detach()).flatten()
        for name, parameter in model.named_parameters()
        if name.endswith("_rho")
    ])

    quantiles = torch.quantile(
        sigma,
        torch.tensor([0.05, 0.5, 0.95], device=sigma.device),
    )

    return {
        "count": sigma.numel(),
        "min": sigma.min().item(),
        "p05": quantiles[0].item(),
        "median": quantiles[1].item(),
        "p95": quantiles[2].item(),
        "max": sigma.max().item(),
        "mean": sigma.mean().item(),
    }


# ============================================================
# Test set
# ============================================================

def test_metrics(
    model,
    test_loader,
    criterion,
    device,
    dataset_size,
    kl_weight,
    sampled_passes,
):
    """
    Reconstruction loss and PSNR with the posterior-mean weights, and
    averaged over several passes with sampled weights.
    """

    mean_weights = evaluate_variational(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        dataset_size=dataset_size,
        kl_weight=kl_weight,
        sample=False,
        description="Test (mean weights)",
    )

    sampled = [
        evaluate_variational(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            dataset_size=dataset_size,
            kl_weight=kl_weight,
            sample=True,
            description="Test (sampled weights)",
        )
        for _ in range(sampled_passes)
    ]

    return (
        {
            "loss": mean_weights["reconstruction"],
            "psnr": mean_weights["psnr"],
        },
        {
            "passes": sampled_passes,
            "loss": sum(m["reconstruction"] for m in sampled) / sampled_passes,
            "psnr": sum(m["psnr"] for m in sampled) / sampled_passes,
        },
    )


# ============================================================
# Uncertainty on one validation batch (same as the notebook cell)
# ============================================================

def uncertainty_summary(
    model,
    val_loader,
    device,
    number_of_samples,
):

    noisy_images, clean_images = next(iter(val_loader))
    noisy_images = noisy_images.to(device)
    clean_images = clean_images.to(device)

    predictive_mean, epistemic_variance = predict_with_uncertainty(
        model,
        noisy_images,
        number_of_samples=number_of_samples,
    )

    error = (predictive_mean - clean_images).abs()
    epistemic_std = epistemic_variance.sqrt()

    correlation = torch.corrcoef(
        torch.stack([epistemic_std.flatten(), error.flatten()])
    )[0, 1].item()

    return {
        "samples": number_of_samples,
        "images": noisy_images.shape[0],
        "mean_epistemic_std": epistemic_std.mean().item(),
        "mean_absolute_error": error.mean().item(),
        "psnr_predictive_mean": calculate_psnr(predictive_mean, clean_images),
        "correlation_std_error": correlation,
    }


# ============================================================
# Other noise levels (same as the notebook cell)
# ============================================================

def noise_level_summary(
    model,
    val_loader,
    device,
    train_std,
    number_of_samples,
):

    noise_levels = sorted({0.0, 0.1, train_std, 0.5})

    clean_images = next(iter(val_loader))[1].to(device)

    rows = []

    for noise_std in noise_levels:

        noisy = torch.clamp(
            clean_images + noise_std * torch.randn_like(clean_images),
            0.0,
            1.0,
        )

        mean_prediction, variance = predict_with_uncertainty(
            model,
            noisy,
            number_of_samples=number_of_samples,
        )

        rows.append({
            "noise_std": noise_std,
            "is_training_level": noise_std == train_std,
            "input_psnr": calculate_psnr(noisy, clean_images),
            "predictive_mean_psnr": calculate_psnr(mean_prediction, clean_images),
            "mean_epistemic_std": variance.sqrt().mean().item(),
        })

    return rows


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", help="folder with config.yaml and best_model.pt")
    parser.add_argument("--samples", type=int, default=30,
                        help="stochastic forward passes for the uncertainty numbers")
    parser.add_argument("--sampled-passes", type=int, default=5,
                        help="passes over the test set with sampled weights")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()

    # The data root in config.yaml is relative to the repository root
    os.chdir(PROJECT_ROOT)

    config = load_config(run_dir / "config.yaml")

    if not config["model"]["name"].startswith("variational"):
        raise SystemExit(
            f"{config['model']['name']} is not a variational model."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = create_dataloaders(config)

    model = create_model(config["model"]).to(device)
    model.load_state_dict(
        torch.load(
            run_dir / "best_model.pt",
            map_location=device,
            weights_only=True,
        )
    )

    criterion = get_loss(config["training"]["loss"])

    # All sampling below is seeded, so the file can be regenerated
    set_seed(config["seed"])

    test_mean, test_sampled = test_metrics(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        dataset_size=len(train_loader.dataset),
        kl_weight=config["training"]["kl_weight"],
        sampled_passes=args.sampled_passes,
    )

    summary = {
        "run": run_dir.name,
        "model": config["model"]["name"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else "cpu"
            ),
        },
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "variational_weights": sum(
                p.numel()
                for name, p in model.named_parameters()
                if name.endswith("_rho")
            ),
        },
        "posterior_std": posterior_std_summary(model),
        "test_mean_weights": test_mean,
        "test_sampled_weights": test_sampled,
        "uncertainty": uncertainty_summary(
            model=model,
            val_loader=val_loader,
            device=device,
            number_of_samples=args.samples,
        ),
        "noise_levels": noise_level_summary(
            model=model,
            val_loader=val_loader,
            device=device,
            train_std=config["noise"]["train_val_std"],
            number_of_samples=args.samples,
        ),
    }

    save_path = run_dir / "variational_summary.json"

    with open(save_path, "w") as file:
        json.dump(summary, file, indent=4)

    print(json.dumps(summary, indent=4))
    print("Saved to:", save_path)


if __name__ == "__main__":
    main()

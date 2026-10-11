"""
Measure how much weight noise a trained plain UNet tolerates.

    python scripts/weight_noise_sensitivity.py results/00_baseline/<unet run>

The run folder must contain config.yaml and best_model.pt of a `unet` run.
The trained weights are copied into a VariationalPartialUNet as its
posterior means. Every sigma in the variational layers is then set to one
fixed value, and the validation loss is measured with sampled weights.

This shows which sigma range is usable before any variational model is
trained. It is how initial_rho for variational_partial_unet was chosen
(docs/variational/06_hyperparameters.md).

The result is written to weight_noise_sensitivity.json in the run folder.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models import VariationalPartialUNet
from train import evaluate_variational, predict_with_uncertainty
from utils.dataloader import create_dataloaders
from utils.misc import load_config, set_seed
from utils.model_configs import get_loss


# ============================================================
# Plain UNet weights -> posterior means of the partial model
# ============================================================

# Position inside DoubleConv.block  ->  name inside VariationalDoubleConv
BLOCK_TO_VARIATIONAL = {
    "block.0.": "conv1.",
    "block.1.": "post1.0.",
    "block.3.": "conv2.",
    "block.4.": "post2.0.",
}


def load_unet_weights_as_means(model, unet_state_dict):
    """
    Copy a plain UNet state dict into a VariationalPartialUNet.

    Deterministic layers have identical names in both models. For the
    variational layers, `weight` / `bias` become `weight_mean` /
    `bias_mean`. The rho parameters are left untouched.
    """

    target = model.state_dict()
    converted = {}

    for name, tensor in unet_state_dict.items():

        if name in target:
            converted[name] = tensor
            continue

        for block_name, variational_name in BLOCK_TO_VARIATIONAL.items():
            name = name.replace(block_name, variational_name)

        if name not in target:
            # A convolution that is variational in the target model
            name = name + "_mean"

        if name not in target:
            raise KeyError(f"No counterpart for UNet parameter {name}")

        converted[name] = tensor

    missing = [
        name
        for name in target
        if name not in converted and not name.endswith("_rho")
    ]

    if missing:
        raise KeyError(f"Not filled from the UNet checkpoint: {missing}")

    model.load_state_dict({**target, **converted})


def set_uniform_sigma(model, sigma):
    """
    Give every variational weight the same posterior std.
    """

    # Inverse of softplus
    rho = math.log(math.expm1(sigma))

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("_rho"):
                parameter.fill_(rho)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", help="folder of a plain unet run")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2],
                        help="values of variational_decoder_levels to test")
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[3e-4, 1e-3, 3e-3, 1e-2])
    parser.add_argument("--draws", type=int, default=5,
                        help="passes over the validation set per sigma")
    parser.add_argument("--samples", type=int, default=30,
                        help="stochastic forward passes for the epistemic std")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()

    # The data root in config.yaml is relative to the repository root
    os.chdir(PROJECT_ROOT)

    config = load_config(run_dir / "config.yaml")

    if config["model"]["name"] != "unet":
        raise SystemExit("This script needs a run of the plain `unet` model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = create_dataloaders(config)

    criterion = get_loss(config["training"]["loss"])

    unet_state_dict = torch.load(
        run_dir / "best_model.pt",
        map_location=device,
        weights_only=True,
    )

    model_config = config["model"]

    noisy_images = next(iter(val_loader))[0].to(device)

    set_seed(config["seed"])

    result = {
        "source_run": run_dir.name,
        "draws": args.draws,
        "samples": args.samples,
        "rows": [],
    }

    for levels in args.levels:

        model = VariationalPartialUNet(
            channels=model_config["channels"],
            depth=model_config["depth"],
            kernel_size=model_config.get("kernel_size", 3),
            activation=model_config.get("activation", "relu"),
            normalization=model_config.get("normalization", "none"),
            num_groups=model_config.get("num_groups", 4),
            variational_decoder_levels=levels,
        ).to(device)

        load_unet_weights_as_means(model, unet_state_dict)

        variational_weights = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.endswith("_rho")
        )

        def validation_loss(sample):
            return evaluate_variational(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                dataset_size=len(train_loader.dataset),
                kl_weight=0.0,
                sample=sample,
                description=f"levels={levels}",
            )["reconstruction"]

        # Without noise this equals the plain UNet's validation loss
        baseline = validation_loss(sample=False)
        result["baseline_val_loss"] = baseline

        for sigma in args.sigmas:

            set_uniform_sigma(model, sigma)

            noisy_loss = sum(
                validation_loss(sample=True) for _ in range(args.draws)
            ) / args.draws

            _, variance = predict_with_uncertainty(
                model,
                noisy_images,
                number_of_samples=args.samples,
            )

            row = {
                "variational_decoder_levels": levels,
                "variational_weights": variational_weights,
                "sigma": sigma,
                "val_loss": noisy_loss,
                "relative_increase": noisy_loss / baseline - 1.0,
                "mean_epistemic_std": variance.sqrt().mean().item(),
            }

            result["rows"].append(row)

            print(
                f"levels={levels}  n={variational_weights:>7,}  "
                f"sigma={sigma:.0e}  "
                f"loss {row['relative_increase']:+.1%}  "
                f"std {row['mean_epistemic_std']:.4f}"
            )

    save_path = run_dir / "weight_noise_sensitivity.json"

    with open(save_path, "w") as file:
        json.dump(result, file, indent=4)

    print("Baseline validation loss:", result["baseline_val_loss"])
    print("Saved to:", save_path)


if __name__ == "__main__":
    main()

"""
Hyperparameter sweep for variational_unet.

    python scripts/sweep_variational_unet.py trials.json --stage stage1

trials.json is a list of {"name": ..., "set": {"training.kl_weight": 1e-8, ...}}.
Each trial starts from config.yaml with model.name = variational_unet, applies
its overrides, trains with train_model_variational and is saved to
results/variational_unet_sweep/<stage>/<name>/ (config.yaml, results.json).
A trial whose results.json already exists is skipped, so an interrupted
sweep can be restarted. The summary of every finished trial is appended to
results/variational_unet_sweep/all_trials.jsonl.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import create_model  # noqa: E402
from train import (  # noqa: E402
    evaluate, evaluate_variational, train_model, train_model_variational,
)
from utils.dataloader import create_dataloaders  # noqa: E402
from utils.experiment import save_history  # noqa: E402
from utils.misc import load_config, set_seed  # noqa: E402
from utils.model_configs import get_loss, get_optimizer, get_scheduler  # noqa: E402

SWEEP_ROOT = PROJECT_ROOT / "results" / "variational_unet_sweep"

# Settings every trial starts from, before its own overrides
BASE = {
    "model.name": "variational_unet",
    "model.initial_rho": -8.0,
    "model.prior_standard_deviation": 0.1,
    "training.kl_weight": 1e-8,
    "training.rho_learning_rate": 0.01,
}


def apply(config, settings):
    for key, value in settings.items():
        node = config
        *parents, last = key.split(".")
        for part in parents:
            node = node[part]
        node[last] = value


def run_trial(stage, trial, keep_checkpoint):

    run_dir = SWEEP_ROOT / stage / trial["name"]

    if (run_dir / "results.json").exists():
        print(f"skip {stage}/{trial['name']} (done)")
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(PROJECT_ROOT / "config.yaml")
    config["data"]["root"] = str(PROJECT_ROOT / config["data"]["root"])
    apply(config, BASE)
    apply(config, trial.get("set", {}))

    with open(run_dir / "config.yaml", "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    set_seed(config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = create_dataloaders(config)

    model = create_model(config["model"]).to(device)
    training = config["training"]

    criterion = get_loss(training["loss"])
    optimizer = get_optimizer(
        training["optimizer"],
        model,
        training["learning_rate"],
        training["weight_decay"],
        training.get("rho_learning_rate"),
    )
    scheduler = get_scheduler(
        training.get("scheduler", "none"), optimizer, training["epochs"]
    )

    start = time.perf_counter()

    # A non-variational model (e.g. unet) can be run as a reference;
    # it ignores the variational keys
    is_variational = config["model"]["name"].startswith("variational")

    common = dict(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=training["epochs"],
        output_dir=run_dir,
        save_best_model=True,
        scheduler=scheduler,
    )

    if is_variational:
        history = train_model_variational(**common, kl_weight=training["kl_weight"])
    else:
        history = train_model(**common)

    checkpoint = run_dir / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    if is_variational:
        test = evaluate_variational(
            model, test_loader, criterion, device,
            dataset_size=len(train_loader.dataset),
            kl_weight=training["kl_weight"], sample=False,
        )
        test = (test["reconstruction"], test["psnr"])
    else:
        test = evaluate(model, test_loader, criterion, device)

    save_history(history, run_dir, test_loss=test[0], test_psnr=test[1])

    if not keep_checkpoint:
        checkpoint.unlink()

    best_epoch = min(range(len(history["val_loss"])),
                     key=lambda i: history["val_loss"][i])

    summary = {
        "stage": stage,
        "name": trial["name"],
        "set": trial.get("set", {}),
        "epochs": training["epochs"],
        "best_epoch": best_epoch + 1,
        "val_psnr_at_best": history["val_psnr"][best_epoch],
        "max_val_psnr": max(history["val_psnr"]),
        "final_val_psnr": history["val_psnr"][-1],
        "best_val_loss": history["val_loss"][best_epoch],
        "test_psnr": test[1],
        "test_loss": test[0],
        "final_train_kl": history.get("train_kl", [0.0])[-1],
        "minutes": (time.perf_counter() - start) / 60,
    }

    with open(SWEEP_ROOT / "all_trials.jsonl", "a") as file:
        file.write(json.dumps(summary) + "\n")

    print(f"done {stage}/{trial['name']}: "
          f"val_psnr {summary['val_psnr_at_best']:.3f} dB "
          f"(max {summary['max_val_psnr']:.3f}), "
          f"test {summary['test_psnr']:.3f} dB, "
          f"{summary['minutes']:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trials")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()

    with open(args.trials) as file:
        trials = json.load(file)

    for trial in trials:
        try:
            run_trial(args.stage, trial, args.keep_checkpoints)
        except Exception as error:  # keep the sweep going
            print(f"FAILED {args.stage}/{trial['name']}: {error!r}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

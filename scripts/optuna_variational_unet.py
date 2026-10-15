"""
Bayesian hyperparameter search for variational_unet with Optuna.

    python scripts/optuna_variational_unet.py --trials 40 --epochs 15

Optuna's TPE sampler models which settings gave a high validation PSNR
and proposes new trials from the promising regions. The median pruner
stops a trial early when its best val PSNR so far is below the median of
earlier trials at the same epoch, so bad settings cost only a few epochs.

The study is stored in results/variational_unet_sweep/optuna/study.db, so
running the command again continues the same study. Every trial's config
and per-epoch history are written next to it in trial_<number>/.
Needs:  pip install optuna  (not part of requirements.txt)
"""

import argparse
import json
import sys
from pathlib import Path

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import TrialState
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import create_model  # noqa: E402
from train import evaluate_variational, train_one_epoch_variational  # noqa: E402
from utils.dataloader import create_dataloaders  # noqa: E402
from utils.misc import load_config, set_seed  # noqa: E402
from utils.model_configs import get_loss, get_optimizer, get_scheduler  # noqa: E402

STUDY_DIR = PROJECT_ROOT / "results" / "variational_unet_sweep" / "optuna"

# Configurations that did well in the one-at-a-time screen, evaluated
# first so the sampler starts from known good points
STARTING_POINTS = [
    {"learning_rate": 1e-3, "batch_size": 8, "initial_rho": -8.0,
     "prior_standard_deviation": 0.1, "kl_weight": 1e-8, "rho_learning_rate": 0.01,
     "activation": "relu", "scheduler": "none"},
    {"learning_rate": 1e-3, "batch_size": 8, "initial_rho": -10.0,
     "prior_standard_deviation": 0.1, "kl_weight": 1e-8, "rho_learning_rate": 0.01,
     "activation": "gelu", "scheduler": "none"},
    {"learning_rate": 1e-3, "batch_size": 16, "initial_rho": -10.0,
     "prior_standard_deviation": 0.1, "kl_weight": 1e-8, "rho_learning_rate": 0.01,
     "activation": "gelu", "scheduler": "none"},
]

# Batch size 4 was dropped after one trial: no better than 8 and six
# times slower per trial
SPACE = {
    "learning_rate": FloatDistribution(2e-4, 1.5e-3, log=True),
    "batch_size": CategoricalDistribution([8, 16]),
    "initial_rho": FloatDistribution(-12.0, -7.0),
    "prior_standard_deviation": FloatDistribution(0.03, 1.0, log=True),
    "kl_weight": FloatDistribution(1e-10, 1e-6, log=True),
    "rho_learning_rate": FloatDistribution(1e-3, 3e-2, log=True),
    "activation": CategoricalDistribution(["relu", "gelu"]),
    "scheduler": CategoricalDistribution(["none", "cosine"]),
}


def suggest(trial):
    params = {}
    for name, distribution in SPACE.items():
        if isinstance(distribution, CategoricalDistribution):
            params[name] = trial.suggest_categorical(name, distribution.choices)
        else:
            params[name] = trial.suggest_float(
                name, distribution.low, distribution.high, log=distribution.log)
    return params


def warm_start(study, old_storage):
    """
    Copy the finished trials of an earlier study whose settings fit
    SPACE, so the sampler does not have to rediscover them.
    """
    def fits(name, value):
        distribution = SPACE[name]
        if isinstance(distribution, CategoricalDistribution):
            return value in distribution.choices
        return distribution.low <= value <= distribution.high

    for study_name in optuna.get_all_study_names(old_storage):
        old = optuna.load_study(study_name=study_name, storage=old_storage)
        for trial in old.get_trials(states=[TrialState.COMPLETE]):
            if not all(name in trial.params and fits(name, trial.params[name])
                       for name in SPACE):
                continue
            study.add_trial(optuna.trial.create_trial(
                params=trial.params,
                distributions=SPACE,
                value=trial.value,
                intermediate_values=trial.intermediate_values,
                user_attrs={"copied_from": f"{study_name} trial {trial.number}"},
            ))
            print(f"warm start: copied trial {trial.number} ({trial.value:.3f} dB)")


def build_config(params, epochs, seed):
    config = load_config(PROJECT_ROOT / "config.yaml")
    config["seed"] = seed
    config["data"]["root"] = str(PROJECT_ROOT / config["data"]["root"])
    config["model"].update({
        "name": "variational_unet",
        "activation": params["activation"],
        "initial_rho": params["initial_rho"],
        "prior_standard_deviation": params["prior_standard_deviation"],
    })
    config["training"].update({
        "epochs": epochs,
        "batch_size": params["batch_size"],
        "learning_rate": params["learning_rate"],
        "kl_weight": params["kl_weight"],
        "rho_learning_rate": params["rho_learning_rate"],
        "scheduler": params["scheduler"],
    })
    return config


def objective(trial, epochs, seed):

    params = suggest(trial)
    config = build_config(params, epochs, seed)

    trial_dir = STUDY_DIR / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / "config.yaml", "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _ = create_dataloaders(config)
    dataset_size = len(train_loader.dataset)

    model = create_model(config["model"]).to(device)
    training = config["training"]
    criterion = get_loss(training["loss"])
    optimizer = get_optimizer(training["optimizer"], model,
                              training["learning_rate"], training["weight_decay"],
                              training["rho_learning_rate"])
    scheduler = get_scheduler(training["scheduler"], optimizer, epochs)

    history = {"val_psnr": [], "val_loss": [], "train_loss": [], "train_kl": []}
    best_psnr = float("-inf")

    try:
        for epoch in range(epochs):
            train_metrics = train_one_epoch_variational(
                model, train_loader, optimizer, criterion, device,
                dataset_size, kl_weight=training["kl_weight"])
            val_metrics = evaluate_variational(
                model, val_loader, criterion, device, dataset_size,
                kl_weight=training["kl_weight"], sample=False)
            if scheduler is not None:
                scheduler.step()

            history["val_psnr"].append(val_metrics["psnr"])
            history["val_loss"].append(val_metrics["reconstruction"])
            history["train_loss"].append(train_metrics["reconstruction"])
            history["train_kl"].append(train_metrics["kl"])

            # The running best is reported, so a one-epoch dip in val
            # PSNR does not get a good trial pruned
            best_psnr = max(best_psnr, val_metrics["psnr"])
            trial.report(best_psnr, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()
    finally:
        with open(trial_dir / "history.json", "w") as file:
            json.dump({"params": params, "history": history}, file, indent=2)

    return best_psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start", metavar="STUDY_DB",
                        help="copy finished trials from an earlier study.db")
    args = parser.parse_args()

    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="variational_unet",
        storage=f"sqlite:///{(STUDY_DIR / 'study.db').as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        # Only clear failures are stopped: best val PSNR below 29.3 dB after
        # 8 epochs. Every screening run that ended above 30.2 dB was above
        # that by epoch 8. Median and percentile pruning were tried first;
        # at this noise level they stopped almost every new trial.
        pruner=optuna.pruners.ThresholdPruner(lower=29.3, n_warmup_steps=7),
    )

    # A trial left RUNNING by an interrupted process would stay that way
    for trial in study.get_trials(states=[TrialState.RUNNING]):
        study._storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)

    if len(study.trials) == 0:
        if args.warm_start:
            warm_start(study, f"sqlite:///{Path(args.warm_start).resolve().as_posix()}")
        else:
            for params in STARTING_POINTS:
                study.enqueue_trial(params)

    study.optimize(lambda trial: objective(trial, args.epochs, args.seed),
                   n_trials=args.trials, gc_after_trial=True)

    print("Best val PSNR:", study.best_value)
    print("Best params:", json.dumps(study.best_params, indent=2))

    trials = [
        {"number": t.number, "state": t.state.name, "value": t.value,
         "epochs_run": len(t.intermediate_values), "params": t.params}
        for t in study.trials
    ]
    trials.sort(key=lambda t: -(t["value"] or float("-inf")))
    with open(STUDY_DIR / "trials.json", "w") as file:
        json.dump(trials, file, indent=2)
    try:
        importance = optuna.importance.get_param_importances(study)
        with open(STUDY_DIR / "param_importance.json", "w") as file:
            json.dump(importance, file, indent=2)
        print("Importance:", json.dumps(importance, indent=2))
    except Exception as error:  # needs enough finished trials
        print("No importance:", error)


if __name__ == "__main__":
    main()

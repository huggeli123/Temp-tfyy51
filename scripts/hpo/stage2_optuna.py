"""
Stage 2: joint multi-objective search with Optuna (TPE).

Objectives, both maximised:
  1. val PSNR of the posterior-mean network (full-size val images)
  2. epistemic std / mean absolute error on 8 val images: how much
     uncertainty the model expresses relative to its error

The result is a Pareto front between accuracy and uncertainty. Every
trial uses its own seed (the trial number), so the search does not tune
itself to one seed's luck. The Adam eps fix for the rho group found in
stage 1b is always on.

    python scripts/hpo/stage2_optuna.py --trials 48 [--workers 4]

Resumable: the study lives in results/hpo/stage2_optuna.journal; rerun
the same command to add trials. Full per-trial records go to
results/hpo/stage2.jsonl.
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results" / "hpo"
JOURNAL = RESULTS / "stage2_optuna.journal"
LOG = RESULTS / "stage2.jsonl"
STUDY = "stage2_variational_unet"

FIXED = {
    "model": "variational_unet",
    "rho_adam_eps": 1e-16,
    "eval_fractions": [0.5, 1.0],
    "crop": 96,
}

# Images seen per trial is kept equal (600 steps x 16), so a larger batch
# gets fewer steps.
IMAGES_PER_TRIAL = 600 * 16


def suggest(trial):
    p = {
        "initial_rho": trial.suggest_float("initial_rho", -11.0, -5.5),
        "kl_weight": trial.suggest_float("kl_weight", 1e-10, 1e-5, log=True),
        "prior_standard_deviation": trial.suggest_float(
            "prior_standard_deviation", 0.01, 1.0, log=True),
        "rho_learning_rate": trial.suggest_float(
            "rho_learning_rate", 1e-3, 3e-2, log=True),
        "learning_rate": trial.suggest_float(
            "learning_rate", 3e-4, 4e-3, log=True),
        "activation": trial.suggest_categorical(
            "activation", ["relu", "leaky_relu", "gelu", "silu"]),
        "normalization": trial.suggest_categorical(
            "normalization", ["none", "group", "batch"]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "loss": trial.suggest_categorical("loss", ["mse", "l1"]),
        "warmup_fraction": trial.suggest_categorical(
            "warmup_fraction", [0.0, 0.05]),
        "kl_anneal_fraction": trial.suggest_categorical(
            "kl_anneal_fraction", [0.0, 0.3]),
        "gradient_clip": trial.suggest_categorical("gradient_clip", [0.0, 1.0]),
    }
    p["steps"] = IMAGES_PER_TRIAL // p["batch_size"]
    return p


# Known points from stages 1/1b, evaluated first so TPE starts informed.
START_POINTS = [
    # best accuracy in stage 1b (ties the plain UNet)
    dict(initial_rho=-10.0, kl_weight=1e-8, prior_standard_deviation=0.1,
         rho_learning_rate=0.01, learning_rate=1e-3, activation="relu",
         normalization="none", batch_size=16, loss="mse",
         warmup_fraction=0.0, kl_anneal_fraction=0.0, gradient_clip=0.0),
    # most uncertainty in stage 1b
    dict(initial_rho=-6.0, kl_weight=1e-8, prior_standard_deviation=0.1,
         rho_learning_rate=0.01, learning_rate=1e-3, activation="relu",
         normalization="none", batch_size=16, loss="mse",
         warmup_fraction=0.0, kl_anneal_fraction=0.0, gradient_clip=0.0),
    # in between
    dict(initial_rho=-8.0, kl_weight=1e-8, prior_standard_deviation=0.1,
         rho_learning_rate=0.01, learning_rate=1e-3, activation="relu",
         normalization="none", batch_size=16, loss="mse",
         warmup_fraction=0.0, kl_anneal_fraction=0.0, gradient_clip=0.0),
]


def storage():
    import optuna
    from optuna.storages.journal import JournalFileBackend
    return optuna.storages.JournalStorage(JournalFileBackend(str(JOURNAL)))


def objective(trial):
    import proxy_train
    p = {**FIXED, **suggest(trial), "seed": trial.number,
         "tag": f"s2_t{trial.number:03d}"}
    r = proxy_train.run_trial(p, str(LOG))
    if r.get("diverged") or "val_psnr" not in r:
        return 0.0, 0.0
    trial.set_user_attr("sampling_cost", r["sampling_cost"])
    trial.set_user_attr("sigma_median", r["sigma"]["median"])
    trial.set_user_attr("corr_std_error", r["uncertainty"]["corr_std_error"])
    return r["val_psnr"], r["uncertainty"]["std_over_error"]


def worker(n_trials, seed):
    sys.path.insert(0, str(HERE))
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(
        study_name=STUDY, storage=storage(),
        sampler=optuna.samplers.TPESampler(
            multivariate=True, constant_liar=True, seed=seed,
            n_startup_trials=12,
        ),
    )

    def done(study, trial):
        print(f"trial {trial.number:3d} {trial.state.name} values={trial.values}",
              flush=True)

    study.optimize(objective, n_trials=n_trials, callbacks=[done],
                   catch=(Exception,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    import optuna
    RESULTS.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY, storage=storage(), load_if_exists=True,
        directions=["maximize", "maximize"],
    )
    if len(study.trials) == 0:
        for point in START_POINTS:
            study.enqueue_trial(point)
    finished = sum(t.state.is_finished() for t in study.trials)
    remaining = max(0, a.trials - finished)
    print(f"{finished} trials finished, running {remaining} more", flush=True)
    if remaining == 0:
        return

    per_worker = [remaining // a.workers + (i < remaining % a.workers)
                  for i in range(a.workers)]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(n, 1000 + i))
             for i, n in enumerate(per_worker) if n]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()

from pathlib import Path
from datetime import datetime
import json
import yaml

from pathlib import Path




def create_experiment(
    config,
    experiment_name,
    results_root="results",
):
    """
    Create a new directory for one experiment run.

    Example:

        results/
        └── compare_models/
            └── 2026-08-19_13-30-22/
                └── config.yaml

    Returns
    -------
    run_dir : Path
        Directory where all results for this run should be saved.
    """

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    run_dir = (
        Path(results_root)
        / experiment_name
        / timestamp
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Save the exact configuration used for this run
    config_path = run_dir / "config.yaml"

    with open(config_path, "w") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    print("Experiment created")
    print("------------------")
    print("Name:      ", experiment_name)
    print("Run:       ", timestamp)
    print("Directory: ", run_dir)

    return run_dir




def save_history(
    history,
    run_dir,
    test_loss=None,
    test_psnr=None,
):
    """
    Save training, validation, and test metrics.

    The file contains:
        - metrics for every training epoch
        - best validation epoch
        - final training metrics
        - final test metrics
        - total and mean per-epoch training time (train + validation)
    """

    run_dir = Path(run_dir)

    # --------------------------------------------------------
    # Find best validation epoch
    # --------------------------------------------------------

    best_epoch_index = min(
        range(len(history["val_loss"])),
        key=lambda i: history["val_loss"][i],
    )

    # --------------------------------------------------------
    # Runtime (older histories may not have epoch_time)
    # --------------------------------------------------------

    epoch_times = history.get("epoch_time", [])

    total_training_time = sum(epoch_times) if epoch_times else None
    mean_epoch_time = (
        total_training_time / len(epoch_times) if epoch_times else None
    )

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    summary = {
        "best_epoch": best_epoch_index + 1,

        "best_val_loss": history["val_loss"][best_epoch_index],
        "best_val_psnr": history["val_psnr"][best_epoch_index],

        "final_train_loss": history["train_loss"][-1],
        "final_train_psnr": history["train_psnr"][-1],

        "test_loss": test_loss,
        "test_psnr": test_psnr,

        "total_training_time_seconds": total_training_time,
        "mean_epoch_time_seconds": mean_epoch_time,
    }

    # --------------------------------------------------------
    # Everything to save
    # --------------------------------------------------------

    results = {
        "summary": summary,
        "history": history,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_path = run_dir / "results.json"

    with open(save_path, "w") as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print("Results saved to:", save_path)
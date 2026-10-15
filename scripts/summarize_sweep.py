"""
Rank the trials of a sweep made by scripts/sweep_variational_unet.py.

    python scripts/summarize_sweep.py [--stage stage1_screen]

Reads results/variational_unet_sweep/all_trials.jsonl and writes
results/variational_unet_sweep/summary.md, sorted by val PSNR at the
best-val-loss epoch.
"""

import argparse
import json
from pathlib import Path

SWEEP_ROOT = Path(__file__).resolve().parents[1] / "results" / "variational_unet_sweep"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="append",
                        help="only these stages (default: all)")
    args = parser.parse_args()

    with open(SWEEP_ROOT / "all_trials.jsonl") as file:
        trials = [json.loads(line) for line in file if line.strip()]

    stages = args.stage or sorted({t["stage"] for t in trials})

    lines = ["# variational_unet sweep", ""]

    for stage in stages:
        rows = sorted((t for t in trials if t["stage"] == stage),
                      key=lambda t: -t["val_psnr_at_best"])
        lines += [f"## {stage}", "",
                  "| # | Trial | Val PSNR (best ckpt) | Max val PSNR | Test PSNR "
                  "| Best epoch | Final KL/N | Min | Overrides |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for rank, t in enumerate(rows, 1):
            overrides = ", ".join(f"{k}={v}" for k, v in t["set"].items()
                                  if k != "training.epochs")
            lines.append(
                f"| {rank} | {t['name']} | {t['val_psnr_at_best']:.3f} "
                f"| {t['max_val_psnr']:.3f} | {t['test_psnr']:.3f} "
                f"| {t['best_epoch']}/{t['epochs']} | {t['final_train_kl']:.3g} "
                f"| {t['minutes']:.1f} | {overrides} |")
        lines.append("")

    text = "\n".join(lines)
    (SWEEP_ROOT / "summary.md").write_text(text, encoding="utf8")
    print(text)


if __name__ == "__main__":
    main()

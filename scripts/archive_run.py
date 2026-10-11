"""
Copy the small text records of a run into the documentation.

    python scripts/archive_run.py results/00_baseline/<run> <name>

results/ is not in git, so a result that the documentation relies on would
otherwise exist on one computer only. This copies config.yaml, results.json
and any summary .json files of the run to docs/variational/runs/<name>/.
Checkpoints and images are left out: they are large and can be regenerated.
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT_ROOT / "docs" / "variational" / "runs"


def main():

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", help="folder written by the notebook")
    parser.add_argument("name", help="descriptive folder name in the archive")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    target = ARCHIVE_ROOT / args.name

    records = [run_dir / "config.yaml", *sorted(run_dir.glob("*.json"))]

    if not (run_dir / "results.json").exists():
        raise SystemExit(f"{run_dir} has no results.json; did the run finish?")

    target.mkdir(parents=True, exist_ok=True)

    for record in records:
        shutil.copy2(record, target / record.name)
        print("Archived:", (target / record.name).relative_to(PROJECT_ROOT))

    # Remember which run this was, since the timestamp is not in the name
    (target / "source.txt").write_text(
        f"results/{run_dir.parent.name}/{run_dir.name}\n"
    )


if __name__ == "__main__":
    main()

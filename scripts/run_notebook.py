"""
Run the experiment notebook from the command line.

    python scripts/run_notebook.py
    python scripts/run_notebook.py --set seed=43 --set training.rho_learning_rate=null
    python scripts/run_notebook.py --set model.name=unet

The notebook is executed as it is, from top to bottom, in a fresh kernel.
The file on disk is not changed and keeps whatever outputs it has.

Each --set changes one config value for this run only. The assignment is
inserted right after the notebook loads config.yaml, so config.yaml is not
edited, and the config snapshot saved with the results records the value
that was actually used.

The seeded runs in docs/variational/runs/ were made this way.
"""

import argparse
import sys
import warnings
from pathlib import Path

import nbformat
import yaml
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "experiments" / "00_baseline.ipynb"

# The line in the notebook after which the overrides are inserted
LOAD_LINE = 'config = load_config("config.yaml")'

# Lines of the notebook's printed output worth repeating in the terminal
REPORTED = ("Run:  ", "Loading Model:", "Test loss:", "Test PSNR:")


def override_statement(setting):
    """
    'training.rho_learning_rate=null' -> "config['training']['rho_learning_rate'] = None"
    """

    key, separator, text = setting.partition("=")

    if not separator or not key:
        raise SystemExit(f"--set expects key=value, got: {setting}")

    value = yaml.safe_load(text)

    # YAML reads 1e-6 as a string; accept it as the number it looks like
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass

    path = "".join(f"[{part!r}]" for part in key.split("."))

    return f"config{path} = {value!r}"


def main():

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", dest="settings", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override one config value, e.g. seed=43")
    parser.add_argument("--notebook", default=str(NOTEBOOK))
    parser.add_argument("--kernel", default="python3",
                        help="Jupyter kernel to run in (default: this environment)")
    parser.add_argument("--save-executed", metavar="PATH",
                        help="also save a copy of the notebook with its outputs")
    args = parser.parse_args()

    notebook_path = Path(args.notebook).resolve()
    notebook = nbformat.read(notebook_path, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    overrides = [override_statement(setting) for setting in args.settings]

    if overrides:

        config_cells = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code" and LOAD_LINE in cell.source
        ]

        if len(config_cells) != 1:
            raise SystemExit(
                f"Expected exactly one cell containing: {LOAD_LINE}"
            )

        config_cells[0].source = config_cells[0].source.replace(
            LOAD_LINE,
            "\n".join([LOAD_LINE, *overrides]),
            1,
        )

        for statement in overrides:
            print("Override:", statement)

    # Harmless notice from the kernel connection on Windows
    warnings.filterwarnings("ignore", message="Proactor event loop")

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=args.kernel,
        # Same working directory as when the notebook is opened in an editor
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )

    try:
        client.execute()
    finally:
        if args.save_executed:
            nbformat.write(notebook, args.save_executed)

    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            for line in output.get("text", "").splitlines():
                if any(key in line for key in REPORTED):
                    print(line.strip())


if __name__ == "__main__":
    sys.exit(main())

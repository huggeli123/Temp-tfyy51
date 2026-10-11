"""
Check that docs/variational/ still matches the code and the archived runs.

    python scripts/check_variational_docs.py

Run it after changing the variational code, the notebook, config defaults
or the archived runs. It exits with status 1 and lists what is out of date
if any of these fail:

    1. every relative link and heading anchor in the docs resolves
    2. every Python excerpt in the docs still exists in the source
    3. numbers that follow from the code or the data (parameter counts,
       defaults, weight scales, dataset size) are the ones the docs state
    4. numbers from experiments are the ones stored in docs/variational/runs/

An excerpt is compared with comments, docstrings and whitespace removed,
so it may be reformatted or shortened. A line holding only `...` marks an
omission. A block that is deliberately not real source (a usage example)
is skipped when the line before it is:

    <!-- docs-check: illustrative -->
"""

import json
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from models import create_model
from models.variational import positive_scale

DOCS = PROJECT_ROOT / "docs" / "variational"
RUNS = DOCS / "runs"

ILLUSTRATIVE = "<!-- docs-check: illustrative -->"

# The archived run whose config is the recommended setting
RECOMMENDED_RUN = "partial_learned_seed42"

# Architecture all the tables in the docs are written for
BASE_MODEL_CONFIG = {
    "channels": 32,
    "depth": 3,
    "kernel_size": 3,
    "activation": "relu",
    "normalization": "none",
    "num_groups": 4,
}

problems = []
number_of_checks = 0


# ============================================================
# Helpers
# ============================================================

def read_doc(name):
    return (DOCS / name).read_text(encoding="utf-8")


def expect(doc_name, *parts, why):
    """
    Some line of the document must contain all the given strings.
    """

    global number_of_checks
    number_of_checks += 1

    parts = [str(part) for part in parts]

    for line in read_doc(doc_name).splitlines():
        if all(part in line for part in parts):
            return

    problems.append(
        f"{doc_name}: no line contains {parts}  ({why})"
    )


def expect_number(doc_name, *parts, value, tolerance, why):
    """
    Some line containing all the given strings must also contain a number
    within the relative tolerance of `value`.
    """

    global number_of_checks
    number_of_checks += 1

    parts = [str(part) for part in parts]
    found_line = False

    for line in read_doc(doc_name).splitlines():

        if not all(part in line for part in parts):
            continue

        found_line = True

        for text in re.findall(r"\d[\d,]*\.?\d*", line):
            number = float(text.replace(",", "").rstrip("."))
            if abs(number - value) <= tolerance * abs(value):
                return

    reason = "no number close to" if found_line else "no line for"

    problems.append(
        f"{doc_name}: {reason} {value:.4g} with {parts}  ({why})"
    )


SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def scientific(value, digits=1):
    """
    0.0067 -> '6.7 × 10⁻³', the notation used in the docs.
    """

    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10 ** exponent

    return f"{mantissa:.{digits}f} × 10{str(exponent).translate(SUPERSCRIPTS)}"


def build(name, **extra):
    torch.manual_seed(0)
    return create_model({"name": name, **BASE_MODEL_CONFIG, **extra})


def variational_weights(model):
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.endswith("_rho")
    )


def total_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def initial_kl(model):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(torch.zeros(1, 1, 32, 32))[1].item()


def load_json(path):
    with open(path) as file:
        return json.load(file)


# ============================================================
# 1. Links and anchors
# ============================================================

def heading_anchors(path):

    anchors = set()

    for line in path.read_text(encoding="utf-8").splitlines():

        match = re.match(r"^#+\s+(.*)", line)

        if match:
            text = match.group(1).strip().lower()
            text = re.sub(r"[^\w\s\-]", "", text)
            anchors.add(text.replace(" ", "-"))

    return anchors


def check_links():

    global number_of_checks

    for path in sorted(DOCS.glob("*.md")):

        text = path.read_text(encoding="utf-8")

        for link in re.findall(r"\]\(([^)\s]+)\)", text):

            if link.startswith(("http://", "https://")):
                continue

            number_of_checks += 1

            target_text, _, anchor = link.replace("\\_", "_").partition("#")
            target = (path.parent / target_text).resolve() if target_text else path

            if not target.exists():
                problems.append(f"{path.name}: link to missing file {link}")

            elif anchor and target.suffix == ".md":
                if anchor not in heading_anchors(target):
                    problems.append(f"{path.name}: link to missing heading {link}")


# ============================================================
# 2. Python excerpts
# ============================================================

def normalise_code(code):
    code = re.sub(r'""".*?"""', "", code, flags=re.S)   # docstrings
    code = re.sub(r"#[^\n]*", "", code)                  # comments
    code = re.sub(r"\s+", "", code)                      # layout
    code = re.sub(r",(?=[)\]}])", "", code)              # trailing commas
    return code


def source_texts():

    paths = [
        *PROJECT_ROOT.glob("*.py"),
        *PROJECT_ROOT.glob("models/*.py"),
        *PROJECT_ROOT.glob("utils/*.py"),
        *PROJECT_ROOT.glob("scripts/*.py"),
    ]

    texts = {
        str(path.relative_to(PROJECT_ROOT)): normalise_code(
            path.read_text(encoding="utf-8")
        )
        for path in paths
    }

    for path in PROJECT_ROOT.glob("experiments/*.ipynb"):

        notebook = json.loads(path.read_text(encoding="utf-8"))

        texts[str(path.relative_to(PROJECT_ROOT))] = normalise_code(
            "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
        )

    return texts


def excerpt_exists(excerpt, sources):
    """
    True if the pieces of the excerpt (split at `...` lines) occur in
    this order in one source file.
    """

    pieces = [
        normalise_code(piece)
        for piece in re.split(r"^\s*\.\.\.\s*$", excerpt, flags=re.M)
    ]
    pieces = [piece for piece in pieces if piece]

    for source in sources.values():

        position = 0

        for piece in pieces:
            position = source.find(piece, position)
            if position < 0:
                break
            position += len(piece)
        else:
            return True

    return False


def check_excerpts():

    global number_of_checks

    sources = source_texts()

    block = re.compile(
        r"^(?P<before>[^\n]*)\n(?:[ \t]*\n)?[ \t]*```python\n(?P<code>.*?)```",
        flags=re.S | re.M,
    )

    for path in sorted(DOCS.glob("*.md")):

        for match in block.finditer(path.read_text(encoding="utf-8")):

            if ILLUSTRATIVE in match.group("before"):
                continue

            number_of_checks += 1

            if not excerpt_exists(match.group("code"), sources):

                first_line = match.group("code").strip().splitlines()[0]

                problems.append(
                    f"{path.name}: Python excerpt not found in the source, "
                    f"starting with: {first_line.strip()}"
                )


# ============================================================
# 3. Numbers that follow from the code and the data
# ============================================================

def check_model_tables():

    # ---- summary table: variational weights and total parameters ----

    models = {
        "variational_simple_cnn": build("variational_simple_cnn"),
        "variational_unet": build("variational_unet"),
        "variational_partial_unet": build("variational_partial_unet"),
    }

    for name, model in models.items():

        expect(
            "03_models.md",
            f"`{name}`",
            f"{variational_weights(model):,}",
            f"{total_parameters(model):,}",
            why="parameter counts in the summary table",
        )

        # ---- factory defaults ----

        expect(
            "03_models.md",
            f"`{name}`",
            f"{model.output_layer.prior_standard_deviation:.1f}",
            f"{model.output_layer.weight_rho.flatten()[0].item():.1f}",
            why="defaults used by create_model",
        )

    expect(
        "03_models.md",
        f"{total_parameters(build('unet')):,}",
        f"{total_parameters(build('simple_cnn')):,}",
        why="parameter counts of the plain models",
    )

    # ---- per-layer table of the fully variational UNet ----

    for name, parameter in models["variational_unet"].named_parameters():

        if not name.endswith("weight_mean"):
            continue

        fan_in = parameter.shape[1] * parameter.shape[2] * parameter.shape[3]

        expect(
            "03_models.md",
            f"`{name.removesuffix('.weight_mean')}`",
            str(tuple(parameter.shape)),
            f"{parameter.numel():,}",
            f"{1 / math.sqrt(3 * fan_in):.3f}",
            why="per-layer table of variational_unet",
        )

    # ---- variational_decoder_levels table ----

    recommended = yaml.safe_load(
        (RUNS / RECOMMENDED_RUN / "config.yaml").read_text()
    )
    training_images = recommended_training_images()

    for levels in range(BASE_MODEL_CONFIG["depth"] + 1):

        model = build(
            "variational_partial_unet",
            initial_rho=recommended["model"]["initial_rho"],
            prior_standard_deviation=recommended["model"]["prior_standard_deviation"],
            variational_decoder_levels=levels,
        )

        for doc_name in ("03_models.md", "06_hyperparameters.md"):
            expect(
                doc_name,
                f"{variational_weights(model):,}",
                why=f"variational weights with {levels} variational decoder levels",
            )

        if training_images:
            expect_number(
                "03_models.md",
                f"| {levels} ",
                f"{variational_weights(model):,}",
                value=initial_kl(model) / training_images,
                tolerance=0.01,
                why="KL per training image at initialisation",
            )


def check_scales():

    # ---- rho -> sigma ----

    for rho in (-5.0, -6.0, -6.5, -8.0, -10.0):

        sigma = positive_scale(torch.tensor(rho)).item()

        expect(
            "06_hyperparameters.md",
            f"| {rho:g} |",
            scientific(sigma),
            why="sigma = softplus(rho)",
        )

    # ---- weight scale by fan-in, and sigma at rho = -5 relative to it ----

    sigma = positive_scale(torch.tensor(-5.0)).item()

    for fan_in in (9, 288, 576, 1152, 2304, 32):

        weight_std = 1 / math.sqrt(3 * fan_in)

        for value, why in (
            (weight_std, "std of Kaiming-uniform means"),
            (100 * sigma / weight_std, "sigma at rho = -5 as a share of it"),
        ):
            expect_number(
                "02_layers.md",
                f"| {fan_in:,} |",
                value=value,
                tolerance=0.06,
                why=why,
            )


def recommended_training_images():

    train_dir = PROJECT_ROOT / "ct_data" / "train"

    if not train_dir.exists():
        return None

    return sum(1 for path in train_dir.iterdir() if path.is_file())


def check_data_and_schedule():

    training_images = recommended_training_images()

    if training_images is None:
        print("ct_data/ is not unpacked: skipping the dataset-size checks.")
        return

    from PIL import Image

    train_dir = PROJECT_ROOT / "ct_data" / "train"
    width, height = Image.open(next(train_dir.iterdir())).size

    expect("README.md", "training images", f"({training_images:,})",
           why="number of training images")
    expect("README.md", "pixels per image", f"{width} × {height}",
           f"{width * height:,}", why="image size")

    training = yaml.safe_load(
        (RUNS / RECOMMENDED_RUN / "config.yaml").read_text()
    )["training"]

    steps = math.ceil(training_images / training["batch_size"])

    expect(
        "04_training.md",
        f"{steps} steps per epoch",
        f"{steps * training['epochs']:,} steps",
        why="optimizer steps in the recommended run",
    )

    # ---- KL scale table: KL per training image at initialisation ----

    settings = [
        (("`variational_unet`", "ρ = -5"),
         build("variational_unet", initial_rho=-5.0, prior_standard_deviation=1.0)),
        (("`variational_unet`", "ρ = -8"),
         build("variational_unet", initial_rho=-8.0, prior_standard_deviation=0.1)),
        (("`variational_partial_unet`", "ρ = -6.5"),
         build("variational_partial_unet", initial_rho=-6.5, prior_standard_deviation=1.0)),
        (("`variational_simple_cnn`", "ρ = -5"),
         build("variational_simple_cnn", initial_rho=-5.0, prior_standard_deviation=1.0)),
    ]

    for parts, model in settings:
        expect_number(
            "06_hyperparameters.md",
            *parts,
            f"{variational_weights(model):,}",
            value=initial_kl(model) / training_images,
            tolerance=0.01,
            why="KL per training image in the scale table",
        )


# ============================================================
# 4. Numbers from the archived runs
# ============================================================

def check_runs():

    environment = None

    for run in sorted(path for path in RUNS.iterdir() if path.is_dir()):

        name = f"`{run.name}`"

        results = load_json(run / "results.json")["summary"]

        expect(
            "07_experiments.md",
            name,
            f"{results['test_loss']:.6f}",
            f"{results['test_psnr']:.2f}",
            why="test loss and PSNR in results.json",
        )

        if (run / "variational_summary.json").exists():

            summary = load_json(run / "variational_summary.json")
            environment = summary["environment"]

            sigma = summary["posterior_std"]
            uncertainty = summary["uncertainty"]

            expect(
                "07_experiments.md",
                name,
                *[
                    f"{sigma[key] * 1e3:.2f}"
                    for key in ("min", "p05", "median", "p95", "max")
                ],
                f"{summary['test_sampled_weights']['loss']:.6f}",
                why="learned sigma and sampled-weight test loss",
            )

            expect(
                "07_experiments.md",
                name,
                f"{uncertainty['mean_epistemic_std']:.5f}",
                f"{uncertainty['mean_absolute_error']:.5f}",
                f"{uncertainty['correlation_std_error']:.2f}",
                why="uncertainty numbers",
            )

            expect(
                "07_experiments.md",
                name,
                *[
                    f"{row['mean_epistemic_std']:.5f}"
                    for row in summary["noise_levels"]
                ],
                why="epistemic std at the four noise levels",
            )

            if run.name == RECOMMENDED_RUN:
                for row in summary["noise_levels"]:
                    expect(
                        "05_uncertainty.md",
                        f"| {row['noise_std']:.1f} ",
                        f"{row['predictive_mean_psnr']:.2f}",
                        f"{row['mean_epistemic_std']:.5f}",
                        why="noise-level table of the recommended run",
                    )

        if (run / "weight_noise_sensitivity.json").exists():

            for row in load_json(run / "weight_noise_sensitivity.json")["rows"]:
                expect(
                    "06_hyperparameters.md",
                    f"| {row['variational_decoder_levels']} |",
                    f"| {row['sigma']:g} |",
                    f"{row['relative_increase'] * 100:+.1f}%",
                    f"{row['mean_epistemic_std']:.4f}",
                    why="weight-noise sensitivity table",
                )

    # ---- the logged numbers of the first fully variational run ----

    history = load_json(RUNS / "full_default_10ep" / "results.json")["history"]

    expect(
        "04_training.md",
        f"{history['train_kl'][0]:,.2f}",
        f"{history['train_total_loss'][0]:,.2f}",
        why="KL and total loss logged in epoch 1 of full_default_10ep",
    )

    for epoch in (1, 5, 10):
        expect(
            "07_experiments.md",
            f"| {epoch} |",
            f"{history['train_loss'][epoch - 1]:.5f}",
            f"{history['train_kl'][epoch - 1]:,.0f}",
            f"{history['train_total_loss'][epoch - 1]:,.2f}",
            why=f"log of full_default_10ep, epoch {epoch}",
        )

    if environment:
        expect(
            "07_experiments.md",
            environment["python"],
            environment["torch"],
            environment["device"],
            why="software and hardware the runs were made with",
        )


def check_quick_start():
    """
    The settings shown in the docs README must be the recommended run's.
    """

    global number_of_checks

    recommended = yaml.safe_load(
        (RUNS / RECOMMENDED_RUN / "config.yaml").read_text()
    )

    match = re.search(r"```yaml\n(.*?)```", read_doc("README.md"), flags=re.S)
    shown = yaml.safe_load(match.group(1))

    for section, settings in shown.items():
        for key, value in settings.items():

            number_of_checks += 1

            if recommended[section].get(key) != value:
                problems.append(
                    f"README.md: quick start shows {section}.{key} = {value}, "
                    f"the recommended run used {recommended[section].get(key)}"
                )


# ============================================================
# Main
# ============================================================

def main():

    check_links()
    check_excerpts()
    check_model_tables()
    check_scales()
    check_data_and_schedule()
    check_runs()
    check_quick_start()

    if problems:
        print(f"{len(problems)} of {number_of_checks} checks failed:\n")
        for message in problems:
            print("  -", message)
        raise SystemExit(1)

    print(f"All {number_of_checks} checks passed.")


if __name__ == "__main__":
    main()

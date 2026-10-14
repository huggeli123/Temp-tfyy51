"""
Compare all runs in results/ in one table.

    python scripts/compare_runs.py
    python scripts/compare_runs.py --sort test_psnr
    python scripts/compare_runs.py --differing --where model.name=unet --params
    python scripts/compare_runs.py --html            # results/comparison.html
    python scripts/compare_runs.py --csv comparison.csv

Each row is one run folder (results/<experiment>/<timestamp>/), with a
column for every setting in config.yaml, grouped by section in the order
of the file. Headings of settings that differ between the listed runs are
highlighted; --differing shows only those, for a narrow table. A value
the run's model does not read (the variational keys for a plain UNet,
num_groups without group normalisation, ...) is shown greyed out, in
parentheses in the terminal, and never counts as a difference.

--html writes a page to open in a browser: sortable columns, PSNR shaded
from worst to best, a validation-PSNR curve per run, and a click on a row
shows that run's plots.

test_loss is in the run's own loss function (training.loss), so it is
only comparable between runs with the same loss; test_psnr always is.
Test numbers are from the best-validation checkpoint; for variational
models they use the posterior-mean network. Folders without results.json
(runs still training, or that crashed or were stopped) are listed
separately at the end.
"""

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_HTML = RESULTS_ROOT / "comparison.html"

VARIATIONAL_KEYS = (
    "model.prior_standard_deviation",
    "model.initial_rho",
    "training.kl_weight",
    "training.rho_learning_rate",
)

# Bookkeeping settings that never change what is trained
IGNORED_KEYS = (
    "data.root",
    "training.num_workers",
    "output.directory",
    "output.save_best_model",
    "output.save_example_images",
)

# Section heading for each top-level config key
SECTION_TITLES = {
    "seed": "general",
    "data": "data",
    "noise": "noise",
    "model": "model",
    "augmentation": "augmentation",
    "training": "training",
    "output": "output",
}

# Result columns: (heading, function of the run record)
RESULT_COLUMNS = [
    ("epochs_run", lambda r: len(r["history"]["val_loss"])),
    ("best_epoch", lambda r: r["summary"]["best_epoch"]),
    ("val_psnr", lambda r: r["summary"]["best_val_psnr"]),
    ("test_loss", lambda r: r["summary"]["test_loss"]),
    ("test_psnr", lambda r: r["summary"]["test_psnr"]),
    ("minutes", lambda r: r["summary"]["total_training_time_seconds"] / 60),
]

# Short column headings; anything not listed uses the last part of its key
LABELS = {
    "model.name": "model",
    "model.normalization": "norm",
    "model.activation": "act",
    "model.kernel_size": "kernel",
    "model.num_groups": "groups",
    "model.initial_rho": "rho0",
    "model.prior_standard_deviation": "prior_std",
    "model.variational_decoder_levels": "var_levels",
    "training.learning_rate": "lr",
    "training.rho_learning_rate": "rho_lr",
    "training.kl_weight": "kl_w",
    "training.batch_size": "batch",
    "training.weight_decay": "wd",
    "training.scheduler": "sched",
    "noise.train_val_std": "noise",
    "noise.test_std": "test_noise",
    "augmentation.enabled": "aug",
    "augmentation.horizontal_flip": "hflip",
    "augmentation.horizontal_flip_probability": "hflip_p",
    "augmentation.rotation_degrees": "rot_deg",
    "training.num_workers": "workers",
    "training.optimizer": "opt",
    "data.root": "data_root",
    "output.directory": "out_dir",
    "output.save_best_model": "save_best",
    "output.save_example_images": "save_imgs",
    "epochs_run": "ran",
    "best_epoch": "best",
    "minutes": "min",
}

# Plots a run folder may hold, shown in the HTML details
RUN_IMAGES = (
    "psnr_curve.png",
    "loss_curve.png",
    "val_denoising_examples.png",
    "test_examples.png",
    "val_uncertainty.png",
)

FAILED = ("exploded", "collapsed", "stalled")


def flatten(config, prefix=""):
    """
    {'model': {'depth': 3}} -> {'model.depth': 3}
    """

    flat = {}

    for key, value in config.items():
        name = f"{prefix}{key}"

        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value

    return flat


def used_settings(flat):
    """
    Drop the settings the chosen model and options do not read.
    """

    model_name = str(flat.get("model.name", "")).lower()
    unused = set(IGNORED_KEYS)

    if not model_name.startswith("variational"):
        unused.update(VARIATIONAL_KEYS)

    if model_name != "variational_partial_unet":
        unused.add("model.variational_decoder_levels")

    if flat.get("model.normalization") != "group":
        unused.add("model.num_groups")

    if model_name == "linear":
        unused.update(("model.activation", "model.normalization"))

    if not flat.get("augmentation.enabled", False):
        unused.update(
            key for key in flat
            if key.startswith("augmentation.") and key != "augmentation.enabled"
        )

    return {
        key: value for key, value in flat.items()
        if key not in unused
    }


def status(record):
    """
    Short warning for runs that clearly failed, empty otherwise.
    """

    history = record["history"]

    if max(history["train_loss"]) > 1.0:
        return "exploded"

    # Ended far below its best: the run broke down part-way
    if history["val_psnr"][-1] < max(history["val_psnr"]) - 3.0:
        return "collapsed"

    # Validation PSNR frozen over the second half: the network stopped learning
    second_half = history["val_psnr"][len(history["val_psnr"]) // 2:]
    if len(second_half) >= 5 and max(second_half) - min(second_half) < 0.01:
        return "stalled"

    return ""


def parameter_count(config):
    """
    Number of parameters, built on the meta device so no memory is used.
    """

    import torch

    sys.path.insert(0, str(PROJECT_ROOT))
    from models import create_model

    with torch.device("meta"):
        model = create_model(config["model"])

    return sum(parameter.numel() for parameter in model.parameters())


def load_runs(results_root):

    finished = []
    unfinished = []

    for config_path in sorted(results_root.glob("*/*/config.yaml")):
        run_dir = config_path.parent
        config = yaml.safe_load(config_path.read_text())
        name = f"{run_dir.parent.name}/{run_dir.name}"
        results_path = run_dir / "results.json"

        if not results_path.exists():
            unfinished.append((name, config))
            continue

        finished.append({
            "name": name,
            "dir": run_dir,
            "config": config,
            "all": flatten(config),
            "settings": used_settings(flatten(config)),
            "record": json.loads(results_path.read_text()),
        })

    return finished, unfinished


def matches(run, conditions):

    for condition in conditions:
        key, _, text = condition.partition("=")
        if str(run["all"].get(key)) != str(yaml.safe_load(text)):
            return False

    return True


def format_value(value, heading=""):

    if value is None:
        return ""

    if heading in ("val_psnr", "test_psnr"):
        return f"{value:.2f}"

    if heading == "minutes":
        return f"{value:.1f}"

    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.3g}"
        return f"{value:.4g}" if abs(value) < 1 else f"{value:.2f}"

    if isinstance(value, int) and not isinstance(value, bool) and value >= 10_000:
        return f"{value:,}"

    return str(value)


def short_label(key, keys):
    """
    Column heading for a setting or result key.
    """

    if key in LABELS:
        return LABELS[key]

    tail = key.split(".")[-1]
    clash = sum(other.split(".")[-1] == tail for other in keys) > 1

    return key if clash else tail


def run_label(name, single_experiment):
    """
    '00_baseline/2026-10-14_08-47-30' -> '10-14 08:47' (plus the experiment
    name when the table mixes experiments).
    """

    experiment, _, stamp = name.partition("/")
    date, _, time = stamp.partition("_")

    if len(date) == 10 and len(time) == 8:
        stamp = f"{date[5:]} {time[:5].replace('-', ':')}"

    return stamp if single_experiment else f"{experiment} {stamp}"


def setting_value(run, key):
    """
    (value, state) of one setting for one run. state is 'used', 'unused'
    (in the config but not read by this model) or 'missing' (not in this
    run's config at all, e.g. a key added after the run).
    """

    if key not in run["all"]:
        return None, "missing"

    return run["all"][key], ("used" if key in run["settings"] else "unused")


def section_of(key):

    top = key.split(".")[0]

    return SECTION_TITLES.get(top, top)


def config_order(keys):
    """
    Sort setting keys in the order they appear in config.yaml; keys the
    current config no longer has go last.
    """

    current = PROJECT_ROOT / "config.yaml"
    order = list(flatten(yaml.safe_load(current.read_text()))) if current.exists() else []
    section_rank = list(SECTION_TITLES)

    def rank(key):
        if key in order:
            return (0, order.index(key), "")
        top = key.split(".")[0]
        section = section_rank.index(top) if top in section_rank else len(section_rank)
        return (1, section, key)

    ordered = sorted(keys, key=rank)

    # Keep each section together even if config.yaml interleaves them
    return sorted(ordered, key=lambda key: next(
        (i for i, other in enumerate(ordered) if section_of(other) == section_of(key)),
    ))


def section_spans(keys):
    """
    [(section title, number of consecutive keys in it), ...]
    """

    spans = []

    for key in keys:
        title = section_of(key)
        if spans and spans[-1][0] == title:
            spans[-1] = (title, spans[-1][1] + 1)
        else:
            spans.append((title, 1))

    return spans


# ============================================================
# Terminal table
# ============================================================

COLOURS = {"green": "32", "red": "31", "yellow": "33", "dim": "2", "bold": "1"}


def print_table(header, rows, groups, styles, colour, header_styles=None):
    """
    rows: lists of plain strings. styles: per row, per cell, a colour name
    or None. groups: (title, number of columns) printed above the header.
    header_styles: per heading, a colour name or None (default bold).
    """

    widths = [
        max(len(cell) for cell in column)
        for column in zip(header, *rows)
    ]

    # Widen a group's last column when its title is longer than the group
    position = 0
    for title, size in groups:
        span = sum(widths[position:position + size]) + 2 * (size - 1)
        if len(title) > span:
            widths[position + size - 1] += len(title) - span
        position += size

    def paint(text, style):
        if not colour or not style:
            return text
        return f"\033[{COLOURS[style]}m{text}\033[0m"

    def line(cells, cell_styles=None):
        parts = []
        position = 0
        for _, size in groups:
            part = "  ".join(
                paint(
                    cells[i].ljust(widths[i]),
                    cell_styles[i] if cell_styles else None,
                )
                for i in range(position, position + size)
            )
            parts.append(part)
            position += size
        return " │ ".join(parts).rstrip()

    titles = []
    position = 0
    for title, size in groups:
        span = sum(widths[position:position + size]) + 2 * (size - 1)
        titles.append(title.ljust(span))
        position += size

    print(paint(" │ ".join(titles).rstrip(), "dim"))
    print(line(header, header_styles or ["bold"] * len(header)))
    print(line(["─" * width for width in widths]))
    for row, row_styles in zip(rows, styles):
        print(line(row, row_styles))


# ============================================================
# HTML page
# ============================================================

HTML_STYLE = """
:root {
  --bg: #ffffff; --fg: #1d2129; --muted: #6b7280; --line: #e5e7eb;
  --head: #f6f7f9; --hover: #f3f6fb; --accent: #2563eb; --accentrgb: 37, 99, 235;
  --good: 22, 163, 74; --bad: #dc2626; --warn: #b45309;
  --badbg: #fde8e8; --warnbg: #fdf1dc; --spark: #2563eb;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #15171c; --fg: #e5e7eb; --muted: #9ca3af; --line: #2c3039;
    --head: #1d2027; --hover: #1f2430; --accent: #60a5fa; --accentrgb: 96, 165, 250;
    --good: 74, 222, 128; --bad: #f87171; --warn: #fbbf24;
    --badbg: #3b1d1d; --warnbg: #3a2c12; --spark: #60a5fa;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #15171c; --fg: #e5e7eb; --muted: #9ca3af; --line: #2c3039;
  --head: #1d2027; --hover: #1f2430; --accent: #60a5fa; --accentrgb: 96, 165, 250;
  --good: 74, 222, 128; --bad: #f87171; --warn: #fbbf24;
  --badbg: #3b1d1d; --warnbg: #3a2c12; --spark: #60a5fa;
}
body { background: var(--bg); color: var(--fg); margin: 0;
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { padding: 24px 16px 48px; }
.sub, .notes { max-width: 1100px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 16px; }
.notes { margin: 0 0 16px; padding: 0; list-style: none; }
.notes li { padding: 8px 12px; border-radius: 6px; margin-bottom: 6px;
  background: var(--warnbg); }
.wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; text-align: left; white-space: nowrap;
  border-bottom: 1px solid var(--line); }
thead th { background: var(--head); position: sticky; top: 0; }
thead tr.groups th { font-weight: 500; color: var(--muted); font-size: 12px;
  text-transform: uppercase; letter-spacing: .04em; border-bottom: 0; }
thead tr.cols th { cursor: pointer; user-select: none; }
thead tr.cols th:hover { color: var(--accent); }
th.sorted::after { content: " \\25BE"; }
th.sorted.asc::after { content: " \\25B4"; }
.sep { border-left: 2px solid var(--line); }
.pin { position: sticky; left: 0; z-index: 1; background: var(--bg);
  box-shadow: inset -1px 0 0 var(--line); }
thead th.pin { z-index: 2; background: var(--head); }
tbody.run > tr.main:hover td.pin { background: var(--hover); }
tr.details td { position: sticky; left: 0; }
td.num, th.num { text-align: right; }
tbody.run > tr.main { cursor: pointer; }
tbody.run > tr.main:hover { background: var(--hover); }
tbody.run.failed > tr.main td { color: var(--muted); }
td.psnr { font-weight: 600;
  background: rgba(var(--good), calc(var(--share, 0) * 0.35)); }
tbody.run.failed td.psnr { background: none; }
td.blank { color: var(--muted); }
th.diff { color: var(--accent); }
th.diff::before { content: "\\25CF "; font-size: 9px; vertical-align: 2px; }
td.diff { background: rgba(var(--accentrgb), .07); }
td.unused, .key-unused { color: var(--muted); opacity: .55; font-style: italic; }
.key-diff { color: var(--accent); font-weight: 600; }
thead tr.cols th:not(.diff):not(.num) { font-weight: 500; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 999px;
  font-size: 12px; font-weight: 600; }
.badge.exploded, .badge.collapsed { background: var(--badbg); color: var(--bad); }
.badge.stalled { background: var(--warnbg); color: var(--warn); }
.badge.best { background: rgba(var(--good), .18); color: rgb(var(--good)); }
svg.spark { display: block; }
svg.spark polyline { fill: none; stroke: var(--spark); stroke-width: 1.5; }
svg.spark circle { fill: var(--spark); }
tr.details { display: none; }
tbody.run.open tr.details { display: table-row; }
tr.details td { white-space: normal; background: var(--head); padding: 16px; }
.imgs { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; }
.imgs figure { margin: 0; flex: 1 1 320px; max-width: 520px; }
.imgs img { width: 100%; border: 1px solid var(--line); border-radius: 6px;
  background: #fff; }
.imgs figcaption { color: var(--muted); font-size: 12px; }
.path { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
.foot { color: var(--muted); margin-top: 16px; }
.foot code { font-size: 12px; }
"""

HTML_SCRIPT = """
document.querySelectorAll("thead tr.cols th").forEach((th, column) => {
  th.addEventListener("click", () => {
    const table = th.closest("table");
    const ascending = th.classList.contains("sorted") && !th.classList.contains("asc");
    table.querySelectorAll("th.sorted").forEach(h => h.classList.remove("sorted", "asc"));
    th.classList.add("sorted");
    if (ascending) th.classList.add("asc");
    const bodies = [...table.querySelectorAll("tbody.run")];
    const value = body => body.rows[0].cells[column].dataset.sort ?? "";
    bodies.sort((a, b) => {
      const x = value(a), y = value(b);
      if (x === "" || y === "") return (x === "") - (y === "");
      const nx = Number(x), ny = Number(y);
      const order = !isNaN(nx) && !isNaN(ny) ? nx - ny : x.localeCompare(y);
      return ascending ? order : -order;
    });
    bodies.forEach(body => table.appendChild(body));
  });
});
document.querySelectorAll("tbody.run > tr.main").forEach(row => {
  row.addEventListener("click", () => row.parentElement.classList.toggle("open"));
});
"""


def sparkline(values, low, high, width=120, height=28):
    """
    Inline SVG of a run's validation PSNR per epoch, on a scale shared by
    all runs so the curves can be compared.
    """

    if not values:
        return ""

    span = (high - low) or 1.0
    step = (width - 4) / max(len(values) - 1, 1)

    points = [
        (2 + i * step, 2 + (height - 4) * (1 - (value - low) / span))
        for i, value in enumerate(values)
    ]
    best = max(range(len(values)), key=values.__getitem__)

    return (
        f'<svg class="spark" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="validation PSNR per epoch">'
        f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}"/>'
        f'<circle cx="{points[best][0]:.1f}" cy="{points[best][1]:.1f}" r="2.5"/>'
        f"</svg>"
    )


def write_html(path, runs, keys, differing, result_headings, labels, notes):

    escape = html.escape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    good = [run for run in runs if run["results"]["status"] not in FAILED]
    scored = good or runs
    psnr_low = min(run["results"]["test_psnr"] for run in scored)
    psnr_high = max(run["results"]["test_psnr"] for run in scored)
    best_name = max(scored, key=lambda run: run["results"]["test_psnr"])["name"]

    curves = [value for run in runs for value in run["record"]["history"]["val_psnr"]]
    curve_low, curve_high = min(curves), max(curves)

    single_experiment = len({run["name"].split("/")[0] for run in runs}) == 1
    shown_results = [h for h in result_headings if h != "status"]

    # Group row: one cell per config section, then the results
    sections = section_spans(keys)
    first_in_section = set()
    position = 0
    groups = [
        '<th class="pin"></th>',
        f'<th class="sep" colspan="{len(shown_results) + 2}">results</th>',
    ]
    for title, size in sections:
        first_in_section.add(keys[position])
        groups.append(f'<th class="sep" colspan="{size}">{escape(title)}</th>')
        position += size
    head_groups = f'<tr class="groups">{"".join(groups)}</tr>'

    columns = ['<th class="pin">run</th>']
    for i, heading in enumerate(shown_results):
        sep = " sep" if i == 0 else ""
        columns.append(
            f'<th class="num{sep}" title="{escape(heading)}">'
            f"{escape(labels[heading])}</th>"
        )
    columns.append('<th title="validation PSNR per epoch">val PSNR curve</th>')
    columns.append("<th>status</th>")
    for key in keys:
        classes = []
        if key in first_in_section:
            classes.append("sep")
        if key in differing:
            classes.append("diff")
        tip = f"{key} (differs between runs)" if key in differing else key
        columns.append(
            f'<th class="{" ".join(classes)}" title="{escape(tip)}">'
            f"{escape(labels[key])}</th>"
        )

    bodies = []

    for run in runs:
        results = run["results"]
        failed = results["status"] in FAILED

        cells = [
            f'<td class="pin" data-sort="{escape(run["name"])}" title="{escape(run["name"])}">'
            f'{escape(run_label(run["name"], single_experiment))}</td>'
        ]
        setting_cells = []

        for key in keys:
            value, state = setting_value(run, key)
            classes = []
            if key in first_in_section:
                classes.append("sep")
            if key in differing:
                classes.append("diff")
            title = ""
            if state == "unused":
                classes.append("unused")
                title = ' title="not used by this model, or does not affect the result"'
            text = "" if state == "missing" else format_value(value)
            sort = "" if state != "used" or value is None else str(value)
            setting_cells.append(
                f'<td class="{" ".join(classes)}"{title} data-sort="{escape(sort)}">'
                f"{escape(text)}</td>"
            )

        for i, heading in enumerate(shown_results):
            value = results[heading]
            classes = ["num"]
            style = ""
            if i == 0:
                classes.append("sep")
            if heading == "test_psnr":
                classes.append("psnr")
                share = (value - psnr_low) / ((psnr_high - psnr_low) or 1.0)
                style = f' style="--share:{max(0.0, min(1.0, share)):.3f}"'
            cells.append(
                f'<td class="{" ".join(classes)}"{style} data-sort="{value}">'
                f"{escape(format_value(value, heading))}</td>"
            )

        cells.append(
            f'<td data-sort="{max(run["record"]["history"]["val_psnr"])}">'
            f'{sparkline(run["record"]["history"]["val_psnr"], curve_low, curve_high)}</td>'
        )

        badge = results["status"]
        if not badge and run["name"] == best_name:
            badge = "best"
        cells.append(
            f'<td data-sort="{escape(badge)}">'
            + (f'<span class="badge {badge}">{badge}</span>' if badge else "")
            + "</td>"
        )
        cells.extend(setting_cells)

        # Details: the run's own plots
        relative = Path(os.path.relpath(run["dir"], path.parent))
        figures = "".join(
            f'<figure><img loading="lazy" src="{escape((relative / image).as_posix())}" '
            f'alt="{escape(image)}"><figcaption>{escape(image)}</figcaption></figure>'
            for image in RUN_IMAGES
            if (run["dir"] / image).exists()
        )
        details = (
            f'<tr class="details"><td colspan="{len(columns)}">'
            f'<div class="path">results/{escape(run["name"])}</div>'
            f'<div class="imgs">{figures or "No plots saved for this run."}</div>'
            f"</td></tr>"
        )

        bodies.append(
            f'<tbody class="run{" failed" if failed else ""}">'
            f'<tr class="main">{"".join(cells)}</tr>{details}</tbody>'
        )

    note_items = "".join(f"<li>{escape(note)}</li>" for note in notes)

    page = f"""<title>Run comparison</title>
<style>{HTML_STYLE}</style>
<main>
<h1>Run comparison</h1>
<p class="sub">{len(runs)} runs, every setting in config.yaml. Columns whose value
differs between runs are <span class="key-diff">highlighted</span>; greyed-out values
are <span class="key-unused">not used</span> by that run's model. Click a heading to
sort, click a row for its plots. Test numbers are from the best-validation checkpoint;
PSNR shading runs from the worst to the best run that did not fail.</p>
{f'<ul class="notes">{note_items}</ul>' if notes else ""}
<div class="wrap"><table>
<thead>{head_groups}<tr class="cols">{"".join(columns)}</tr></thead>
{"".join(bodies)}
</table></div>
<p class="foot">Generated by <code>scripts/compare_runs.py --html</code>; run it
again to refresh.</p>
</main>
<script>{HTML_SCRIPT}</script>
"""

    path.write_text("<!doctype html>\n<meta charset=\"utf-8\">\n" + page)

    return path


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results", default=str(RESULTS_ROOT),
        help="results folder (default: results/)",
    )
    parser.add_argument(
        "--sort", default="name",
        help="column to sort by, e.g. test_psnr, test_loss, model.depth",
    )
    parser.add_argument(
        "--where", action="append", default=[], metavar="KEY=VALUE",
        help="only runs with this setting, e.g. model.name=unet (repeatable)",
    )
    parser.add_argument(
        "--differing", action="store_true",
        help="only the settings that differ between the listed runs",
    )
    parser.add_argument(
        "--params", action="store_true",
        help="add the parameter count of each model (needs torch)",
    )
    parser.add_argument(
        "--html", nargs="?", const=str(DEFAULT_HTML), metavar="PATH",
        help="also write an HTML page (default: results/comparison.html)",
    )
    parser.add_argument(
        "--csv", metavar="PATH",
        help="also write the table to a CSV file",
    )
    parser.add_argument(
        "--no-colour", action="store_true",
        help="plain terminal output",
    )
    args = parser.parse_args()

    finished, unfinished = load_runs(Path(args.results))
    runs = [run for run in finished if matches(run, args.where)]

    if not runs:
        raise SystemExit("No finished runs match.")

    # Every setting any listed run has, in the order of config.yaml
    keys = config_order({key for run in runs for key in run["all"]})

    # A setting differs when the runs that use it do not all agree
    differing = {
        key for key in keys
        if len({repr(run["settings"].get(key)) for run in runs}) > 1
    }

    if args.differing:
        keys = [key for key in keys if key in differing]
        keys.sort(key=lambda key: key != "model.name")

    for run in runs:
        run["results"] = {
            heading: function(run["record"])
            for heading, function in RESULT_COLUMNS
        }
        run["results"]["status"] = status(run["record"])
        if args.params:
            run["results"]["params"] = parameter_count(run["config"])

    result_headings = [heading for heading, _ in RESULT_COLUMNS]

    # 'ran' only matters when some run stopped before its configured epochs
    if all(
        run["results"]["epochs_run"] == run["all"].get("training.epochs")
        for run in runs
    ):
        result_headings.remove("epochs_run")

    if args.params:
        result_headings.append("params")
    result_headings.append("status")

    def sort_key(run):
        if args.sort == "name":
            value = run["name"]
        elif args.sort in run["results"]:
            value = run["results"][args.sort]
        else:
            value = run["settings"].get(args.sort)
        # Missing values last; numbers and text sorted separately
        return (value is None, isinstance(value, str), value if value is not None else 0)

    descending = args.sort in ("test_psnr", "val_psnr")
    runs.sort(key=sort_key, reverse=descending)
    if descending:
        # Keep missing values last after reversing
        runs.sort(key=lambda run: sort_key(run)[0])

    labels = {key: short_label(key, keys) for key in keys + result_headings}

    # Best run: highest test PSNR among the runs that did not fail
    good = [run for run in runs if run["results"]["status"] not in FAILED]
    best = max(good or runs, key=lambda run: run["results"]["test_psnr"])

    single_experiment = len({run["name"].split("/")[0] for run in runs}) == 1

    header = ["run"] + [labels[h] for h in result_headings] + [labels[key] for key in keys]
    header_styles = (
        ["bold"] * (1 + len(result_headings))
        + ["yellow" if key in differing else "dim" for key in keys]
    )
    rows = []
    styles = []

    for run in runs:
        marker = "best" if run is best else run["results"]["status"]
        setting_cells = []
        setting_styles = []

        for key in keys:
            value, state = setting_value(run, key)
            if state == "missing":
                setting_cells.append("")
            elif state == "unused":
                setting_cells.append(f"({format_value(value)})")
            else:
                setting_cells.append(format_value(value))
            setting_styles.append("dim" if state == "unused" else None)

        row = (
            [run_label(run["name"], single_experiment)]
            + [
                marker if heading == "status"
                else format_value(run["results"][heading], heading)
                for heading in result_headings
            ]
            + setting_cells
        )

        if run is best:
            style = ["green"] * len(row)
        elif run["results"]["status"] in FAILED:
            style = ["dim"] * len(row)
            style[len(result_headings)] = (
                "red" if run["results"]["status"] != "stalled" else "yellow"
            )
        else:
            style = [None] * (1 + len(result_headings)) + setting_styles

        rows.append(row)
        styles.append(style)

    groups = [("", 1), ("results", len(result_headings))] + section_spans(keys)

    colour = sys.stdout.isatty() and not args.no_colour and "NO_COLOR" not in os.environ
    print_table(header, rows, groups, styles, colour, header_styles)

    print(
        "\n" + ("Highlighted headings" if colour else "Settings")
        + " that differ between runs: "
        + (", ".join(labels[key] for key in keys if key in differing) or "none")
        + ". Values in (parentheses) are not used by that run's model."
    )

    notes = []

    losses = {run["settings"].get("training.loss") for run in runs}
    if len(losses) > 1:
        notes.append(
            "These runs use different loss functions ("
            + ", ".join(sorted(map(str, losses)))
            + "), so test_loss is not comparable between them; compare test PSNR."
        )

    if unfinished:
        notes.append(
            "No results.json yet (still running, crashed or stopped): " + ", ".join(
                f"{name} ({config['model']['name']}, depth {config['model']['depth']})"
                for name, config in unfinished
            )
        )

    for note in notes:
        print("\nNote: " + note)

    if args.csv:
        with open(args.csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["run"] + keys + result_headings)
            for run in runs:
                writer.writerow(
                    [run["name"]]
                    + [run["all"].get(key) for key in keys]
                    + [run["results"][heading] for heading in result_headings]
                )
        print(f"\nWrote {args.csv}")

    if args.html:
        page = write_html(
            args.html, runs, keys, differing, result_headings, labels, notes,
        )
        print(f"\nWrote {page}\nOpen it in a browser: file://{page.resolve()}")


if __name__ == "__main__":
    main()

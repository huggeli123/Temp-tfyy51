"""
Print a table of proxy trials from one or more results.jsonl files.

    python scripts/hpo/summarize.py docs/variational/hpo/stage1.jsonl [--sort]
"""

import json
import sys

KEYS = [
    "kl_weight", "initial_rho", "prior_standard_deviation",
    "rho_learning_rate", "learning_rate",
]


def load(paths):
    rows = []
    for path in paths:
        for line in open(path):
            rows.append(json.loads(line))
    return rows


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = load(paths)
    if "--sort" in sys.argv:
        rows.sort(key=lambda r: -(r.get("val_psnr") or 0))
    print(f"{'tag':34s} {'psnr':>6s} {'half':>6s} {'samp%':>6s} "
          f"{'std':>7s} {'std/err':>7s} {'sig_med':>8s} {'sig_p95':>8s} "
          f"{'min':>5s}  " + " ".join(f"{k[:10]:>10s}" for k in KEYS))
    for r in rows:
        p = r["params"]
        if "error" in r:
            print(f"{p['tag']:34s} ERROR {r['error'][:80]}")
            continue
        u = r.get("uncertainty") or {}
        s = r.get("sigma") or {}
        half = r["curve"][0]["val_psnr"] if r.get("curve") else None
        print(
            f"{p['tag']:34s} {fmt(r.get('val_psnr')):>6s} {fmt(half):>6s} "
            f"{fmt(100 * r['sampling_cost'] if 'sampling_cost' in r else None):>6s} "
            f"{fmt(u.get('mean_epistemic_std')):>7s} {fmt(u.get('std_over_error')):>7s} "
            f"{fmt(s.get('median')):>8s} {fmt(s.get('p95')):>8s} "
            f"{r['seconds'] / 60:5.1f}  "
            + " ".join(f"{fmt(p.get(k)):>10s}" for k in KEYS)
        )


if __name__ == "__main__":
    main()

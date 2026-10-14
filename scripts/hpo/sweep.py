"""
Run a list of proxy trials in parallel worker processes.

    python scripts/hpo/sweep.py <plan.json> <results.jsonl> [--workers 4]

<plan.json> is a list of override dicts for proxy_train.run_trial, each
with a unique "tag". Trials whose tag is already in <results.jsonl> are
skipped, so an interrupted sweep can be restarted with the same command.
Each trial runs in a fresh single-threaded process (4 x 1 thread gave
about 30% more throughput than 1 x 4 threads on the 4-core machine).
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _worker(args):
    overrides, log = args
    import proxy_train
    t = time.time()
    try:
        r = proxy_train.run_trial(overrides, log)
        return overrides["tag"], r.get("val_psnr"), time.time() - t
    except Exception as e:  # keep the sweep going, record the failure
        with open(log, "a") as f:
            f.write(json.dumps({"params": overrides, "error": repr(e)}) + "\n")
        return overrides["tag"], None, time.time() - t


def done_tags(log):
    tags = set()
    if Path(log).exists():
        for line in open(log):
            tags.add(json.loads(line)["params"].get("tag"))
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("log")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    plan = json.load(open(a.plan))
    finished = done_tags(a.log)
    todo = [t for t in plan if t["tag"] not in finished]
    print(f"{len(todo)} of {len(plan)} trials to run", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for tag, psnr, sec in pool.imap_unordered(
            _worker, [(t, a.log) for t in todo]
        ):
            print(f"{tag:40s} val_psnr={psnr} ({sec / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()

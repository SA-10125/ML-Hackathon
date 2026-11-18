"""Grid-search HMM decoder parameters on TRAIN trips only.

Objective: micro-averaged point-level Top-1 accuracy (filled predictions)
against the route-aligned reference labels.

Usage: python scripts/tune_hmm.py --max-points 6000 --workers 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.data import list_trips, DATA_ROOT  # noqa: E402
from pathsnap.hmm import HMMParams, fill_unmatched  # noqa: E402
from pathsnap import pipeline  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
GRID = {"sigma": [3.0, 5.0, 8.0, 12.0], "beta": [2.0, 5.0, 10.0, 20.0], "heading_w": [0.0, 2.0]}

_ctxs = None


def _init(trip_ids):
    global _ctxs
    _ctxs = [pipeline.prepare(t, "train", HMMParams()) for t in trip_ids]


def _score(combo):
    p = HMMParams(**combo)
    correct = total = 0
    t0 = time.perf_counter()
    for ctx in _ctxs:
        res, _ = pipeline.run_hmm(ctx, p)
        pred = fill_unmatched(res.arc)
        m = ctx.labels.arc >= 0
        correct += int((pred[m] == ctx.labels.arc[m]).sum())
        total += int(m.sum())
    return {**combo, "train_top1_acc": correct / total, "n_points": total, "seconds": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-points", type=int, default=6000, help="skip very long train trips to keep tuning fast")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    trips = []
    for t in list_trips("train"):
        n = sum(1 for _ in open(DATA_ROOT / "train" / t / f"{t}.track"))
        if n <= args.max_points:
            trips.append(t)
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"{len(trips)} train trips, {len(combos)} combos", flush=True)

    with Pool(args.workers, initializer=_init, initargs=(trips,)) as pool:
        rows = []
        for r in pool.imap_unordered(_score, combos):
            rows.append(r)
            print(f"sigma={r['sigma']:5.1f} beta={r['beta']:5.1f} heading_w={r['heading_w']:.1f} "
                  f"-> acc={r['train_top1_acc']:.4f}", flush=True)

    df = pd.DataFrame(rows).sort_values("train_top1_acc", ascending=False)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "hmm_tuning_train.csv", index=False)
    best = df.iloc[0][list(GRID)].to_dict()
    (RESULTS / "hmm_best_params.json").write_text(json.dumps({**HMMParams().dict(), **best}, indent=2))
    print("best:", best, "acc:", df.iloc[0]["train_top1_acc"])


if __name__ == "__main__":
    main()

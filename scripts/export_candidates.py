"""Export the shared candidate table (features + labels) for classifier training.

One row per (GPS point, candidate arc). The HMM uses exactly these
candidates, so classifier probabilities must be produced for these rows.
Join key: (trip, point_idx, arc_id).

Usage: python scripts/export_candidates.py --partition train
Output: data_processed/candidates_<partition>.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import list_trips  # noqa: E402
from pathsnap.hmm import HMMParams  # noqa: E402
from pathsnap import pipeline  # noqa: E402


def trip_table(ctx) -> pd.DataFrame:
    c, tr = ctx.cands, ctx.trip
    pid = c.point_index()
    n_cand = np.diff(c.ptr)
    rank = np.arange(len(pid)) - c.ptr[pid]
    min_d = c.dist[c.ptr[:-1][pid]]

    step = np.hypot(*np.diff(tr.xy, axis=0).T)
    dt = np.maximum(np.diff(tr.times), 1e-6)
    speed = np.concatenate([[0.0], step / dt])

    # distance to the arc averaged over a +-2 point window (smooths single-point noise)
    a = tr.node_xy[tr.arcs[c.arc, 0]]
    b = tr.node_xy[tr.arcs[c.arc, 1]]
    win = []
    from pathsnap.geometry import project_points
    for k in (-2, -1, 1, 2):
        q = np.clip(pid + k, 0, len(tr.xy) - 1)
        win.append(project_points(tr.xy[q], a, b)[0])
    win_mean = (np.sum(win, axis=0) + c.dist) / 5

    df = pd.DataFrame({
        "trip": tr.trip_id,
        "point_idx": pid,
        "time": tr.times[pid],
        "arc_id": c.arc,
        "dist": c.dist,
        "dist_rank": rank,
        "dist_minus_min": c.dist - min_d,
        "dist_window_mean": win_mean,
        "n_cand": n_cand[pid],
        "offset": c.offset,
        "t_along": c.t,
        "at_arc_end": ((c.t <= 0.0) | (c.t >= 1.0)).astype(np.int8),
        "arc_len": ctx.arc_len[c.arc],
        "heading_cos": c.heading_cos,
        "speed": speed[pid],
    })
    if ctx.labels is not None:
        lab = ctx.labels.arc[pid]
        df["has_label"] = (lab >= 0).astype(np.int8)
        df["label"] = (c.arc == lab).astype(np.int8)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="train", choices=["train", "student_test"])
    args = ap.parse_args()
    p = HMMParams()
    parts = []
    for tid in list_trips(args.partition):
        parts.append(trip_table(pipeline.prepare(tid, args.partition, p)))
        print(tid, len(parts[-1]), flush=True)
    df = pd.concat(parts, ignore_index=True)
    out = ROOT / "data_processed" / f"candidates_{args.partition}.parquet"
    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} rows={len(df)} positives={int(df['label'].sum())} params radius={p.radius} k={p.k}")


if __name__ == "__main__":
    main()

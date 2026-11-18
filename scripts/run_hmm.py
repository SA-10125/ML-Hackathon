"""Run nearest-arc baseline vs HMM+Viterbi on a partition and log metrics.

Usage:
  python scripts/run_hmm.py --partition student_test
  python scripts/run_hmm.py --partition train --sigma 4 --beta 3 --tag tuned
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.data import list_trips  # noqa: E402
from pathsnap.hmm import HMMParams, classifier_log_emission  # noqa: E402
from pathsnap import pipeline, metrics  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test", choices=["train", "student_test"])
    ap.add_argument("--trips", nargs="*", help="subset of trip ids")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--probs", help="parquet/csv with trip, point_idx, arc_id, prob from a classifier")
    for f, v in HMMParams().dict().items():
        ap.add_argument(f"--{f}", type=type(v), default=v)
    args = ap.parse_args()
    p = HMMParams(**{f: getattr(args, f) for f in HMMParams().dict()})

    probs = None
    if args.probs:
        probs = pd.read_parquet(args.probs) if args.probs.endswith(".parquet") else pd.read_csv(args.probs, dtype={"trip": str})
        probs["trip"] = probs["trip"].astype(str).str.zfill(8)
    methods = ["near", "hmm"] + (["clf", "clf_hmm"] if probs is not None else [])

    rows = []
    for tid in args.trips or list_trips(args.partition):
        t0 = time.perf_counter()
        ctx = pipeline.prepare(tid, args.partition, p)
        prep_s = time.perf_counter() - t0
        miss = metrics.candidate_miss_rate(ctx.cands, ctx.labels.arc)

        near = pipeline.evaluate(ctx, pipeline.nearest_arc(ctx.cands))
        res, dec_s = pipeline.run_hmm(ctx, p)
        hmm = pipeline.evaluate(ctx, res.arc)

        row = {"trip": tid, "n_points": hmm["n_points"], "n_labelled": hmm["n_labelled"],
               "cand_miss": miss, "unmatched": hmm["unmatched_frac"],
               "prep_s": prep_s, "decode_s": dec_s, "breaks": res.n_breaks, "gap_splits": res.n_gap_splits}
        evals = {"near": near, "hmm": hmm}
        if probs is not None:
            pr = pipeline.probs_for_trip(ctx, probs)
            c = ctx.cands
            clf_pred = np.full(c.n_points, -1)
            for i in range(c.n_points):
                sl = c.of(i)
                if sl.start < sl.stop:
                    clf_pred[i] = c.arc[sl][np.argmax(pr[sl])]
            evals["clf"] = pipeline.evaluate(ctx, clf_pred)
            res_c, _ = pipeline.run_hmm(ctx, p, classifier_log_emission(pr, p))
            evals["clf_hmm"] = pipeline.evaluate(ctx, res_c.arc)
        for name, m in evals.items():
            for k in ["top1_acc", "continuity", "route_f1"]:
                row[f"{name}_{k}"] = m[k]
        rows.append(row)
        print(f"{tid}: n={row['n_points']:5d} miss={miss:.3f} breaks={res.n_breaks} | " +
              " | ".join(f"{n} acc={m['top1_acc']:.3f} cont={m['continuity']:.3f}" for n, m in evals.items()) +
              f" | {prep_s:.1f}s + {dec_s:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"hmm_{args.partition}_{args.tag}.csv"
    df.to_csv(out, index=False)

    w = df["n_labelled"]
    summary = {
        "partition": args.partition, "tag": args.tag, "params": p.dict(), "trips": len(df),
        "cand_miss_rate": float(np.average(df["cand_miss"], weights=w)),
    }
    for name in methods:
        summary[f"{name}_top1_acc_micro"] = float(np.average(df[f"{name}_top1_acc"], weights=w))
        summary[f"{name}_top1_acc_trip_mean"] = float(df[f"{name}_top1_acc"].mean())
        summary[f"{name}_continuity_mean"] = float(df[f"{name}_continuity"].mean())
        summary[f"{name}_route_f1_mean"] = float(df[f"{name}_route_f1"].mean())
    summary["decode_s_total"] = float(df["decode_s"].sum())
    (RESULTS / f"hmm_{args.partition}_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

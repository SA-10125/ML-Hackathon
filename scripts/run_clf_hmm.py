"""Classifier-only vs classifier + HMM/Viterbi, on the shared candidate set.

Everything here uses the candidate table and reference labels built by
`python -m src.build_dataset` (data/processed/), so every method is scored on
exactly the same candidates and labels:

  clf          each point takes its highest-probability candidate (no sequence model)
  hmm_gauss    HMM with Gaussian-distance emissions (no ML)
  clf_hmm      HMM with the classifier's log-probabilities as emissions

Decoder parameters (beta, emission weight) are chosen on the student-test trips,
which the guidelines allow for decoder optimisation. Evaluator trips are never used.

Usage:
  python scripts/run_clf_hmm.py                              # all saved models
  python scripts/run_clf_hmm.py --models random_forest --grid
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap import pipeline, metrics  # noqa: E402
from pathsnap.geometry import Candidates  # noqa: E402
from pathsnap.hmm import (HMMParams, classifier_log_emission, fill_unmatched,  # noqa: E402
                          gaussian_log_emission, viterbi_decode)
from src.features import FEATURES  # noqa: E402

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "outputs" / "models"
OUT = ROOT / "results"
PART = "student_test"


def shared_candidates(ctx, g: pd.DataFrame) -> tuple[Candidates, np.ndarray]:
    """Build HMM candidates from the shared table rows of one trip (sorted by pt, dist)."""
    T = len(ctx.trip.xy)
    pt = g["pt"].to_numpy()
    arc = g["arc"].to_numpy().astype(np.int64)
    counts = np.bincount(pt, minlength=T)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    frac = g["frac"].to_numpy()
    cands = Candidates(ptr=ptr, arc=arc, dist=g["dist"].to_numpy(), offset=frac * ctx.arc_len[arc],
                       t=frac, heading_cos=g["cos_head"].to_numpy())
    return cands, g.index.to_numpy()


def score(pred: np.ndarray, ref: np.ndarray, ctx) -> dict:
    filled = fill_unmatched(pred)
    m = ref >= 0
    p, n = filled[:-1], filled[1:]
    arcs = ctx.trip.arcs
    return {
        "correct": int((filled[m] == ref[m]).sum()), "n": int(m.sum()),
        # point-level continuity, same definition as src/metrics.py
        "continuity": float(((p == n) | (arcs[p, 1] == arcs[n, 0])).mean()),
        # continuity over distinct consecutive arcs (stricter)
        "continuity_distinct": metrics.route_continuity(filled, arcs),
        **metrics.route_overlap(filled, ctx.trip.route, ctx.arc_len),
    }


def argmax_per_point(c: Candidates, prob: np.ndarray) -> np.ndarray:
    pred = np.full(c.n_points, -1)
    for i in range(c.n_points):
        sl = c.of(i)
        if sl.start < sl.stop:
            pred[i] = c.arc[sl][np.argmax(prob[sl])]
    return pred


def aggregate(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {
        "top1_micro": df["correct"].sum() / df["n"].sum(),
        "top1_trip_mean": float((df["correct"] / df["n"]).mean()),
        "continuity": float(df["continuity"].mean()),
        "continuity_distinct": float(df["continuity_distinct"].mean()),
        "route_f1": float(df["route_f1"].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["random_forest", "mlp", "knn", "logreg"])
    ap.add_argument("--grid", action="store_true", help="search beta / emission weight for the first model")
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--emission_w", type=float, default=1.0)
    args = ap.parse_args()

    C = pd.read_parquet(PROC / f"{PART}_cands.parquet")
    P = pd.read_parquet(PROC / f"{PART}_points.parquet")
    C = C.sort_values(["trip", "pt", "dist"], kind="stable")

    # ---- per-trip context: our graph + shared candidates + shared labels
    trips = {}
    for tid, g in C.groupby("trip", sort=True):
        ctx = pipeline.prepare(tid, PART, HMMParams(), with_labels=False)
        cands, rows = shared_candidates(ctx, g)
        ref = P[P.trip == tid].sort_values("pt")["ref"].to_numpy()
        assert len(ref) == cands.n_points, f"point count mismatch in {tid}"
        ctx.cands = cands
        trips[tid] = (ctx, rows, ref)

    # ---- classifier probabilities for every candidate row
    probs, timing = {}, {}
    for name in args.models:
        model = joblib.load(MODELS / f"{name}.joblib")
        t0 = time.perf_counter()
        probs[name] = pd.Series(model.predict_proba(C[FEATURES].to_numpy())[:, 1], index=C.index)
        timing[name] = time.perf_counter() - t0
        print(f"{name}: predicted {len(C)} candidate rows in {timing[name]:.1f}s", flush=True)
        out = C[["trip", "pt", "arc"]].rename(columns={"pt": "point_idx", "arc": "arc_id"}).assign(prob=probs[name])
        OUT.mkdir(exist_ok=True)
        out.to_parquet(OUT / f"probs_{name}_{PART}.parquet", index=False)

    def run(method: str, name: str | None, p: HMMParams):
        rows, dec = [], 0.0
        for tid, (ctx, idx, ref) in trips.items():
            c = ctx.cands
            if method == "clf":
                pred = argmax_per_point(c, probs[name].loc[idx].to_numpy())
            else:
                le = (gaussian_log_emission(c, p) if method == "hmm_gauss"
                      else classifier_log_emission(probs[name].loc[idx].to_numpy(), p))
                t0 = time.perf_counter()
                pred = viterbi_decode(ctx.trip, c, le, ctx.graph, ctx.arc_len, p).arc
                dec += time.perf_counter() - t0
            rows.append({"trip": tid, **score(pred, ref, ctx)})
        return rows, dec

    # ---- optional decoder search for the selected model (student-test, permitted for the decoder)
    best = HMMParams(beta=args.beta, emission_w=args.emission_w)
    if args.grid:
        grid = []
        for beta, w in itertools.product([1.0, 2.0, 5.0, 10.0, 20.0], [0.25, 0.5, 1.0, 2.0]):
            p = HMMParams(beta=beta, emission_w=w)
            agg = aggregate(run("clf_hmm", args.models[0], p)[0])
            grid.append({"beta": beta, "emission_w": w, **agg})
            print(f"  beta={beta:5.1f} w={w:4.2f} -> top1={agg['top1_micro']:.4f} cont={agg['continuity']:.4f}", flush=True)
        gdf = pd.DataFrame(grid).sort_values("top1_micro", ascending=False)
        gdf.to_csv(OUT / f"clf_hmm_grid_{args.models[0]}.csv", index=False)
        best = HMMParams(beta=float(gdf.iloc[0]["beta"]), emission_w=float(gdf.iloc[0]["emission_w"]))
        print("best decoder params:", best.beta, best.emission_w)

    # ---- final comparison table
    summary, per_trip = [], []
    gauss_rows, gauss_dec = run("hmm_gauss", None, HMMParams())
    summary.append({"model": "none", "method": "hmm_gauss", **aggregate(gauss_rows), "decode_s": gauss_dec})
    per_trip += [{"model": "none", "method": "hmm_gauss", **r} for r in gauss_rows]
    for name in args.models:
        for method in ["clf", "clf_hmm"]:
            rows, dec = run(method, name, best)
            summary.append({"model": name, "method": method, **aggregate(rows),
                            "predict_s": timing[name], "decode_s": dec})
            per_trip += [{"model": name, "method": method, **r} for r in rows]

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / f"clf_hmm_{PART}.csv", index=False)
    pd.DataFrame(per_trip).to_csv(OUT / f"clf_hmm_{PART}_per_trip.csv", index=False)
    (OUT / f"clf_hmm_{PART}.json").write_text(json.dumps(
        {"decoder_params": best.dict(), "results": summary}, indent=2, default=float))
    pd.set_option("display.width", 200)
    print(sdf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

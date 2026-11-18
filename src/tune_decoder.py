"""Tune the HMM decoder for a classifier without overfitting.

  python -m src.tune_decoder --model random_forest

Protocol (no student-test or evaluator data is used for any choice):
  1. Out-of-fold probabilities on ALL 80 train trips: 5-fold GroupKFold by trip, so every
     train trip is scored by a classifier that never saw it (same model spec as src/models.py).
     Cached in data/processed/oof_<model>.parquet.
  2. Grid search over decoder settings on those 80 trips:
        beta            transition tolerance (m)
        switch_penalty  log-penalty for changing arc between steps
        min_move        stationary pooling radius (m); 0 = off
     Objective: point-level Top-1 on reliably labelled points (micro over all train points).
     Fold-wise scores are recorded to check the choice is stable, not a lucky fold.
  3. The chosen setting is evaluated ONCE on the 10 student-test trips with the saved model
     (fitted on all 80 train trips), next to the previous default (beta=3, no extras).
Outputs: outputs/hmm/<model>_decoder_grid_train_oof.csv, outputs/hmm/<model>_decoder_tuned.json
"""
import argparse
import itertools
import json
import time
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .data_io import load_trip
from .features import FEATURES
from .hmm import FLOOR, RoadGraph, viterbi
from .metrics import continuity
from .models import MODELS

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "hmm"
SEED = 42

GRID = {"beta": [2.0, 3.0, 5.0, 8.0],
        "switch_penalty": [0.0, 0.5, 1.0, 2.0, 4.0],
        "min_move": [0.0, 4.0, 8.0, 12.0]}
GAP_S = 60.0
BASELINE = {"beta": 3.0, "switch_penalty": 0.0, "min_move": 0.0}


def oof_probs(model_name, C, n_folds=5):
    path = PROC / f"oof_{model_name}.parquet"
    if path.exists():
        print(f"using cached {path.name}")
        return pd.read_parquet(path), json.loads((PROC / f"oof_{model_name}_folds.json").read_text())
    spec = MODELS[model_name]
    prob = np.full(len(C), np.nan)
    fold_of = {}
    trips = C.trip.values
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(C, groups=trips)):
        t0 = time.time()
        fit = C.iloc[tr]
        fit = fit[fit.ref_valid == 1]
        idx = np.arange(len(fit))
        rows = spec.get("max_train_rows")
        if rows and len(idx) > rows:
            idx = np.sort(np.random.default_rng(SEED).choice(idx, rows, replace=False))
        m = spec["make"]()
        m.fit(fit[FEATURES].values[idx], fit.label.values[idx])
        prob[va] = m.predict_proba(C.iloc[va][FEATURES].values)[:, 1]
        for t in np.unique(trips[va]):
            fold_of[t] = k
        print(f"  OOF fold {k}: {len(np.unique(trips[va]))} trips, {time.time() - t0:.0f}s", flush=True)
    out = C[["trip", "pt", "arc"]].assign(prob=prob)
    out.to_parquet(path, index=False)
    (PROC / f"oof_{model_name}_folds.json").write_text(json.dumps(fold_of))
    return out, fold_of


# ---------------------------------------------------------------- per-trip worker
_W = {}


def _init(part, cands_path, probs_path, points_path):
    _W["part"] = part
    _W["C"] = pd.read_parquet(cands_path, columns=["trip", "pt", "arc", "frac"])
    _W["prob"] = pd.read_parquet(probs_path)["prob"].values if probs_path else None
    _W["P"] = pd.read_parquet(points_path, columns=["trip", "pt", "ref"])


def _trip_grid(args):
    tid, combos = args
    C, P = _W["C"], _W["P"]
    mask = (C.trip == tid).values
    c = C[mask].assign(prob=_W["prob"][mask]).sort_values(["pt", "arc"]).reset_index(drop=True)
    ref = P[P.trip == tid].sort_values("pt").ref.values
    trip = load_trip(tid, _W["part"])
    g = RoadGraph(trip)
    le = np.log(np.clip(c.prob.values, FLOOR, 1))
    ok = ref >= 0
    out = []
    for combo in combos:
        pred, nb = viterbi(trip, c, le, gap_s=GAP_S, graph=g, **combo)
        has = pred >= 0
        out.append({**combo, "trip": tid, "correct": int((pred[ok] == ref[ok]).sum()), "n": int(ok.sum()),
                    "continuity": continuity(pred[has], trip.arcs), "breaks": nb})
    return out


def run_grid(part, trips, combos, cands_path, probs_path, points_path, workers):
    tasks = [(t, combos) for t in trips]
    rows = []
    with Pool(workers, initializer=_init, initargs=(part, cands_path, probs_path, points_path)) as pool:
        for i, r in enumerate(pool.imap_unordered(_trip_grid, tasks), 1):
            rows += r
            print(f"  decoded trip {i}/{len(trips)}", end="\r", flush=True)
    print()
    return pd.DataFrame(rows)


def summarise(df, keys):
    g = df.groupby(keys)
    s = g.apply(lambda x: pd.Series({"top1": x.correct.sum() / x.n.sum(),
                                     "continuity": x.continuity.mean(),
                                     "breaks": x.breaks.sum()}), include_groups=False)
    return s.reset_index()


def main(model_name, workers, grid_override=None, tag=""):
    global GRID
    if grid_override:
        GRID = grid_override
    OUT.mkdir(parents=True, exist_ok=True)
    Ctr = pd.read_parquet(PROC / "train_cands.parquet")
    oof, fold_of = oof_probs(model_name, Ctr)
    probs_path = PROC / f"oof_{model_name}.parquet"

    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    trips = sorted(Ctr.trip.unique())
    print(f"grid: {len(combos)} decoder settings x {len(trips)} train trips (OOF probabilities)")
    t0 = time.time()
    raw = run_grid("train", trips, combos, PROC / "train_cands.parquet", probs_path,
                   PROC / "train_points.parquet", workers)
    raw["fold"] = raw.trip.map(fold_of)
    keys = list(GRID)
    grid = summarise(raw, keys)
    per_fold = summarise(raw, keys + ["fold"]).pivot_table(index=keys, columns="fold", values="top1")
    per_fold.columns = [f"top1_fold{c}" for c in per_fold.columns]
    grid = grid.merge(per_fold.reset_index(), on=keys)
    grid["top1_fold_std"] = grid.filter(like="top1_fold").std(axis=1)
    grid = grid.sort_values(["top1", "continuity"], ascending=False)
    grid.to_csv(OUT / f"{model_name}_decoder_grid_train_oof{tag}.csv", index=False)
    print(f"grid done in {time.time() - t0:.0f}s")
    print(grid.head(10).round(4).to_string(index=False))

    best = {k: float(grid.iloc[0][k]) for k in keys}
    base_row = grid[(grid[keys] == pd.Series(BASELINE)).all(axis=1)].iloc[0]
    print("best on train OOF:", best, "top1=%.4f" % grid.iloc[0].top1,
          "| baseline top1=%.4f" % base_row.top1)

    # ---- single final check on student-test with the saved full-train model
    Cte = pd.read_parquet(PROC / "student_test_cands.parquet")
    te_probs = PROC / f"student_test_probs_{model_name}.parquet"
    if model_name in MODELS:  # single-stage model: score from the base features
        model = joblib.load(ROOT / "outputs" / "models" / f"{model_name}.joblib")
        Cte[["trip", "pt", "arc"]].assign(prob=model.predict_proba(Cte[FEATURES].values)[:, 1]).to_parquet(te_probs, index=False)
    elif not te_probs.exists():  # multi-stage models write their own test probabilities
        raise FileNotFoundError(f"{te_probs} missing: run the training script for {model_name} first")
    te = run_grid("student_test", sorted(Cte.trip.unique()), [BASELINE, best],
                  PROC / "student_test_cands.parquet", te_probs, PROC / "student_test_points.parquet", workers)
    te["setting"] = np.where((te[keys] == pd.Series(best)).all(axis=1), "tuned", "baseline")
    if best == BASELINE:
        te["setting"] = "tuned"
    te.to_csv(OUT / f"{model_name}_decoder_tuned_student_test_per_trip{tag}.csv", index=False)
    te_sum = summarise(te, ["setting"]).set_index("setting")

    # classifier-only reference on both partitions
    def cls_top1(C, probs, P):
        d = C[["trip", "pt", "arc"]].assign(s=probs)
        b = d.loc[d.groupby(["trip", "pt"]).s.idxmax(), ["trip", "pt", "arc"]]
        m = P[P.ref >= 0].merge(b, on=["trip", "pt"], how="left")
        return float((m.arc == m.ref).mean())

    Ptr = pd.read_parquet(PROC / "train_points.parquet")
    Pte = pd.read_parquet(PROC / "student_test_points.parquet")
    summary = {
        "model": model_name, "gap_s": GAP_S, "grid": GRID, "selection": "max Top-1 on 80 train trips, out-of-fold",
        "best": best, "baseline": BASELINE,
        "train_oof": {"classifier_only_top1": cls_top1(Ctr, oof.prob.values, Ptr),
                      "baseline_top1": float(base_row.top1), "tuned_top1": float(grid.iloc[0].top1),
                      "tuned_top1_fold_std": float(grid.iloc[0].top1_fold_std)},
        "student_test": {"classifier_only_top1": cls_top1(Cte, pd.read_parquet(te_probs).prob.values, Pte),
                         **{f"{k}_top1": float(v) for k, v in te_sum.top1.items()},
                         **{f"{k}_continuity": float(v) for k, v in te_sum.continuity.items()}},
    }
    (OUT / f"{model_name}_decoder_tuned{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="random_forest")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--grid", help='JSON dict overriding GRID, e.g. {"beta":[1,2],"switch_penalty":[4,8],"min_move":[0]}')
    ap.add_argument("--tag", default="", help="suffix for output files")
    a = ap.parse_args()
    main(a.model, a.workers, json.loads(a.grid) if a.grid else None, a.tag)

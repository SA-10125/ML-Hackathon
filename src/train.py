"""Train and evaluate one candidate classifier.

  python -m src.train --model logreg

Protocol
  * Fit / cross-validate on the 80 training trips only, with GroupKFold by trip
    (no trip appears in both a training and a validation fold -> no leakage).
  * Rows whose reference label is unreliable (ref == -1) are excluded.
  * Student-test trips are used only for the development-stage comparison.
  * Results, configs, per-fold scores and the fitted model are saved in outputs/.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .data_io import RAW
from .features import FEATURES
from .metrics import continuity, row_metrics, top1_accuracy, top1_predictions
from .models import MODELS

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
SEED = 42


def load(part):
    C = pd.read_parquet(PROC / f"{part}_cands.parquet")
    P = pd.read_parquet(PROC / f"{part}_points.parquet")
    return C[C.ref_valid == 1].reset_index(drop=True), P


def route_continuity(points, pred, part):
    vals = []
    for tid, g in pred.merge(points[["trip", "pt"]], on=["trip", "pt"]).groupby("trip"):
        arcs = np.loadtxt(RAW / part / tid / f"{tid}.arcs", dtype=np.int64, ndmin=2)
        vals.append(continuity(g.sort_values("pt").pred.values.astype(np.int64), arcs))
    return float(np.mean(vals))


def fit_model(spec, X, y, groups, idx):
    """Fit a fresh model on rows idx, honouring subsampling / group options."""
    rows = spec.get("max_train_rows")
    if rows and len(idx) > rows:
        idx = np.sort(np.random.default_rng(SEED).choice(idx, rows, replace=False))
    m = spec["make"]()
    if spec.get("needs_groups"):
        m.fit(X[idx], y[idx], groups=groups[idx])
    else:
        m.fit(X[idx], y[idx])
    return m


def main(name, n_folds=5):
    spec = MODELS[name]
    Ctr, Ptr = load("train")
    Cte, Pte = load("student_test")
    X, y, groups = Ctr[FEATURES].values, Ctr.label.values, Ctr.trip.values

    # ---- group-aware cross-validation on the 80 training trips ----
    folds = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        m = fit_model(spec, X, y, groups, tr)
        p = m.predict_proba(X[va])[:, 1]
        r = row_metrics(y[va], p)
        r["top1"] = top1_accuracy(Ptr[Ptr.trip.isin(set(groups[va]))],
                                  top1_predictions(Ctr.iloc[va], p))
        r["fold"] = k
        folds.append(r)
        print(f"  fold {k}: macroF1={r['macro_f1']:.4f} top1={r['top1']:.4f} logloss={r['log_loss']:.4f}")
    folds = pd.DataFrame(folds)

    # ---- refit on all training trips, evaluate on student-test ----
    t0 = time.time(); model = fit_model(spec, X, y, groups, np.arange(len(y)))
    train_time = time.time() - t0
    t0 = time.time(); p_te = model.predict_proba(Cte[FEATURES].values)[:, 1]
    infer_ms_per_pt = (time.time() - t0) / Pte.shape[0] * 1000
    te = row_metrics(Cte.label.values, p_te)
    pred = top1_predictions(Cte, p_te)
    te["top1"] = top1_accuracy(Pte, pred)
    te["continuity"] = route_continuity(Pte, pred, "student_test")

    # training-set score on a fixed 200k-row sample (for over/underfitting check;
    # scoring all 1.46M rows is too slow for KNN)
    s = np.random.default_rng(SEED).choice(len(y), min(200_000, len(y)), replace=False)
    res = {
        "model": name, "family": spec["family"], "config": spec["config"],
        "cv_macro_f1": folds.macro_f1.mean(), "cv_macro_f1_std": folds.macro_f1.std(),
        "cv_top1": folds.top1.mean(), "cv_log_loss": folds.log_loss.mean(),
        "train_macro_f1": row_metrics(y[s], model.predict_proba(X[s])[:, 1])["macro_f1"],
        **{f"test_{k}": v for k, v in te.items()},
        "train_time_s": train_time, "infer_ms_per_point": infer_ms_per_pt,
    }

    (OUT / "models").mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    # compressed: an uncompressed random forest is ~260 MB (GitHub limit is 100 MB)
    joblib.dump(model, OUT / "models" / f"{name}.joblib", compress=("xz", 3))
    folds.to_csv(OUT / "results" / f"{name}_cv_folds.csv", index=False)
    if hasattr(model, "history_"):  # epoch-wise models only (MLP)
        pd.DataFrame(model.history_).to_csv(OUT / "results" / f"{name}_loss_history.csv", index=False)
        res["best_epoch"] = model.best_epoch_
    pred.to_csv(OUT / "results" / f"{name}_student_test_top1.csv", index=False)
    with open(OUT / "results" / f"{name}_summary.json", "w") as f:
        json.dump(res, f, indent=2, default=str)

    table = OUT / "results" / "model_comparison.csv"
    old = pd.read_csv(table) if table.exists() else pd.DataFrame()
    if len(old):
        old = old[old.model != name]
    pd.concat([old, pd.DataFrame([res])]).to_csv(table, index=False)

    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    main(a.model, a.folds)

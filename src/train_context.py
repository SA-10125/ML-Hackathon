"""Train the two-stage Random Forest (RF -> context features -> RF).

  python -m src.train_context

Steps (no student-test or evaluator data is used for any fitting or choice):
  1. Stage-1 probabilities on train = the cached OUT-OF-FOLD RF probabilities from
     src.tune_decoder (data/processed/oof_random_forest.parquet).
  2. Context features (src/context.py) from those probabilities.
  3. Stage-2 RF (same configuration as the stage-1 RF) cross-validated with the SAME
     GroupKFold trip folds -> out-of-fold stage-2 probabilities for decoder tuning
     (data/processed/oof_random_forest_ctx.parquet).
  4. Stage-2 refit on all 80 train trips; saved as a TwoStage model with the full stage-1 RF.
  5. Student-test probabilities from the saved TwoStage model (one final check only).
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .context import FEATURES2, TwoStage, add_context
from .data_io import RAW
from .features import FEATURES
from .metrics import row_metrics
from .models import MODELS

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
NAME = "random_forest_ctx"


def arcs_for(part, trips):
    return {t: np.loadtxt(RAW / part / t / f"{t}.arcs", dtype=np.int64, ndmin=2) for t in trips}


def top1(C, prob, P):
    d = C[["trip", "pt", "arc"]].assign(s=prob)
    b = d.loc[d.groupby(["trip", "pt"]).s.idxmax(), ["trip", "pt", "arc"]]
    m = P[P.ref >= 0].merge(b, on=["trip", "pt"], how="left")
    return float((m.arc == m.ref).mean())


def main():
    spec = MODELS["random_forest"]
    Ctr = pd.read_parquet(PROC / "train_cands.parquet")
    Ptr = pd.read_parquet(PROC / "train_points.parquet")
    oof1 = pd.read_parquet(PROC / "oof_random_forest.parquet")
    fold_of = json.loads((PROC / "oof_random_forest_folds.json").read_text())
    assert (oof1.trip.values == Ctr.trip.values).all() and (oof1.arc.values == Ctr.arc.values).all()

    t0 = time.time()
    Xtr = add_context(Ctr, oof1.prob.values, arcs_for("train", Ctr.trip.unique()))
    print(f"context features (train): {time.time() - t0:.0f}s", flush=True)

    # ---- stage-2 out-of-fold with the same trip folds as stage 1
    folds = Xtr.trip.map(fold_of).values
    oof2 = np.full(len(Xtr), np.nan)
    fold_rows = []
    for k in sorted(set(folds)):
        t0 = time.time()
        tr = (folds != k) & (Xtr.ref_valid.values == 1)
        va = folds == k
        m = spec["make"]()
        m.fit(Xtr.loc[tr, FEATURES2].values, Xtr.loc[tr, "label"].values)
        oof2[va] = m.predict_proba(Xtr.loc[va, FEATURES2].values)[:, 1]
        vv = va & (Xtr.ref_valid.values == 1)
        r = row_metrics(Xtr.label.values[vv], oof2[vv])
        r.update(fold=int(k), top1=top1(Ctr[va], oof2[va], Ptr[Ptr.trip.isin(set(Ctr.trip[va]))]),
                 seconds=time.time() - t0)
        fold_rows.append(r)
        print(f"  stage-2 fold {k}: top1={r['top1']:.4f} macroF1={r['macro_f1']:.4f} ({r['seconds']:.0f}s)", flush=True)
    Ctr[["trip", "pt", "arc"]].assign(prob=oof2).to_parquet(PROC / f"oof_{NAME}.parquet", index=False)
    (PROC / f"oof_{NAME}_folds.json").write_text(json.dumps(fold_of))
    pd.DataFrame(fold_rows).to_csv(OUT / "results" / f"{NAME}_cv_folds.csv", index=False)

    # ---- refit stage 2 on all train trips; stage 1 = RF already fitted on all train trips
    t0 = time.time()
    ok = Xtr.ref_valid.values == 1
    stage2 = spec["make"]()
    stage2.fit(Xtr.loc[ok, FEATURES2].values, Xtr.loc[ok, "label"].values)
    train_s = time.time() - t0
    stage1 = joblib.load(OUT / "models" / "random_forest.joblib")
    joblib.dump(TwoStage(stage1, stage2), OUT / "models" / f"{NAME}.joblib", compress=("xz", 3))

    # ---- student-test: one final check
    Cte = pd.read_parquet(PROC / "student_test_cands.parquet")
    Pte = pd.read_parquet(PROC / "student_test_points.parquet")
    t0 = time.time()
    p1 = stage1.predict_proba(Cte[FEATURES].values)[:, 1]
    Xte = add_context(Cte, p1, arcs_for("student_test", Cte.trip.unique()))
    p2 = stage2.predict_proba(Xte[FEATURES2].values)[:, 1]
    infer_ms = (time.time() - t0) / len(Pte) * 1000
    Cte[["trip", "pt", "arc"]].assign(prob=p2).to_parquet(PROC / f"student_test_probs_{NAME}.parquet", index=False)

    v = Cte.ref_valid.values == 1
    te = row_metrics(Cte.label.values[v], p2[v])
    summary = {
        "model": NAME, "config": "RF stage 1 (OOF on train) -> 11 context features -> RF stage 2 (same config)",
        "cv_top1_stage1_oof": top1(Ctr, oof1.prob.values, Ptr),
        "cv_top1_stage2_oof": top1(Ctr, oof2, Ptr),
        "cv_macro_f1": float(np.mean([r["macro_f1"] for r in fold_rows])),
        "cv_macro_f1_std": float(np.std([r["macro_f1"] for r in fold_rows], ddof=1)),
        "test_top1_stage1": top1(Cte, p1, Pte), "test_top1": top1(Cte, p2, Pte),
        **{f"test_{k}": v for k, v in te.items()},
        "train_time_s_stage2": train_s, "infer_ms_per_point": infer_ms,
    }
    (OUT / "results" / f"{NAME}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

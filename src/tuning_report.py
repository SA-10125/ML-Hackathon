"""Tuned vs untuned comparison + Optuna history plot.

  python -m src.tuning_report
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eval_hmm import decode_partition, weighted

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tuning"
PROC = ROOT / "data" / "processed"


def main():
    T = pd.read_csv(OUT / "trials.csv")
    done = T[T.state == "COMPLETE"].sort_values("number")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(done.number, done.value, s=18, label="trial (3-fold GroupKFold log loss)")
    ax.plot(done.number, done.value.cummin(), c="#d7191c", label="best so far")
    ax.axhline(done[done.number == 0].value.iloc[0], ls="--", c="grey", label="untuned (trial 0)")
    ax.set_xlabel("Trial"); ax.set_ylabel("CV log loss (lower is better)"); ax.legend()
    ax.set_title(f"Optuna TPE optimisation of LightGBM ({len(T)} trials, "
                 f"{(T.state == 'PRUNED').sum()} pruned)")
    fig.tight_layout(); fig.savefig(OUT / "optuna_history.png", dpi=150); plt.close(fig)

    res = {r["model"]: r for r in pd.read_csv(ROOT / "outputs/results/model_comparison.csv").to_dict("records")}
    cfg = json.load(open(ROOT / "outputs/hmm/hmm_config.json"))
    C = pd.read_parquet(PROC / "student_test_cands.parquet")
    P = pd.read_parquet(PROC / "student_test_points.parquet")
    rows = []
    for name in ["lightgbm", "lightgbm_tuned"]:
        m = joblib.load(ROOT / f"outputs/models/{name}.joblib")
        df, _ = decode_partition("student_test", m, sorted(C.trip.unique()), cfg["beta"], cfg["gap_s"], C, P)
        h = weighted(df, "hmm")
        r = res[name]
        rows.append({"model": name, "cv_macro_f1": r["cv_macro_f1"], "cv_log_loss": r["cv_log_loss"],
                     "test_top1": r["test_top1"], "test_macro_f1": r["test_macro_f1"],
                     "test_log_loss": r["test_log_loss"], "train_macro_f1": r["train_macro_f1"],
                     "infer_ms_per_point": r["infer_ms_per_point"], "train_time_s": r["train_time_s"],
                     "hmm_test_top1": h["top1"], "hmm_test_continuity": h["continuity"],
                     "hmm_test_route_f1": h["route_f1"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    print(out.T.round(4).to_string())


if __name__ == "__main__":
    main()

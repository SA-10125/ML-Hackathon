"""Evidence for the feature choices.

  python -m src.feature_analysis

1. Distributions of key features for correct vs wrong candidates (train trips only).
2. Single-feature rule baselines: pick the candidate with the best value of one feature
   (e.g. nearest arc) and measure Top-1 — what the ML model has to beat.
3. LightGBM gain importance (if outputs/models/lightgbm.joblib exists).
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import FEATURES
from .metrics import top1_accuracy, top1_predictions

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "features"


def load(part):
    C = pd.read_parquet(PROC / f"{part}_cands.parquet")
    P = pd.read_parquet(PROC / f"{part}_points.parquet")
    return C[C.ref_valid == 1].reset_index(drop=True), P


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Ctr, Ptr = load("train")
    Cte, Pte = load("student_test")

    # 1. distributions, train only
    panels = [
        ("dist", "Distance to arc (m)", None, (0, 40)),
        ("head_diff", "Heading difference (deg), moving points only", "moving", (0, 180)),
        ("dist_win", "Mean distance of arc to neighbouring fixes (m)", None, (0, 60)),
        ("dist_minus_min", "Distance minus nearest candidate's distance (m)", None, (0, 30)),
        ("dist_rank", "Distance rank among the point's candidates", None, (1, 15)),
        ("speed", "Speed (m/s)", None, (0, 35)),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, title, cond, rng) in zip(axs.ravel(), panels):
        d = Ctr[Ctr[cond] == 1] if cond else Ctr
        v = np.degrees(d[col]) if col == "head_diff" else d[col]
        bins = np.arange(rng[0], rng[1] + 1) if col == "dist_rank" else np.linspace(*rng, 50)
        ax.hist(v[d.label == 0], bins=bins, density=True, alpha=0.6, label="wrong candidate", color="#d7191c")
        ax.hist(v[d.label == 1], bins=bins, density=True, alpha=0.6, label="correct (reference) arc", color="#1a9641")
        ax.set_title(title, fontsize=10)
    axs[0, 0].legend()
    fig.suptitle("Feature distributions: correct vs wrong candidates (train trips)")
    fig.tight_layout(); fig.savefig(OUT / "feature_distributions.png", dpi=150); plt.close(fig)

    # 2. single-feature rule baselines (lower is better for these features)
    rules = {"nearest arc (min dist)": "dist", "min dist_win": "dist_win",
             "min heading difference": "head_diff"}
    rows = []
    for name, col in rules.items():
        r = {"rule": name}
        for part, C, P in [("train", Ctr, Ptr), ("student_test", Cte, Pte)]:
            r[f"{part}_top1"] = top1_accuracy(P, top1_predictions(C, -C[col].values))
        rows.append(r)
    base = pd.DataFrame(rows)
    base.to_csv(OUT / "rule_baselines.csv", index=False)
    print(base.round(4).to_string(index=False))

    # 3. LightGBM importance
    mp = ROOT / "outputs" / "models" / "lightgbm.joblib"
    if mp.exists():
        m = joblib.load(mp)
        imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES)
        imp = (imp / imp.sum()).sort_values()
        imp.to_csv(OUT / "lightgbm_gain_importance.csv", header=["share_of_gain"])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(imp.index, imp.values * 100, color="#2c7bb6")
        ax.set_xlabel("Share of total split gain (%)")
        ax.set_title("LightGBM feature importance (fitted on all train trips)")
        fig.tight_layout(); fig.savefig(OUT / "lightgbm_importance.png", dpi=150); plt.close(fig)
        print((imp[::-1] * 100).round(1).to_string())


if __name__ == "__main__":
    main()

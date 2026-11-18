"""Model-comparison plots from outputs/results (actual experiment outputs only).

  python -m src.plot_results
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(RES / "model_comparison.csv")
    names = T.model.tolist()
    x = np.arange(len(names))

    # 1. CV macro F1 per fold (box) — shows stability across folds
    folds = [pd.read_csv(RES / f"{n}_cv_folds.csv").macro_f1.values for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(folds, tick_labels=names)
    ax.set_ylabel("Macro F1 (5-fold GroupKFold, train trips)")
    ax.set_title("Cross-validation Macro F1")
    fig.tight_layout(); fig.savefig(FIG / "cv_macro_f1.png", dpi=150)

    # 2. student-test Top-1 and Macro F1
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    ax.bar(x - w / 2, T.test_top1, w, label="Top-1 accuracy")
    ax.bar(x + w / 2, T.test_macro_f1, w, label="Macro F1")
    for i, (a, b) in enumerate(zip(T.test_top1, T.test_macro_f1)):
        ax.text(i - w / 2, a + 0.003, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.003, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, names); ax.set_ylim(0.8, 1.0); ax.legend()
    ax.set_title("Student-test performance (10 trips)")
    fig.tight_layout(); fig.savefig(FIG / "test_top1_f1.png", dpi=150)

    # 3. log loss (CV and test)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, T.cv_log_loss, w, label="CV")
    ax.bar(x + w / 2, T.test_log_loss, w, label="Student-test")
    ax.set_xticks(x, names); ax.set_ylabel("Log loss (lower is better)"); ax.legend()
    ax.set_title("Probability quality")
    fig.tight_layout(); fig.savefig(FIG / "log_loss.png", dpi=150)

    # 4. train vs CV macro F1 — over/underfitting evidence
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, T.train_macro_f1, w, label="Train (200k-row sample)")
    ax.bar(x + w / 2, T.cv_macro_f1, w, label="CV")
    ax.set_xticks(x, names); ax.set_ylim(0.85, 1.0); ax.legend()
    ax.set_title("Train vs CV Macro F1 (gap = overfitting)")
    fig.tight_layout(); fig.savefig(FIG / "train_vs_cv.png", dpi=150)

    # 5. MLP epoch-wise loss (only model with a real training history)
    h = RES / "mlp_loss_history.csv"
    if h.exists():
        H = pd.read_csv(h)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(H.epoch, H.train_loss, label="Train")
        ax.plot(H.epoch, H.val_loss, label="Validation (held-out train trips)")
        ax.axvline(H.val_loss.idxmin() + 1, ls="--", c="grey", label="Best epoch")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Log loss"); ax.legend()
        ax.set_title("MLP training history (final refit)")
        fig.tight_layout(); fig.savefig(FIG / "mlp_loss_curve.png", dpi=150)

    cols = ["model", "family", "cv_macro_f1", "cv_macro_f1_std", "test_top1", "test_macro_f1",
            "test_log_loss", "test_continuity", "train_time_s", "infer_ms_per_point"]
    print(T[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

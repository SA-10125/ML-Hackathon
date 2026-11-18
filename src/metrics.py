"""Evaluation metrics.

Row level (candidate classification): accuracy, precision, recall, macro F1, log loss.
Point level:  Top-1 accuracy = for each GPS point with a reliable reference arc,
              is the highest-scoring candidate the reference arc? Points whose
              reference is missing from the candidate set count as wrong.
Route level:  continuity = fraction of consecutive predicted arc pairs that are
              the same arc or physically connected (to_node(prev) == from_node(next)).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score)


def row_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro"),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def top1_predictions(cands: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Return one row per (trip, pt): the arc with the highest score."""
    df = cands[["trip", "pt", "arc"]].copy()
    df["score"] = score
    idx = df.groupby(["trip", "pt"], sort=False)["score"].idxmax()
    return df.loc[idx, ["trip", "pt", "arc", "score"]].rename(columns={"arc": "pred"})


def top1_accuracy(points: pd.DataFrame, pred: pd.DataFrame) -> float:
    pts = points[points.ref >= 0][["trip", "pt", "ref"]]
    m = pts.merge(pred, on=["trip", "pt"], how="left")
    return float((m.pred == m.ref).mean())


def continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """pred_arcs: predicted arc per point in time order (no NaNs)."""
    if len(pred_arcs) < 2:
        return 1.0
    p, n = pred_arcs[:-1], pred_arcs[1:]
    ok = (p == n) | (arcs[p, 1] == arcs[n, 0])
    return float(ok.mean())

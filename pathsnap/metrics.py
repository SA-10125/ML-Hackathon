"""Route-matching metrics."""
from __future__ import annotations

import numpy as np

from .geometry import Candidates


def top1_accuracy(pred: np.ndarray, label: np.ndarray) -> float:
    m = label >= 0
    return float((pred[m] == label[m]).mean()) if m.any() else float("nan")


def candidate_miss_rate(c: Candidates, label: np.ndarray) -> float:
    """Share of labelled points whose reference arc is not among their candidates."""
    pid = c.point_index()
    hit = np.zeros(c.n_points, bool)
    hit[pid[c.arc == label[pid]]] = True
    m = label >= 0
    return float((~hit[m]).mean()) if m.any() else float("nan")


def route_continuity(pred: np.ndarray, arcs: np.ndarray) -> float:
    """Share of consecutive *distinct* predicted arcs that are directly connected."""
    seq = pred[pred >= 0]
    if len(seq) < 2:
        return float("nan")
    seq = seq[np.concatenate([[True], seq[1:] != seq[:-1]])]
    if len(seq) < 2:
        return 1.0
    return float((arcs[seq[:-1], 1] == arcs[seq[1:], 0]).mean())


def route_overlap(pred: np.ndarray, route: np.ndarray, arc_len: np.ndarray) -> dict:
    """Length-weighted precision / recall / F1 between predicted and reference arc sets."""
    p = np.unique(pred[pred >= 0])
    r = np.unique(route)
    inter = np.intersect1d(p, r)
    lp, lr, li = arc_len[p].sum(), arc_len[r].sum(), arc_len[inter].sum()
    prec = li / lp if lp else 0.0
    rec = li / lr if lr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": float(prec), "route_recall": float(rec), "route_f1": float(f1)}

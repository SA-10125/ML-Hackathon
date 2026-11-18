"""Glue: load a trip, build its index/graph, generate candidates, match."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .data import Trip, load_trip
from .geometry import Candidates, SegmentIndex, generate_candidates
from .graph import RoadGraph
from .hmm import HMMParams, gaussian_log_emission, viterbi_decode, fill_unmatched, DecodeResult
from .labels import Labels, align_route
from . import metrics


@dataclass
class TripContext:
    trip: Trip
    arc_len: np.ndarray
    index: SegmentIndex
    graph: RoadGraph
    cands: Candidates
    labels: Labels | None


def prepare(trip_id: str, partition: str, p: HMMParams, with_labels: bool = True) -> TripContext:
    trip = load_trip(trip_id, partition)
    a = trip.node_xy[trip.arcs[:, 0]]
    b = trip.node_xy[trip.arcs[:, 1]]
    arc_len = np.hypot(*(b - a).T)
    index = SegmentIndex(a, b)
    graph = RoadGraph(len(trip.node_xy), trip.arcs, arc_len)
    cands = generate_candidates(trip, index, p.radius, p.k)
    labels = align_route(trip) if (with_labels and trip.route is not None) else None
    return TripContext(trip, arc_len, index, graph, cands, labels)


def nearest_arc(c: Candidates) -> np.ndarray:
    """Baseline: each point independently takes its closest candidate."""
    first = c.ptr[:-1]
    has = c.ptr[1:] > first
    return np.where(has, c.arc[np.minimum(first, len(c.arc) - 1)], -1)


def run_hmm(ctx: TripContext, p: HMMParams, log_em: np.ndarray | None = None) -> tuple[DecodeResult, float]:
    if log_em is None:
        log_em = gaussian_log_emission(ctx.cands, p)
    t0 = time.perf_counter()
    res = viterbi_decode(ctx.trip, ctx.cands, log_em, ctx.graph, ctx.arc_len, p)
    return res, time.perf_counter() - t0


def probs_for_trip(ctx: TripContext, table) -> np.ndarray:
    """Align a classifier output table (columns trip, point_idx, arc_id, prob) with ctx.cands rows.

    Candidates missing from the table get a tiny probability.
    """
    c = ctx.cands
    pid = c.point_index()
    t = table[table["trip"] == ctx.trip.trip_id]
    lookup = dict(zip(zip(t["point_idx"].to_numpy(), t["arc_id"].to_numpy()), t["prob"].to_numpy()))
    return np.array([lookup.get((i, a), 1e-9) for i, a in zip(pid.tolist(), c.arc.tolist())])


def evaluate(ctx: TripContext, pred: np.ndarray) -> dict:
    trip, lab = ctx.trip, ctx.labels
    filled = fill_unmatched(pred)
    out = {
        "n_points": len(pred),
        "unmatched_frac": float((pred < 0).mean()),
        "continuity": metrics.route_continuity(filled, trip.arcs),
    }
    if lab is not None:
        out["top1_acc"] = metrics.top1_accuracy(filled, lab.arc)
        out["n_labelled"] = int((lab.arc >= 0).sum())
        out.update(metrics.route_overlap(filled, trip.route, ctx.arc_len))
    return out

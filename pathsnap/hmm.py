"""HMM map matching decoded with Viterbi (Newson & Krumm 2009 style).

Hidden state at time t : one candidate arc of GPS point t (plus where on the arc).
Emission               : log-score of the GPS point given that arc. Pluggable:
                         Gaussian on distance (classic), or a classifier's
                         probability once the ML models are trained.
Transition i -> j      : -|d_route - d_gps| / beta, where d_route is the
                         shortest driving distance on the directed graph from
                         the projection on arc i to the projection on arc j,
                         and d_gps is the straight-line GPS displacement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from .geometry import Candidates
from .graph import RoadGraph


@dataclass
class HMMParams:
    sigma: float = 5.0          # GPS noise std (m), Gaussian emission
    beta: float = 5.0           # transition tolerance (m)
    radius: float = 50.0        # candidate search radius (m)
    k: int = 16                 # max candidates per point (two-way roads use 2 slots: A->B and B->A)
    heading_w: float = 0.0      # weight of the heading agreement term in the emission
    back_tol: float = 10.0      # allowed backward drift on the same arc (m), GPS jitter
    gap_s: float = 60.0         # time gap that starts a new HMM segment (s)
    vmax: float = 0.0           # optional hard speed limit (m/s); 0 = off. GPS spikes exceed it, so off by default
    cutoff_factor: float = 3.0  # Dijkstra search limit = factor * d_gps + slack
    cutoff_slack: float = 200.0
    emission_w: float = 1.0     # weight on external (classifier) log-probabilities

    def dict(self):
        return asdict(self)


def gaussian_log_emission(c: Candidates, p: HMMParams) -> np.ndarray:
    le = -0.5 * (c.dist / p.sigma) ** 2
    if p.heading_w:
        hc = np.nan_to_num(c.heading_cos, nan=1.0)  # unknown heading => no penalty
        le = le + p.heading_w * (hc - 1.0)
    return le


def classifier_log_emission(prob: np.ndarray, p: HMMParams) -> np.ndarray:
    """prob: classifier P(candidate is the true arc), aligned with Candidates rows."""
    return p.emission_w * np.log(np.clip(prob, 1e-9, 1.0))


@dataclass
class DecodeResult:
    arc: np.ndarray        # (T,) predicted arc per GPS point, -1 if no candidates
    choice: np.ndarray     # (T,) index into Candidates rows, -1 if none
    n_breaks: int          # times the chain had no feasible transition
    n_gap_splits: int      # times a long time gap started a new segment


def _log_transition(trip, c: Candidates, graph: RoadGraph, arc_len, ti: int, tj: int,
                    p: HMMParams) -> np.ndarray:
    si, sj = c.of(ti), c.of(tj)
    arc_i, off_i = c.arc[si], c.offset[si]
    arc_j, off_j = c.arc[sj], c.offset[sj]
    from_j = trip.arcs[arc_j, 0].tolist()
    d_gps = float(np.hypot(*(trip.xy[tj] - trip.xy[ti])))
    dt = max(float(trip.times[tj] - trip.times[ti]), 1e-6)
    cutoff = p.cutoff_factor * d_gps + p.cutoff_slack

    d_route = np.empty((len(arc_i), len(arc_j)))
    for r, (ai, oi) in enumerate(zip(arc_i.tolist(), off_i.tolist())):
        dists = graph.dists_from(int(trip.arcs[ai, 1]), cutoff)
        between = np.array([dists.get(n, math.inf) for n in from_j])
        row = (arc_len[ai] - oi) + between + off_j          # leave arc i, drive, enter arc j
        same = (arc_j == ai) & (off_j >= oi - p.back_tol)   # stay on the same arc
        row[same] = np.maximum(off_j[same] - oi, 0.0)
        d_route[r] = row

    lt = -np.abs(d_route - d_gps) / p.beta
    bad = ~np.isfinite(d_route)
    if p.vmax:
        bad |= d_route / dt > p.vmax
    lt[bad] = -np.inf
    return lt


def viterbi_decode(trip, c: Candidates, log_em: np.ndarray, graph: RoadGraph,
                   arc_len: np.ndarray, p: HMMParams) -> DecodeResult:
    T = c.n_points
    choice = np.full(T, -1, np.int64)
    n_breaks = n_gaps = 0

    seg: list[tuple[int, np.ndarray | None]] = []
    score, prev_t = None, None

    def close_segment():
        if not seg:
            return
        j = int(np.argmax(score))
        for t, bp in reversed(seg):
            choice[t] = c.ptr[t] + j
            if bp is not None:
                j = int(bp[j])
        seg.clear()

    for t in range(T):
        sl = c.of(t)
        if sl.start == sl.stop:
            continue  # no road nearby (outlier): skip, filled in afterwards
        if prev_t is None or trip.times[t] - trip.times[prev_t] > p.gap_s:
            if prev_t is not None:
                n_gaps += 1
            close_segment()
            score = log_em[sl].copy()
            seg.append((t, None))
            prev_t = t
            continue
        lt = _log_transition(trip, c, graph, arc_len, prev_t, t, p)
        cand = score[:, None] + lt
        best = cand.argmax(0)
        new = cand[best, np.arange(cand.shape[1])] + log_em[sl]
        if np.all(np.isneginf(new)):
            n_breaks += 1  # HMM break: no road path explains the move -> restart here
            close_segment()
            score = log_em[sl].copy()
            seg.append((t, None))
        else:
            score = new
            seg.append((t, best))
        prev_t = t
    close_segment()

    arc = np.where(choice >= 0, c.arc[np.maximum(choice, 0)], -1)
    return DecodeResult(arc, choice, n_breaks, n_gaps)


def fill_unmatched(arc: np.ndarray) -> np.ndarray:
    """Give skipped points the prediction of the nearest earlier (else later) matched point."""
    out = arc.copy()
    idx = np.nonzero(out >= 0)[0]
    if len(idx) == 0:
        return out
    pos = np.searchsorted(idx, np.arange(len(out)), side="right") - 1
    out = np.where(pos >= 0, out[idx[np.maximum(pos, 0)]], out[idx[0]])
    return out

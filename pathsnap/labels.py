"""Point-level reference labels: which route arc was each GPS point on?

The .route file is only an ordered list of arcs, so it does not say which GPS
point belongs to which arc. We align the two with a small monotonic HMM whose
states are *positions in the route list* (not arc ids, because a route can
pass the same arc twice). Positions may only move forward in time, and the
distance travelled along the route should match the GPS displacement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import SegmentIndex


@dataclass
class Labels:
    pos: np.ndarray   # (T,) index into trip.route, -1 if unaligned
    arc: np.ndarray   # (T,) reference arc id, -1 if unaligned
    dist: np.ndarray  # (T,) GPS distance to that arc (m), nan if unaligned


def align_route(trip, sigma: float = 5.0, beta: float = 10.0, radius: float = 50.0,
                back_tol: float = 10.0, gap_s: float = 60.0) -> Labels:
    route = trip.route
    T = len(trip.xy)
    a = trip.node_xy[trip.arcs[route, 0]]
    b = trip.node_xy[trip.arcs[route, 1]]
    seg_len = np.hypot(*(b - a).T)
    start = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])  # along-route offset of each position

    index = SegmentIndex(a, b)
    ptr, pos, dist, off, _ = index.query(trip.xy, radius, k=40, fallback_k=3)
    along = start[pos] + off
    log_em = -0.5 * (dist / sigma) ** 2

    back: list[tuple[int, np.ndarray]] = []  # (t, backpointer into previous kept point)
    prev_t, score = None, None
    for t in range(T):
        sl = slice(ptr[t], ptr[t + 1])
        if sl.start == sl.stop:
            continue
        if prev_t is None:
            prev_t, score = t, log_em[sl].copy()
            back.append((t, None))
            continue
        psl = slice(ptr[prev_t], ptr[prev_t + 1])
        diff = along[sl][None, :] - along[psl][:, None]
        ok = (pos[sl][None, :] >= pos[psl][:, None]) & (diff >= -back_tol)
        if trip.times[t] - trip.times[prev_t] > gap_s:
            lt = np.where(ok, 0.0, -np.inf)
        else:
            d_gps = np.hypot(*(trip.xy[t] - trip.xy[prev_t]))
            lt = np.where(ok, -np.abs(np.maximum(diff, 0.0) - d_gps) / beta, -np.inf)
        cand = score[:, None] + lt
        best = cand.argmax(0)
        new = cand[best, np.arange(cand.shape[1])] + log_em[sl]
        if np.all(np.isneginf(new)):
            continue  # cannot be placed consistently: leave this point unlabelled
        back.append((t, best))
        prev_t, score = t, new

    lab_pos = np.full(T, -1, np.int64)
    lab_dist = np.full(T, np.nan)
    if back:
        j = int(score.argmax())
        for t, bp in reversed(back):
            k = ptr[t] + j
            lab_pos[t], lab_dist[t] = pos[k], dist[k]
            if bp is not None:
                j = int(bp[j])
    lab_arc = np.where(lab_pos >= 0, route[np.maximum(lab_pos, 0)], -1)
    return Labels(lab_pos, lab_arc, lab_dist)

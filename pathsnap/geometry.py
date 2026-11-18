"""Spatial index over road segments and candidate generation."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import numpy as np
from scipy.spatial import cKDTree


def project_points(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Project points p onto segments a->b (all (n, 2)).

    Returns (distance, offset_along_segment_m, t in [0, 1]).
    """
    v = b - a
    L2 = (v * v).sum(1)
    t = ((p - a) * v).sum(1) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * v
    d = np.hypot(*(p - proj).T)
    return d, t * np.sqrt(L2), t


class SegmentIndex:
    """KD-tree over points sampled every `step` metres along each segment.

    Any segment within distance r of a query point has a sample within
    r + step/2, so a ball query on samples followed by an exact
    point-to-segment distance gives exact radius search.
    """

    def __init__(self, a: np.ndarray, b: np.ndarray, step: float = 10.0):
        self.a, self.b, self.step = a, b, step
        length = np.hypot(*(b - a).T)
        n = np.ceil(length / step).astype(np.int64) + 1
        seg = np.repeat(np.arange(len(a)), n)
        start = np.cumsum(n) - n
        t = (np.arange(n.sum()) - np.repeat(start, n)) / np.repeat(np.maximum(n - 1, 1), n)
        samples = a[seg] + t[:, None] * (b - a)[seg]
        self.sample_seg = seg
        self.tree = cKDTree(samples)
        self.n_seg = len(a)

    def query(self, xy: np.ndarray, radius: float, k: int, fallback_k: int = 0):
        """Candidate segments for every point.

        Returns CSR-style arrays: ptr (T+1,), seg, dist, offset, t
        sorted by distance within each point, at most k per point.
        If fallback_k > 0, points with no segment within radius get their
        fallback_k nearest segments instead (used for label alignment).
        """
        res = self.tree.query_ball_point(xy, radius + self.step / 2 + 1e-6, workers=-1)
        lens = np.fromiter(map(len, res), np.int64, len(res))
        flat = np.fromiter(chain.from_iterable(res), np.int64, int(lens.sum()))
        pid = np.repeat(np.arange(len(xy)), lens)
        seg = self.sample_seg[flat]

        if fallback_k:
            empty = np.nonzero(lens == 0)[0]
            if len(empty):
                kk = min(fallback_k * 8, len(self.sample_seg))
                _, nn = self.tree.query(xy[empty], k=kk, workers=-1)
                nn = np.atleast_2d(nn)
                pid = np.concatenate([pid, np.repeat(empty, nn.shape[1])])
                seg = np.concatenate([seg, self.sample_seg[nn.ravel()]])

        key = np.unique(pid * self.n_seg + seg)
        pid, seg = key // self.n_seg, key % self.n_seg
        dist, off, t = project_points(xy[pid], self.a[seg], self.b[seg])

        keep = dist <= radius
        if fallback_k:
            has_near = np.zeros(len(xy), bool)
            has_near[pid[keep]] = True
            keep |= ~has_near[pid]
        pid, seg, dist, off, t = pid[keep], seg[keep], dist[keep], off[keep], t[keep]

        order = np.lexsort((dist, pid))
        pid, seg, dist, off, t = pid[order], seg[order], dist[order], off[order], t[order]
        counts = np.bincount(pid, minlength=len(xy))
        start = np.cumsum(counts) - counts
        rank = np.arange(len(pid)) - start[pid]
        limit = k if not fallback_k else max(k, fallback_k)
        keep = rank < limit
        pid, seg, dist, off, t = pid[keep], seg[keep], dist[keep], off[keep], t[keep]

        ptr = np.zeros(len(xy) + 1, np.int64)
        ptr[1:] = np.cumsum(np.bincount(pid, minlength=len(xy)))
        return ptr, seg, dist, off, t


@dataclass
class Candidates:
    """Candidate arcs for every GPS point, stored flat (CSR layout).

    Candidates of point i are rows ptr[i]:ptr[i+1], nearest first.
    """
    ptr: np.ndarray
    arc: np.ndarray
    dist: np.ndarray      # perpendicular distance to arc (m)
    offset: np.ndarray    # distance from arc start to projection (m)
    t: np.ndarray         # offset / arc length, in [0, 1]
    heading_cos: np.ndarray  # cos(angle between GPS heading and arc direction); nan if unknown

    def of(self, i: int) -> slice:
        return slice(self.ptr[i], self.ptr[i + 1])

    @property
    def n_points(self) -> int:
        return len(self.ptr) - 1

    def point_index(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_points), np.diff(self.ptr))


def gps_heading(xy: np.ndarray, min_move: float = 2.0) -> np.ndarray:
    """Unit heading vector per point from its neighbours; nan where the car barely moved."""
    nxt = np.vstack([xy[1:], xy[-1:]])
    prv = np.vstack([xy[:1], xy[:-1]])
    v = nxt - prv
    n = np.hypot(*v.T)
    out = np.full_like(v, np.nan)
    ok = n >= min_move
    out[ok] = v[ok] / n[ok, None]
    return out


def generate_candidates(trip, index: SegmentIndex, radius: float, k: int) -> Candidates:
    ptr, arc, dist, off, t = index.query(trip.xy, radius, k)
    pid = np.repeat(np.arange(len(trip.xy)), np.diff(ptr))
    a = trip.node_xy[trip.arcs[arc, 0]]
    b = trip.node_xy[trip.arcs[arc, 1]]
    u = b - a
    u /= np.maximum(np.hypot(*u.T), 1e-9)[:, None]
    h = gps_heading(trip.xy)[pid]
    heading_cos = (u * h).sum(1)
    return Candidates(ptr, arc, dist, off, t, heading_cos)

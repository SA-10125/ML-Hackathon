"""Candidate generation and feature engineering.

For each GPS point, every arc within RADIUS metres becomes a candidate.
Each (point, candidate) pair is one row; label = 1 if the candidate is the
point's reference arc. The classifier learns P(correct | features), which
later becomes the HMM emission probability.
"""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from .geo import angle_diff, bearing, point_segment

RADIUS = 40.0     # metres; trade-off between candidate miss rate and candidate count
HEAD_K = 2        # heading/speed estimated from points i-K .. i+K
WIN = 3           # temporal window for neighbouring-point distance feature


def motion(trip, k: int = HEAD_K):
    """Heading (rad) and speed (m/s) from a centred difference of +-k points."""
    N = len(trip.xy)
    lo = np.clip(np.arange(N) - k, 0, N - 1)
    hi = np.clip(np.arange(N) + k, 0, N - 1)
    dxy = trip.xy[hi] - trip.xy[lo]
    dt = np.maximum(trip.t[hi] - trip.t[lo], 1e-6)
    dist = np.linalg.norm(dxy, axis=1)
    return bearing(dxy[:, 0], dxy[:, 1]), dist / dt, dist


def generate_candidates(trip, radius: float = RADIUS):
    a, b = trip.arc_a, trip.arc_b
    lines = shapely.linestrings(np.stack([a, b], axis=1))
    tree = STRtree(lines)
    pts = shapely.points(trip.xy)
    pi, ai = tree.query(pts, predicate="dwithin", distance=radius)
    order = np.lexsort((ai, pi))
    return pi[order], ai[order]


def build_features(trip, radius: float = RADIUS) -> pd.DataFrame:
    pi, ai = generate_candidates(trip, radius)
    if len(pi) == 0:
        return pd.DataFrame()
    a, b = trip.arc_a[ai], trip.arc_b[ai]
    p = trip.xy[pi]
    dist, frac = point_segment(p, a, b)
    arc_vec = b - a
    arc_len = np.linalg.norm(arc_vec, axis=1)
    arc_head = bearing(arc_vec[:, 0], arc_vec[:, 1])

    head, speed, disp = motion(trip)
    hd = angle_diff(head[pi], arc_head)
    moving = (disp[pi] > 3.0).astype(np.int8)  # heading meaningless when stationary

    # temporal consistency: mean distance of this arc to neighbouring points
    N = len(trip.xy)
    acc = np.zeros(len(pi))
    for off in range(-WIN, WIN + 1):
        if off == 0:
            continue
        q = trip.xy[np.clip(pi + off, 0, N - 1)]
        acc += point_segment(q, a, b)[0]
    dist_win = acc / (2 * WIN)

    dt_prev = np.r_[np.inf, np.diff(trip.t)][pi]
    dt_next = np.r_[np.diff(trip.t), np.inf][pi]

    df = pd.DataFrame({
        "pt": pi, "arc": ai,
        "dist": dist, "frac": frac, "arc_len": arc_len,
        "head_diff": hd, "cos_head": np.cos(hd), "moving": moving,
        "speed": speed[pi], "dist_win": dist_win,
        "dt_prev": np.minimum(dt_prev, 600), "dt_next": np.minimum(dt_next, 600),
    })
    g = df.groupby("pt")
    df["n_cand"] = g["dist"].transform("size")
    df["dist_rank"] = g["dist"].rank(method="min")
    df["dist_minus_min"] = df["dist"] - g["dist"].transform("min")
    df["distwin_minus_min"] = df["dist_win"] - g["dist_win"].transform("min")
    df["head_rank"] = g["head_diff"].rank(method="min")
    return df


FEATURES = [
    "dist", "frac", "arc_len", "head_diff", "cos_head", "moving", "speed",
    "dist_win", "dt_prev", "dt_next", "n_cand", "dist_rank", "dist_minus_min",
    "distwin_minus_min", "head_rank",
]

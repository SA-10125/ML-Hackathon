"""Vectorised planar geometry (inputs already in metres)."""
import numpy as np


def point_segment(p, a, b):
    """Distance from points p to segments a->b, plus projection fraction.

    p, a, b broadcast to (..., 2). Returns (dist, frac) where frac in [0,1]
    is how far along the segment the closest point lies.
    """
    ab = b - a
    L2 = np.einsum("...i,...i->...", ab, ab)
    frac = np.einsum("...i,...i->...", p - a, ab) / np.where(L2 == 0, 1, L2)
    frac = np.clip(frac, 0.0, 1.0)
    proj = a + frac[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1), frac


def bearing(dx, dy):
    """Planar bearing in radians, 0 = +x (east), counter-clockwise."""
    return np.arctan2(dy, dx)


def angle_diff(a, b):
    """Absolute smallest difference between angles, in [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)

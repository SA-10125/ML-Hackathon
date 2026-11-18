"""Loading raw PathSnap trips and projecting them to metres.

Raw files (whitespace separated, no headers):
  .track  lon lat time_s       GPS observations
  .nodes  lon lat              row index = node_id
  .arcs   from_node to_node    row index = arc_id (directed)
  .route  arc_id               ground-truth arc sequence (may be absent)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "Student folder" / "Student folder"
EARTH_R = 6_371_000.0


@dataclass
class Trip:
    trip_id: str
    partition: str
    lonlat: np.ndarray       # (T, 2) GPS lon/lat after cleaning
    times: np.ndarray        # (T,)   seconds
    orig_idx: np.ndarray     # (T,)   row in the raw .track file
    node_lonlat: np.ndarray  # (N, 2)
    arcs: np.ndarray         # (A, 2) int  from_node, to_node
    route: np.ndarray | None # (M,)   int  arc ids, chronological
    origin: tuple[float, float]  # (lon0, lat0) of the local projection
    xy: np.ndarray = None        # (T, 2) metres
    node_xy: np.ndarray = None   # (N, 2) metres

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        return lonlat_to_xy(lonlat, *self.origin)

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        return xy_to_lonlat(xy, *self.origin)


def lonlat_to_xy(lonlat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Local equirectangular projection centred on (lon0, lat0).

    Trips are spread worldwide, so one global projection would distort distances.
    Centring the projection on each trip keeps the scale error tiny at the
    metre scale we care about (point-to-road and point-to-point distances).
    """
    k = np.cos(np.radians(lat0))
    x = np.radians(lonlat[:, 0] - lon0) * EARTH_R * k
    y = np.radians(lonlat[:, 1] - lat0) * EARTH_R
    return np.column_stack([x, y])


def xy_to_lonlat(xy: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    k = np.cos(np.radians(lat0))
    lon = np.degrees(xy[:, 0] / (EARTH_R * k)) + lon0
    lat = np.degrees(xy[:, 1] / EARTH_R) + lat0
    return np.column_stack([lon, lat])


def list_trips(partition: str) -> list[str]:
    return sorted(p.name for p in (DATA_ROOT / partition).iterdir() if p.is_dir())


def _load(path: Path, dtype=float) -> np.ndarray | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return np.loadtxt(path, dtype=dtype, ndmin=2)


def load_trip(trip_id: str, partition: str) -> Trip:
    d = DATA_ROOT / partition / trip_id
    track = _load(d / f"{trip_id}.track")
    nodes = _load(d / f"{trip_id}.nodes")
    arcs = _load(d / f"{trip_id}.arcs", dtype=np.int64)
    route = _load(d / f"{trip_id}.route", dtype=np.int64)
    if track is None or nodes is None or arcs is None:
        raise FileNotFoundError(f"trip {trip_id} in {partition} is missing files")

    # --- cleaning: drop non-finite rows, sort by time, drop duplicate timestamps
    ok = np.isfinite(track).all(axis=1)
    idx = np.nonzero(ok)[0]
    idx = idx[np.argsort(track[idx, 2], kind="stable")]
    _, first = np.unique(track[idx, 2], return_index=True)
    idx = idx[np.sort(first)]
    track = track[idx]

    lon0, lat0 = float(track[:, 0].mean()), float(track[:, 1].mean())
    trip = Trip(
        trip_id=trip_id,
        partition=partition,
        lonlat=track[:, :2],
        times=track[:, 2],
        orig_idx=idx,
        node_lonlat=nodes,
        arcs=arcs,
        route=None if route is None else route[:, 0],
        origin=(lon0, lat0),
    )
    trip.xy = trip.to_xy(trip.lonlat)
    trip.node_xy = trip.to_xy(nodes)
    return trip

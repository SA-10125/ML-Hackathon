"""Raw-file loading and per-trip projection to metres.

File formats (headerless, whitespace-separated):
  .track  lon lat t        GPS observations
  .nodes  lon lat          road vertices   (row index = node_id)
  .arcs   from to          directed arcs   (row index = arc_id)
  .route  arc_id           ground-truth arc sequence (reference only)

Trips are spread across the globe, so each trip is projected into its own
UTM zone (chosen from the GPS centroid). Inside a UTM zone, Euclidean
distance in metres is accurate to well under 0.1% at city scale.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer

_REPO = Path(__file__).resolve().parents[1]
# dataset location: data/raw if present, else the "Student folder" committed at the repo root
RAW = _REPO / "data" / "raw" if (_REPO / "data" / "raw").exists() else _REPO / "Student folder"


@dataclass
class Trip:
    trip_id: str
    track_ll: np.ndarray   # (N,2) lon,lat
    t: np.ndarray          # (N,)  seconds
    nodes_ll: np.ndarray   # (V,2) lon,lat
    arcs: np.ndarray       # (E,2) from_node,to_node
    route: np.ndarray | None  # (M,) arc ids, None if unavailable
    xy: np.ndarray = None      # (N,2) projected GPS, metres
    nodes_xy: np.ndarray = None  # (V,2) projected nodes, metres
    epsg: int = None

    @property
    def arc_a(self):  # start point of each arc, metres
        return self.nodes_xy[self.arcs[:, 0]]

    @property
    def arc_b(self):  # end point of each arc, metres
        return self.nodes_xy[self.arcs[:, 1]]


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def load_trip(trip_id: str, partition: str, root: Path = RAW, with_route: bool = True) -> Trip:
    d = Path(root) / partition / trip_id
    tr = np.loadtxt(d / f"{trip_id}.track", ndmin=2)
    nodes = np.loadtxt(d / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(d / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
    rp = d / f"{trip_id}.route"
    route = np.loadtxt(rp, dtype=np.int64, ndmin=1) if (with_route and rp.exists()) else None

    # basic cleaning: sort by time, drop exact duplicate timestamps
    order = np.argsort(tr[:, 2], kind="stable")
    tr = tr[order]
    keep = np.r_[True, np.diff(tr[:, 2]) > 0]
    tr = tr[keep]

    trip = Trip(trip_id, tr[:, :2], tr[:, 2], nodes, arcs, route)
    lon0, lat0 = np.median(trip.track_ll, axis=0)
    trip.epsg = utm_epsg(lon0, lat0)
    tf = Transformer.from_crs(4326, trip.epsg, always_xy=True)  # always_xy => (lon,lat) order
    trip.xy = np.column_stack(tf.transform(trip.track_ll[:, 0], trip.track_ll[:, 1]))
    trip.nodes_xy = np.column_stack(tf.transform(nodes[:, 0], nodes[:, 1]))
    return trip


def list_trips(partition: str, root: Path = RAW) -> list[str]:
    return sorted(p.name for p in (Path(root) / partition).iterdir() if p.is_dir())

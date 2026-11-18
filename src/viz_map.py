"""Folium map: raw GPS, road network near the trip, reference route, predicted arcs.

  python -m src.viz_map --trip 00000000 --part student_test --pred outputs/results/logreg_student_test_top1.csv
"""
import argparse
from pathlib import Path

import folium
import numpy as np
import pandas as pd

from .data_io import load_trip

OUT = Path(__file__).resolve().parents[1] / "outputs" / "maps"


def arc_lines(trip, arc_ids):
    ll = trip.nodes_ll[:, ::-1]  # folium wants [lat, lon]
    return [[ll[trip.arcs[a, 0]].tolist(), ll[trip.arcs[a, 1]].tolist()] for a in arc_ids]


def make_map(trip_id, part, pred_csv=None, label="prediction"):
    trip = load_trip(trip_id, part)
    lat, lon = trip.track_ll[:, 1], trip.track_ll[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=14, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", attr="Tiles &copy; Esri")

    # road network: only arcs near the trajectory bounding box, to keep the file small
    pad = 0.003
    nl = trip.nodes_ll
    inbox = ((nl[:, 0] > lon.min() - pad) & (nl[:, 0] < lon.max() + pad) &
             (nl[:, 1] > lat.min() - pad) & (nl[:, 1] < lat.max() + pad))
    near = np.where(inbox[trip.arcs[:, 0]] & inbox[trip.arcs[:, 1]])[0]
    g = folium.FeatureGroup("Road network", show=True)
    for seg in arc_lines(trip, near):
        folium.PolyLine(seg, color="#bbbbbb", weight=1).add_to(g)
    g.add_to(m)

    g = folium.FeatureGroup("Reference route (.route)")
    for seg in arc_lines(trip, trip.route):
        folium.PolyLine(seg, color="#1a9850", weight=6, opacity=0.6).add_to(g)
    g.add_to(m)

    if pred_csv:
        pr = pd.read_csv(pred_csv, dtype={"trip": str})
        pr = pr[pr.trip == trip_id]
        g = folium.FeatureGroup(f"Predicted arcs ({label})")
        for seg in arc_lines(trip, np.unique(pr.pred.values.astype(int))):
            folium.PolyLine(seg, color="#d73027", weight=3).add_to(g)
        g.add_to(m)

    g = folium.FeatureGroup("Raw GPS")
    for la, lo in zip(lat[::2], lon[::2]):
        folium.CircleMarker([la, lo], radius=2, color="#2c7bb6", fill=True, weight=0,
                            fill_opacity=0.7).add_to(g)
    g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{part}_{trip_id}_{label}.html"
    m.save(path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", required=True)
    ap.add_argument("--part", default="student_test")
    ap.add_argument("--pred")
    ap.add_argument("--label", default="logreg")
    a = ap.parse_args()
    print(make_map(a.trip, a.part, a.pred, a.label))

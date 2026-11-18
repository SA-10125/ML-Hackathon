"""Folium map: raw GPS, road network, reference route, nearest-arc baseline and HMM route.

Usage: python scripts/map_trip.py --partition student_test --trip 00000000
Output: results/maps/<partition>_<trip>.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import folium
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.hmm import HMMParams, fill_unmatched  # noqa: E402
from pathsnap import pipeline  # noqa: E402


def arc_lines(trip, arc_ids):
    """List of [[lat, lon], [lat, lon]] lines for the given arcs."""
    ll = trip.node_lonlat
    a, b = trip.arcs[arc_ids, 0], trip.arcs[arc_ids, 1]
    return [[[ll[i, 1], ll[i, 0]], [ll[j, 1], ll[j, 0]]] for i, j in zip(a, b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--trip", default="00000000")
    ap.add_argument("--sigma", type=float, default=HMMParams.sigma)
    ap.add_argument("--beta", type=float, default=HMMParams.beta)
    ap.add_argument("--max-network-arcs", type=int, default=15000)
    args = ap.parse_args()

    p = HMMParams(sigma=args.sigma, beta=args.beta)
    ctx = pipeline.prepare(args.trip, args.partition, p)
    tr = ctx.trip
    res, _ = pipeline.run_hmm(ctx, p)
    hmm = fill_unmatched(res.arc)
    near = fill_unmatched(pipeline.nearest_arc(ctx.cands))

    lat, lon = tr.lonlat[:, 1], tr.lonlat[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], tiles="OpenStreetMap",control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    # road network around the track (arcs whose midpoint is within 150 m of the track's bounding box)
    mid = (tr.node_xy[tr.arcs[:, 0]] + tr.node_xy[tr.arcs[:, 1]]) / 2
    lo, hi = tr.xy.min(0) - 150, tr.xy.max(0) + 150
    near_box = np.nonzero(((mid >= lo) & (mid <= hi)).all(1))[0][: args.max_network_arcs]
    fg = folium.FeatureGroup("Road network", show=True)
    folium.PolyLine(arc_lines(tr, near_box), color="#9e9e9e", weight=1, opacity=0.6).add_to(fg)
    fg.add_to(m)

    if tr.route is not None:
        fg = folium.FeatureGroup("Reference route (ground truth)")
        folium.PolyLine(arc_lines(tr, tr.route), color="#2e7d32", weight=7, opacity=0.6).add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup("Nearest-arc baseline", show=False)
    folium.PolyLine(arc_lines(tr, np.unique(near[near >= 0])), color="#f57c00", weight=3).add_to(fg)
    fg.add_to(m)

    fg = folium.FeatureGroup("HMM + Viterbi route")
    folium.PolyLine(arc_lines(tr, np.unique(hmm[hmm >= 0])), color="#c62828", weight=3).add_to(fg)
    fg.add_to(m)

    fg = folium.FeatureGroup("Raw GPS points")
    step = max(1, len(tr.xy) // 1500)
    for i in range(0, len(tr.xy), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color="#1565c0", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"pt {i}  t={tr.times[i]:.0f}s").add_to(fg)
    fg.add_to(m)

    if ctx.labels is not None:
        wrong = np.nonzero((ctx.labels.arc >= 0) & (hmm != ctx.labels.arc))[0]
        fg = folium.FeatureGroup(f"HMM mismatches ({len(wrong)})", show=False)
        for i in wrong[:: max(1, len(wrong) // 800)]:
            folium.CircleMarker([lat[i], lon[i]], radius=3, color="#000000", fill=True,
                                tooltip=f"pt {i}: pred {hmm[i]} ref {ctx.labels.arc[i]}").add_to(fg)
        fg.add_to(m)

    legend = """<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;
      padding:8px 12px;border:1px solid #888;border-radius:6px;font:13px sans-serif;line-height:1.6">
      <b>Trip {trip}</b><br>
      <span style="color:#1565c0">&#9679;</span> Raw GPS<br>
      <span style="color:#2e7d32">&#9644;</span> Reference route<br>
      <span style="color:#c62828">&#9644;</span> HMM + Viterbi<br>
      <span style="color:#f57c00">&#9644;</span> Nearest arc (baseline)<br>
      <span style="color:#9e9e9e">&#9644;</span> Road network</div>""".format(trip=tr.trip_id)
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)

    out = ROOT / "results" / "maps" / f"{args.partition}_{args.trip}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print("saved", out)


if __name__ == "__main__":
    main()

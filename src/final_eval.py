"""Final pipeline (tuned LightGBM + HMM) on student-test: per-trip results, what the HMM
fixed vs broke, and Folium maps with every layer the report asks for.

  python -m src.final_eval
"""
import json
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from .data_io import load_trip
from .eval_hmm import classifier_only, trip_metrics
from .features import FEATURES
from .hmm import viterbi

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "final"
MODEL = "lightgbm_tuned"


def folium_map(trip, cls, hmm, ref, title, path):
    lat, lon = trip.track_ll[:, 1], trip.track_ll[:, 0]
    ll = trip.nodes_ll[:, ::-1]
    seg = lambda a: [ll[trip.arcs[a, 0]].tolist(), ll[trip.arcs[a, 1]].tolist()]
    m = folium.Map(tiles=None)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", attr="Tiles &copy; Esri", name="Streets (Esri)").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}", attr="Tiles &copy; Esri", name="Light basemap (Esri)").add_to(m)

    tree = STRtree(shapely.linestrings(np.stack([trip.arc_a, trip.arc_b], axis=1)))
    _, near = tree.query(shapely.points(trip.xy[::5]), predicate="dwithin", distance=150)
    g = folium.FeatureGroup("Road network (within 150 m)", show=True)
    for a in np.unique(near):
        folium.PolyLine(seg(a), color="#9e9e9e", weight=1).add_to(g)
    g.add_to(m)

    layers = [("Reference route (.route)", trip.route, "#1a9850", 8, 0.45, True),
              ("Classifier only (tuned LightGBM)", np.unique(cls[cls >= 0]), "#f5a524", 3, 0.9, False),
              ("Predicted route (LightGBM + HMM)", np.unique(hmm[hmm >= 0]), "#d01c8b", 4, 0.95, True)]
    for name, arcs, col, w, op, show in layers:
        g = folium.FeatureGroup(name, show=show)
        for a in arcs:
            folium.PolyLine(seg(a), color=col, weight=w, opacity=op).add_to(g)
        g.add_to(m)

    g = folium.FeatureGroup("GPS observations", show=True)
    step = max(1, len(lat) // 2500)
    for i in range(0, len(lat), step):
        wrong = ref[i] >= 0 and hmm[i] != ref[i]
        folium.CircleMarker([lat[i], lon[i]], radius=2.5 if wrong else 2, weight=0, fill=True,
                            fill_opacity=0.85, color="#d7191c" if wrong else "#2c7bb6",
                            tooltip=f"#{i} t={trip.t[i]-trip.t[0]:.0f}s "
                                    f"{'HMM arc wrong' if wrong else 'HMM arc correct'}").add_to(g)
    g.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    legend = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
              f'padding:8px 12px;border:1px solid #888;border-radius:6px;font:12px sans-serif;line-height:1.7">'
              f'<b>{title}</b><br>'
              f'<span style="color:#1a9850">━━</span> Reference route<br>'
              f'<span style="color:#d01c8b">━━</span> Predicted route (LightGBM + HMM)<br>'
              f'<span style="color:#f5a524">━━</span> Classifier only (toggle in layers)<br>'
              f'<span style="color:#9e9e9e">━━</span> Road network<br>'
              f'<span style="color:#2c7bb6">●</span> GPS fix, HMM arc correct &nbsp;'
              f'<span style="color:#d7191c">●</span> GPS fix, HMM arc wrong</div>')
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    m.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "maps").mkdir(exist_ok=True)
    model = joblib.load(ROOT / "outputs" / "models" / f"{MODEL}.joblib")
    cfg = json.load(open(ROOT / "outputs" / "hmm" / "hmm_config.json"))
    C = pd.read_parquet(PROC / "student_test_cands.parquet")
    P = pd.read_parquet(PROC / "student_test_points.parquet")
    rows, keep = [], {}
    for tid in sorted(C.trip.unique()):
        trip = load_trip(tid, "student_test")
        c = C[C.trip == tid].sort_values(["pt", "arc"]).reset_index(drop=True)
        ref = P[P.trip == tid].sort_values("pt").ref.values
        prob = model.predict_proba(c[FEATURES].values)[:, 1]
        cls = classifier_only(c, prob, len(ref))
        hmm, nb = viterbi(trip, c, np.log(np.clip(prob, 1e-6, 1)), beta=cfg["beta"], gap_s=cfg["gap_s"])
        ok = ref >= 0
        changed = ok & (cls != hmm)
        mc, mh = trip_metrics(trip, cls, ref), trip_metrics(trip, hmm, ref)
        rows.append({"trip": tid, "points": int(ok.sum()),
                     "cls_top1": mc["top1"], "hmm_top1": mh["top1"], "gain_pts": 100 * (mh["top1"] - mc["top1"]),
                     "cls_continuity": mc["continuity"], "hmm_continuity": mh["continuity"],
                     "cls_route_f1": mc["route_f1"], "hmm_route_f1": mh["route_f1"],
                     "hmm_changed": int(changed.sum()),
                     "hmm_fixed": int((changed & (hmm == ref)).sum()),
                     "hmm_broke": int((changed & (cls == ref)).sum()),
                     "both_wrong_changed": int((changed & (cls != ref) & (hmm != ref)).sum()),
                     "hmm_breaks": nb})
        keep[tid] = (trip, cls, hmm, ref)
    df = pd.DataFrame(rows)
    tot = {"trip": "ALL", "points": df.points.sum()}
    for k in ["cls_top1", "hmm_top1"]:
        tot[k] = np.average(df[k], weights=df.points)
    tot["gain_pts"] = 100 * (tot["hmm_top1"] - tot["cls_top1"])
    for k in ["cls_continuity", "hmm_continuity", "cls_route_f1", "hmm_route_f1"]:
        tot[k] = df[k].mean()
    for k in ["hmm_changed", "hmm_fixed", "hmm_broke", "both_wrong_changed", "hmm_breaks"]:
        tot[k] = df[k].sum()
    df = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
    df.to_csv(OUT / "final_student_test_per_trip.csv", index=False)
    print(df.round(4).to_string(index=False))

    best = df[df.trip != "ALL"].sort_values("gain_pts", ascending=False).iloc[0].trip
    for tid, label in [(best, "largest HMM gain"), ("00000000", "typical trip")]:
        trip, cls, hmm, ref = keep[tid]
        r = df[df.trip == tid].iloc[0]
        folium_map(trip, cls, hmm, ref,
                   f"student_test {tid} ({label}) · Top-1 classifier {r.cls_top1:.1%} → HMM {r.hmm_top1:.1%}",
                   OUT / "maps" / f"final_student_test_{tid}.html")
    print("maps:", sorted(p.name for p in (OUT / "maps").iterdir()))


if __name__ == "__main__":
    main()

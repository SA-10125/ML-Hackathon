"""Exploratory data analysis for the raw PathSnap dataset.

  python -m src.eda

Partitions used: train (80 trips) and student_test (10 trips). Evaluator trips are not
present in the repository and are never touched. Every table/plot states its partition.

Outputs -> outputs/eda/
  inventory.csv            per-trip counts, sizes, gaps, extent, network stats
  partition_summary.csv    aggregated statistics per partition
  radius_tradeoff.csv      candidate miss rate and candidates/point vs search radius (train)
  data_quality.md          anomalies found and decisions taken
  *.png                    plots
  maps/*.html              Folium maps
"""
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import STRtree

from .data_io import RAW, list_trips, load_trip
from .labels import align_route

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "eda"
PARTS = ["train", "student_test"]
COLORS = {"train": "#2c7bb6", "student_test": "#d7191c"}


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def trip_stats(tid, part):
    d = RAW / part / tid
    raw = np.loadtxt(d / f"{tid}.track", ndmin=2)  # untouched, for validity checks
    trip = load_trip(tid, part)
    dt = np.diff(trip.t)
    step = np.linalg.norm(np.diff(trip.xy, axis=0), axis=1)
    speed = step / np.maximum(dt, 1e-6)

    V, E = len(trip.nodes_ll), len(trip.arcs)
    A = coo_matrix((np.ones(E), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(V, V)).tocsr()
    n_weak, lab_w = connected_components(A, directed=True, connection="weak")
    n_strong, lab_s = connected_components(A, directed=True, connection="strong")
    deg = np.bincount(trip.arcs.ravel(), minlength=V)
    arc_set = set(map(tuple, trip.arcs))
    bidir = np.mean([(b, a) in arc_set for a, b in trip.arcs])
    arc_len = np.linalg.norm(trip.arc_b - trip.arc_a, axis=1)

    r = trip.route
    route_cont = float(np.mean(trip.arcs[r[:-1], 1] == trip.arcs[r[1:], 0])) if len(r) > 1 else 1.0
    route_len_km = arc_len[r].sum() / 1000

    ref, _, ref_dist = align_route(trip)

    return {
        "partition": part, "trip": tid,
        "track_bytes": (d / f"{tid}.track").stat().st_size,
        "n_points": len(raw),
        "n_nan": int(np.isnan(raw).any(axis=1).sum()),
        "n_out_of_range": int(((np.abs(raw[:, 0]) > 180) | (np.abs(raw[:, 1]) > 90)).sum()),
        "n_dup_or_unsorted_t": int(len(raw) - len(trip.t)),
        "duration_s": trip.t[-1] - trip.t[0],
        "frac_dt_1s": float(np.mean(dt == 1)),
        "median_dt": float(np.median(dt)),
        "max_gap_s": float(dt.max()),
        "n_gaps_gt5s": int((dt > 5).sum()),
        "n_gaps_gt60s": int((dt > 60).sum()),
        "gps_path_km": step.sum() / 1000,
        "n_speed_gt50ms": int((speed > 50).sum()),  # >180 km/h between fixes = implausible jump
        "median_speed_ms": float(np.median(speed)),
        "lon_c": float(np.median(trip.track_ll[:, 0])), "lat_c": float(np.median(trip.track_ll[:, 1])),
        "extent_km": float(haversine_km(*trip.track_ll.min(0), *trip.track_ll.max(0))),
        "utm_epsg": trip.epsg,
        "n_nodes": V, "n_arcs": E, "n_isolated_nodes": int((deg == 0).sum()),
        "frac_bidirectional": float(bidir), "median_arc_len_m": float(np.median(arc_len)),
        "n_weak_components": int(n_weak), "largest_weak_frac": float(np.bincount(lab_w).max() / V),
        "n_strong_components": int(n_strong), "largest_strong_frac": float(np.bincount(lab_s).max() / V),
        "route_len_arcs": len(r), "route_len_km": route_len_km,
        "route_continuity": route_cont,
        "median_gps_to_ref_m": float(np.median(ref_dist)),
        "p95_gps_to_ref_m": float(np.percentile(ref_dist, 95)),
        "frac_unreliable_label": float(np.mean(ref < 0)),
    }, dt, speed, ref_dist[ref >= 0]


def radius_tradeoff(radii=(10, 15, 20, 30, 40, 50, 60, 80)):
    """Train only. Miss rate(R) = share of reliably-labelled points whose reference arc
    is farther than R. Candidates/point(R) counted with the same STRtree query used in
    candidate generation."""
    miss = {R: [] for R in radii}
    cnt = {R: [] for R in radii}
    for tid in list_trips("train"):
        trip = load_trip(tid, "train")
        ref, _, ref_dist = align_route(trip)
        ok = ref >= 0
        tree = STRtree(shapely.linestrings(np.stack([trip.arc_a, trip.arc_b], axis=1)))
        pts = shapely.points(trip.xy[ok])
        for R in radii:
            miss[R].append(ref_dist[ok] > R)
            pi, _ = tree.query(pts, predicate="dwithin", distance=R)
            cnt[R].append(np.bincount(pi, minlength=ok.sum()))
    return pd.DataFrame({
        "radius_m": radii,
        "candidate_miss_rate": [np.concatenate(miss[R]).mean() for R in radii],
        "mean_candidates_per_point": [np.concatenate(cnt[R]).mean() for R in radii],
    })


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def make_plots(inv, dts, speeds, refd, rad):
    # 1 points per trip
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in PARTS:
        ax.hist(inv[inv.partition == p].n_points, bins=np.logspace(2, 4.5, 25), alpha=0.7,
                label=f"{p} ({(inv.partition == p).sum()} trips)", color=COLORS[p])
    ax.set_xscale("log"); ax.set_xlabel("GPS observations per trip"); ax.set_ylabel("Trips")
    ax.set_title("Trajectory length (train + student_test)"); ax.legend()
    save(fig, "01_points_per_trip.png")

    # 2 sampling intervals
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in PARTS:
        d = np.concatenate(dts[p])
        vals, counts = np.unique(np.minimum(d, 31), return_counts=True)
        ax.bar(vals + (0.2 if p == "student_test" else -0.2), counts / counts.sum(), 0.4,
               label=p, color=COLORS[p])
    ax.set_yscale("log"); ax.set_xlabel("Seconds between consecutive fixes (31 = 31 or more)")
    ax.set_ylabel("Fraction of steps (log)"); ax.set_title("Sampling interval distribution")
    ax.legend(); save(fig, "02_sampling_intervals.png")

    # 3 gaps per trip
    fig, ax = plt.subplots(figsize=(10, 4))
    s = inv.sort_values("max_gap_s", ascending=False).head(30)
    ax.bar(s.trip + "\n" + s.partition.str[:5], s.max_gap_s, color=[COLORS[p] for p in s.partition])
    ax.set_yscale("log"); ax.set_ylabel("Largest time gap (s, log)")
    ax.set_title("30 trips with the largest GPS gaps"); ax.tick_params(axis="x", labelsize=6, rotation=90)
    save(fig, "03_largest_gaps.png")

    # 4 GPS noise: distance to aligned reference arc
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in PARTS:
        ax.hist(np.concatenate(refd[p]), bins=np.arange(0, 51, 1), density=True, alpha=0.6,
                label=p, color=COLORS[p])
    ax.set_xlabel("Distance from GPS fix to its reference arc (m)"); ax.set_ylabel("Density")
    ax.set_title("GPS noise relative to the reference route (reliable labels only)"); ax.legend()
    save(fig, "04_gps_noise.png")

    # 5 speeds
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in PARTS:
        sp = np.concatenate(speeds[p]) * 3.6
        ax.hist(np.clip(sp, 0, 200), bins=100, density=True, alpha=0.6, label=p, color=COLORS[p])
    ax.axvline(180, ls="--", c="k", lw=1, label="180 km/h (implausible)")
    ax.set_yscale("log"); ax.set_xlabel("Implied speed between fixes (km/h, clipped at 200)")
    ax.set_ylabel("Density (log)"); ax.set_title("Implied speed distribution"); ax.legend()
    save(fig, "05_speed.png")

    # 6 radius trade-off (justifies the 40 m candidate radius)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(rad.radius_m, rad.candidate_miss_rate * 100, "o-", c="#d7191c")
    ax1.set_xlabel("Candidate search radius (m)"); ax1.set_ylabel("Candidate miss rate (%)", color="#d7191c")
    ax2 = ax1.twinx()
    ax2.plot(rad.radius_m, rad.mean_candidates_per_point, "s-", c="#2c7bb6")
    ax2.set_ylabel("Mean candidates per point", color="#2c7bb6")
    ax1.axvline(40, ls="--", c="grey"); ax1.axvspan(50, rad.radius_m.max(), color="grey", alpha=0.12)
    ax1.text(51, ax1.get_ylim()[1] * 0.8, "miss rate 0 by construction\n(labels > 50 m excluded)", fontsize=8)
    ax1.set_title("Search radius trade-off (train)")
    save(fig, "06_radius_tradeoff.png")

    # 7 network size and connectivity
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for p in PARTS:
        q = inv[inv.partition == p]
        axs[0].scatter(q.n_nodes, q.n_arcs, label=p, c=COLORS[p], s=18)
        axs[1].scatter(q.n_nodes, q.largest_strong_frac, label=p, c=COLORS[p], s=18)
    axs[0].set_xlabel("Nodes"); axs[0].set_ylabel("Arcs"); axs[0].set_title("Road-network size per trip")
    axs[1].set_xlabel("Nodes"); axs[1].set_ylabel("Share of nodes in largest strongly connected component")
    axs[1].set_title("Road-network connectivity"); axs[0].legend()
    save(fig, "07_network.png")

    # 8 where the trips are
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for p in PARTS:
        q = inv[inv.partition == p]
        ax.scatter(q.lon_c, q.lat_c, c=COLORS[p], s=25, label=p)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.grid(alpha=0.3)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend()
    ax.set_title("Trip locations (median GPS position) — worldwide, so per-trip UTM projection")
    save(fig, "08_trip_locations.png")


def trip_map(tid, part, title):
    trip = load_trip(tid, part)
    lat, lon = trip.track_ll[:, 1], trip.track_ll[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    ll = trip.nodes_ll[:, ::-1]
    seg = lambda a: [ll[trip.arcs[a, 0]].tolist(), ll[trip.arcs[a, 1]].tolist()]
    # only arcs within 150 m of the GPS track (a bounding box explodes on long trips)
    tree = STRtree(shapely.linestrings(np.stack([trip.arc_a, trip.arc_b], axis=1)))
    _, near = tree.query(shapely.points(trip.xy[::5]), predicate="dwithin", distance=150)
    near = np.unique(near)
    g = folium.FeatureGroup("Road network (.nodes/.arcs)")
    for a in near:
        folium.PolyLine(seg(a), color="#999999", weight=1).add_to(g)
    g.add_to(m)
    g = folium.FeatureGroup("Reference route (.route)")
    for a in trip.route:
        folium.PolyLine(seg(a), color="#1a9850", weight=6, opacity=0.6).add_to(g)
    g.add_to(m)
    ref, _, _ = align_route(trip)
    g = folium.FeatureGroup("GPS fixes (blue = labelled, orange = >50 m from reference)")
    step = max(1, len(lat) // 3000)
    for i in range(0, len(lat), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2, weight=0, fill=True, fill_opacity=0.8,
                            color="#2c7bb6" if ref[i] >= 0 else "#fdae61",
                            tooltip=f"#{i} t={trip.t[i]:.0f}s").add_to(g)
    g.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:white;'
        f'padding:6px 10px;border:1px solid #888;font:13px sans-serif">{title}</div>'))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    (OUT / "maps").mkdir(parents=True, exist_ok=True)
    m.save(OUT / "maps" / f"{part}_{tid}.html")


def overview_map(inv):
    m = folium.Map(location=[30, 0], zoom_start=2, tiles="OpenStreetMap")
    for _, r in inv.iterrows():
        folium.CircleMarker([r.lat_c, r.lon_c], radius=5, color=COLORS[r.partition], fill=True,
                            tooltip=f"{r.partition} {r.trip}: {r.n_points} pts, "
                                    f"{r.route_len_km:.1f} km route").add_to(m)
    m.save(OUT / "maps" / "overview_all_trips.html")


def quality_report(inv, rad):
    t = inv[inv.partition == "train"]
    lines = ["# Data-quality record (train + student_test)", "",
             "| Check | Result | Decision |", "|---|---|---|"]
    lines.append(f"| Missing / NaN GPS values | {inv.n_nan.sum()} rows | none to handle |")
    lines.append(f"| Coordinates out of range | {inv.n_out_of_range.sum()} rows | none to handle |")
    lines.append(f"| Duplicate or unsorted timestamps | {inv.n_dup_or_unsorted_t.sum()} rows | "
                 "sort by time, drop exact duplicates (in loader) |")
    lines.append(f"| Steps exactly 1 s apart | train {np.average(t.frac_dt_1s, weights=t.n_points):.1%} | "
                 "nominal 1 Hz confirmed |")
    big = inv[inv.max_gap_s > 600].sort_values("max_gap_s", ascending=False)
    lines.append(f"| Trips with a gap > 10 min | {len(big)}: " +
                 ", ".join(f"{r.trip} ({r.max_gap_s/3600:.1f} h)" for _, r in big.iterrows()) +
                 " | dt features capped at 600 s; HMM must break/relax transitions across gaps |")
    lines.append(f"| Implied speed > 180 km/h | {inv.n_speed_gt50ms.sum()} steps | GPS jumps; "
                 "kept (no ground truth to delete them), HMM transitions penalise them |")
    bad = inv[inv.frac_unreliable_label > 0.05].sort_values("frac_unreliable_label", ascending=False)
    lines.append("| Reference route does not cover the GPS track | " +
                 ", ".join(f"{r.trip} ({r.frac_unreliable_label:.0%} of fixes)" for _, r in bad.iterrows()) +
                 " | fixes > 50 m from their aligned reference arc labelled unreliable (-1), "
                 "excluded from training and scoring |")
    nc = inv[inv.route_continuity < 1]
    lines.append(f"| Reference route not fully connected | {len(nc)} trips (min continuity "
                 f"{inv.route_continuity.min():.3f}) | treated as reference with known gaps, "
                 "not error-free truth |")
    lines.append(f"| Isolated nodes | {inv.n_isolated_nodes.sum()} across all trips | ignored "
                 "(no arcs, never candidates) |")
    lines.append(f"| Directed arcs with a reverse twin | median {inv.frac_bidirectional.median():.0%} | "
                 "heading feature needed to separate A->B from B->A |")
    lines.append(f"| Weakly connected components | median {inv.n_weak_components.median():.0f} per trip | "
                 "transitions between components are impossible -> HMM must handle |")
    r40 = rad[rad.radius_m == 40].iloc[0]
    lines.append(f"| Candidate radius | 40 m: miss rate {r40.candidate_miss_rate:.2%}, "
                 f"{r40.mean_candidates_per_point:.1f} candidates/point | chosen R = 40 m (train only). "
                 "Note: miss rate is 0 for R >= 50 m by construction, because labels > 50 m "
                 "are excluded, so only R < 50 m is informative |")
    lines += ["", "Trips span several continents, so distances use a per-trip UTM projection "
              "(metres), never raw degrees. Coordinates are stored lon, lat."]
    (OUT / "data_quality.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, dts, speeds, refd = [], {p: [] for p in PARTS}, {p: [] for p in PARTS}, {p: [] for p in PARTS}
    for p in PARTS:
        for tid in list_trips(p):
            r, dt, sp, rd = trip_stats(tid, p)
            rows.append(r); dts[p].append(dt); speeds[p].append(sp); refd[p].append(rd)
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)

    agg = inv.groupby("partition").agg(
        trips=("trip", "size"), total_points=("n_points", "sum"),
        median_points=("n_points", "median"), min_points=("n_points", "min"), max_points=("n_points", "max"),
        median_duration_min=("duration_s", lambda s: s.median() / 60),
        trips_gap_gt60s=("n_gaps_gt60s", lambda s: (s > 0).sum()),
        median_nodes=("n_nodes", "median"), median_arcs=("n_arcs", "median"),
        median_route_km=("route_len_km", "median"),
        median_gps_to_ref_m=("median_gps_to_ref_m", "median"),
        nan_rows=("n_nan", "sum"),
    )
    agg.to_csv(OUT / "partition_summary.csv")
    print(agg.T.to_string())

    rad = radius_tradeoff()
    rad.to_csv(OUT / "radius_tradeoff.csv", index=False)
    print(rad.to_string(index=False))

    make_plots(inv, dts, speeds, refd, rad)
    quality_report(inv, rad)
    trip_map("00000000", "student_test", "student_test 00000000 — typical trip")
    trip_map("00000054", "train", "train 00000054 — GPS continues beyond the reference route")
    trip_map("00000090", "train", "train 00000090 — 42 h gap inside the track")
    overview_map(inv)
    print((OUT / "data_quality.md").read_text())


if __name__ == "__main__":
    main()

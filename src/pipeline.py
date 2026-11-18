"""End-to-end inference for one trip, with ground truth kept strictly separate.

predict_route() sees only .track, .nodes and .arcs. The .route file is opened only by
score_route(), after prediction, on the server. Nothing from .route is ever returned
to the UI except the aggregate metrics.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from .data_io import load_trip
from .features import FEATURES, build_features
from .hmm import viterbi
from .labels import align_route
from .metrics import continuity

ROOT = Path(__file__).resolve().parents[1]


def load_artifacts(model_path=None, hmm_config=None):
    mdir = ROOT / "outputs" / "models"
    if model_path is None:
        tuned = mdir / "lightgbm_tuned.joblib"
        model_path = tuned if tuned.exists() else mdir / "lightgbm.joblib"
    cfg_path = Path(hmm_config or ROOT / "outputs" / "hmm" / "hmm_config.json")
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {"beta": 3.0, "gap_s": 60.0}
    return {"model": joblib.load(model_path), "model_path": str(model_path),
            "beta": float(cfg["beta"]), "gap_s": float(cfg["gap_s"])}


def predict_route(trip_dir: Path, trip_id: str, art):
    """Prediction only. The trip is loaded WITHOUT its .route file."""
    t0 = time.time()
    trip = load_trip(trip_id, trip_dir.name, root=trip_dir.parent, with_route=False)
    assert trip.route is None  # ground truth must not enter the prediction path
    t1 = time.time()
    cands = build_features(trip).sort_values(["pt", "arc"]).reset_index(drop=True)
    prob = art["model"].predict_proba(cands[FEATURES].values)[:, 1]
    t2 = time.time()

    best = cands[["pt", "arc"]].assign(s=prob).groupby("pt").s.idxmax()
    cls_pred = np.full(len(trip.xy), -1, dtype=np.int64)
    cls_pred[cands.pt.values[best.values]] = cands.arc.values[best.values]
    # classifier confidence per point = probability of its top candidate (NaN if no candidates)
    conf = np.full(len(trip.xy), np.nan)
    conf[cands.pt.values[best.values]] = prob[best.values]

    hmm_pred, breaks = viterbi(trip, cands, np.log(np.clip(prob, 1e-6, 1)),
                               beta=art["beta"], gap_s=art["gap_s"])
    t3 = time.time()
    timing = {"load_s": t1 - t0, "features_and_classifier_s": t2 - t1, "viterbi_s": t3 - t2,
              "total_s": t3 - t0}
    return trip, cls_pred, hmm_pred, breaks, timing, len(cands), conf


def score_route(trip_dir: Path, trip_id: str, trip, pred):
    """Server-side evaluation against the internal reference route."""
    ref_trip = load_trip(trip_id, trip_dir.name, root=trip_dir.parent, with_route=True)
    if ref_trip.route is None:
        return None
    ref, _, _ = align_route(ref_trip)
    ok = ref >= 0
    has = pred >= 0
    P, R = set(pred[has].tolist()), set(ref_trip.route.tolist())
    prec = len(P & R) / max(len(P), 1)
    rec = len(P & R) / max(len(R), 1)
    return {
        "top1_accuracy": float(np.mean(pred[ok] == ref[ok])),
        "route_continuity": continuity(pred[has], trip.arcs),
        "route_precision": prec, "route_recall": rec,
        "route_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "scored_points": int(ok.sum()),
    }


def route_geometry(trip, pred):
    """Predicted path as an ordered list of [lat, lon] polylines (one per unbroken run)."""
    ll = trip.nodes_ll[:, ::-1]
    lines, cur, last = [], [], None
    for a in pred:
        if a < 0 or a == last:
            continue
        u, v = trip.arcs[a]
        if last is not None and trip.arcs[last, 1] != u:  # not connected -> new polyline
            if cur:
                lines.append(cur)
            cur = []
        if not cur:
            cur.append(ll[u].round(6).tolist())
        cur.append(ll[v].round(6).tolist())
        last = a
    if cur:
        lines.append(cur)
    return lines

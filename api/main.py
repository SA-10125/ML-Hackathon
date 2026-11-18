"""PathSnap API — runs on Laptop 1.

  uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables
  PATHSNAP_TRIPS_DIR  folder holding the evaluation trips, one sub-folder per trip id
                      (default: "Student folder/student_test" for development; point it
                      at the evaluator trips on the day)
  PATHSNAP_MODEL      path to a .joblib model (default: tuned LightGBM if present)

Endpoints
  GET  /health            status, model and decoder configuration
  GET  /trips             trip ids available for evaluation (ids only)
  POST /predict {trip_id} full pipeline -> predicted route + server-side metrics

Reference routes never leave the server: the response contains the GPS track, the
predicted route and aggregate accuracy numbers only.
"""
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import load_artifacts, predict_route, route_geometry, score_route  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")

TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", ROOT / "Student folder" / "student_test"))
if not TRIPS_DIR.is_absolute():
    TRIPS_DIR = ROOT / TRIPS_DIR
ART = load_artifacts(os.environ.get("PATHSNAP_MODEL"))
log.info("model=%s beta=%.1f gap=%.0fs trips_dir=%s", ART["model_path"], ART["beta"],
         ART["gap_s"], TRIPS_DIR)

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def available_trips():
    if not TRIPS_DIR.exists():
        return []
    return sorted(p.name for p in TRIPS_DIR.iterdir()
                  if p.is_dir() and (p / f"{p.name}.track").exists())


@app.get("/health")
def health():
    return {"status": "ok", "model": Path(ART["model_path"]).name, "beta": ART["beta"],
            "gap_s": ART["gap_s"], "trips_dir": str(TRIPS_DIR), "n_trips": len(available_trips())}


@app.get("/trips")
def trips():
    return {"trips": available_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in available_trips():
        log.warning("REQUEST trip=%s rejected: unknown trip id", tid)
        raise HTTPException(404, f"unknown trip id {tid}")
    log.info("REQUEST  /predict trip=%s", tid)
    t0 = time.time()
    trip, cls_pred, hmm_pred, breaks, timing, n_cands = predict_route(TRIPS_DIR, tid, ART)
    metrics_hmm = score_route(TRIPS_DIR, tid, trip, hmm_pred)
    metrics_cls = score_route(TRIPS_DIR, tid, trip, cls_pred)
    step = max(1, len(trip.xy) // 4000)
    gps = trip.track_ll[::step, ::-1].round(6).tolist()
    resp = {
        "trip_id": tid,
        "n_points": int(len(trip.xy)),
        "n_candidates": int(n_cands),
        "points_without_candidates": int((hmm_pred < 0).sum()),
        "hmm_chain_breaks": int(breaks),
        "gps": gps,
        "route_hmm": route_geometry(trip, hmm_pred),
        "route_classifier": route_geometry(trip, cls_pred),
        "metrics": {"hmm": metrics_hmm, "classifier_only": metrics_cls},
        "timing": {**{k: round(v, 3) for k, v in timing.items()},
                   "request_s": round(time.time() - t0, 3)},
        "model": Path(ART["model_path"]).name,
    }
    if metrics_hmm:
        log.info("RESULT   trip=%s points=%d cands=%d top1_cls=%.4f top1_hmm=%.4f "
                 "continuity=%.4f route_f1=%.4f time=%.2fs", tid, len(trip.xy), n_cands,
                 metrics_cls["top1_accuracy"], metrics_hmm["top1_accuracy"],
                 metrics_hmm["route_continuity"], metrics_hmm["route_f1"], resp["timing"]["request_s"])
    else:
        log.info("RESULT   trip=%s points=%d (no reference route on server) time=%.2fs",
                 tid, len(trip.xy), resp["timing"]["request_s"])
    return resp

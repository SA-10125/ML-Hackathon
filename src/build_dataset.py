"""Build candidate-level datasets for train and student_test partitions.

Outputs (data/processed/):
  {part}_cands.parquet   one row per (GPS point, candidate arc) with features + label
  {part}_points.parquet  one row per GPS point: reference arc, alignment distance,
                         whether the reference arc is in the candidate set

Usage:  python -m src.build_dataset [--radius 40]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import list_trips, load_trip
from .features import RADIUS, build_features
from .labels import align_route

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"


def process(trip_id, part, radius):
    trip = load_trip(trip_id, part)
    ref, _, ref_dist = align_route(trip)
    df = build_features(trip, radius)
    df["label"] = (df["arc"].values == ref[df["pt"].values]).astype(np.int8)
    df["ref_valid"] = (ref[df["pt"].values] >= 0).astype(np.int8)
    df.insert(0, "trip", trip_id)

    in_set = np.zeros(len(ref), bool)
    in_set[df.loc[df.label == 1, "pt"].values] = True
    has_cand = np.zeros(len(ref), bool)
    has_cand[df["pt"].unique()] = True
    pts = pd.DataFrame({"trip": trip_id, "pt": np.arange(len(ref)), "t": trip.t,
                        "ref": ref, "ref_dist": ref_dist, "has_cand": has_cand,
                        "ref_in_cands": in_set})
    return df, pts


def main(radius):
    OUT.mkdir(parents=True, exist_ok=True)
    for part in ["train", "student_test"]:
        t0 = time.time()
        C, P = [], []
        for tid in list_trips(part):
            c, p = process(tid, part, radius)
            C.append(c); P.append(p)
        C, P = pd.concat(C, ignore_index=True), pd.concat(P, ignore_index=True)
        C.to_parquet(OUT / f"{part}_cands.parquet")
        P.to_parquet(OUT / f"{part}_points.parquet")
        v = P[P.ref >= 0]
        print(f"[{part}] trips={P.trip.nunique()} points={len(P)} cand_rows={len(C)} "
              f"cands/pt={len(C)/len(P):.1f} unreliable_labels={(P.ref<0).mean():.2%} "
              f"candidate_miss_rate={1-v.ref_in_cands.mean():.2%} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=RADIUS)
    main(ap.parse_args().radius)

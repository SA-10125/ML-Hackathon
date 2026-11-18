"""Classifier-only vs classifier + HMM/Viterbi.

  python -m src.eval_hmm --model lightgbm

1. Decoder tuning (beta, gap) on 20 TRAIN trips held out from a classifier fitted on the
   other 60 train trips (out-of-fold probabilities, so beta is not tuned on overconfident
   in-sample scores). Student-test is not used for choosing beta.
2. Final comparison on the 10 student-test trips with the saved model.

Metrics
  top1        point-level accuracy on reliably labelled points
  continuity  share of consecutive predicted arcs that are equal or connected
  route_p/r/f1  sequence level: predicted set of traversed arcs vs reference .route arcs
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data_io import load_trip
from .features import FEATURES
from .hmm import viterbi
from .metrics import continuity
from .models import MODELS

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "hmm"
SEED = 42


def trip_metrics(trip, pred, ref):
    ok = ref >= 0
    has = pred >= 0
    top1 = float(np.mean(pred[ok] == ref[ok]))
    p = pred[has]
    cont = continuity(p, trip.arcs)
    P, R = set(p.tolist()), set(trip.route.tolist())
    prec = len(P & R) / max(len(P), 1)
    rec = len(P & R) / max(len(R), 1)
    return {"top1": top1, "continuity": cont, "route_p": prec, "route_r": rec,
            "route_f1": 2 * prec * rec / max(prec + rec, 1e-9), "n_valid": int(ok.sum())}


def classifier_only(cands, prob, N):
    df = cands[["pt", "arc"]].assign(s=prob)
    best = df.loc[df.groupby("pt").s.idxmax()]
    pred = np.full(N, -1, dtype=np.int64)
    pred[best.pt.values] = best.arc.values
    return pred


def decode_partition(part, model, trips, beta, gap, C, P, with_cls=True):
    rows, preds = [], {}
    for tid in trips:
        trip = load_trip(tid, part)
        c = C[C.trip == tid].sort_values(["pt", "arc"]).reset_index(drop=True)
        pr = P[P.trip == tid].sort_values("pt")
        ref = pr.ref.values
        prob = model.predict_proba(c[FEATURES].values)[:, 1]
        t0 = time.time()
        hm, nb = viterbi(trip, c, np.log(np.clip(prob, 1e-6, 1)), beta=beta, gap_s=gap)
        dt = time.time() - t0
        r = {"trip": tid, **{f"hmm_{k}": v for k, v in trip_metrics(trip, hm, ref).items()},
             "hmm_breaks": nb, "hmm_seconds": dt}
        if with_cls:
            cl = classifier_only(c, prob, len(ref))
            r.update({f"cls_{k}": v for k, v in trip_metrics(trip, cl, ref).items()})
            preds[tid] = (cl, hm)
        rows.append(r)
    return pd.DataFrame(rows), preds


def weighted(df, prefix):
    w = df[f"{prefix}_n_valid"]
    return {m: float(np.average(df[f"{prefix}_{m}"], weights=w if m == "top1" else None))
            for m in ["top1", "continuity", "route_p", "route_r", "route_f1"]}


def main(model_name, betas, gaps):
    OUT.mkdir(parents=True, exist_ok=True)
    Ctr = pd.read_parquet(PROC / "train_cands.parquet")
    Ptr = pd.read_parquet(PROC / "train_points.parquet")
    Cte = pd.read_parquet(PROC / "student_test_cands.parquet")
    Pte = pd.read_parquet(PROC / "student_test_points.parquet")

    # ---- 1. decoder tuning on held-out train trips ----
    trips = np.array(sorted(Ctr.trip.unique()))
    rng = np.random.default_rng(SEED)
    hold = sorted(rng.choice(trips, 20, replace=False).tolist())
    fit = Ctr[(~Ctr.trip.isin(hold)) & (Ctr.ref_valid == 1)]
    oof_model = MODELS[model_name]["make"]()
    oof_model.fit(fit[FEATURES].values, fit.label.values)
    sweep = []
    for gap in gaps:
        for beta in betas:
            df, _ = decode_partition("train", oof_model, hold, beta, gap, Ctr, Ptr, with_cls=False)
            r = {"beta": beta, "gap_s": gap, **weighted(df, "hmm")}
            sweep.append(r)
            print(f"  tune beta={beta:>5} gap={gap:>4}: top1={r['top1']:.4f} cont={r['continuity']:.4f}")
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "decoder_sweep_train_heldout.csv", index=False)
    best = sweep.sort_values(["top1", "continuity"], ascending=False).iloc[0]
    beta, gap = float(best.beta), float(best.gap_s)
    # classifier-only on the same held-out trips, for reference
    df_cls, _ = decode_partition("train", oof_model, hold, beta, gap, Ctr, Ptr)
    tune_ref = {"classifier_only": weighted(df_cls, "cls"), "hmm": weighted(df_cls, "hmm")}

    # ---- 2. student-test comparison with the saved model ----
    model = joblib.load(ROOT / "outputs" / "models" / f"{model_name}.joblib")
    df, preds = decode_partition("student_test", model, sorted(Cte.trip.unique()), beta, gap, Cte, Pte)
    df.to_csv(OUT / f"{model_name}_student_test_per_trip.csv", index=False)
    for tid, (cl, hm) in preds.items():
        pd.DataFrame({"pt": np.arange(len(cl)), "cls_pred": cl, "hmm_pred": hm}).to_csv(
            OUT / f"pred_{model_name}_{tid}.csv", index=False)
    summary = {"model": model_name, "beta": beta, "gap_s": gap, "heldout_train_trips": hold,
               "heldout_train": tune_ref,
               "student_test": {"classifier_only": weighted(df, "cls"), "hmm": weighted(df, "hmm")},
               "decode_seconds_total": float(df.hmm_seconds.sum())}
    json.dump(summary, open(OUT / "hmm_config.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "heldout_train_trips"}, indent=2))
    print(df[["trip", "cls_top1", "hmm_top1", "cls_continuity", "hmm_continuity", "hmm_breaks"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--betas", default="3,10,30")
    ap.add_argument("--gaps", default="60")
    a = ap.parse_args()
    main(a.model, [float(x) for x in a.betas.split(",")], [float(x) for x in a.gaps.split(",")])

"""Bayesian hyperparameter optimisation of LightGBM with Optuna (TPE sampler).

  python -m src.tune --timeout 1500

How TPE works: after a few random trials, it splits past trials into "good" and "bad"
by objective value, fits a density model to each, and proposes parameters where
p(good)/p(bad) is highest — so each new trial uses what earlier trials revealed,
unlike grid or random search.

Objective: mean log loss over 3-fold GroupKFold on the 80 training trips (lower is
better). Log loss is chosen over Top-1 because the classifier's probabilities become
HMM emission probabilities, so their calibration matters, and log loss is a smoother
objective than Top-1. Top-1 and macro F1 are recorded for every trial as well.
Student-test and evaluator trips are never used here.

Outputs: outputs/tuning/trials.csv, best_params.json, optuna.db (resumable study)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from .features import FEATURES
from .metrics import row_metrics, top1_accuracy, top1_predictions
from .train import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tuning"
SEED = 42


def make_objective(C, P):
    X, y, g = C[FEATURES].values, C.label.values, C.trip.values
    splits = list(GroupKFold(n_splits=3).split(X, y, g))

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        t0 = time.time()
        ll, t1, f1 = [], [], []
        for k, (tr, va) in enumerate(splits):
            m = LGBMClassifier(**params, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]
            r = row_metrics(y[va], p)
            ll.append(r["log_loss"]); f1.append(r["macro_f1"])
            t1.append(top1_accuracy(P[P.trip.isin(set(g[va]))], top1_predictions(C.iloc[va], p)))
            trial.report(float(np.mean(ll)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_top1", float(np.mean(t1)))
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1)))
        trial.set_user_attr("seconds", time.time() - t0)
        return float(np.mean(ll))

    return objective


def main(timeout, n_trials):
    OUT.mkdir(parents=True, exist_ok=True)
    C, P = load("train")
    study = optuna.create_study(
        study_name="lgbm_logloss", direction="minimize",
        storage=f"sqlite:///{OUT / 'optuna.db'}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1),
    )
    # trial 0 = the untuned configuration, so tuned vs untuned share the same protocol
    if len(study.trials) == 0:
        study.enqueue_trial(dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                 min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1e-3))
    study.optimize(make_objective(C, P), timeout=timeout, n_trials=n_trials, gc_after_trial=True)

    df = study.trials_dataframe()
    df.to_csv(OUT / "trials.csv", index=False)
    best = study.best_trial
    json.dump({"params": best.params, "cv_log_loss": best.value, **best.user_attrs,
               "n_trials": len(study.trials),
               "n_complete": int((df.state == "COMPLETE").sum()),
               "n_pruned": int((df.state == "PRUNED").sum())},
              open(OUT / "best_params.json", "w"), indent=2)
    print(json.dumps(json.load(open(OUT / "best_params.json")), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--trials", type=int, default=60)
    a = ap.parse_args()
    main(a.timeout, a.trials)

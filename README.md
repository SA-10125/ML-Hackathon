# PathSnap: Map Matching with Candidate Classification + HMM/Viterbi

Problem walkthrough for the team: [PROBLEM_EXPLAINED.md](PROBLEM_EXPLAINED.md)

## Setup

```bash
pip install -r requirements.txt
```

Raw data is expected at `Student folder/Student folder/{train,student_test}/<trip_id>/`.

## Repository layout

| Path | Contents |
|---|---|
| `pathsnap/data.py` | Load and clean a trip (sort by time, drop duplicate timestamps), local metric projection |
| `pathsnap/geometry.py` | KD-tree segment index, candidate generation (radius 50 m, K=16) |
| `pathsnap/graph.py` | Directed road graph, bounded and cached Dijkstra |
| `pathsnap/labels.py` | Point-level reference labels: monotonic alignment of GPS points to `.route` |
| `pathsnap/hmm.py` | HMM emissions/transitions and Viterbi with breaks and gap splitting |
| `pathsnap/metrics.py` | Top-1 accuracy, candidate miss rate, route continuity, route overlap F1 |
| `pathsnap/pipeline.py` | Glue code: prepare a trip, run baselines and the HMM, evaluate |
| `scripts/run_hmm.py` | Nearest-arc vs HMM (and classifier vs classifier+HMM with `--probs`) on a partition |
| `scripts/tune_hmm.py` | Grid search of decoder parameters on **train trips only** |
| `scripts/export_candidates.py` | Shared candidate and feature table for classifier training |
| `scripts/map_trip.py` | Folium map: GPS, road network, reference route, baseline and HMM route |
| `results/` | Experiment CSV/JSON logs and maps |

## How to run

```bash
# 1. HMM (Gaussian-distance emission) vs nearest-arc baseline
python scripts/run_hmm.py --partition student_test --tag baseline

# 2. Tune decoder parameters on train (writes results/hmm_tuning_train.csv)
python scripts/tune_hmm.py --max-points 6000 --workers 8

# 3. Export candidate tables for the classifiers
python scripts/export_candidates.py --partition train
python scripts/export_candidates.py --partition student_test

# 4. Plug a trained classifier into the HMM
python scripts/run_hmm.py --partition student_test --probs path/to/probs.parquet --tag lgbm

# 5. Map
python scripts/map_trip.py --partition student_test --trip 00000000
```

## Contract between the classifiers and the HMM

`data_processed/candidates_<partition>.parquet` has one row per **(GPS point, candidate arc)**:

| Column | Meaning |
|---|---|
| `trip`, `point_idx`, `arc_id` | **Join key.** Do not change |
| `dist`, `dist_rank`, `dist_minus_min`, `dist_window_mean`, `n_cand` | Distance features |
| `offset`, `t_along`, `at_arc_end`, `arc_len` | Position along the arc |
| `heading_cos` | cos(GPS heading, arc direction); NaN when nearly stationary. Separates A→B from B→A |
| `speed` | m/s from the previous point |
| `has_label` | 1 if the point could be aligned to the reference route |
| `label` | **Target**: 1 if this candidate is the reference arc |

Rules for classifier work:
- Train on rows with `has_label == 1`.
- Use `GroupKFold(groups=trip)` so no trip is split across folds.
- Don't use `trip`, `point_idx`, `arc_id` or `time` as features.

To feed a classifier into the HMM, save a parquet with columns **`trip, point_idx, arc_id, prob`**, where `prob` = P(label=1), for the student_test rows. Then run step 4.

## Current results (student_test, 10 trips)

| Method | Top-1 acc (micro) | Top-1 acc (trip mean) | Route continuity | Route F1 |
|---|---|---|---|---|
| Nearest arc (no ML, no HMM) | 0.510 | 0.541 | 0.430 | 0.640 |
| HMM, Gaussian emission (σ=5, β=5) | 0.873 | 0.903 | 0.908 | 0.969 |
| HMM, tuned on train (σ=3, β=2) | 0.872 | 0.902 | 0.908 | 0.968 |

Candidate miss rate: 0.12%. Total decode time for all 10 trips: about 2–6 s on CPU.

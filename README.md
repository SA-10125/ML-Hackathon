<<<<<<< HEAD
# PathSnap — Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University.

## Setup
```bash
pip install -r requirements.txt
# the dataset is read from "Student folder/" at the repo root (or data/raw/ if you create it)
```

## Pipeline
```bash
python -m src.build_dataset            # labels + candidates + features -> data/processed/
python -m src.train --model logreg     # CV on train, eval on student_test -> outputs/
python -m src.viz_map --trip 00000065 --pred outputs/results/logreg_student_test_top1.csv
```

## Layout
| Path | Purpose |
|---|---|
| `src/data_io.py` | Load raw files; project each trip to its own UTM zone (metres) |
| `src/labels.py` | Monotonic DP alignment of GPS points to the ordered `.route` → per-point reference arc |
| `src/features.py` | Candidate generation (arcs within 40 m, STRtree) + features |
| `src/build_dataset.py` | Builds candidate/point tables for each partition |
| `src/models.py` | Model registry |
| `src/train.py` | GroupKFold-by-trip CV, student-test evaluation, result logging |
| `src/metrics.py` | Row metrics, observation-level Top-1, route continuity |
| `src/plot_results.py` | Comparison plots + MLP loss curve |
| `src/viz_map.py` | Folium maps (GPS, road network, reference, prediction) |
| `outputs/results/` | `model_comparison.csv`, per-fold CV scores, predictions |
| `outputs/models/` | Saved models |

## Key decisions
- **Distances:** per-trip UTM projection (trips are worldwide; degrees are not distances).
- **Labels:** `.route` is an arc sequence, not per-point labels. Points are aligned with a
  monotonic DP (route position can only stay or advance). Points >50 m from their aligned
  arc are marked unreliable (-1) and excluded (e.g. train trip 54: GPS continues ~5 h past the
  end of the reference route).
- **Candidates:** all arcs within 40 m → ~7 candidates/point, candidate miss rate 0.55% (train), 0.07% (student-test).
- **No leakage:** GroupKFold by trip; scaler fitted inside the pipeline; evaluator trips never used.

## Results so far (student-test)
| Model | CV macro F1 | Test Top-1 | Test macro F1 | Test log loss | Continuity |
|---|---|---|---|---|---|
| Logistic Regression | 0.904 ± 0.016 | 0.867 | 0.911 | 0.101 | 0.931 |
| Random Forest | 0.928 ± 0.013 | 0.890 | 0.934 | 0.068 | 0.952 |
| MLP (64,32) | 0.924 ± 0.013 | 0.882 | 0.928 | 0.073 | 0.948 |
| KNN (k=25, distance) | 0.913 ± 0.015 | 0.871 | 0.919 | 0.107 | 0.939 |

Plots: `python -m src.plot_results` → `outputs/figures/`.
Models are saved xz-compressed (uncompressed random forest is ~260 MB, over GitHub's limit).
=======
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
>>>>>>> hmm

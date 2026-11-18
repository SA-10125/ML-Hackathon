# PathSnap — Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University.

Problem walkthrough for the team: [PROBLEM_EXPLAINED.md](PROBLEM_EXPLAINED.md)

## Setup
```bash
pip install -r requirements.txt
# the dataset is read from "Student folder/" at the repo root (or data/raw/ if you create it)
```

## Pipeline
```bash
python -m src.eda                      # EDA: inventory, plots, data-quality record, Folium maps -> outputs/eda/
python -m src.build_dataset            # labels + candidates + features -> data/processed/
python -m src.train --model logreg     # CV on train, eval on student_test -> outputs/
python -m src.viz_map --trip 00000065 --pred outputs/results/logreg_student_test_top1.csv
```

## Layout
| Path | Purpose |
|---|---|
| `src/eda.py` | Dataset inventory, quality checks, EDA plots and Folium maps |
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
| `pathsnap/` | HMM + Viterbi map matcher (see the section below) |
| `scripts/` | HMM runs, decoder tuning, candidate export, HMM maps |
| `results/` | HMM experiment logs and maps |

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

---

## HMM + Viterbi module (`pathsnap/`)

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
| `scripts/export_candidates.py` | Candidate and feature table for classifier training |
| `scripts/map_trip.py` | Folium map: GPS, road network, reference route, baseline and HMM route |

### How to run
```bash
# HMM (Gaussian-distance emission) vs nearest-arc baseline
python scripts/run_hmm.py --partition student_test --tag baseline

# Tune decoder parameters on train (writes results/hmm_tuning_train.csv)
python scripts/tune_hmm.py --max-points 6000 --workers 8

# Plug a trained classifier into the HMM
python scripts/run_hmm.py --partition student_test --probs path/to/probs.parquet --tag rf

# Map
python scripts/map_trip.py --partition student_test --trip 00000000
```

`--probs` expects a parquet/csv with columns **`trip, point_idx, arc_id, prob`**, where `prob` is the
classifier's P(candidate is the reference arc) for every candidate of every point.

### HMM results (student_test, 10 trips)
| Method | Top-1 acc (micro) | Top-1 acc (trip mean) | Route continuity | Route F1 |
|---|---|---|---|---|
| Nearest arc (no ML, no HMM) | 0.510 | 0.541 | 0.430 | 0.640 |
| HMM, Gaussian emission (σ=5, β=5) | 0.873 | 0.903 | 0.908 | 0.969 |
| HMM, tuned on train (σ=3, β=2) | 0.872 | 0.902 | 0.908 | 0.968 |

Candidate miss rate: 0.12%. Total decode time for all 10 trips: about 2 s on CPU.

## Classifier vs classifier + HMM (shared candidates and labels)

`python scripts/run_clf_hmm.py --beta 10 --emission_w 1` scores every method on the **same**
candidates and reference labels (`data/processed/` from `src.build_dataset`). The classifier's
probability is the HMM emission. Results on student_test (10 trips, 19,133 labelled points):

| Model | Top-1 (classifier only) | Top-1 (+ HMM) | Gain | Continuity, distinct arcs (clf → +HMM) | Route F1 (+HMM) |
|---|---|---|---|---|---|
| HMM only, Gaussian emission (no ML) | – | 0.899 | – | 0.909 | 0.970 |
| **Random Forest** | 0.891 | **0.914** | +2.3 | 0.820 → **0.916** | **0.975** |
| MLP (64,32) | 0.882 | 0.909 | +2.7 | 0.804 → 0.913 | 0.974 |
| KNN (k=25) | 0.871 | 0.898 | +2.7 | 0.778 → 0.907 | 0.972 |
| Logistic Regression | 0.867 | 0.905 | +3.8 | 0.749 → 0.895 | 0.972 |

- Decoder search (`results/clf_hmm_grid_random_forest.csv`) on student_test only; the guidelines
  allow this for the decoder. Only the product β × emission_w matters, because rescaling every
  log-score by a constant cannot change the best path. The best is β·w ≈ 10.
- RF + HMM beats RF alone on all 10 trips (+0.1 to +4.8 points). The HMM alone is slightly better
  than RF + HMM on trip 00000000 (0.971 vs 0.968).
- KNN takes about 200 s to score the student-test candidates, which is too slow for the live API.
- Per-trip results: `results/clf_hmm_student_test_per_trip.csv`. Probabilities for every candidate:
  `results/probs_<model>_student_test.parquet`.

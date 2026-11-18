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

# PathSnap — Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University.

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
| `src/feature_analysis.py` | Feature distributions, rule baselines, LightGBM importance |
| `src/hmm.py` | HMM + Viterbi decoder (emissions = classifier, transitions = network vs GPS distance) |
| `src/eval_hmm.py` | Decoder tuning on held-out train trips; classifier-only vs HMM on student-test |
| `src/tuning_report.py` | Tuned vs untuned comparison, optimisation history plot |
| `src/tune.py` | Optuna (TPE) Bayesian optimisation of LightGBM |
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
| **LightGBM** | **0.929 ± 0.011** | **0.891** | **0.935** | **0.066** | **0.953** |
| Rule: nearest arc | – | 0.540 | – | – | – |

Plots: `python -m src.plot_results` → `outputs/figures/`.
Models are saved xz-compressed (uncompressed random forest is ~260 MB, over GitHub's limit).

## Route reconstruction (student-test, LightGBM, beta=3 tuned on held-out train trips)
| | Top-1 | Continuity | Route F1 |
|---|---|---|---|
| Classifier only | 0.891 | 0.953 | 0.900 |
| Classifier + HMM | **0.925** | **0.979** | **0.935** |

## Bayesian optimisation (Optuna TPE, LightGBM)
Objective: 3-fold GroupKFold log loss on train trips (calibrated probabilities feed the HMM).
19 trials (15 complete, 3 pruned). Run `python -m src.tune`, then `python -m src.tuning_report`.

| Student-test | Classifier Top-1 | Test log loss | HMM Top-1 | Inference |
|---|---|---|---|---|
| Untuned LightGBM | 0.891 | 0.0656 | 0.925 | 0.099 ms/pt |
| **Tuned LightGBM** (deployed) | 0.890 | 0.0663 | **0.938** | 0.064 ms/pt |

## Deployment (two laptops)

**Laptop 1: API**
```bash
pip install -r requirements.txt
# development: serves the 10 student_test trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
# evaluation day: point it at the evaluator trips folder (one sub-folder per trip id)
PATHSNAP_TRIPS_DIR="/path/to/evaluator_trips" uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Find Laptop 1's IP: `ipconfig getifaddr en0` (macOS) or `ipconfig` (Windows). Every request
and its server-side accuracy is logged in this terminal.

**Laptop 2: UI**
```bash
cd ui
python3 -m http.server 5500
```
Open `http://localhost:5500`, enter `http://<laptop-1-ip>:8000`, click Connect, choose a trip,
click Reconstruct route.

**Endpoints:** `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`.
Reference routes stay on the API server; the UI receives only the GPS track, the predicted
route and aggregate metrics.

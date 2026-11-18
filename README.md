# PathSnap — Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University.

Problem walkthrough for the team: [PROBLEM_EXPLAINED.md](PROBLEM_EXPLAINED.md)

## Final model

**Random Forest classifier → HMM + Viterbi decoder** (β = 2, switch penalty = 4, gap split = 60 s).

| | Classifier only | **RF + HMM** | Continuity (RF + HMM) |
|---|---|---|---|
| Train, 80 trips, out-of-fold (used for every choice) | 0.885 | **0.945** (fold std 0.013) | – |
| Student-test, 10 trips | 0.890 | **0.942** | 0.973 |
| Evaluator-test, 10 trips (one run of the frozen model) | 0.862 | **0.959** | 0.962 |

Top-1 = share of reliably labelled GPS points whose predicted arc is the reference arc.

- **Deployed config:** [`outputs/hmm/deployed_config.json`](outputs/hmm/deployed_config.json). The API and
  `src/final_eval.py` both load the model through this file (the UI shows it via `GET /health`), so the
  demo and the reported numbers use the same model.
- **How it was chosen:** `python -m src.tune_decoder --model random_forest`. RF probabilities are produced
  out-of-fold (5-fold GroupKFold by trip) for all 80 train trips. The decoder grid (β, switch penalty,
  stationary pooling) is searched on those, and student-test is checked once. The best setting is an
  interior optimum of the grid (`outputs/hmm/random_forest_decoder_grid_train_oof*.csv`).
- **Why Random Forest, not LightGBM:** both classifiers reach the same Top-1 (0.890 vs 0.891). Through the
  same decoder, RF + HMM ≥ LightGBM + HMM, and the decoder tuning was done with RF.
- **What the HMM changes on student-test:** it corrects 1,393 points and breaks 406, a net gain of +5.2 points
  (`outputs/final/final_student_test_per_trip.csv`).
- **Tried and rejected** (selection on train OOF only):
  - Stationary pooling: no gain at 4 m, worse at 8–12 m.
  - Two-stage context RF (`src/context.py`, `src/train_context.py`): +0.08 points on train OOF, which is
    within fold noise, and 0.931 on student-test. The simpler model is kept.
- **Evaluator-test** was used only once, after the model and decoder were frozen. To reproduce, run the API
  on the evaluator folder; every request logs its server-side accuracy.

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
                                       # (also: random_forest, mlp, knn, lightgbm, lightgbm_tuned)
python -m src.tune_decoder --model random_forest   # OOF probs + decoder grid on train -> outputs/hmm/
python -m src.final_eval               # deployed model on student_test: per-trip table + final maps -> outputs/final/
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
| `src/hmm.py` | HMM + Viterbi decoder (emissions = classifier, transitions = network vs GPS distance, switch penalty, bounded cached Dijkstra) |
| `src/tune_decoder.py` | **Final decoder tuning:** out-of-fold probabilities on all 80 train trips, grid search, one student-test check |
| `src/context.py`, `src/train_context.py` | Two-stage context classifier (evaluated and rejected; see "Final model") |
| `src/pipeline.py` | Inference for the API: loads `outputs/hmm/deployed_config.json`, predicts, scores server-side |
| `src/eval_hmm.py` | Earlier decoder tuning on 20 held-out train trips (LightGBM, β only) |
| `src/tuning_report.py` | Tuned vs untuned comparison, optimisation history plot |
| `src/final_eval.py` | Final pipeline per-trip results, HMM fixed/broke analysis, final Folium maps |
| `src/tune.py` | Optuna (TPE) Bayesian optimisation of LightGBM |
| `src/plot_results.py` | Comparison plots + MLP loss curve |
| `src/viz_map.py` | Folium maps (GPS, road network, reference, prediction) |
| `outputs/results/` | `model_comparison.csv`, per-fold CV scores, predictions |
| `outputs/models/` | Saved models (`random_forest.joblib` is deployed) |
| `outputs/hmm/` | Decoder grids, tuned settings, `deployed_config.json` |
| `outputs/final/` | Final per-trip results and Folium maps of the deployed pipeline |
| `api/main.py` | FastAPI backend (Laptop 1) |
| `ui/index.html` | Evaluator UI (Laptop 2), with Leaflet bundled in `ui/vendor/` |
| `pathsnap/` | First HMM prototype and experiments (see the section below) |
| `scripts/` | Prototype HMM runs, classifier vs classifier+HMM comparison, candidate export, maps |
| `results/` | Prototype HMM experiment logs and maps |

## Key decisions
- **Distances:** per-trip UTM projection (trips are worldwide; degrees are not distances).
- **Labels:** `.route` is an arc sequence, not per-point labels. Points are aligned with a
  monotonic DP (route position can only stay or advance). Points >50 m from their aligned
  arc are marked unreliable (-1) and excluded (e.g. train trip 54: GPS continues ~5 h past the
  end of the reference route).
- **Candidates:** all arcs within 40 m → ~7 candidates/point, candidate miss rate 0.55% (train), 0.07% (student-test).
- **No leakage:** GroupKFold by trip; scaler fitted inside the pipeline; every model and decoder choice
  made on train trips. Student-test is used for comparisons. Evaluator trips are used once, for the final
  run of the frozen model, and are git-ignored.

## Classifier comparison (student-test, classifier only, no HMM)
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

On their own, Random Forest, LightGBM and tuned LightGBM are tied (Top-1 0.890–0.891). The final choice is
made after the HMM (next section).

## Route reconstruction (student-test, classifier + HMM)

| Classifier | Decoder settings | Chosen on | Top-1 |
|---|---|---|---|
| LightGBM | β = 3 | 20 held-out train trips (`src/eval_hmm.py`) | 0.925 |
| Tuned LightGBM | β = 3 | 20 held-out train trips | 0.938 |
| Random Forest | β = 3 | (same decoder, for comparison) | 0.939 |
| **Random Forest (final)** | **β = 2, switch penalty = 4** | **all 80 train trips, out-of-fold (`src/tune_decoder.py`)** | **0.942** |

After the HMM, differences of under about 1 point are within noise. Untuned and tuned LightGBM have almost
identical probabilities yet differ by 1.3 points after decoding.

## Bayesian optimisation (Optuna TPE, LightGBM)
Objective: 3-fold GroupKFold log loss on train trips (calibrated probabilities feed the HMM).
19 trials (15 complete, 3 pruned). Run `python -m src.tune`, then `python -m src.tuning_report`.

| Student-test | Classifier Top-1 | Test log loss | HMM Top-1 | Inference |
|---|---|---|---|---|
| Untuned LightGBM | 0.891 | 0.0656 | 0.925 | 0.099 ms/pt |
| Tuned LightGBM | 0.890 | 0.0663 | 0.938 | 0.064 ms/pt |

Tuning improved the CV objective (log loss 0.0778 → 0.0768) and reduced overfitting (train macro F1
0.950 → 0.943), but did not change classifier Top-1. The HMM difference is within decoding noise.
The deployed model is the Random Forest (see "Final model").

---

## HMM prototype (`pathsnap/`): earlier experiments

This is the first HMM implementation, used to develop and compare decoding ideas. The final decoder is
`src/hmm.py`. The results in this section use the prototype's own candidates/labels, or β chosen on
student-test, so they are not the final numbers.

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
- The prototype decoder treats more than 10 m of backward movement on the same arc as impossible. Allowing it,
  as `src/hmm.py` does, adds about +2.3 points (GPS jitter while the car is stopped). That is why
  `src/hmm.py` scores higher.

---

## Deployment (two laptops)

**Laptop 1: API.** It serves the deployed model from `outputs/hmm/deployed_config.json` (Random Forest,
β = 2, switch penalty = 4). The startup log line and `GET /health` show the model and every decoder setting.
```bash
pip install -r requirements.txt
# development: serves the 10 student_test trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
# evaluation day: point it at the evaluator trips folder (one sub-folder per trip id)
PATHSNAP_TRIPS_DIR="/path/to/evaluator_trips" uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Windows PowerShell:
```powershell
$env:PATHSNAP_TRIPS_DIR = "C:\path\to\evaluator_test"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Find Laptop 1's IP: `ipconfig getifaddr en0` (macOS) or `ipconfig` (Windows; use the **Wi-Fi** adapter's
IPv4, not VirtualBox/VMware addresses). Every request and its server-side accuracy is logged in this terminal.

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

**Pre-demo checklist (run these on the actual Wi-Fi you will use)**
1. Both laptops on the same network. College Wi-Fi often blocks laptop-to-laptop traffic
   ("client isolation"); if step 4 fails, put both laptops on one phone hotspot.
2. Laptop 1: start the API with `--host 0.0.0.0` (not the default 127.0.0.1, which only
   accepts connections from the same machine).
3. Laptop 1 firewall. macOS: System Settings → Network → Firewall: off, or allow incoming connections
   for Python. Windows: allow Python on Public and Private networks, or in an admin PowerShell run
   `New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any`.
4. Laptop 2: `curl http://<laptop-1-ip>:8000/health` must return `"n_trips": 10` before
   opening the UI. If it hangs, the problem is the network or firewall, not the code.
5. The UI offers **Dark** and **Streets** basemaps (Esri tiles). Laptop 2 needs internet for the map
   background; Leaflet is bundled, so routes and metrics still work without it.
6. Install the same library versions on Laptop 1 as were used to train the model
   (`pip freeze | grep -iE "scikit|lightgbm"` on the training machine): the pickled
   Random Forest (scikit-learn) model warns or can break under a different scikit-learn version.
7. The API finds trips in `Student folder/student_test` or
   `Student folder/Student folder/student_test` automatically; for the evaluator trips set
   `PATHSNAP_TRIPS_DIR`. Take the report screenshots of the Laptop 1 terminal logs here.

# PathSnap: Problem Statement Explained

> A team guide to the UE24CS352A ML Hackathon: what we're building, what the data looks like, how to clean it, which models to try, and what we must submit.
> Source: the five PDFs in `PathSnap_student_materials/`, plus our own scan of all 90 trips in `Student folder/`.

---

## Table of Contents

1. [The Problem in Plain Words](#1-the-problem-in-plain-words)
2. [The Dataset](#2-the-dataset)
3. [The Full Pipeline](#3-the-full-pipeline)
4. [Data Cleaning and Preparation](#4-data-cleaning-and-preparation)
5. [Models to Train](#5-models-to-train)
6. [HMM + Viterbi Route Reconstruction](#6-hmm--viterbi-route-reconstruction)
7. [Deployment (Demo Day)](#7-deployment-demo-day)
8. [Deliverables Checklist](#8-deliverables-checklist)
9. [Rules That Can Cost Us Heavily](#9-rules-that-can-cost-us-heavily)
10. [Suggested Order of Work](#10-suggested-order-of-work)
11. [Glossary](#11-glossary)

---

## 1. The Problem in Plain Words

GPS doesn't record exactly where a car is. Tall buildings, bad signal and tunnels push the reading a few metres off, and sometimes hundreds of metres. If you plot raw GPS points on a map, the car looks like it's driving through buildings or on the next street over.

**Map-matching** means taking those noisy GPS dots and working out which roads the car actually drove on. The output is a clean, connected route along the road network.

```
Noisy GPS dots:   •  •   •  • •    •   •
Road network:    ───A───┬───B───────C───
                        └───D───
Our answer:      point1→A, point2→A, point3→B, ...  (a connected route)
```

We aren't predicting one number or one class per trip. For **every GPS point**, we predict **which road segment (arc)** it belongs to, and the sequence of those segments has to form a route a car could really drive.

**The project is called PathSnap.** The system has to:

1. Train an ML classifier that scores how likely each nearby road is to be the true road for a GPS point.
2. Use an **HMM + Viterbi** decoder to turn those scores into one connected route.
3. Serve it through a **FastAPI** backend on one laptop and a **UI** on another laptop.

---

## 2. The Dataset

### 2.1 Folder layout

```
Student folder/Student folder/
├── train/          80 trips   (00000001, 00000002, ...)
└── student_test/   10 trips   (for comparing models during development)

(Evaluators hold 10 more hidden trips for the final demo.)
```

| Partition | Trips | Allowed use |
|---|---|---|
| Training | 80 | Training, cross-validation, tuning |
| Student testing | 10 | Comparing models during development |
| Evaluator testing | 10 | **Final evaluation only. We must never train or tune on these** |

### 2.2 Files inside each trip folder

Each trip folder holds four files. They have **no headers** and are separated by whitespace.

| File | Columns | Meaning | Example row |
|---|---|---|---|
| `.track` | longitude, latitude, time (s) | Noisy GPS readings, about one per second | `54.963519  56.251559  1.0` |
| `.nodes` | longitude, latitude | Road intersections and shape points. **The row number is the node ID** (0-based) | `54.335879  56.117280` → node 0 |
| `.arcs` | from_node, to_node | Road segments between two nodes. **The row number is the arc ID**. Arcs are **directed** | `0  2433` → arc 0 goes from node 0 to node 2433 |
| `.route` | arc_id | **Ground truth**: the ordered list of arcs the car drove | `17630, 17633, 211, ...` |

```
.track  ──(GPS point is near)──►  .arcs  ──(made of)──►  .nodes
                                    ▲
.route  ──(list of arc IDs)─────────┘
```

> ⚠️ **The key catch:** a route has about 280 arcs while the track has about 1,500 GPS points. **Row *i* of the GPS file does not line up with row *i* of the route.** No label tells us "GPS point 37 → arc 211". We have to build that point-to-arc label ourselves (see [Section 4, Step 5](#step-5--build-labels-from-the-route)).

> ⚠️ **Each trip has its own local road network.** Node and arc IDs mean nothing across trips: arc 211 in trip 1 is a different road from arc 211 in trip 2.

### 2.3 What the scan of all 90 trips shows

| Fact | Value | Why it matters |
|---|---|---|
| GPS points per trip | 361 to **19,313** (median about 1,500) | Some trips are large, so the code has to be efficient |
| Road network size | 649 to **307,392 arcs** per trip | We need a spatial index (KD-tree or STRtree), not brute-force search |
| Sampling rate | Median trip has 99% of steps exactly 1 s; the worst has only 53% | Mostly clean 1 Hz data |
| Large time gaps | One trip has a **153,064 s (about 42 h)** gap; another has 52,810 s | Split tracks at gaps or handle them in the HMM |
| Missing values (NaN) | **None** | No imputation needed |
| Out-of-order timestamps | **None** | Good |
| Geographic spread | Longitude −100 to 137, latitude −34 to 62 | **Trips are spread around the world**, so each trip needs its own local projection |
| GPS distance to the true route | Median **4.5 m**, 95th percentile about 14 m, worst **558 m** | A 30–50 m search radius catches nearly every true road. 4 trips have bad outlier points |
| Arcs with a reverse twin | About 90% | Most roads are two-way and stored as 2 arcs (A→B and B→A), so **direction matters** |
| Consecutive route arcs that connect | 98–100% | The ground truth itself has small breaks. The docs warn that reference routes "may have limitations" |
| Routes that reuse an arc | 10 trips | Loops and U-turns happen |
| Isolated nodes | None | Every node belongs to at least one arc |

### 2.4 EDA the organisers expect us to show

- Trip counts, file counts and GPS points per trip
- Summary statistics for trajectory length and sampling intervals, plus a missing-value analysis
- Timestamp ordering and irregular gaps
- Geographical extent of each trajectory
- Road-network size, connectivity and disconnected components
- Trips with anomalies
- **Folium maps** of GPS tracks over the road network
- A short written log of data-quality issues, assumptions and decisions

---

## 3. The Full Pipeline

```
Raw files
   │
   ▼
Clean + project to metres
   │
   ▼
Candidate generation  (nearby arcs for each GPS point)
   │
   ▼
Label alignment  (which candidate is the true arc?)
   │
   ▼
Feature engineering
   │
   ▼
Train ≥ 5 classifiers  ──►  pick the best  ──►  Bayesian hyperparameter tuning
   │
   ▼
HMM + Viterbi  (turns per-point scores into one connected route)
   │
   ▼
Export model  ──►  FastAPI (Laptop 1)  ◄──►  UI (Laptop 2)
```

---

## 4. Data Cleaning and Preparation

### Step 1: Load and check
Read every file and confirm the column order (**longitude first**), the row counts, and that every arc's node IDs exist in `.nodes`.

### Step 2: Convert to metres
Distances in degrees are meaningless because a degree of longitude shrinks toward the poles. Since trips are worldwide, project **each trip** into its own local metric system:
- the trip's **UTM zone** via `pyproj`, or
- a **local equirectangular** projection centred on the trip, which is accurate enough over a few km.

The docs ask us to **justify** this choice in the report.

### Step 3: Clean the GPS
- Remove duplicate timestamps and any zero-length time steps.
- **Split the trip into segments wherever there's a large time gap** (for example over 30–60 s). The 42-hour gap is clearly two separate drives.
- Flag or remove impossible jumps, such as an implied speed over about 70 m/s.
- *Optionally* **downsample**. At 1 Hz, consecutive points are only about 1–15 m apart. Keeping one point every 3–5 s makes the HMM much faster with little loss in accuracy.

### Step 4: Generate candidates
For each GPS point, find all arcs within radius **R** (for example 50 m), or the **K nearest** arcs (for example 8–15). These are the roads the point *might* be on.

Track the **Candidate Miss Rate**: the share of points whose true arc isn't in the candidate set. It's a required metric.

### Step 5: Build labels from the route
Each GPS point needs to be paired with its true arc.
- Restrict the candidates to arcs that appear in the trip's `.route`.
- Walk the GPS points and the route **together, in order** (the route is chronological). Assign each point to the nearest route arc while only ever moving *forward* along the route. A small dynamic-programming alignment does this cleanly.
- Plain "nearest route arc" is simpler, but it breaks where a route doubles back on itself, which happens in 10 trips.

### Step 6: Build the classification table
Each row is one **(GPS point, candidate arc)** pair:

| trip | point | candidate_arc | features… | label |
|---|---|---|---|---|
| 1 | 37 | 211 | … | **1** (true arc) |
| 1 | 37 | 212 | … | 0 |
| 1 | 37 | 5004 | … | 0 |

This makes it a **binary classification** problem: *is this candidate the correct road?* The classes are imbalanced, with about 1 positive per 10 candidates.

### Step 7: Engineer features

| Feature | Intuition |
|---|---|
| Perpendicular distance from point to arc (m) | Closer arcs are more likely. This is the strongest single feature |
| Distance rank among the point's candidates | Relative closeness matters more than absolute distance |
| Distance minus the nearest candidate's distance | Is this arc clearly worse than the best one? |
| **Heading difference** between GPS travel direction and arc direction | Separates A→B from B→A, and parallel roads from crossing roads. Very important |
| Speed (from consecutive points) | Fast cars are on main roads; at low speed the heading is noisy |
| Projection position *t* ∈ [0, 1] along the arc, and whether it's clamped at an end | Points past a segment's end probably belong to the next one |
| Arc length | Short arcs near intersections are easy to confuse |
| Number of candidates near the point | Dense areas are harder |
| Distance to the arc averaged over neighbouring points (±2 window) | Smooths out single-point noise |
| Is this arc reachable from the previous point's best arc? | Adds route-context information |

> 🔒 Fit any learned preprocessing, such as scalers or encoders, **on training data only**.

---

## 5. Models to Train

The rules require **at least 5 models from different algorithm families**, compared with real experimental results. Picking a model without experiments costs marks.

### 5.1 Proposed set and expected ranking

| # | Model | Family | Expected performance | Why include it |
|---|---|---|---|---|
| 1 | **LightGBM / XGBoost** | Boosting | ⭐ **Likely best** | Handles mixed-scale features, nonlinear interactions such as distance × heading, and class imbalance. Fast, with good probabilities. Our likely final model |
| 2 | **Random Forest / Extra Trees** | Ensemble | Strong, close second | Stable and gives feature importances. Probabilities are less well calibrated |
| 3 | **HistGradientBoosting** (sklearn) | Boosting | Similar to LightGBM | A second boosting comparison |
| 4 | **MLP** (2–3 hidden layers) | Neural network | Good if features are scaled | Produces epoch-wise loss curves, which the report requires for models that have them |
| 5 | **Logistic Regression** | Linear | Decent baseline | Well-calibrated probabilities and easy to interpret |
| 6 | KNN (distance-weighted) | Instance-based | Moderate | Sensitive to scaling and slow at inference |
| 7 | Gaussian Naive Bayes | Probabilistic | Weakest | Assumes independent features, but distance and rank are correlated. Useful to show *why* it fails |

**Plan:** train Logistic Regression, Naive Bayes or KNN, Random Forest, MLP and LightGBM (plus XGBoost if time allows). If the results confirm it, tune **LightGBM** as the final model. Gradient-boosted trees usually win on engineered tabular features like these, but **the decision has to come from our measured results**.

### 5.2 How we evaluate

- **Group-aware cross-validation:** use `GroupKFold` with groups = trip ID. A random row split leaks points from the same trip into both train and validation and makes scores falsely high.
- **Primary selection metric: observation-level Top-1 accuracy.** For each GPS point, take the highest-scoring candidate and check whether it's the true arc. This reflects the real task better than Macro F1, which is computed over thousands of easy negative rows.
- **Log loss matters** because the HMM consumes the classifier's probabilities, so well-calibrated probabilities give better routes.
- We still have to report all of these: Accuracy, Precision, Recall, F1, Confusion Matrix, Log Loss, Top-1 Accuracy, Candidate Miss Rate, Route Continuity, Sequence-level performance, Training time and Inference time.

### 5.3 Bayesian hyperparameter optimisation

Use **Optuna**; its default TPE sampler is a Bayesian method. Run it with GroupKFold on the 80 training trips only.

Example search space for LightGBM:

| Hyperparameter | Range |
|---|---|
| `num_leaves` | 15 – 255 |
| `learning_rate` | 0.01 – 0.3 (log scale) |
| `n_estimators` | 100 – 1500 |
| `min_child_samples` | 5 – 200 |
| `subsample`, `colsample_bytree` | 0.5 – 1.0 |
| `reg_lambda` | 1e-3 – 10 (log scale) |

Run 50–100 trials and **save the trial history as a CSV**. Then compare tuned vs untuned on CV Macro F1, student-test Top-1, student-test Macro F1, Log Loss and inference time.

---

## 6. HMM + Viterbi Route Reconstruction

### Why we need it
The classifier judges each GPS point **independently**. It might say point 10 → road A, point 11 → road X (a parallel street), point 12 → road A again. A car can't do that. An HMM looks at the **whole sequence** and uses road connectivity to choose a route that makes sense.

### The parts

| HMM part | In PathSnap |
|---|---|
| **Hidden states** | The candidate arcs at each GPS point (the true road we can't see directly) |
| **Observations** | The GPS points |
| **Emission probability** | How well this GPS point fits this arc. **Our classifier's predicted probability goes here**; this is where the ML model plugs in |
| **Transition probability** | How plausible it is to move from arc *i* (at point *t*) to arc *j* (at point *t+1*). Compare the **shortest-path driving distance** through the road graph with the **straight-line distance** between the two GPS points. If they're similar the transition is likely; if the road path is much longer or doesn't exist, it's unlikely. A common form is `p ∝ exp(−│d_route − d_gps│ / β)` (Newson & Krumm, 2009) |
| **Viterbi** | Dynamic programming that finds the single most probable sequence of arcs across all points |

### Viterbi in one line
```
score[t][j] = log emission[t][j] + max_i ( score[t-1][i] + log transition[i→j] )
```
Keep a back-pointer to the best `i` for each `(t, j)`, then trace back from the best final state.

### Edge cases we must handle (the report asks about each)
- **No candidates near a point:** skip the point, or widen the search radius.
- **No road path between consecutive candidates:** give that transition a tiny probability. If every transition is impossible, restart the HMM from that point (an "HMM break").
- **Large GPS gaps:** split the trip there, or allow longer transitions.
- **Decoder parameters** (β, the GPS noise σ, the search radius): tune them on train and student_test data only.

### Required comparison
Report **classifier-only** vs **classifier + HMM** on Top-1 accuracy, sequence-level performance and **route continuity** (the share of consecutive predicted arcs that actually connect). Also discuss cases where the HMM *didn't* help.

---

## 7. Deployment (Demo Day)

| Laptop | Runs | Responsibilities |
|---|---|---|
| **Laptop 1** | FastAPI backend | Holds the trained model, HMM/Viterbi, road networks and the **hidden ground truth (server-side only)**. Runs the full pipeline and computes accuracy |
| **Laptop 2** | UI (for example Streamlit or HTML/JS) | Lists the 10 evaluation trips, sends the chosen trip ID to the API, and shows the predicted route on a map along with the accuracy |

**Demo flow:**
1. The evaluator opens the UI on Laptop 2.
2. They pick any of the 10 evaluation trips.
3. The UI sends the trip ID to the API on Laptop 1.
4. The API runs the full pipeline and computes accuracy against its internal reference data.
5. The UI displays the route and the results.
6. **We show the API logs on Laptop 1** to prove the request was really processed there.

It must work for **all 10 trips**. No hardcoded outputs, and the UI must never bypass the API.

---

## 8. Deliverables Checklist

### Deliverable 1: Public GitHub repository
- [ ] Data processing (read, inspect, clean, preprocess)
- [ ] Feature engineering code
- [ ] Training code for **every** model we tried
- [ ] Evaluation code: metrics, plots, confusion matrices
- [ ] Bayesian optimisation code, search configuration and trial logs
- [ ] HMM / Viterbi code
- [ ] FastAPI backend, including model loading
- [ ] UI code
- [ ] Deployment config and `requirements.txt`
- [ ] Folium map-generation and plotting scripts
- [ ] Result CSVs, experiment logs and config files
- [ ] Saved model weights and configs (within size limits)
- [ ] README with install, run and deploy instructions
- [ ] Clearly named folders and **no secrets committed**

### Deliverable 2: One consolidated PDF report
- [ ] **EDA**: preprocessing, observed patterns, chosen features and why, missing-value analysis, GPS plots, road-network plots, at least one Folium map, and which partition each analysis used
- [ ] **Model comparison table** (≥ 5 models): Model, Family, Key Config, CV Macro F1, Top-1 Accuracy, Route Continuity, Student-Test Macro F1, Log Loss, Inference Time
- [ ] **Per-model subsection** answering:
  - Why this algorithm suits the problem, plus its strengths and limitations
  - Important hyperparameters and their initial values, and the loss or objective
  - Student-test performance (Top-1, Macro F1, Log Loss)
  - Overfitting or underfitting, with evidence
- [ ] **Plots**: CV Macro F1 across models, student-test Top-1 and F1, log loss comparison, optional confusion matrices, and train/validation loss curves **only** for epoch-based models
- [ ] **Bayesian optimisation**: search space, objective, number of trials, best values, tuned vs untuned
- [ ] **HMM section**: states, emissions, transitions, gaps, disconnected segments, and the improvement backed by numbers
- [ ] **Folium maps** with raw GPS, road network, reference route, predicted route, a legend and a layer control
- [ ] **Screenshots** of the working UI and of the API logs on the receiving laptop

> 📌 There is a **first-phase evaluation before the lunch break**.

---

## 9. Rules That Can Cost Us Heavily

- 🚫 **Never use evaluator-test data** for training, feature selection, tuning or choosing thresholds.
- 🚫 **No hardcoded predictions.** Repositories will be checked for them.
- 🚫 **Ground truth must never reach the model or the UI.** It stays on the server.
- 🚫 **No sharing** code, models, configs or results with other teams. Submissions are compared across teams and sections.
- 🚫 **No fabricated results.** Every number and graph in the report must be reproducible from the repo. No fake training curves.
- ✅ **Cross-validation must split by trip** (GroupKFold).
- ✅ **Every team member must be able to explain** the dataset, preprocessing, features, model choice, Bayesian optimisation, HMM, Viterbi, API and UI in the viva.
- ✅ Disclose AI assistance as required by the event policy.

---

## 10. Suggested Order of Work

| # | Task | Output |
|---|---|---|
| 1 | EDA notebook covering everything in Section 2, plus Folium maps | Plots, stats, data-quality log |
| 2 | Projection, cleaning and candidate generation | Candidate table and miss rate |
| 3 | Label alignment (GPS point → true arc) | Labelled candidates |
| 4 | Feature engineering | Feature table (parquet) |
| 5 | Train 5–7 models with GroupKFold, logging metrics and timings | Comparison table and plots |
| 6 | Optuna tuning on the best model | Trial log, tuned model |
| 7 | HMM + Viterbi, compared with classifier-only | Continuity and accuracy gains |
| 8 | Export the model (`joblib`), then build the FastAPI endpoint and UI | Working two-laptop demo |
| 9 | Write the report | Final PDF |

**A possible split for the team:**
- **Person A:** EDA, cleaning, Folium maps
- **Person B:** candidate generation, labels, features
- **Person C:** model training, comparison, Optuna
- **Person D:** HMM/Viterbi, API, UI

Everyone has to understand every part for the viva.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Map-matching** | Snapping noisy GPS points onto the roads that were actually driven |
| **Node** | A point in the road network, such as an intersection or a bend |
| **Arc** | A directed road segment between two nodes |
| **Route** | The ground-truth ordered list of arcs the vehicle drove |
| **Candidate** | A nearby arc that a GPS point *might* belong to |
| **Candidate Miss Rate** | The share of GPS points whose true arc isn't among their candidates |
| **Top-1 Accuracy** | The share of GPS points where the highest-scored candidate is the true arc |
| **Route Continuity** | The share of consecutive predicted arcs that are actually connected in the road graph |
| **GroupKFold** | Cross-validation that keeps whole trips together in the same fold |
| **HMM** | Hidden Markov Model: hidden true roads that generate the observed GPS points |
| **Emission probability** | P(GPS point │ the car is on this arc) |
| **Transition probability** | P(moving from arc *i* to arc *j* between two consecutive points) |
| **Viterbi** | Dynamic-programming algorithm that finds the most likely hidden-state sequence |
| **Bayesian optimisation** | Hyperparameter search that uses past trials to choose promising next ones (for example Optuna's TPE) |
| **Folium** | Python library for interactive Leaflet maps |
| **UTM** | Universal Transverse Mercator, a projection that turns lon/lat into metres |

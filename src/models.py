"""Model registry. Add a model = add one entry here, then
   python -m src.train --model <name>

Optional spec keys
  max_train_rows : fit on a random subsample of this many rows (for models whose
                   cost grows badly with data size). Recorded in the results.
  needs_groups   : fit() receives trip ids, used for a trip-held-out validation split.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


class EpochMLP(BaseEstimator, ClassifierMixin):
    """MLP trained epoch by epoch so we record real train/validation log-loss curves.

    Validation = 10% of the *training trips* (group split, so no trip is in both),
    used for early stopping and the loss plot. The scaler is fitted on the
    training part only.
    """

    def __init__(self, hidden=(64, 32), alpha=1e-4, lr=1e-3, batch_size=512,
                 max_epochs=60, patience=6, random_state=SEED):
        self.hidden, self.alpha, self.lr = hidden, alpha, lr
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        if groups is None:
            groups = np.arange(len(y)) % 50
        tr, va = next(GroupShuffleSplit(1, test_size=0.1, random_state=self.random_state)
                      .split(X, y, groups))
        self.scaler_ = StandardScaler().fit(X[tr])
        Xtr, Xva = self.scaler_.transform(X[tr]), self.scaler_.transform(X[va])
        ytr, yva = y[tr], y[va]
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha,
                                  learning_rate_init=self.lr, batch_size=self.batch_size,
                                  random_state=self.random_state)
        rng = np.random.default_rng(self.random_state)
        self.history_ = {"epoch": [], "train_loss": [], "val_loss": []}
        best, best_state, wait = np.inf, None, 0
        for ep in range(self.max_epochs):
            perm = rng.permutation(len(ytr))
            for s in range(0, len(perm), 20000):  # chunks; partial_fit does minibatches inside
                idx = perm[s:s + 20000]
                self.mlp_.partial_fit(Xtr[idx], ytr[idx], classes=[0, 1])
            trl = log_loss(ytr, self.mlp_.predict_proba(Xtr)[:, 1], labels=[0, 1])
            val = log_loss(yva, self.mlp_.predict_proba(Xva)[:, 1], labels=[0, 1])
            self.history_["epoch"].append(ep + 1)
            self.history_["train_loss"].append(trl)
            self.history_["val_loss"].append(val)
            if val < best - 1e-4:
                best, wait = val, 0
                best_state = ([w.copy() for w in self.mlp_.coefs_],
                              [b.copy() for b in self.mlp_.intercepts_])
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.mlp_.coefs_, self.mlp_.intercepts_ = best_state  # restore best epoch
        self.best_epoch_ = int(np.argmin(self.history_["val_loss"])) + 1
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(self.scaler_.transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


MODELS = {
    # Model 1 — linear/probabilistic baseline.
    # StandardScaler is fitted inside the pipeline, so in CV it only ever sees
    # the training folds (no preprocessing leakage).
    "logreg": {
        "family": "Linear / probabilistic",
        "config": "StandardScaler + LogisticRegression(C=1.0, lbfgs, max_iter=1000)",
        "make": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        ),
    },
    # Model 2 — bagged trees. No scaling needed (splits are threshold-based).
    # min_samples_leaf=5 limits memorisation of individual noisy GPS points.
    "random_forest": {
        "family": "Ensemble (bagging)",
        "config": "RandomForest(n_estimators=100, max_features='sqrt', min_samples_leaf=5)",
        "make": lambda: RandomForestClassifier(
            n_estimators=100, max_features="sqrt", min_samples_leaf=5,
            n_jobs=-1, random_state=SEED),
    },
    # Model 3 — neural network with epoch-wise loss history (trip-held-out validation).
    "mlp": {
        "family": "Neural network",
        "config": "MLP(64,32), ReLU, Adam lr=1e-3, alpha=1e-4, batch=512, early stop patience=6, 300k-row sample",
        "make": lambda: EpochMLP(hidden=(64, 32)),
        "max_train_rows": 300_000,
        "needs_groups": True,
    },
    # Model 4 — instance-based. Distances are meaningless without scaling.
    # Fitted on a sample: KNN stores the data and its prediction cost grows with it.
    "knn": {
        "family": "Instance-based",
        "config": "StandardScaler + KNN(k=25, weights='distance'), 200k-row sample",
        "make": lambda: make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        "max_train_rows": 200_000,
    },
}

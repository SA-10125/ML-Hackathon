"""Second-stage (stacked) candidate classifier with temporal / network context.

The first-stage classifier scores each (GPS point, candidate arc) on its own. The second
stage adds context computed from first-stage probabilities of NEIGHBOURING points:

  p0             first-stage probability of this candidate
  p_norm         p0 / sum of p0 over the point's candidates
  p_gap          p0 - best p0 at this point
  p_rank         rank of p0 among the point's candidates (1 = best)
  same_k{1,2,5}  mean first-stage probability of the SAME arc at points t-k..t+k (0 if absent)
  prev_conn      1 if this arc equals / directly follows the best arc of point t-1
  next_conn      1 if this arc equals / directly leads into the best arc of point t+1
  prev_compat    best p0 among t-1 candidates that equal / lead into this arc
  next_compat    best p0 among t+1 candidates that equal / follow from this arc

Leakage control: on train trips the first-stage probabilities are OUT-OF-FOLD (each trip is
scored by a model that never saw it), so the second stage learns from probabilities that look
like the ones it will get on unseen trips.
"""
import numpy as np
import pandas as pd

from .features import FEATURES

CTX = ["p0", "p_norm", "p_gap", "p_rank", "same_k1", "same_k2", "same_k5",
       "prev_conn", "next_conn", "prev_compat", "next_compat"]
FEATURES2 = FEATURES + CTX


def _trip_context(c: pd.DataFrame, arcs: np.ndarray) -> pd.DataFrame:
    """c: one trip's candidate rows with columns pt, arc, p0 (any order). Returns CTX columns."""
    c = c[["pt", "arc", "p0"]].copy()
    g = c.groupby("pt").p0
    c["p_norm"] = c.p0 / g.transform("sum").clip(lower=1e-9)
    c["p_gap"] = c.p0 - g.transform("max")
    c["p_rank"] = g.rank(ascending=False, method="min")

    key = c[["pt", "arc", "p0"]]
    for k in (1, 2, 5):
        acc = np.zeros(len(c))
        for o in [x for x in range(-k, k + 1) if x != 0]:
            m = c[["pt", "arc"]].assign(pt=c.pt + o).merge(key, on=["pt", "arc"], how="left")
            acc += m.p0.fillna(0.0).to_numpy()
        c[f"same_k{k}"] = acc / (2 * k)

    best = c.loc[c.groupby("pt").p0.idxmax(), ["pt", "arc"]].set_index("pt").arc
    frm, to = arcs[:, 0], arcs[:, 1]
    a = c.arc.to_numpy()
    pb = c.pt.map(lambda p: best.get(p - 1, -1)).to_numpy()
    nb = c.pt.map(lambda p: best.get(p + 1, -1)).to_numpy()
    c["prev_conn"] = ((pb >= 0) & ((pb == a) | (to[np.maximum(pb, 0)] == frm[a]))).astype(np.int8)
    c["next_conn"] = ((nb >= 0) & ((nb == a) | (to[a] == frm[np.maximum(nb, 0)]))).astype(np.int8)

    # best compatible neighbour probability (pairwise over consecutive points)
    for side, o in (("prev", -1), ("next", 1)):
        nbr = key.rename(columns={"arc": "arc_n", "p0": "p_n"}).assign(pt=key.pt - o)
        m = c[["pt", "arc"]].reset_index().merge(nbr, on="pt", how="inner")
        an, ac = m.arc_n.to_numpy(), m.arc.to_numpy()
        ok = (an == ac) | ((to[an] == frm[ac]) if side == "prev" else (to[ac] == frm[an]))
        best_p = m[ok].groupby("index").p_n.max()
        c[f"{side}_compat"] = best_p.reindex(c.index).fillna(0.0).to_numpy()
    return c[CTX]


def add_context(C: pd.DataFrame, p0: np.ndarray, arcs_by_trip: dict) -> pd.DataFrame:
    """Return C with CTX columns added (row order preserved)."""
    out = C.copy()
    out["p0"] = p0
    parts = [_trip_context(out[out.trip == t], arcs_by_trip[t]) for t in out.trip.unique()]
    ctx = pd.concat(parts).reindex(out.index)
    for col in CTX:
        out[col] = ctx[col].to_numpy()
    return out


class TwoStage:
    """stage1 (base features) -> context features -> stage2. Used by the API pipeline."""

    def __init__(self, stage1, stage2):
        self.stage1, self.stage2 = stage1, stage2

    def predict_proba_trip(self, cands: pd.DataFrame, arcs: np.ndarray) -> np.ndarray:
        p0 = self.stage1.predict_proba(cands[FEATURES].values)[:, 1]
        c = cands.assign(trip="_")
        c = add_context(c, p0, {"_": arcs})
        return self.stage2.predict_proba(c[FEATURES2].values)[:, 1]

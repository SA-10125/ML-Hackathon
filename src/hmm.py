"""HMM + Viterbi route reconstruction.

Hidden states   at time t: the candidate arcs of GPS point t (from features.generate_candidates).
                Each state keeps its arc id, so the candidate <-> road-segment correspondence
                is preserved end to end.
Emission        log P(correct | features) from the selected classifier (LightGBM).
Transition      between candidate i at t-1 and candidate j at t:
                  d_route = network distance from i's projected position to j's projected
                            position (rest of arc i + shortest path + start of arc j);
                  d_gc    = straight-line distance between the two GPS fixes;
                  log P   = -|d_route - d_gc| / beta        (Newson & Krumm, 2009)
                A real car drives roughly the straight-line distance between two fixes 1 s
                apart, so a candidate pair needing a long detour (or no path at all, e.g.
                disconnected network components) is unlikely / impossible.
Viterbi         delta_t(j) = max_i [ delta_{t-1}(i) + logT(i,j) ] + logE_t(j), with
                backpointers; the best final state is traced back to give one arc per point.
Edge cases
  * point with no candidates  -> skipped (pred = -1), chain continues across it
  * time gap > gap_s seconds  -> chain broken; segments decoded independently
  * every transition impossible (-inf) -> chain broken at that point and restarted from
    emissions only, so one bad step cannot kill the whole route
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

NEG = -1e18


class RoadGraph:
    def __init__(self, trip):
        a, b = trip.arc_a, trip.arc_b
        self.arcs = trip.arcs
        self.len = np.linalg.norm(b - a, axis=1)
        V = len(trip.nodes_xy)
        self.G = coo_matrix((np.maximum(self.len, 1e-3), (self.arcs[:, 0], self.arcs[:, 1])),
                            shape=(V, V)).tocsr()

    def node_dists(self, sources, limit):
        """Shortest network distance from each source node to every node (inf beyond limit)."""
        return dijkstra(self.G, directed=True, indices=sources, limit=limit)


def transition_logp(g, arcs_i, frac_i, arcs_j, frac_j, d_gc, beta, slack=200.0):
    """Matrix (len(i), len(j)) of transition log-probabilities."""
    Li, Lj = g.len[arcs_i], g.len[arcs_j]
    limit = 2.0 * d_gc + slack
    src = g.arcs[arcs_i, 1]                       # end node of each previous arc
    usrc, inv = np.unique(src, return_inverse=True)
    D = g.node_dists(usrc, limit)                 # (n_unique_src, V)
    between = D[inv][:, g.arcs[arcs_j, 0]]        # end of arc i -> start of arc j
    d_route = (1 - frac_i)[:, None] * Li[:, None] + between + (frac_j * Lj)[None, :]
    same = arcs_i[:, None] == arcs_j[None, :]
    d_same = np.abs(frac_j[None, :] - frac_i[:, None]) * Li[:, None]
    d_route = np.where(same, d_same, d_route)
    lp = -np.abs(d_route - d_gc) / beta
    return np.where(np.isfinite(lp), lp, NEG)


def viterbi(trip, cands, logp_emis, beta=10.0, gap_s=60.0, emis_weight=1.0):
    """cands: DataFrame with columns pt, arc, frac (sorted by pt). Returns pred arc per point."""
    g = RoadGraph(trip)
    N = len(trip.xy)
    pred = np.full(N, -1, dtype=np.int64)
    pts = cands.pt.values
    starts = np.r_[0, np.flatnonzero(np.diff(pts)) + 1]
    ends = np.r_[starts[1:], len(pts)]
    steps = [(pts[s], cands.arc.values[s:e], cands.frac.values[s:e], emis_weight * logp_emis[s:e])
             for s, e in zip(starts, ends)]

    seg = []          # list of (pt, arcs, backptr) for the current unbroken segment
    delta = None
    n_breaks = 0

    def flush():
        if not seg:
            return
        j = int(np.argmax(delta))
        for pt, arcs, back in reversed(seg):
            pred[pt] = arcs[j]
            j = back[j] if back is not None else j
        seg.clear()

    prev = None
    for pt, arcs, frac, le in steps:
        if prev is None or (trip.t[pt] - trip.t[prev[0]]) > gap_s:
            if prev is not None:
                n_breaks += 1
            flush()
            delta = le.copy()
            seg.append((pt, arcs, None))
            prev = (pt, arcs, frac)
            continue
        d_gc = float(np.linalg.norm(trip.xy[pt] - trip.xy[prev[0]]))
        T = transition_logp(g, prev[1], prev[2], arcs, frac, d_gc, beta)
        scores = delta[:, None] + T
        back = np.argmax(scores, axis=0)
        best = scores[back, np.arange(len(arcs))]
        if np.all(best <= NEG / 2):           # nothing reachable: break and restart
            n_breaks += 1
            flush()
            delta = le.copy()
            seg.append((pt, arcs, None))
        else:
            delta = best + le
            seg.append((pt, arcs, back))
        prev = (pt, arcs, frac)
    flush()
    return pred, n_breaks

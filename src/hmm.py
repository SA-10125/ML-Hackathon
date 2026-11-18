"""HMM + Viterbi route reconstruction.

Hidden states   at time t: the candidate arcs of GPS point t (from features.generate_candidates).
                Each state keeps its arc id, so the candidate <-> road-segment correspondence
                is preserved end to end.
Emission        log P(correct | features) from the selected classifier.
Transition      between candidate i at t-1 and candidate j at t:
                  d_route = network distance from i's projected position to j's projected
                            position (rest of arc i + shortest path + start of arc j);
                  d_gc    = straight-line distance between the two GPS fixes;
                  log P   = -|d_route - d_gc| / beta        (Newson & Krumm, 2009)
                            - switch_penalty  if the arc changes
                A real car drives roughly the straight-line distance between two fixes 1 s
                apart, so a candidate pair needing a long detour (or no path at all, e.g.
                disconnected network components) is unlikely / impossible. The optional
                switch penalty discourages flipping between neighbouring arcs on GPS jitter.
Stationary      with min_move > 0, consecutive fixes within min_move metres of the first fix of
pooling         a run (the car is stopped or crawling) are decoded as ONE HMM step: their
                classifier log-probabilities are averaged per candidate arc and every fix in
                the run gets the same arc. This stops GPS jitter at a standstill from being
                read as movement onto neighbouring arcs.
Viterbi         delta_t(j) = max_i [ delta_{t-1}(i) + logT(i,j) ] + logE_t(j), with
                backpointers; the best final state is traced back to give one arc per point.
Edge cases
  * point with no candidates  -> skipped (pred = -1), chain continues across it
  * time gap > gap_s seconds  -> chain broken; segments decoded independently
  * every transition impossible (-inf) -> chain broken at that point and restarted from
    emissions only, so one bad step cannot kill the whole route
"""
import heapq
import math

import numpy as np

NEG = -1e18
FLOOR = 1e-6  # probability floor used for log emissions


class RoadGraph:
    """Directed road graph with a bounded, cached Dijkstra.

    Only nodes within the search limit are explored, and results are cached per source
    node, so decoding cost does not grow with the size of the whole road network.
    """

    def __init__(self, trip):
        a, b = trip.arc_a, trip.arc_b
        self.arcs = trip.arcs
        self.len = np.linalg.norm(b - a, axis=1)
        V = len(trip.nodes_xy)
        order = np.argsort(self.arcs[:, 0], kind="stable")
        counts = np.bincount(self.arcs[:, 0], minlength=V)
        self._ptr = np.concatenate([[0], np.cumsum(counts)]).tolist()
        self._nbr = self.arcs[order, 1].tolist()
        self._w = np.maximum(self.len, 1e-3)[order].tolist()
        self._cache = {}

    def dists_from(self, src, limit):
        """{node: shortest network distance from src} for every node within `limit` metres."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        cutoff = 250.0 * math.ceil(limit / 250.0)  # bucketed so the cache is reused
        ptr, nbr, w = self._ptr, self._nbr, self._w
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for e in range(ptr[u], ptr[u + 1]):
                nd = d + w[e]
                if nd <= cutoff and nd < dist.get(nbr[e], math.inf):
                    dist[nbr[e]] = nd
                    heapq.heappush(heap, (nd, nbr[e]))
        self._cache[src] = (cutoff, dist)
        return dist


def transition_logp(g, arcs_i, frac_i, arcs_j, frac_j, d_gc, beta, slack=200.0,
                    switch_penalty=0.0):
    """Matrix (len(i), len(j)) of transition log-probabilities."""
    Li, Lj = g.len[arcs_i], g.len[arcs_j]
    limit = 2.0 * d_gc + slack
    start_j = g.arcs[arcs_j, 0].tolist()
    between = np.empty((len(arcs_i), len(arcs_j)))
    for r, a in enumerate(arcs_i.tolist()):
        d = g.dists_from(int(g.arcs[a, 1]), limit)          # end of arc i -> start of arc j
        between[r] = [d.get(n, math.inf) for n in start_j]
    between[between > limit] = math.inf                     # same search limit as before
    d_route = (1 - frac_i)[:, None] * Li[:, None] + between + (frac_j * Lj)[None, :]
    same = arcs_i[:, None] == arcs_j[None, :]
    d_same = np.abs(frac_j[None, :] - frac_i[:, None]) * Li[:, None]
    d_route = np.where(same, d_same, d_route)
    lp = -np.abs(d_route - d_gc) / beta
    if switch_penalty:
        lp = np.where(same, lp, lp - switch_penalty)
    return np.where(np.isfinite(lp), lp, NEG)


def _steps(trip, cands, logp_emis, emis_weight, min_move, gap_s):
    """Group candidate rows into HMM steps: (anchor_pt, last_pt, members, arcs, frac, log_emission)."""
    pts = cands.pt.values
    arcs_all, frac_all = cands.arc.values, cands.frac.values
    starts = np.r_[0, np.flatnonzero(np.diff(pts)) + 1]
    ends = np.r_[starts[1:], len(pts)]
    per_pt = [(int(pts[s]), arcs_all[s:e], frac_all[s:e], logp_emis[s:e]) for s, e in zip(starts, ends)]

    steps = []
    for pt, arcs, frac, le in per_pt:
        if steps and min_move > 0:
            anchor = steps[-1]
            near = np.linalg.norm(trip.xy[pt] - trip.xy[anchor["pt"]]) < min_move
            no_gap = trip.t[pt] - trip.t[anchor["last"]] <= gap_s
            if near and no_gap:
                lookup = dict(zip(arcs.tolist(), le.tolist()))
                anchor["sum"] += np.array([lookup.get(a, math.log(FLOOR)) for a in anchor["arcs"].tolist()])
                anchor["n"] += 1
                anchor["members"].append(pt)
                anchor["last"] = pt
                continue
        steps.append({"pt": pt, "last": pt, "members": [pt], "arcs": arcs, "frac": frac,
                      "sum": le.astype(float).copy(), "n": 1})
    return [(s["pt"], s["last"], s["members"], s["arcs"], s["frac"], emis_weight * s["sum"] / s["n"])
            for s in steps]


def viterbi(trip, cands, logp_emis, beta=10.0, gap_s=60.0, emis_weight=1.0,
            switch_penalty=0.0, min_move=0.0, graph=None):
    """cands: DataFrame with columns pt, arc, frac (sorted by pt). Returns pred arc per point."""
    g = graph or RoadGraph(trip)
    N = len(trip.xy)
    pred = np.full(N, -1, dtype=np.int64)
    steps = _steps(trip, cands, logp_emis, emis_weight, min_move, gap_s)

    seg = []          # (members, arcs, backptr) for the current unbroken segment
    delta = None
    n_breaks = 0

    def flush():
        if not seg:
            return
        j = int(np.argmax(delta))
        for members, arcs, back in reversed(seg):
            pred[members] = arcs[j]
            j = back[j] if back is not None else j
        seg.clear()

    prev = None       # (anchor_pt, last_pt, arcs, frac)
    for pt, last, members, arcs, frac, le in steps:
        if prev is None or (trip.t[pt] - trip.t[prev[1]]) > gap_s:
            if prev is not None:
                n_breaks += 1
            flush()
            delta = le.copy()
            seg.append((members, arcs, None))
            prev = (pt, last, arcs, frac)
            continue
        d_gc = float(np.linalg.norm(trip.xy[pt] - trip.xy[prev[0]]))
        T = transition_logp(g, prev[2], prev[3], arcs, frac, d_gc, beta,
                            switch_penalty=switch_penalty)
        scores = delta[:, None] + T
        back = np.argmax(scores, axis=0)
        best = scores[back, np.arange(len(arcs))]
        if np.all(best <= NEG / 2):           # nothing reachable: break and restart
            n_breaks += 1
            flush()
            delta = le.copy()
            seg.append((members, arcs, None))
        else:
            delta = best + le
            seg.append((members, arcs, back))
        prev = (pt, last, arcs, frac)
    flush()
    return pred, n_breaks

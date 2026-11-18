"""Directed road graph with bounded, cached Dijkstra."""
from __future__ import annotations

import heapq
import math

import numpy as np


class RoadGraph:
    def __init__(self, n_nodes: int, arcs: np.ndarray, arc_len: np.ndarray):
        order = np.argsort(arcs[:, 0], kind="stable")
        counts = np.bincount(arcs[:, 0], minlength=n_nodes)
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).tolist()
        self.nbr = arcs[order, 1].tolist()
        self.w = arc_len[order].tolist()
        self._cache: dict[int, tuple[float, dict[int, float]]] = {}

    def dists_from(self, src: int, cutoff: float) -> dict[int, float]:
        """Shortest driving distance from node `src` to every node within `cutoff` metres."""
        cutoff = 250.0 * math.ceil(cutoff / 250.0)  # bucket cutoffs so the cache is reused
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        indptr, nbr, w = self.indptr, self.nbr, self.w
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for e in range(indptr[u], indptr[u + 1]):
                nd = d + w[e]
                if nd <= cutoff:
                    v = nbr[e]
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
        self._cache[src] = (cutoff, dist)
        return dist

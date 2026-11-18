"""Derive a per-GPS-point reference arc from the ordered .route sequence.

.route gives the arcs in driving order but NOT which point lies on which arc.
Snapping each point to its nearest route arc is wrong when the route passes
the same street twice or runs next to itself. Instead we solve a monotonic
alignment: assign point i to route position k_i with k_0 <= k_1 <= ... and
minimise the total point-to-arc distance. This is an O(N*M) dynamic program
(cost[i,k] = d[i,k] + min_{k'<=k} cost[i-1,k']), i.e. a Viterbi in which the
only allowed transition is "stay or move forward along the route".

Points further than MAX_LABEL_DIST from their aligned arc are marked
unreliable (-1): the reference itself may be imperfect there.
"""
import numpy as np

from .geo import point_segment

MAX_LABEL_DIST = 50.0  # metres


def align_route(trip, max_dist: float = MAX_LABEL_DIST):
    route = trip.route
    a = trip.arc_a[route]          # (M,2)
    b = trip.arc_b[route]
    N, M = len(trip.xy), len(route)
    idx = np.arange(M)
    back = np.empty((N, M), dtype=np.int32)

    d0, _ = point_segment(trip.xy[0][None, :], a, b)
    cost = d0.astype(np.float64)
    for i in range(1, N):
        di, _ = point_segment(trip.xy[i][None, :], a, b)
        pm = np.minimum.accumulate(cost)
        # argmin of prefix: last index <= k where cost equals the prefix min
        back[i] = np.maximum.accumulate(np.where(cost == pm, idx, 0))
        cost = di + pm

    k = np.empty(N, dtype=np.int64)
    k[-1] = int(np.argmin(cost))
    for i in range(N - 1, 0, -1):
        k[i - 1] = back[i, k[i]]

    arc = route[k]
    dist, _ = point_segment(trip.xy, trip.arc_a[arc], trip.arc_b[arc])
    label = np.where(dist <= max_dist, arc, -1)
    return label, k, dist

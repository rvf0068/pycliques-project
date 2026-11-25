"""Exact pair-neighborhood covering obstruction for octahedral completion.

Let H be the complement of the host graph, and T the vertices not yet
dominated by the selected antipodal pairs. Each additional antipodal pair
ab must be an edge of H[T] and can cover at most
T intersect (N_H(a) union N_H(b)).
For any nonnegative integer weights w and r remaining pairs, a necessary
condition is W <= r M, where W is the weight of T and M is the maximum
weight covered by one eligible pair. Every rejection is exact; a negative
answer from this helper is inconclusive.
"""

from __future__ import annotations

from collections.abc import Sequence
from heapq import nlargest

from ._bitsets import iter_indices as _bits


def pair_cover_obstructs(nonedge: Sequence[int], common: int, remaining: int) -> bool:
    """Return True only when no dominating octahedral completion is possible.

    Bound computational overhead: apply the screen only when 1-4 pairs
    remain and at most 80 uncovered vertices are present. Weighted
    screens are further restricted to at most 64 vertices and 1200 edges.
    Returning False does not certify that a completion exists.
    """
    if remaining < 1 or remaining > 4:
        return False
    count = common.bit_count()
    if count < 2 * remaining or count > 80:
        return False

    vertices = list(_bits(common))
    open_neighbors = {u: nonedge[u] & common for u in vertices}
    degrees = {u: mask.bit_count() for u, mask in open_neighbors.items()}
    if any(degree == 0 for degree in degrees.values()):
        return True

    covers = [
        open_neighbors[a] | open_neighbors[b]
        for a in vertices
        for b in _bits(open_neighbors[a])
        if a < b
    ]
    if len(covers) < remaining:
        return True

    # The best r individual pair coverages are an upper bound even
    # without accounting for overlap or pairwise incompatibility.
    if sum(nlargest(remaining, (mask.bit_count() for mask in covers))) < count:
        return True

    if remaining == 1 or count > 64 or len(covers) > 1200:
        return False

    # Integer-only fractional-cover dual witnesses. These weights
    # emphasize vertices with few potential complement-neighbors.
    for power in (1, 2):
        weights = {v: max(1, (2000 // degrees[v]) ** power) for v in vertices}
        total = sum(weights.values())
        maximum = max(sum(weights[v] for v in _bits(mask)) for mask in covers)
        if total > remaining * maximum:
            return True
    return False

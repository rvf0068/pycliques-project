"""Exact bounded reduced mod-2 homology of clique complexes of graphs.

An independent GF(2) boundary reduction. Unlike a homotopy-type heuristic,
successful completion certifies each returned Betti number. Under a face
budget overflow the result is None, NEVER an invented zero.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import networkx as nx


def _rank_gf2(columns) -> int:
    """Rank of binary integer-encoded columns by sparse pivot reduction."""
    pivots: dict[int,int] = {}
    for column in columns:
        while column:
            pivot = column.bit_length()-1
            if pivot in pivots:
                column ^= pivots[pivot]
            else:
                pivots[pivot] = column
                break
    return len(pivots)


def bounded_reduced_betti_mod2(
    graph: nx.Graph,
    *,
    max_faces: int = 100_000,
    max_degree: int | None = None,
) -> dict[int, int] | None:
    """Compute exact reduced Betti numbers through a specified degree.

    If max_degree is None, return the FULL Betti vector including known
    zero dimensions through the maximal simplex dimension. Otherwise
    compute every degree 0..max_degree, using faces through max_degree+1
    to calculate the last boundary rank. Returns None on any face-budget
    overflow, not a partially valid list. The empty graph returns {}.

    Each result includes only dimensions proved exact. Users may safely
    cache the returned mapping, but should never infer ranks in omitted
    dimensions when max_degree was specified.
    """
    if max_faces < 1 or (max_degree is not None and max_degree < 0):
        raise ValueError("max_faces must be positive, max_degree nonnegative")
    if graph.number_of_nodes() == 0:
        return {}

    order = {v:i for i,v in enumerate(graph)}
    by_dimension: dict[int,set[frozenset]] = defaultdict(set)
    max_facet_dimension = -1
    faces_seen = 0
    for maximal_clique in nx.find_cliques(graph):
        vertices = tuple(sorted(maximal_clique,key=order.__getitem__))
        max_facet_dimension = max(max_facet_dimension,len(vertices)-1)
        max_size = len(vertices) if max_degree is None else min(
            len(vertices), max_degree+2,
        )
        for k in range(1,max_size+1):
            for face in combinations(vertices,k):
                container = by_dimension[k-1]
                signature = frozenset(face)
                if signature not in container:
                    container.add(signature)
                    faces_seen += 1
                    if faces_seen > max_faces:
                        return None

    last_degree = (
        max_facet_dimension if max_degree is None else max_degree
    )
    # The highest rank needed is that of d_(last_degree+1).
    top_boundary = min(max_facet_dimension,last_degree+1)
    ranks = {0:0}
    for degree in range(1,top_boundary+1):
        lower_faces = sorted(
            by_dimension.get(degree-1,()),
            key=lambda s:tuple(sorted(order[v] for v in s)),
        )
        rows = {face:j for j,face in enumerate(lower_faces)}
        upper_faces = by_dimension.get(degree,())
        columns = (
            sum(1 << rows[face-{v}] for v in face)
            for face in upper_faces
        )
        # Every boundary simplex has distinct codimension-one faces:
        # the bit sum coincides with their binary mod-2 incidence vector.
        ranks[degree] = _rank_gf2(columns)

    betti = {}
    for degree in range(last_degree+1):
        ordinary = (
            len(by_dimension.get(degree,()))
            - ranks.get(degree,0) - ranks.get(degree+1,0)
        )
        betti[degree] = ordinary-(degree==0)
        if betti[degree] < 0:
            raise AssertionError("Computed invalid negative reduced Betti number")
    return betti

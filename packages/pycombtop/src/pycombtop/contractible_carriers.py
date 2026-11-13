"""Replayable contractible carrier certificates for simplicial link images.

A finite flag complex is contractible if its graph reduces to one vertex
through a sequence of *vertex s-dismantlings*. Each such move is certified
by a dominated-vertex sequence in the removed vertex's open neighborhood.

For a target graph and an image-vertex set, search only induced carrier
subgraphs formed by adding at most two target vertices. Every proof can
be checked without rerunning the search. Bounded search failure means
UNKNOWN; it is never interpreted as a mathematical obstruction.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from itertools import combinations

import networkx as nx

from .s_collapses import (
    dismantling_certificate,
    verify_dismantling_certificate,
    verify_s_collapse_move,
)

CarrierMove = tuple[Hashable, tuple[tuple[Hashable, Hashable], ...]]
CarrierProof = tuple[tuple[CarrierMove, ...], Hashable]


def verify_contractible_carrier(
    graph: nx.Graph,
    moves: Sequence[CarrierMove],
    survivor: Hashable,
) -> bool:
    """Replay all local certificates and require exactly one survivor."""
    try:
        current = graph.copy()
        if not current:
            return False
        for move in moves:
            if not isinstance(move, (tuple, list)) or len(move) != 2:
                return False
            vertex, local = move
            if not verify_s_collapse_move(
                current,
                {"kind": "vertex", "vertex": vertex, "dismantling": local},
            ):
                return False
        return current.number_of_nodes() == 1 and survivor in current
    except (TypeError, ValueError, KeyError):
        return False


def bounded_carrier_collapse(
    graph: nx.Graph,
    *,
    max_vertices: int = 12,
    max_checks: int = 64,
    max_moves: int = 11,
) -> CarrierProof | None:
    """Search for a replayable contraction using strict local work caps."""
    if (
        graph.number_of_nodes() == 0
        or graph.number_of_nodes() > max_vertices
        or max_checks < 0
        or max_moves < 0
        or not nx.is_connected(graph)
    ):
        return None
    current = graph.copy()
    moves: list[CarrierMove] = []
    checks = 0
    while current.number_of_nodes() > 1:
        if len(moves) >= max_moves:
            return None
        found = False
        for vertex in tuple(current):
            if checks >= max_checks:
                return None
            checks += 1
            neighbors = current.subgraph(list(current.neighbors(vertex))).copy()
            local = dismantling_certificate(neighbors)
            if local is None:
                continue
            if not verify_dismantling_certificate(neighbors, local):
                return None
            moves.append((vertex, local))
            current.remove_node(vertex)
            found = True
            break
        if not found:
            return None
    survivor = next(iter(current))
    result = (tuple(moves), survivor)
    if not verify_contractible_carrier(graph, *result):
        return None
    return result


def bounded_contractible_carrier(
    graph: nx.Graph,
    image_vertices: Sequence[Hashable],
    *,
    max_carrier_vertices: int = 12,
    max_extra_vertices: int = 2,
    max_candidates: int = 36,
    max_checks_per_candidate: int = 64,
    max_moves: int = 11,
) -> tuple[tuple[Hashable, ...], tuple[CarrierMove, ...], Hashable] | None:
    """Find a small contractible induced subcomplex containing the image.

    Try the image-induced complex first, then add one or two vertices.
    Each candidate has an independently replayable s-collapse. The
    number of candidates, vertices and local s-move checks is bounded.
    """
    try:
        image_set = set(image_vertices)
        if (
            not image_set
            or not image_set <= set(graph)
            or len(image_set) > max_carrier_vertices
            or max_extra_vertices < 0
            or max_candidates < 1
        ):
            return None
        image = tuple(v for v in graph if v in image_set)
        remaining = tuple(v for v in graph if v not in image_set)
        candidates = 0
        for extra_count in range(min(max_extra_vertices, len(remaining)) + 1):
            for extras in combinations(remaining, extra_count):
                if candidates >= max_candidates:
                    return None
                candidates += 1
                if len(image) + len(extras) > max_carrier_vertices:
                    continue
                selected = image_set | set(extras)
                carrier = tuple(v for v in graph if v in selected)
                subgraph = graph.subgraph(carrier).copy()
                proof = bounded_carrier_collapse(
                    subgraph,
                    max_vertices=max_carrier_vertices,
                    max_checks=max_checks_per_candidate,
                    max_moves=max_moves,
                )
                if proof is not None:
                    moves, survivor = proof
                    return carrier, moves, survivor
        return None
    except (TypeError, ValueError, KeyError):
        return None

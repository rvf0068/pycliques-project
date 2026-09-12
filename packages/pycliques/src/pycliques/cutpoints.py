"""Edge operations from Theorems 6.1 and 6.2.

This module implements the elementary graph operations appearing in
Frías-Armenta, Larrión, Neumann-Lara, and Pizaña (2013),
"Edge contraction and edge removal on iterated clique graphs."

Theorem 6.1:
    If ``uv`` is a local bridge of ``G``, then ``G`` and ``G/uv`` have the
    same clique behavior.

Theorem 6.2:
    If ``uv`` is an edge of ``G`` contained in no triangle, then ``G - uv``
    is below ``G`` in the clique-behavior ordering.

The functions in this module implement the structural hypotheses and the
corresponding graph operations. They do not classify clique behavior.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from typing import cast

import networkx as nx


def is_local_bridge(
    graph: nx.Graph,
    u: Hashable,
    v: Hashable,
) -> bool:
    """Return whether ``uv`` is a local bridge.

    By the equivalent characterization from Theorem 6.1, removing ``uv``
    must leave ``u`` and ``v`` at distance at least four. Infinite distance
    is allowed.
    """
    if not graph.has_edge(u, v):
        return False

    graph_without_edge = graph.copy()
    graph_without_edge.remove_edge(u, v)

    try:
        distance = cast(int, nx.shortest_path_length(graph_without_edge, u, v))
    except nx.NetworkXNoPath:
        return True

    return distance >= 4


def local_bridges(graph: nx.Graph) -> Iterator[tuple[Hashable, Hashable]]:
    """Yield all local bridges of ``graph``."""
    for u, v in graph.edges():
        if is_local_bridge(graph, u, v):
            yield u, v


def contract_local_bridge(
    graph: nx.Graph,
    u: Hashable,
    v: Hashable,
) -> nx.Graph:
    """Return a copy of ``graph`` with the local bridge ``uv`` contracted.

    Raises:
        ValueError: If ``uv`` is not an edge or is not a local bridge.
    """
    if not graph.has_edge(u, v):
        raise ValueError(f"{u!r}-{v!r} is not an edge")
    if not is_local_bridge(graph, u, v):
        raise ValueError(f"{u!r}-{v!r} is not a local bridge")

    return nx.contracted_edge(graph, (u, v), self_loops=False)


def local_bridge_contractions(graph: nx.Graph) -> Iterator[nx.Graph]:
    """Yield the contractions of all local bridges of ``graph``."""
    for u, v in local_bridges(graph):
        yield contract_local_bridge(graph, u, v)


def edge_in_triangle(
    graph: nx.Graph,
    u: Hashable,
    v: Hashable,
) -> bool:
    """Return whether the edge ``uv`` is contained in a triangle."""
    if not graph.has_edge(u, v):
        return False
    return bool(set(graph[u]).intersection(graph[v]))


def edges_in_no_triangle(
    graph: nx.Graph,
) -> Iterator[tuple[Hashable, Hashable]]:
    """Yield all edges of ``graph`` contained in no triangle."""
    for u, v in graph.edges():
        if not edge_in_triangle(graph, u, v):
            yield u, v


def remove_edge_not_in_triangle(
    graph: nx.Graph,
    u: Hashable,
    v: Hashable,
) -> nx.Graph:
    """Return a copy with the triangle-free edge ``uv`` removed.

    Raises:
        ValueError: If ``uv`` is not an edge or is contained in a triangle.
    """
    if not graph.has_edge(u, v):
        raise ValueError(f"{u!r}-{v!r} is not an edge")
    if edge_in_triangle(graph, u, v):
        raise ValueError(f"{u!r}-{v!r} is contained in a triangle")

    result = graph.copy()
    result.remove_edge(u, v)
    return result


def non_triangle_edge_removals(graph: nx.Graph) -> Iterator[nx.Graph]:
    """Yield graphs obtained by removing every edge in no triangle."""
    for u, v in edges_in_no_triangle(graph):
        yield remove_edge_not_in_triangle(graph, u, v)

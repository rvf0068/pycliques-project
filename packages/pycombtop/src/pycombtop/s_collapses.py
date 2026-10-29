"""S-collapses for graphs: vertex and edge s-dismantling.

An *s-dismantlable vertex* is one whose open neighbourhood is dismantlable
(collapses to a single vertex under iterated dominated-vertex removal).
Dually, an *s-dismantlable edge* is one whose common neighbourhood is
dismantlable.

Removing all such vertices/edges preserves the simple-homotopy type of the
clique complex.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, cast

import networkx as nx
from pycliques.dominated import is_dismantlable
from pycliques.surfaces import open_neighborhood

# ---------------------------------------------------------------------------
# Vertex s-collapses
# ---------------------------------------------------------------------------


def dismantling_certificate(
    graph: nx.Graph,
) -> tuple[tuple[Hashable, Hashable], ...] | None:
    """Return a deterministic dominated-vertex sequence to one vertex.

    An empty graph is not dismantlable; a one-vertex graph has the empty
    certificate. Each pair is ``(removed_vertex, dominator)``.
    """
    if graph.number_of_nodes() == 0:
        return None
    current = graph.copy()
    pairs: list[tuple[Hashable, Hashable]] = []
    while current.number_of_nodes() > 1:
        found: tuple[Hashable, Hashable] | None = None
        for removed in current:
            removed_closed = set(current[removed]) | {removed}
            for dominator in current[removed]:
                if dominator == removed:
                    continue
                dominator_closed = set(current[dominator]) | {dominator}
                if removed_closed <= dominator_closed:
                    found = (removed, dominator)
                    break
            if found is not None:
                break
        if found is None:
            return None
        pairs.append(found)
        current.remove_node(found[0])
    return tuple(pairs)


def verify_dismantling_certificate(
    graph: nx.Graph,
    pairs: Sequence[tuple[Hashable, Hashable]],
) -> bool:
    """Replay a dominated-vertex sequence and require exactly one survivor."""
    if graph.number_of_nodes() == 0:
        return False
    current = graph.copy()
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return False
        removed, dominator = pair
        if removed not in current or dominator not in current or removed == dominator:
            return False
        if not current.has_edge(removed, dominator):
            return False
        removed_closed = set(current[removed]) | {removed}
        dominator_closed = set(current[dominator]) | {dominator}
        if not removed_closed <= dominator_closed:
            return False
        current.remove_node(removed)
    return bool(current.number_of_nodes() == 1)


def complete_s_collapse_with_certificate(
    graph: nx.Graph,
) -> tuple[
    nx.Graph,
    tuple[tuple[Hashable, tuple[tuple[Hashable, Hashable], ...]], ...],
]:
    """Remove s-dismantlable vertices and retain each local proof."""
    current = graph.copy()
    moves: list[tuple[Hashable, tuple[tuple[Hashable, Hashable], ...]]] = []
    while True:
        for vertex in current:
            neighborhood = open_neighborhood(current, vertex)
            local = dismantling_certificate(neighborhood)
            if local is not None:
                moves.append((vertex, local))
                current.remove_node(vertex)
                break
        else:
            return current, tuple(moves)


def is_s_dismantlable_vertex(graph: nx.Graph, v: Hashable) -> bool:
    """Return whether vertex *v* is s-dismantlable in *graph*.

    A vertex is s-dismantlable when its open neighbourhood is a
    dismantlable graph.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    v
        Vertex to test.

    .. rubric:: Returns

    bool
        ``True`` if the open neighbourhood of *v* is dismantlable.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.s_collapses import is_s_dismantlable_vertex
    >>> is_s_dismantlable_vertex(nx.complete_graph(4), 0)
    True
    >>> is_s_dismantlable_vertex(nx.cycle_graph(6), 0)
    False
    """
    neigh = open_neighborhood(graph, v)
    return is_dismantlable(neigh)


def has_s_dismantlable_vertex(graph: nx.Graph) -> Hashable | None:
    """Return an s-dismantlable vertex of *graph*, or ``None``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.s_collapses import has_s_dismantlable_vertex
    >>> has_s_dismantlable_vertex(nx.complete_graph(4)) is not None
    True
    """
    for v in graph.nodes():
        if is_s_dismantlable_vertex(graph, v):
            return cast(Hashable, v)
    return None


def remove_s_dismantlable_vertex(graph: nx.Graph) -> nx.Graph:
    """Remove one s-dismantlable vertex from *graph* if one exists.

    Returns a copy of *graph* with the first s-dismantlable vertex
    removed, or a copy of *graph* unchanged if none exists.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.s_collapses import remove_s_dismantlable_vertex
    >>> g = remove_s_dismantlable_vertex(nx.complete_graph(4))
    >>> g.order()
    3
    """
    graph_aux = graph.copy()
    v = has_s_dismantlable_vertex(graph)
    if v is not None:
        graph_aux.remove_node(v)
    return graph_aux


def complete_s_collapse(graph: nx.Graph) -> nx.Graph:
    """Remove all s-dismantlable vertices from *graph*.

    Repeatedly removes s-dismantlable vertices until none remain.
    The result has the same simple-homotopy type (of its clique complex)
    as the original graph.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.

    .. rubric:: Returns

    networkx.Graph
        A graph with no s-dismantlable vertices.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.s_collapses import complete_s_collapse
    >>> g = complete_s_collapse(nx.circulant_graph(7, [1, 2]))
    >>> g.order()
    4
    """
    graph_aux = graph.copy()
    while True:
        n = graph_aux.order()
        graph_aux = remove_s_dismantlable_vertex(graph_aux)
        if n == graph_aux.order():
            return graph_aux


# ---------------------------------------------------------------------------
# Edge s-collapses
# ---------------------------------------------------------------------------


def is_s_dismantlable_edge(graph: nx.Graph, e: tuple[Hashable, Hashable]) -> bool:
    """Return whether edge *e* is s-dismantlable in *graph*.

    An edge is s-dismantlable when the subgraph induced by the common
    neighbours of its endpoints is dismantlable.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.s_collapses import is_s_dismantlable_edge
    >>> is_s_dismantlable_edge(nx.complete_graph(3), (0, 1))
    True
    >>> is_s_dismantlable_edge(nx.cycle_graph(4), (0, 1))
    False
    """
    inter = graph.subgraph(set(graph[e[0]]).intersection(graph[e[1]])).copy()
    return is_dismantlable(inter)


def has_s_dismantlable_edge(graph: nx.Graph) -> tuple[Hashable, Hashable] | None:
    """Return an s-dismantlable edge of *graph*, or ``None``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.s_collapses import has_s_dismantlable_edge
    >>> has_s_dismantlable_edge(nx.complete_graph(3)) is not None
    True
    >>> has_s_dismantlable_edge(nx.cycle_graph(4)) is None
    True
    """
    for e in graph.edges():
        if is_s_dismantlable_edge(graph, e):
            return cast(tuple[Hashable, Hashable], e)
    return None


def remove_s_dismantlable_edge(graph: nx.Graph) -> nx.Graph:
    """Remove one s-dismantlable edge from *graph* if one exists.

    Returns a copy of *graph* with the first s-dismantlable edge
    removed, or a copy of *graph* unchanged if none exists.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.s_collapses import remove_s_dismantlable_edge
    >>> g = remove_s_dismantlable_edge(nx.complete_graph(3))
    >>> g.size()
    2
    """
    graph_aux = graph.copy()
    e = has_s_dismantlable_edge(graph)
    if e is not None:
        graph_aux.remove_edge(*e)
    return graph_aux


def complete_s_collapse_edges(graph: nx.Graph) -> nx.Graph:
    """Remove all s-dismantlable edges from *graph*.

    Repeatedly removes s-dismantlable edges until none remain.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.

    .. rubric:: Returns

    networkx.Graph
        A graph with no s-dismantlable edges.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.s_collapses import complete_s_collapse_edges
    >>> g = complete_s_collapse_edges(nx.complete_graph(4))
    >>> g.size()
    3
    """
    graph_aux = graph.copy()
    while True:
        n = graph_aux.size()
        graph_aux = remove_s_dismantlable_edge(graph_aux)
        if n == graph_aux.size():
            return graph_aux


def complete_s_collapse_edges_with_certificate(
    graph: nx.Graph,
) -> tuple[
    nx.Graph,
    tuple[tuple[tuple[Hashable, Hashable], tuple[tuple[Hashable, Hashable], ...]], ...],
]:
    """Remove s-dismantlable edges and retain each local proof."""
    current = graph.copy()
    moves: list[
        tuple[tuple[Hashable, Hashable], tuple[tuple[Hashable, Hashable], ...]]
    ] = []
    while True:
        for first, second in current.edges():
            common = set(current[first]) & set(current[second])
            local_graph = current.subgraph(common).copy()
            local = dismantling_certificate(local_graph)
            if local is not None:
                moves.append(((first, second), local))
                current.remove_edge(first, second)
                break
        else:
            return current, tuple(moves)


def verify_s_collapse_move(
    graph: nx.Graph,
    move: dict[str, Any],
) -> bool:
    """Check one recorded vertex or edge s-move on the current graph."""
    try:
        kind = move["kind"]
        local_pairs = move["dismantling"]
        if kind == "vertex":
            vertex = move["vertex"]
            if vertex not in graph:
                return False
            local = open_neighborhood(graph, vertex)
            if not verify_dismantling_certificate(local, local_pairs):
                return False
            graph.remove_node(vertex)
            return True
        if kind == "edge":
            first, second = move["edge"]
            if not graph.has_edge(first, second):
                return False
            common = set(graph[first]) & set(graph[second])
            local = graph.subgraph(common).copy()
            if not verify_dismantling_certificate(local, local_pairs):
                return False
            graph.remove_edge(first, second)
            return True
    except (KeyError, TypeError, ValueError):
        return False
    return False

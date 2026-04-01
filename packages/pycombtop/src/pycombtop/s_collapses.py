"""S-collapses for graphs: vertex and edge s-dismantling.

An *s-dismantlable vertex* is one whose open neighbourhood is dismantlable
(collapses to a single vertex under iterated dominated-vertex removal).
Dually, an *s-dismantlable edge* is one whose common neighbourhood is
dismantlable.

Removing all such vertices/edges preserves the simple-homotopy type of the
clique complex.
"""

from __future__ import annotations

import copy

import networkx as nx
from pycliques.dominated import is_dismantlable
from pycliques.surfaces import open_neighborhood

# ---------------------------------------------------------------------------
# Vertex s-collapses
# ---------------------------------------------------------------------------


def is_s_dismantlable_vertex(graph: nx.Graph, v: int) -> bool:
    """Return whether vertex *v* is s-dismantlable in *graph*.

    A vertex is s-dismantlable when its open neighbourhood is a
    dismantlable graph.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    v : int
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


def has_s_dismantlable_vertex(graph: nx.Graph) -> int | None:
    """Return an s-dismantlable vertex of *graph*, or ``None``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.s_collapses import has_s_dismantlable_vertex
    >>> has_s_dismantlable_vertex(nx.complete_graph(4)) is not None
    True
    """
    for v in graph.nodes():
        if is_s_dismantlable_vertex(graph, v):
            return int(v)
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
    graph_aux = copy.deepcopy(graph)
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
    graph_aux = copy.deepcopy(graph)
    while True:
        n = graph_aux.order()
        graph_aux = remove_s_dismantlable_vertex(graph_aux)
        if n == graph_aux.order():
            return graph_aux


# ---------------------------------------------------------------------------
# Edge s-collapses
# ---------------------------------------------------------------------------


def is_s_dismantlable_edge(graph: nx.Graph, e: tuple[int, int]) -> bool:
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


def has_s_dismantlable_edge(graph: nx.Graph) -> tuple[int, int] | None:
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
            return (int(e[0]), int(e[1]))
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
    graph_aux = copy.deepcopy(graph)
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
    graph_aux = copy.deepcopy(graph)
    while True:
        n = graph_aux.size()
        graph_aux = remove_s_dismantlable_edge(graph_aux)
        if n == graph_aux.size():
            return graph_aux

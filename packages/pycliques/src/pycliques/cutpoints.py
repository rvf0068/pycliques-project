"""Local cutpoints and edge operations at cutpoints.

A *local cutpoint* of a graph *G* is a vertex *x* such that the open
neighborhood *N(x)* is disconnected.  This module provides detection of
local cutpoints and reduced graphs via edge removal and edge contraction
at cutpoints, following the approach of Frías-Armenta, Larrión,
Neumann-Lara and Pizaña (2013).
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterator

import networkx as nx

from pycliques.dominated import completely_pared_graph
from pycliques.retractions import retracts
from pycliques.surfaces import open_neighborhood

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def local_cutpoints(graph: nx.Graph) -> Iterator[Hashable]:
    """Yield the local cutpoints of *graph*.

    A vertex *x* is a local cutpoint when its open neighborhood
    :math:`N(x)` is not connected.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.

    .. rubric:: Yields

    Hashable
        Vertices whose open neighborhood is disconnected.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cutpoints import local_cutpoints
    >>> sorted(local_cutpoints(nx.path_graph(4)))
    [1, 2]
    >>> list(local_cutpoints(nx.complete_graph(4)))
    []
    """
    for v in graph:
        nbhd = open_neighborhood(graph, v)
        if nbhd.order() >= 2 and not nx.is_connected(nbhd):
            yield v


def has_local_cutpoints(graph: nx.Graph) -> bool:
    """Return whether *graph* has at least one local cutpoint.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.

    .. rubric:: Returns

    bool
        ``True`` if *graph* contains a local cutpoint.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cutpoints import has_local_cutpoints
    >>> has_local_cutpoints(nx.path_graph(4))
    True
    >>> has_local_cutpoints(nx.complete_graph(4))
    False
    """
    for v in graph:
        nbhd = open_neighborhood(graph, v)
        if nbhd.order() >= 2 and not nx.is_connected(nbhd):
            return True
    return False


def neighborhood_components(graph: nx.Graph, x: Hashable) -> list[set[Hashable]]:
    """Return the connected components of the open neighborhood of *x*.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    x : Hashable
        A vertex of *graph*.

    .. rubric:: Returns

    list[set[Hashable]]
        Each element is the vertex set of a connected component of
        *N(x)*.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cutpoints import neighborhood_components
    >>> neighborhood_components(nx.path_graph(5), 2)
    [{1}, {3}]
    """
    nbhd = open_neighborhood(graph, x)
    return [set(c) for c in nx.connected_components(nbhd)]


def cutpoint_edge_removals(graph: nx.Graph) -> Iterator[nx.Graph]:
    """Yield pared connected subgraphs from edge removal at local cutpoints.

    For each local cutpoint *x* and each connected component *C* of
    *N(x)*, remove all edges between *x* and *C*.  Each connected
    component of the resulting graph with at least two vertices is pared
    (dominated vertices removed) and yielded.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.

    .. rubric:: Yields

    networkx.Graph
        Non-trivial pared connected subgraphs produced by the edge
        removals.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cutpoints import cutpoint_edge_removals
    >>> reductions = list(cutpoint_edge_removals(nx.path_graph(5)))
    >>> len(reductions) > 0
    True
    """
    for x in local_cutpoints(graph):
        components = neighborhood_components(graph, x)
        for comp in components:
            g = graph.copy()
            for y in comp:
                g.remove_edge(x, y)
            for cc in nx.connected_components(g):
                sub = g.subgraph(cc).copy()
                if sub.order() >= 2:
                    pared = completely_pared_graph(sub)
                    if pared.order() >= 2:
                        yield pared


def cutpoint_edge_contractions(graph: nx.Graph) -> Iterator[nx.Graph]:
    """Yield pared graphs from edge contraction at local cutpoints.

    For each local cutpoint *x* and each neighbor *y* of *x*, contract
    the edge *xy*, pare the result, and yield it when non-trivial.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.

    .. rubric:: Yields

    networkx.Graph
        Non-trivial pared graphs produced by the edge contractions.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cutpoints import cutpoint_edge_contractions
    >>> reductions = list(cutpoint_edge_contractions(nx.path_graph(5)))
    >>> len(reductions) > 0
    True
    """
    for x in local_cutpoints(graph):
        for y in list(graph.neighbors(x)):
            g = nx.contracted_edge(graph, (x, y), self_loops=False)
            g = nx.convert_node_labels_to_integers(g)
            pared = completely_pared_graph(g)
            if pared.order() >= 2:
                yield pared


def cutpoint_reductions(graph: nx.Graph) -> Iterator[nx.Graph]:
    """Yield all cutpoint reductions (edge removals and contractions).

    This is the union of :func:`cutpoint_edge_removals` and
    :func:`cutpoint_edge_contractions`.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.

    .. rubric:: Yields

    networkx.Graph
        Non-trivial pared graphs from both operations.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cutpoints import cutpoint_reductions
    >>> list(cutpoint_reductions(nx.complete_graph(4)))
    []
    """
    yield from cutpoint_edge_removals(graph)
    yield from cutpoint_edge_contractions(graph)


def reduction_retracts_to(graph: nx.Graph, target: nx.Graph) -> bool:
    """Check whether any cutpoint reduction of *graph* retracts to *target*.

    Applies edge removal and edge contraction at every local cutpoint
    (following Frías-Armenta, Larrión, Neumann-Lara and Pizaña, 2013),
    and for each resulting graph tests whether it retracts to *target*.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph (should already be completely pared).
    target : networkx.Graph
        Target graph to retract onto.

    .. rubric:: Returns

    bool
        ``True`` if some cutpoint reduction retracts to *target*.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cutpoints import reduction_retracts_to
    >>> reduction_retracts_to(nx.path_graph(5), nx.path_graph(2))
    True
    >>> reduction_retracts_to(nx.complete_graph(4), nx.cycle_graph(4))
    False
    """
    target_order = target.order()
    for reduced in cutpoint_reductions(graph):
        if reduced.order() >= target_order and retracts(reduced, target):
            _logger.debug("Cutpoint reduction retracts to target")
            return True
    return False

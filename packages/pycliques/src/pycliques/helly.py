r"""
A collection :math:`\mathcal{C}` of subsets of a set :math:`X` is called
*intersecting* if the intersection of any two elements of
:math:`\mathcal{C}` is non empty. The collection :math:`\mathcal{C}` is called
*Helly* if any intersecting subcollection :math:`\mathcal{C}'` of
:math:`\mathcal{C}` has non empty intersection. A graph is called *Helly*
if the collection of its cliques is Helly.
"""

from __future__ import annotations

from collections.abc import Hashable

import networkx as nx

from .dominated import closed_neighborhood


def n_open(graph: nx.Graph, u: Hashable, v: Hashable) -> set[Hashable]:
    """Return the intersection of the open neighborhoods of ``u`` and ``v``.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    u : Hashable
        First vertex.
    v : Hashable
        Second vertex.

    .. rubric:: Returns

    set[Hashable]
        The set of common neighbors of ``u`` and ``v``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.helly import n_open
    >>> sorted(n_open(nx.complete_graph(4), 0, 1))
    [2, 3]
    >>> n_open(nx.path_graph(4), 0, 2)
    {1}
    """
    return set(graph[u]) & set(graph[v])


def n_closed(graph: nx.Graph, u: Hashable, v: Hashable) -> set[Hashable]:
    """Return the intersection of the closed neighborhoods of ``u`` and ``v``.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    u : Hashable
        First vertex.
    v : Hashable
        Second vertex.

    .. rubric:: Returns

    set[Hashable]
        The common neighbors of ``u`` and ``v``, plus ``u`` and ``v``
        themselves.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.helly import n_closed
    >>> sorted(n_closed(nx.complete_graph(4), 0, 1))
    [0, 1, 2, 3]
    >>> sorted(n_closed(nx.path_graph(4), 0, 2))
    [0, 1, 2]
    """
    return n_open(graph, u, v) | {u, v}


def u_open(graph: nx.Graph, u: Hashable, v: Hashable) -> set[Hashable]:
    """Return the universal vertices in the set :func:`n_open`.

    A vertex ``w`` in :func:`n_open` is *universal* if :func:`n_open` is a
    subset of the closed neighborhood of ``w``.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    u : Hashable
        First vertex.
    v : Hashable
        Second vertex.

    .. rubric:: Returns

    set[Hashable]
        Universal vertices of the open common neighborhood.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.helly import u_open
    >>> sorted(u_open(nx.complete_graph(4), 0, 1))
    [2, 3]
    """
    s_open = n_open(graph, u, v)
    # A vertex is universal in S if S is a subset of its closed neighborhood
    return {w for w in s_open if s_open.issubset(closed_neighborhood(graph, w))}


def u_closed(graph: nx.Graph, u: Hashable, v: Hashable) -> set[Hashable]:
    """Return the universal vertices in the set :func:`n_closed`.

    A vertex ``w`` in :func:`n_closed` is *universal* if :func:`n_closed` is a
    subset of the closed neighborhood of ``w``.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph.
    u : Hashable
        First vertex.
    v : Hashable
        Second vertex.

    .. rubric:: Returns

    set[Hashable]
        Universal vertices of the closed common neighborhood.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.helly import u_closed
    >>> sorted(u_closed(nx.complete_graph(4), 0, 1))
    [0, 1, 2, 3]
    """
    s_closed = n_closed(graph, u, v)
    return {w for w in s_closed if s_closed.issubset(closed_neighborhood(graph, w))}


def is_clique_helly(graph: nx.Graph) -> bool:
    """Return whether the graph is clique-Helly.

    .. rubric:: Parameters

    graph : networkx.Graph
        An undirected graph.

    .. rubric:: Returns

    bool
        ``True`` if the graph is clique-Helly, ``False`` otherwise.

    .. rubric:: Notes

    This implementation is from [2]_.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.helly import is_clique_helly
    >>> is_clique_helly(nx.complete_graph(4))
    True
    >>> is_clique_helly(nx.cycle_graph(5))
    True
    """
    # Precompute u_closed for all edges using frozenset for order-independent lookup
    uclosed = {frozenset({u, v}): u_closed(graph, u, v) for u, v in graph.edges()}

    for u, v in graph.edges():
        for w in n_open(graph, u, v):
            # A triangle is formed by {u,v}, {v,w}, {w,u}
            e1 = frozenset({u, v})
            e2 = frozenset({v, w})
            e3 = frozenset({w, u})

            # The intersection of their u_closed sets must be non-empty
            if not (uclosed[e1] & uclosed[e2] & uclosed[e3]):
                return False

    return True


def is_hereditary_clique_helly(graph: nx.Graph) -> bool:
    """Return whether the graph is hereditary clique-Helly.

    .. rubric:: Parameters

    graph : networkx.Graph
        An undirected graph.

    .. rubric:: Returns

    bool
        ``True`` if the graph is hereditary clique-Helly, ``False`` otherwise.

    .. rubric:: Notes

    This implementation is from [2]_.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.helly import is_clique_helly, is_hereditary_clique_helly
    >>> from pyg6data.lists import list_graphs
    >>> is_hereditary_clique_helly(nx.complete_graph(4))
    True
    >>> g = list_graphs(7)[645]
    >>> is_clique_helly(g)
    True
    >>> is_hereditary_clique_helly(g)
    False

    .. rubric:: References

    .. [2] Lin, M. C., & Szwarcfiter, J. L., Faster recognition of clique-Helly
       and hereditary clique-Helly graphs, Information Processing Letters,
       103(1), 40–43 (2007).
    """
    uopen = {frozenset({u, v}): u_open(graph, u, v) for u, v in graph.edges()}

    for u, v in graph.edges():
        for w in n_open(graph, u, v):
            e1 = frozenset({u, v})
            e2 = frozenset({v, w})
            e3 = frozenset({w, u})

            if not (uopen[e1] & uopen[e2] & uopen[e3]):
                return False

    return True

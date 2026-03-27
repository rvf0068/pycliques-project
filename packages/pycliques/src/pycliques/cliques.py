from __future__ import annotations

import itertools
import math
from collections.abc import Hashable
from functools import singledispatch
from typing import TypeAlias

import networkx as nx


class Clique(frozenset):
    """Base class for a clique in a graph.

    This class derives from :class:`frozenset` but overrides ``__repr__`` so
    that instances display like plain set literals instead of ``frozenset``.

    .. rubric:: Examples

    Inspecting a non-empty clique produces a clean set-style representation::

        >>> from pycliques import Clique
        >>> Clique({1, 2, 3})
        {1, 2, 3}

    Empty cliques render as ``{}``, which keeps doctest outputs short and lets
    documentation examples double as regression tests::

        >>> Clique([])
        {}
    """

    def __repr__(self) -> str:
        """Return a set-style string representation."""
        u = set(self)
        if len(u) == 0:
            return "{}"
        else:
            return f"{u}"


NodeH: TypeAlias = tuple[int, Clique]


@singledispatch
def clique_graph(graph: nx.Graph, bound: int | float = math.inf) -> nx.Graph | None:
    """Produce the clique graph of an undirected NetworkX graph.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph whose cliques will become nodes of the output graph.
    bound : int, optional
        Maximum number of cliques before aborting (default: ``math.inf``).

    .. rubric:: Returns

    networkx.Graph | None
        The clique graph if the clique count stays within ``bound``; ``None``
        otherwise.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques import Clique, clique_graph
    >>> g = clique_graph(nx.octahedral_graph())
    >>> g.number_of_nodes()
    8
    >>> g.degree[Clique({0, 1, 2})]
    6
    >>> clique_graph(nx.cycle_graph(4), bound=2) is None
    True
    """
    it_cliques = nx.find_cliques(graph)
    cliques = []
    K = nx.Graph()
    while True:
        try:
            clique = next(it_cliques)
            cliques.append(Clique(clique))
            if len(cliques) > bound:
                return None
        except StopIteration:
            break
    K.add_nodes_from(cliques)

    # Fast edge generation: group cliques by the vertices they contain
    for v in graph:
        # Find all cliques containing this vertex
        cliques_with_v = [c for c in cliques if v in c]
        # Link all of them together in the clique graph
        K.add_edges_from(itertools.combinations(cliques_with_v, 2))

    return K


def homotopy_clique_graph(graph: nx.Graph) -> nx.Graph:
    """The homotopy clique graph

    .. rubric:: Parameters

    graph : NetworkX graph
            An undirected graph

    .. rubric:: Returns

    NetworkX graph
        the homotopy clique graph of graph

    .. rubric:: Notes

    This is the operator :math:`H` defined in [Larrion08]_.

     .. [Larrion08] F. Larrion, M. A. Pizana and R. Villarroel-Flores. Posets,
         clique graphs and their homotopy type. European Journal of
         Combinatorics, 29(1), (2008) pp. 334-342.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques import homotopy_clique_graph
    >>> G = nx.path_graph(3)
    >>> H = homotopy_clique_graph(G)
    >>> # The nodes of H are pairs (vertex, clique)
    >>> len(H)
    4
    >>> nx.is_connected(H)
    True
    """

    H = nx.Graph()

    # 1. Extract cliques and precompute which cliques contain which vertices
    # Using sets for fast intersection later
    cliques = [Clique(c) for c in nx.find_cliques(graph)]
    cliques_of: dict[Hashable, set[Clique]] = {v: set() for v in graph}

    for c in cliques:
        for v in c:
            cliques_of[v].add(c)

    # 2. Add nodes and "same-vertex" edges (where v == w)
    for v, v_cliques in cliques_of.items():
        v_nodes = [(v, c) for c in v_cliques]
        H.add_nodes_from(v_nodes)

        # (v, C) is always adjacent to (v, D) because v in D and v in C
        H.add_edges_from(itertools.combinations(v_nodes, 2))

    # 3. Add "cross-vertex" edges (where v != w)
    for v, w in graph.edges():
        # A cross edge exists between (v, C) and (w, D) iff
        # both C and D contain both v and w.
        shared_cliques = cliques_of[v].intersection(cliques_of[w])

        if not shared_cliques:  # pragma: no cover
            continue

        cross_nodes_v = [(v, c) for c in shared_cliques]
        cross_nodes_w = [(w, c) for c in shared_cliques]

        # Connect every valid v-node to every valid w-node
        H.add_edges_from(itertools.product(cross_nodes_v, cross_nodes_w))

    return H

from __future__ import annotations

from functools import singledispatch
import itertools
import math
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
def clique_graph(
        graph: nx.Graph,
        bound: int | float = math.inf
) -> nx.Graph | None:
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
    clique_pairs = itertools.combinations(cliques, 2)
    K.add_edges_from((c1, c2) for (c1, c2) in clique_pairs if c1 & c2)
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
    # A node in H is a (vertex, clique) tuple

    def _ady(c1: NodeH, c2: NodeH) -> bool:
        return (c1[0] in c2[1]) and (c2[0] in c1[1])

    H = nx.Graph()
    cliques: list[Clique] = [Clique(q) for q in nx.find_cliques(graph)]
    vertices: list[NodeH] = [
        (x, q) for x in graph.nodes() for q in cliques if x in q
    ]
    H.add_nodes_from(vertices)
    vertex_pairs = itertools.combinations(vertices, 2)
    H.add_edges_from((c1, c2) for (c1, c2) in vertex_pairs if _ady(c1, c2))
    return H

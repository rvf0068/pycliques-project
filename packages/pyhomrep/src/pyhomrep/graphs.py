"""Graph utility functions for representation theory.

This module provides graph constructions useful for studying representations
on simplicial complexes, particularly matching graphs.
"""

from __future__ import annotations

import networkx as nx


def matching_graph(n: int) -> nx.Graph:
    """Construct the matching graph from a complete graph.

    The matching graph M(K_n) has as vertices the edges of the complete graph
    K_n, and two vertices are adjacent if and only if the corresponding edges
    are disjoint (form a matching).

    .. rubric:: Parameters

    n : int
        The number of vertices in the complete graph.

    .. rubric:: Returns

    networkx.Graph
        The matching graph of K_n.

    .. rubric:: Raises

    NetworkXError
        If n is negative.

    .. rubric:: Examples

    >>> from pyhomrep.graphs import matching_graph
    >>> G = matching_graph(4)
    >>> G.number_of_nodes()
    6
    >>> G.number_of_edges()
    3
    """
    k_n = nx.complete_graph(n)
    G = nx.Graph()
    for i in k_n.edges():
        G.add_node(i)
    w: list = []
    for i in k_n.edges():
        for j in k_n.edges():
            if (
                (j[0] not in i)
                and (j[1] not in i)
                and ((i, j) not in w)
                and ((j, i) not in w)
            ):
                w.append((i, j))
                G.add_edge(i, j)
    return G

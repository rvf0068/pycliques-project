"""Graph homomorphism complex Hom(G, H).

This module provides utilities for enumerating graph homomorphisms and
building the graph ``Hom(G, H)`` whose vertices are graph homomorphisms
from *G* to *H* and whose edges encode a compatibility condition on
those homomorphisms.
"""

from __future__ import annotations

import itertools
from typing import Any

import networkx as nx


def graph_homomorphisms(g: nx.Graph, h: nx.Graph) -> list[dict[Any, Any]]:
    """Return all graph homomorphisms from *g* to *h*.

    A graph homomorphism is a mapping of the vertices of *g* to the
    vertices of *h* that sends every edge of *g* to an edge of *h*.
    Each homomorphism is returned as a :class:`dict` mapping every vertex
    of *g* to its image in *h*.

    The search uses backtracking: vertices of *g* are assigned one by one
    and each partial assignment is validated against already-assigned
    neighbors before recursing, pruning large parts of the search space.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycombtop.hom_complex import graph_homomorphisms
    >>> G = nx.path_graph(2)
    >>> H = nx.complete_graph(2)
    >>> homs = graph_homomorphisms(G, H)
    >>> len(homs)
    2
    >>> {0: 0, 1: 1} in homs
    True
    """
    g_nodes: list[Any] = list(g.nodes())
    h_nodes: list[Any] = list(h.nodes())
    n = len(g_nodes)

    # Precompute adjacency sets of H for O(1) edge lookup.
    h_adj: dict[Any, set[Any]] = {v: set(h.neighbors(v)) for v in h.nodes()}

    # For each position k in g_nodes, record which earlier positions share
    # an edge in g.  These are the constraints checked during backtracking.
    node_to_idx: dict[Any, int] = {v: i for i, v in enumerate(g_nodes)}
    earlier_neighbors: list[list[int]] = [[] for _ in range(n)]
    for u, v in g.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        if i < j:
            earlier_neighbors[j].append(i)
        else:
            earlier_neighbors[i].append(j)

    result: list[dict[Any, Any]] = []
    current: list[Any] = [None] * n

    def backtrack(pos: int) -> None:
        if pos == n:
            result.append(dict(zip(g_nodes, current)))
            return
        for h_v in h_nodes:
            if all(h_v in h_adj[current[j]] for j in earlier_neighbors[pos]):
                current[pos] = h_v
                backtrack(pos + 1)
        current[pos] = None

    backtrack(0)
    return result


def hom_graph(g: nx.Graph, h: nx.Graph) -> nx.Graph:
    """Build the graph ``Hom(G, H)``.

    The vertices of the resulting graph are all graph homomorphisms from
    *g* to *h* (see :func:`graph_homomorphisms`).  Two homomorphisms *f*
    and *phi* are declared adjacent when, for **every** edge ``{x, y}``
    in *g*, both

    * ``f(x)`` is adjacent to ``phi(y)`` in *h*, and
    * ``f(y)`` is adjacent to ``phi(x)`` in *h*.

    The construction is an induced subgraph of the classical *exponential
    graph* ``H^G``.

    Each node of the returned graph is a :class:`frozenset` of
    ``(vertex, image)`` pairs; use ``dict(node)`` to recover the
    homomorphism as a plain dictionary.

    .. rubric:: Parameters

    g : networkx.Graph
        The domain graph.
    h : networkx.Graph
        The codomain graph.

    .. rubric:: Returns

    networkx.Graph
        The hom graph whose nodes are frozen homomorphism mappings.

    .. rubric:: Examples

    A single-vertex domain with no edges means any map is a homomorphism
    and the adjacency condition is vacuously satisfied, so the result is
    the complete graph on the vertices of *H*::

        >>> import networkx as nx
        >>> from pycombtop.hom_complex import hom_graph
        >>> G = nx.complete_graph(1)
        >>> H = nx.complete_graph(2)
        >>> result = hom_graph(G, H)
        >>> result.number_of_nodes()
        2
        >>> result.number_of_edges()
        1

    When *G* is a single edge and *H* is also a single edge (K_2), only
    two homomorphisms exist and they are not adjacent::

        >>> G = nx.path_graph(2)
        >>> H = nx.complete_graph(2)
        >>> result = hom_graph(G, H)
        >>> result.number_of_nodes()
        2
        >>> result.number_of_edges()
        0
    """
    homomorphisms = graph_homomorphisms(g, h)

    result: nx.Graph = nx.Graph()
    result.add_nodes_from(frozenset(f.items()) for f in homomorphisms)

    for f, phi in itertools.combinations(homomorphisms, 2):
        if all(
            h.has_edge(f[u], phi[v]) and h.has_edge(f[v], phi[u]) for u, v in g.edges()
        ):
            result.add_edge(frozenset(f.items()), frozenset(phi.items()))

    return result

r"""
A *coaffination* of a graph :math:`G` is an automorphism
:math:`\sigma\colon G\to G` such that the distance from
:math:`\sigma(x)` to :math:`x` is at least 2 for each vertex
:math:`x`.
"""

from __future__ import annotations

import math
import networkx as nx
from networkx.algorithms import isomorphism
from typing import Dict, Iterator

from .cliques import Clique, clique_graph


class CoaffinePair(object):
    """Bundle a graph with one of its coaffinations.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques import CoaffinePair, clique_graph
    >>> g = nx.cycle_graph(4)
    >>> pair = CoaffinePair(g, {0: 2, 1: 3, 2: 0, 3: 1})
    >>> pair.graph.number_of_nodes()
    4
    >>> pair.coaffination[0]
    2
    >>> kpair = clique_graph(pair)
    >>> kpair.coaffination
    {{0, 1}: {2, 3}, {0, 3}: {1, 2}, {1, 2}: {0, 3}, {2, 3}: {0, 1}}
    """
    def __init__(self, graph: nx.Graph, coaffination: dict[int, int]):
        """Store the base graph and its associated automorphism."""
        self.graph = graph
        self.coaffination = coaffination


@clique_graph.register
def _(
        pair: CoaffinePair,
        bound: int | float = math.inf
) -> CoaffinePair | None:
    """Return the clique graph of a :class:`CoaffinePair` as another pair.

    .. rubric:: Parameters

    pair : CoaffinePair
        Pair consisting of the original graph and one of its coaffinations.
    bound : int, optional
        Maximum number of cliques allowed before returning ``None``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques import CoaffinePair, clique_graph
    >>> g = nx.cycle_graph(4)
    >>> pair = CoaffinePair(g, {0: 2, 1: 3, 2: 0, 3: 1})
    >>> result = clique_graph(pair)
    >>> isinstance(result, CoaffinePair)
    True
    """
    g = pair.graph
    sigma = pair.coaffination
    # We call the original generic clique_graph logic on the underlying graph
    kg = clique_graph.registry[object](g, bound)
    if kg is None:
        return None
    coaf_k: Dict[Clique, Clique] = {}
    for q in kg:
        coaf_k[q] = Clique([sigma[x] for x in q])
    return CoaffinePair(kg, coaf_k)


def automorphisms(graph: nx.Graph) -> Iterator[dict[int, int]]:
    """Yield every automorphism of ``graph`` as a dict mapping.

    .. rubric:: Parameters

    graph : networkx.Graph
        Graph whose automorphisms will be generated.

    .. rubric:: Yields

    dict[int, int]
        A mapping that maps vertices according to an automorphism.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques import automorphisms
    >>> autos = list(automorphisms(nx.cycle_graph(3)))
    >>> autos[0]
    {0: 0, 1: 1, 2: 2}
    >>> len(autos)
    6
    >>> autos[0][0]
    0

    """
    GM = isomorphism.GraphMatcher(graph, graph)
    yield from GM.subgraph_isomorphisms_iter()


def coaffinations(graph: nx.Graph, k: int) -> Iterator[dict[int, int]]:
    """Yield automorphisms that map a vertex outside its closed neighborhood.

    .. rubric:: Parameters

    graph : networkx.Graph
        Graph under study.
    k : int
        Minimum distance between each vertex and its image.

    .. rubric:: Yields

    dict[int, int]
        A coaffination that satisfies the distance constraint.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques import coaffinations
    >>> cycle = nx.cycle_graph(4)
    >>> cof = list(coaffinations(cycle, 2))
    >>> cof == [{0: 2, 1: 3, 2: 0, 3: 1}]
    True
    >>> all(nx.shortest_path_length(cycle, v, mapping[v]) >= 2
    ...     for mapping in cof for v in mapping)
    True

    """
    the_automorphisms = automorphisms(graph)
    distance = dict(nx.all_pairs_shortest_path_length(graph))
    for auto in the_automorphisms:
        for v in graph:
            if distance[v][auto[v]] < k:
                break
        else:
            yield auto

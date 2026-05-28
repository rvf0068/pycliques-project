r"""
A *coaffination* of a graph :math:`G` is an automorphism
:math:`\sigma\colon G\to G` such that the distance from
:math:`\sigma(x)` to :math:`x` is at least 2 for each vertex
:math:`x`.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterator

import networkx as nx
from grandiso import find_motifs_iter
from networkx.algorithms import isomorphism

from .cliques import Clique, clique_graph


class CoaffinePair:
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

    def __init__(self, graph: nx.Graph, coaffination: dict[Hashable, Hashable]):
        """Store the base graph and its associated automorphism."""
        self.graph = graph
        self.coaffination = coaffination


@clique_graph.register
def _(pair: CoaffinePair, bound: int | float = math.inf) -> CoaffinePair | None:
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
    coaf_k: dict[Hashable, Hashable] = {}
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


def is_coaffine_map(
    small_pair: CoaffinePair,
    large_pair: CoaffinePair,
    mono: dict[Hashable, Hashable],
) -> bool:
    """Return ``True`` if *mono* is a coaffine map from *small_pair* to *large_pair*.

    A graph monomorphism ``f: G -> H`` is *coaffine* with respect to coaffine
    pairs ``(G, s)`` and ``(H, t)`` when it satisfies the equivariance
    condition ``f(s(v)) = t(f(v))`` for every vertex ``v`` in ``G``.

    .. rubric:: Parameters

    small_pair : CoaffinePair
        The domain pair ``(G, s)``.
    large_pair : CoaffinePair
        The codomain pair ``(H, t)``.
    mono : dict
        A monomorphism mapping vertices of ``G`` to vertices of ``H``.

    .. rubric:: Returns

    bool
        ``True`` if the equivariance condition holds for every vertex,
        ``False`` otherwise.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques import CoaffinePair, is_coaffine_map
    >>> g = nx.cycle_graph(4)
    >>> sigma = {0: 2, 1: 3, 2: 0, 3: 1}
    >>> pair = CoaffinePair(g, sigma)
    >>> is_coaffine_map(pair, pair, sigma)
    True
    >>> other = CoaffinePair(g, {0: 3, 1: 0, 2: 1, 3: 2})
    >>> is_coaffine_map(pair, other, {0: 0, 1: 1, 2: 2, 3: 3})
    False

    """
    sigma = small_pair.coaffination
    tau = large_pair.coaffination
    return all(mono[sigma[v]] == tau[mono[v]] for v in small_pair.graph)


def coaffine_monomorphism(
    large_pair: CoaffinePair,
    small_pair: CoaffinePair,
    algorithm: str = "GM",
) -> dict[Hashable, Hashable] | bool:
    """Find a coaffine monomorphism from *small_pair* to *large_pair*.

    Searches for a graph monomorphism ``f: G -> H`` (where ``G`` is the graph
    of *small_pair* and ``H`` is the graph of *large_pair*) that is equivariant
    with respect to the stored coaffinations, i.e. satisfies
    ``f(s(v)) = t(f(v))`` for every vertex ``v`` in ``G``.

    .. rubric:: Parameters

    large_pair : CoaffinePair
        The codomain coaffine pair ``(H, t)``.
    small_pair : CoaffinePair
        The domain coaffine pair ``(G, s)``.
    algorithm : str, optional
        Subgraph search backend.  ``"GM"`` (default) uses NetworkX's
        :class:`~networkx.algorithms.isomorphism.GraphMatcher`;
        ``"Grandiso"`` uses :func:`grandiso.find_motifs_iter`.

    .. rubric:: Returns

    dict | False
        A dict mapping each vertex of ``G`` to a vertex of ``H`` when a
        coaffine monomorphism is found, or ``False`` when none exists.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques import CoaffinePair, coaffine_monomorphism
    >>> g = nx.cycle_graph(4)
    >>> sigma = {0: 2, 1: 3, 2: 0, 3: 1}
    >>> pair = CoaffinePair(g, sigma)
    >>> f = coaffine_monomorphism(pair, pair)
    >>> f is not False
    True
    >>> from pycliques import is_coaffine_map
    >>> is_coaffine_map(pair, pair, f)
    True

    """
    g1 = small_pair.graph
    g2 = large_pair.graph
    if algorithm == "GM":
        gm = isomorphism.GraphMatcher(g2, g1)
        the_iter = gm.subgraph_monomorphisms_iter()
    elif algorithm == "Grandiso":
        the_iter = find_motifs_iter(g1, g2)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
    for mono in the_iter:
        if algorithm == "GM":
            mono = {v: k for k, v in mono.items()}
        if is_coaffine_map(small_pair, large_pair, mono) is True:
            return dict(mono)
    return False

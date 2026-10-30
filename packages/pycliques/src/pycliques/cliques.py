from __future__ import annotations

import itertools
import math
from collections.abc import Hashable
from functools import lru_cache, singledispatch
from math import comb
from typing import TypeAlias

import networkx as nx

from .octahedral_boolean import enumerate_maximal_intersecting_families


class Clique(frozenset):
    """Base class for a clique in a graph.

    This class derives from :class:`frozenset` but overrides ``__repr__`` so
    that instances display like plain set literals instead of ``frozenset``.

    .. rubric:: Examples

    Inspecting a non-empty clique produces a clean set-style representation::

        >>> from pycliques import Clique
        >>> Clique({1, 2, 3})
        {1, 2, 3}
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


NodeH: TypeAlias = tuple[Hashable, Clique]

_MAXIMAL_INTERSECTING_FAMILY_COUNTS = (0, 1, 2, 4, 12, 81, 2646)


def _cliques_by_vertex(
    graph: nx.Graph, bound: int | float = math.inf
) -> tuple[list[Clique], dict[Hashable, set[Clique]]] | None:
    """Return cliques and the input vertices contained in each."""
    cliques: list[Clique] = []
    cliques_of: dict[Hashable, set[Clique]] = {v: set() for v in graph}

    for clique in nx.find_cliques(graph):
        clique_node = Clique(clique)
        cliques.append(clique_node)
        for vertex in clique_node:
            cliques_of[vertex].add(clique_node)
        if len(cliques) > bound:
            return None

    return cliques, cliques_of


@singledispatch
def clique_graph(
    graph: nx.Graph,
    bound: int | float = math.inf,
    *,
    max_edges: int | None = None,
) -> nx.Graph | None:
    """Produce the clique graph of an undirected NetworkX graph.

    .. rubric:: Parameters

    graph : networkx.Graph
        Input graph whose cliques become the vertices of the output graph.
        Two output vertices are adjacent exactly when those cliques
        intersect.
    bound : int, optional
        Maximum number of cliques before aborting (default: ``math.inf``).
    max_edges : int, optional
        Maximum number of edges in the output graph. When exceeded, return
        ``None`` before constructing additional edges. The default is unbounded.

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
    if max_edges is not None and max_edges < 0:
        raise ValueError("max_edges must be nonnegative or None")
    indexed = _cliques_by_vertex(graph, bound)
    if indexed is None:
        return None
    cliques, cliques_of = indexed
    clique_graph = nx.Graph()
    clique_graph.add_nodes_from(cliques)

    # Link clique vertices that share an input-graph vertex.
    edge_count = 0
    for cliques_with_v in cliques_of.values():
        for first, second in itertools.combinations(cliques_with_v, 2):
            if not clique_graph.has_edge(first, second):
                if max_edges is not None and edge_count >= max_edges:
                    return None
                clique_graph.add_edge(first, second)
                edge_count += 1

    return clique_graph


def homotopy_clique_graph(graph: nx.Graph) -> nx.Graph:
    """Produce the homotopy clique graph of an undirected NetworkX graph.

    .. rubric:: Parameters

    graph : NetworkX graph
            An undirected graph

    .. rubric:: Returns

    networkx.Graph
        Graph whose vertices are ``(v, C)`` pairs with ``v`` in a maximal
        clique ``C`` of *graph*.

    .. rubric:: Notes

    This is the operator :math:`H` defined in [Larrion08]_.

     .. [Larrion08] F. Larrión, M. A. Pizaña and R. Villarroel-Flores. Posets,
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
    indexed = _cliques_by_vertex(graph)
    assert indexed is not None
    _, cliques_of = indexed

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


def double_bound_graph(
    graph: nx.Graph,
    *,
    max_faces: int | None = None,
    max_candidates: int | None = None,
    max_edges: int | None = None,
) -> nx.Graph | None:
    """Build the full double-bound graph ``DB(P(G))``.

    The vertices are all distinct nonempty faces of the clique complex of
    *graph*. Two faces are adjacent exactly when they intersect and their
    union is a face. This is the full face-poset construction, not the
    coatomic compact realization ``DB(P(G)'')``.

    ``max_faces`` bounds the number of distinct face vertices,
    ``max_candidates`` bounds pairs of faces tested for adjacency, and
    ``max_edges`` bounds the number of output edges. Exceeding any configured
    bound returns ``None``; it does not mean the construction is invalid.
    """
    for name, value in (
        ("max_faces", max_faces),
        ("max_candidates", max_candidates),
        ("max_edges", max_edges),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be nonnegative or None")

    vertex_order = {vertex: index for index, vertex in enumerate(graph.nodes())}
    faces: set[frozenset] = set()
    for maximal_clique in nx.find_cliques(graph):
        for size in range(1, len(maximal_clique) + 1):
            for face in itertools.combinations(maximal_clique, size):
                faces.add(frozenset(face))
                if max_faces is not None and len(faces) > max_faces:
                    return None

    ordered_faces = sorted(
        faces,
        key=lambda face: tuple(sorted(vertex_order[vertex] for vertex in face)),
    )
    result = nx.Graph()
    result.add_nodes_from(ordered_faces)
    candidate_count = 0
    edge_count = 0
    for first, second in itertools.combinations(ordered_faces, 2):
        candidate_count += 1
        if max_candidates is not None and candidate_count > max_candidates:
            return None
        if first.isdisjoint(second):
            continue
        union = first | second
        if not all(graph.has_edge(u, v) for u, v in itertools.combinations(union, 2)):
            continue
        if max_edges is not None and edge_count >= max_edges:
            return None
        result.add_edge(first, second)
        edge_count += 1
    return result


@lru_cache(maxsize=6)
def _maximal_intersecting_family_library(n: int) -> tuple[int, ...] | None:
    """Return canonical maximal intersecting families through rank six."""
    if n > 6:
        return None
    result = enumerate_maximal_intersecting_families(n, max_subset_vertices=63)
    if not result.is_exact:
        return None
    return result.families


def _maximal_intersecting_family_count(n: int) -> int | None:
    """Return canonical family counts without expanding family contents."""
    if not 1 <= n < len(_MAXIMAL_INTERSECTING_FAMILY_COUNTS):
        return None
    return _MAXIMAL_INTERSECTING_FAMILY_COUNTS[n]


def clique_graph_of_double_bound(
    graph: nx.Graph,
    *,
    max_vertices: int | None = 10_000,
    max_edges: int | None = 100_000,
) -> nx.Graph | None:
    """Build ``K(DB(P(G)))`` directly from maximal cliques of *graph*.

    This avoids constructing the usually much larger graph ``DB(P(G))``.
    A vertex is represented by its maximal intersecting family of nonempty
    faces, and two vertices are adjacent exactly when those families share
    a face. Families are generated from a cached canonical rank-specific
    library for supports of size at most six.

    If either output budget is exceeded, or a maximal clique has size greater
    than six, return ``None`` rather than a partial graph. Set either budget
    to ``None`` to remove that output limit.
    """
    for name, value in (("max_vertices", max_vertices), ("max_edges", max_edges)):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer or None")

    support_sets = tuple(frozenset(clique) for clique in nx.find_cliques(graph))
    family_counts: list[int] = []
    for support in support_sets:
        family_count = _maximal_intersecting_family_count(len(support))
        if family_count is None:
            return None
        family_counts.append(family_count)

    vertex_count = sum(family_counts)
    if max_vertices is not None and vertex_count > max_vertices:
        return None
    within_support_edge_count = sum(comb(count, 2) for count in family_counts)
    if max_edges is not None and within_support_edge_count > max_edges:
        return None

    supports: list[tuple[frozenset, tuple[frozenset, ...]]] = []
    for support, family_count in zip(support_sets, family_counts, strict=True):
        family_codes = _maximal_intersecting_family_library(len(support))
        if family_codes is None or len(family_codes) != family_count:
            raise RuntimeError("maximal intersecting family count metadata mismatch")

        support_vertices = tuple(support)
        full_mask = (1 << len(support_vertices)) - 1
        families = tuple(
            frozenset(
                frozenset(
                    support_vertices[index]
                    for index in range(len(support_vertices))
                    if subset_mask & (1 << index)
                )
                for subset_mask in range(1, full_mask + 1)
                if family_code & (1 << subset_mask)
            )
            for family_code in family_codes
        )
        supports.append((support, families))

    result = nx.Graph()
    for _, families in supports:
        result.add_nodes_from(families)

    edge_count = 0
    for _, families in supports:
        support_edges = len(families) * (len(families) - 1) // 2
        if max_edges is not None and edge_count + support_edges > max_edges:
            return None
        result.add_edges_from(itertools.combinations(families, 2))
        edge_count += support_edges

    for first_index, (first_support, first_families) in enumerate(supports):
        for second_support, second_families in supports[first_index + 1 :]:
            common_support = first_support & second_support
            if not common_support:
                continue
            for first_family in first_families:
                if common_support not in first_family:
                    continue
                for second_family in second_families:
                    if common_support not in second_family:
                        continue
                    if max_edges is not None and edge_count >= max_edges:
                        return None
                    result.add_edge(first_family, second_family)
                    edge_count += 1

    return result

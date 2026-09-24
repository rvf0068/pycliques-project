"""Edge operations from Theorems 6.1 and 6.2, and inverse cutpoint extensions.

This module implements the elementary graph operations appearing in
Frías-Armenta, Larrión, Neumann-Lara, and Pizaña (2013),
"Edge contraction and edge removal on iterated clique graphs."

Theorem 6.1:
    If ``uv`` is a local bridge of ``G``, then ``G`` and ``G/uv`` have the
    same clique behavior.

Theorem 6.2:
    If ``uv`` is an edge of ``G`` contained in no triangle, then ``G - uv``
    is below ``G`` in the clique-behavior ordering.

The functions in this module implement the structural hypotheses and the
corresponding graph operations. They do not classify clique behavior.

Inverse cutpoint extension
--------------------------

The paper also studies the *opposite* operations to edge contraction and
edge removal, and relates vertex identification to cutting a graph at a
*local cutpoint* -- a vertex ``x`` whose open neighborhood ``N(x)`` is
disconnected. This module provides a restricted, theorem-motivated search
for such a construction:

Given a graph ``H`` with a local cutpoint ``w``, partition the connected
components of ``N(w)`` into two nonempty groups, replace ``w`` by two
vertices ``u`` and ``v`` attached respectively to each group, and add the
edge ``uv``. Call the result ``H'``.

By construction, contracting ``uv`` in ``H'`` reproduces ``H`` exactly
(``w``'s neighborhood is simply the union of the two groups again).
Consequently, this construction is a genuine instance of Theorem 6.1 read
in reverse: it is admissible -- and only then does it certify that ``H``
and ``H'`` share the same clique behavior -- when ``uv`` is actually a
local bridge of ``H'``, i.e. when :func:`is_local_bridge` holds for the
constructed edge. This is the same, already-verified hypothesis used by
:func:`contract_local_bridge`; no unverified theorem is assumed here.
An arbitrary split of a cutpoint that fails this check is only a
candidate, never a proof step, and is not treated as admissible.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from itertools import combinations
from typing import cast

import networkx as nx

from pycliques.surfaces import open_neighborhood


def is_local_bridge(
    graph: nx.Graph,
    u: Hashable,
    v: Hashable,
) -> bool:
    """Return whether ``uv`` is a local bridge.

    By the equivalent characterization from Theorem 6.1, removing ``uv``
    must leave ``u`` and ``v`` at distance at least four. Infinite distance
    is allowed.
    """
    if not graph.has_edge(u, v):
        return False

    graph_without_edge = graph.copy()
    graph_without_edge.remove_edge(u, v)

    try:
        distance = cast(int, nx.shortest_path_length(graph_without_edge, u, v))
    except nx.NetworkXNoPath:
        return True

    return distance >= 4


def local_bridges(graph: nx.Graph) -> Iterator[tuple[Hashable, Hashable]]:
    """Yield all local bridges of ``graph``."""
    for u, v in graph.edges():
        if is_local_bridge(graph, u, v):
            yield u, v


def contract_local_bridge(
    graph: nx.Graph,
    u: Hashable,
    v: Hashable,
) -> nx.Graph:
    """Return a copy of ``graph`` with the local bridge ``uv`` contracted.

    Raises:
        ValueError: If ``uv`` is not an edge or is not a local bridge.
    """
    if not graph.has_edge(u, v):
        raise ValueError(f"{u!r}-{v!r} is not an edge")
    if not is_local_bridge(graph, u, v):
        raise ValueError(f"{u!r}-{v!r} is not a local bridge")

    return nx.contracted_edge(graph, (u, v), self_loops=False)


def local_bridge_contractions(graph: nx.Graph) -> Iterator[nx.Graph]:
    """Yield the contractions of all local bridges of ``graph``."""
    for u, v in local_bridges(graph):
        yield contract_local_bridge(graph, u, v)


def edge_in_triangle(
    graph: nx.Graph,
    u: Hashable,
    v: Hashable,
) -> bool:
    """Return whether the edge ``uv`` is contained in a triangle."""
    if not graph.has_edge(u, v):
        return False
    return bool(set(graph[u]).intersection(graph[v]))


def edges_in_no_triangle(
    graph: nx.Graph,
) -> Iterator[tuple[Hashable, Hashable]]:
    """Yield all edges of ``graph`` contained in no triangle."""
    for u, v in graph.edges():
        if not edge_in_triangle(graph, u, v):
            yield u, v


def remove_edge_not_in_triangle(
    graph: nx.Graph,
    u: Hashable,
    v: Hashable,
) -> nx.Graph:
    """Return a copy with the triangle-free edge ``uv`` removed.

    Raises:
        ValueError: If ``uv`` is not an edge or is contained in a triangle.
    """
    if not graph.has_edge(u, v):
        raise ValueError(f"{u!r}-{v!r} is not an edge")
    if edge_in_triangle(graph, u, v):
        raise ValueError(f"{u!r}-{v!r} is contained in a triangle")

    result = graph.copy()
    result.remove_edge(u, v)
    return result


def non_triangle_edge_removals(graph: nx.Graph) -> Iterator[nx.Graph]:
    """Yield graphs obtained by removing every edge in no triangle."""
    for u, v in edges_in_no_triangle(graph):
        yield remove_edge_not_in_triangle(graph, u, v)


# ---------------------------------------------------------------------------
# Local cutpoints and inverse cutpoint extensions
# ---------------------------------------------------------------------------


def local_cutpoints(graph: nx.Graph) -> Iterator[Hashable]:
    """Yield the local cutpoints of ``graph``.

    A vertex ``x`` is a local cutpoint when its open neighborhood ``N(x)``
    is disconnected.

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


def neighborhood_components(graph: nx.Graph, x: Hashable) -> list[frozenset[Hashable]]:
    """Return the connected components of the open neighborhood of ``x``.

    .. rubric:: Examples

    >>> import networkx as nx
    >>> from pycliques.cutpoints import neighborhood_components
    >>> neighborhood_components(nx.path_graph(5), 2) == [frozenset({1}), frozenset({3})]
    True
    """
    nbhd = open_neighborhood(graph, x)
    return [frozenset(c) for c in nx.connected_components(nbhd)]


def _fresh_label(graph: nx.Graph, base: Hashable, tag: str) -> Hashable:
    """Return a node label not already used in ``graph``."""
    candidate: Hashable = (base, tag)
    suffix = 0
    while candidate in graph:
        suffix += 1
        candidate = (base, tag, suffix)
    return candidate


@dataclass(frozen=True)
class InverseCutpointExtension:
    """A candidate inverse cutpoint extension.

    This is a structural candidate only: :func:`is_admissible_inverse_extension`
    must hold before it may be used to justify a clique-behavior conclusion
    (see the module docstring for the precise hypothesis).

    .. rubric:: Attributes

    cutpoint : Hashable
        The local cutpoint of the original graph that was split.
    u, v : Hashable
        The two fresh vertices replacing ``cutpoint``.
    u_branch, v_branch : frozenset[Hashable]
        The vertices of ``N(cutpoint)`` attached to ``u`` and to ``v``
        respectively; each is a union of one or more connected components of
        ``N(cutpoint)``.
    graph : networkx.Graph
        The candidate extended graph, i.e. the original graph with
        ``cutpoint`` replaced by ``u`` and ``v`` (attached according to
        ``u_branch``/``v_branch``) and the edge ``uv`` added.
    """

    cutpoint: Hashable
    u: Hashable
    v: Hashable
    u_branch: frozenset[Hashable]
    v_branch: frozenset[Hashable]
    graph: nx.Graph


def is_admissible_inverse_extension(extension: InverseCutpointExtension) -> bool:
    """Return whether Theorem 6.1's hypothesis holds for a candidate extension.

    The candidate is admissible exactly when the added edge ``uv`` is a
    local bridge of ``extension.graph``. Only then does Theorem 6.1 certify
    that the candidate extension has the same clique behavior as the graph
    it was built from.
    """
    return is_local_bridge(extension.graph, extension.u, extension.v)


def _build_extension(
    graph: nx.Graph,
    cutpoint: Hashable,
    u_branch: frozenset[Hashable],
    v_branch: frozenset[Hashable],
) -> InverseCutpointExtension:
    u = _fresh_label(graph, cutpoint, "u")
    v = _fresh_label(graph, cutpoint, "v")
    extended = graph.copy()
    extended.remove_node(cutpoint)
    extended.add_node(u)
    extended.add_node(v)
    for y in u_branch:
        extended.add_edge(u, y)
    for y in v_branch:
        extended.add_edge(v, y)
    extended.add_edge(u, v)
    return InverseCutpointExtension(
        cutpoint=cutpoint,
        u=u,
        v=v,
        u_branch=u_branch,
        v_branch=v_branch,
        graph=extended,
    )


def inverse_cutpoint_extensions_at(
    graph: nx.Graph,
    cutpoint: Hashable,
    *,
    max_candidates: int = 64,
) -> Iterator[InverseCutpointExtension]:
    """Yield candidate inverse cutpoint extensions splitting ``cutpoint``.

    Each candidate partitions the connected components of
    ``N(cutpoint)`` into two nonempty groups and replaces ``cutpoint`` by two
    vertices attached to each group, joined by a new edge. Candidates are
    structural only; test :func:`is_admissible_inverse_extension` before
    relying on one for a clique-behavior conclusion.

    At most ``max_candidates`` candidates are generated, to keep the search
    bounded for cutpoints with many neighborhood components.
    """
    components = neighborhood_components(graph, cutpoint)
    k = len(components)
    if k < 2:
        return

    first, rest = components[0], components[1:]
    produced = 0
    for r in range(len(rest)):
        for subset in combinations(range(len(rest)), r):
            if produced >= max_candidates:
                return
            chosen = set(subset)
            u_branch = (
                frozenset(first).union(*(rest[i] for i in chosen))
                if chosen
                else frozenset(first)
            )
            v_branch = frozenset().union(
                *(rest[i] for i in range(len(rest)) if i not in chosen)
            )
            if not u_branch or not v_branch:
                continue
            produced += 1
            yield _build_extension(graph, cutpoint, u_branch, v_branch)


def inverse_cutpoint_extensions(
    graph: nx.Graph,
    *,
    max_candidates: int = 64,
) -> Iterator[InverseCutpointExtension]:
    """Yield candidate inverse cutpoint extensions at every local cutpoint.

    See :func:`inverse_cutpoint_extensions_at` for the candidate
    construction and :func:`is_admissible_inverse_extension` for the
    theorem-backed admissibility check.

    ``max_candidates`` bounds the number of candidates generated per
    cutpoint, not the total across all cutpoints.
    """
    for cutpoint in local_cutpoints(graph):
        yield from inverse_cutpoint_extensions_at(
            graph, cutpoint, max_candidates=max_candidates
        )

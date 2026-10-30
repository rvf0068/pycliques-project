"""Closed-clique enumeration and certificates for poset homotopy checks."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import networkx as nx

from .cliques import clique_graph, double_bound_graph


@dataclass(frozen=True)
class ClosedCliqueIntersections:
    """Result of enumerating nonempty intersections of maximal cliques."""

    closed_sets: tuple[frozenset, ...]
    is_exact: bool

    @property
    def is_limited(self) -> bool:
        """Whether a configured limit prevented exhaustive enumeration."""
        return not self.is_exact


class ThreeHellyStatus(Enum):
    """Outcome of the exact four-vertex recognition test for 3-Helly."""

    PROVED = "proved"
    OBSTRUCTION = "obstruction"
    INCONCLUSIVE_BUDGET = "inconclusive_budget"


@dataclass(frozen=True)
class ThreeHellyResult:
    """Auditable outcome of checking whether maximal cliques are 3-Helly."""

    status: ThreeHellyStatus
    vertex_count: int
    maximal_cliques_count: int
    four_sets_checked: int
    max_maximal_cliques: int | None
    max_4sets: int | None
    maximal_cliques_complete: bool = True
    witness: frozenset[Hashable] | None = None
    relevant_cliques: tuple[frozenset[Hashable], ...] = ()


def maximal_clique_hypergraph_is_3_helly(
    graph: nx.Graph,
    *,
    max_maximal_cliques: int | None = None,
    max_4sets: int | None = None,
) -> ThreeHellyResult:
    """Check the 3-Helly property using the published four-vertex test.

    For each four-set ``W``, the subfamily of maximal cliques meeting ``W``
    in at least three vertices must be empty or have nonempty total
    intersection. A clique-enumeration or four-set budget exit is
    inconclusive; partial clique lists are never used to prove the property.
    """
    for name, value in (
        ("max_maximal_cliques", max_maximal_cliques),
        ("max_4sets", max_4sets),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer or None")

    vertices = tuple(graph.nodes())
    vertex_count = len(vertices)
    if vertex_count < 4:
        return ThreeHellyResult(
            ThreeHellyStatus.PROVED,
            vertex_count,
            maximal_cliques_count=0,
            four_sets_checked=0,
            max_maximal_cliques=max_maximal_cliques,
            max_4sets=max_4sets,
        )

    vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
    clique_masks: list[int] = []
    for clique in nx.find_cliques(graph):
        if max_maximal_cliques is not None and len(clique_masks) >= max_maximal_cliques:
            return ThreeHellyResult(
                ThreeHellyStatus.INCONCLUSIVE_BUDGET,
                vertex_count,
                maximal_cliques_count=len(clique_masks),
                four_sets_checked=0,
                max_maximal_cliques=max_maximal_cliques,
                max_4sets=max_4sets,
                maximal_cliques_complete=False,
            )
        clique_masks.append(sum(1 << vertex_index[vertex] for vertex in clique))

    cliques = tuple(
        frozenset(
            vertex for index, vertex in enumerate(vertices) if mask & (1 << index)
        )
        for mask in clique_masks
    )
    checked = 0
    for four_indices in combinations(range(vertex_count), 4):
        if max_4sets is not None and checked >= max_4sets:
            return ThreeHellyResult(
                ThreeHellyStatus.INCONCLUSIVE_BUDGET,
                vertex_count,
                maximal_cliques_count=len(clique_masks),
                four_sets_checked=checked,
                max_maximal_cliques=max_maximal_cliques,
                max_4sets=max_4sets,
            )
        checked += 1
        four_mask = sum(1 << index for index in four_indices)
        relevant_indices = [
            index
            for index, mask in enumerate(clique_masks)
            if (mask & four_mask).bit_count() >= 3
        ]
        if not relevant_indices:
            continue

        common_mask = clique_masks[relevant_indices[0]]
        for index in relevant_indices[1:]:
            common_mask &= clique_masks[index]
        if common_mask:
            continue

        witness = frozenset(vertices[index] for index in four_indices)
        relevant_cliques = tuple(cliques[index] for index in relevant_indices)
        direct_relevant = tuple(
            clique for clique in cliques if len(clique & witness) >= 3
        )
        direct_intersection = set.intersection(
            *(set(clique) for clique in direct_relevant)
        )
        if (
            not witness
            or len(witness) != 4
            or direct_relevant != relevant_cliques
            or direct_intersection
        ):
            raise RuntimeError(
                "four-vertex obstruction failed its direct intersection check"
            )
        return ThreeHellyResult(
            ThreeHellyStatus.OBSTRUCTION,
            vertex_count,
            maximal_cliques_count=len(clique_masks),
            four_sets_checked=checked,
            max_maximal_cliques=max_maximal_cliques,
            max_4sets=max_4sets,
            witness=witness,
            relevant_cliques=relevant_cliques,
        )

    return ThreeHellyResult(
        ThreeHellyStatus.PROVED,
        vertex_count,
        maximal_cliques_count=len(clique_masks),
        four_sets_checked=checked,
        max_maximal_cliques=max_maximal_cliques,
        max_4sets=max_4sets,
    )


def h_closure_from_neighbors(
    graph: nx.Graph, complete: Iterable[Hashable]
) -> frozenset[Hashable]:
    """Return ``h(complete)`` using common neighbors, without maximal cliques.

    *complete* must be a nonempty clique of *graph*. The common open
    neighborhood ``W`` of its vertices induces a graph whose universal
    vertices are exactly the additional vertices in the intersection of all
    maximal cliques containing *complete*.
    """
    clique = frozenset(complete)
    if not clique:
        raise ValueError("complete must be a nonempty clique")
    if not clique <= graph.nodes:
        raise ValueError("complete must contain only vertices of graph")
    if any(
        not graph.has_edge(first, second) for first, second in combinations(clique, 2)
    ):
        raise ValueError("complete must be a clique")

    common_neighbors = set.intersection(
        *(set(graph.neighbors(vertex)) for vertex in clique)
    )
    universal_neighbors = {
        vertex
        for vertex in common_neighbors
        if all(
            vertex == other or graph.has_edge(vertex, other)
            for other in common_neighbors
        )
    }
    return frozenset(clique | universal_neighbors)


def closed_clique_intersections(
    graph: nx.Graph,
    *,
    max_maximal_cliques: int | None = None,
    max_closed_sets: int | None = None,
) -> ClosedCliqueIntersections:
    """Enumerate all nonempty intersections of maximal cliques of *graph*.

    The returned ``is_exact`` flag is false if either bound is exceeded.
    An incomplete result must not be used as evidence that all closed sets
    were checked. The empty graph has no nonempty closed sets and is exact.
    """
    for name, value in (
        ("max_maximal_cliques", max_maximal_cliques),
        ("max_closed_sets", max_closed_sets),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be nonnegative or None")

    vertices = list(graph.nodes())
    vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
    closed_masks: set[int] = set()
    maximal_count = 0
    for clique in nx.find_cliques(graph):
        if max_maximal_cliques is not None and maximal_count >= max_maximal_cliques:
            return _closed_intersection_result(closed_masks, vertices, is_exact=False)
        maximal_count += 1
        clique_mask = sum(1 << vertex_index[vertex] for vertex in clique)
        additions = {clique_mask}
        additions.update(
            previous & clique_mask
            for previous in closed_masks
            if previous & clique_mask
        )
        for mask in sorted(additions - closed_masks):
            if max_closed_sets is not None and len(closed_masks) >= max_closed_sets:
                return _closed_intersection_result(
                    closed_masks, vertices, is_exact=False
                )
            closed_masks.add(mask)

    return _closed_intersection_result(closed_masks, vertices, is_exact=True)


def _closed_intersection_result(
    masks: set[int],
    vertices: Sequence[Hashable],
    *,
    is_exact: bool,
) -> ClosedCliqueIntersections:
    vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
    closed_sets = tuple(
        sorted(
            (
                frozenset(
                    vertex
                    for index, vertex in enumerate(vertices)
                    if mask & (1 << index)
                )
                for mask in masks
            ),
            key=lambda closed: tuple(sorted(vertex_index[vertex] for vertex in closed)),
        )
    )
    return ClosedCliqueIntersections(closed_sets, is_exact)


def _maximal_nonempty_facets(
    facets: Iterable[Iterable[Hashable]],
) -> frozenset[frozenset[Hashable]]:
    unique = {frozenset(facet) for facet in facets if facet}
    return frozenset(
        facet for facet in unique if not any(facet < other for other in unique)
    )


def _dominated_move(
    facets: frozenset[frozenset[Hashable]],
) -> tuple[Hashable, Hashable] | None:
    vertices = set().union(*facets) if facets else set()
    for vertex in sorted(vertices, key=repr):
        incident_facets = [facet for facet in facets if vertex in facet]
        dominators = set.intersection(*(set(facet) for facet in incident_facets))
        dominators.discard(vertex)
        if dominators:
            return vertex, min(dominators, key=repr)
    return None


def _delete_facet_vertex(
    facets: frozenset[frozenset[Hashable]], vertex: Hashable
) -> frozenset[frozenset[Hashable]]:
    return _maximal_nonempty_facets(facet - {vertex} for facet in facets)


def strong_collapse_certificate(
    facets: Iterable[Iterable[Hashable]],
) -> tuple[tuple[Hashable, Hashable], ...] | None:
    """Find a vertex-dominated collapse sequence ending at one vertex.

    Returns moves ``(v, w)`` where every current maximal facet containing
    ``v`` also contains ``w``. Returns ``None`` if this search cannot find a
    sequence; that is inconclusive, not a noncontractibility certificate.
    """
    current = _maximal_nonempty_facets(facets)
    if not current:
        return None
    moves: list[tuple[Hashable, Hashable]] = []
    while len(set().union(*current)) > 1:
        move = _dominated_move(current)
        if move is None:
            return None
        vertex = move[0]
        moves.append(move)
        current = _delete_facet_vertex(current, vertex)
    return tuple(moves)


def verify_strong_collapse_certificate(
    facets: Iterable[Iterable[Hashable]],
    moves: Iterable[tuple[Hashable, Hashable]],
) -> bool:
    """Independently replay a strong-collapse sequence and check its endpoint."""
    current = _maximal_nonempty_facets(facets)
    if not current:
        return False
    active_vertices = set().union(*current)
    for move in moves:
        try:
            vertex, dominator = move
        except (TypeError, ValueError):
            return False
        if (
            vertex == dominator
            or vertex not in active_vertices
            or dominator not in active_vertices
        ):
            return False
        if any(vertex in facet and dominator not in facet for facet in current):
            return False
        current = _delete_facet_vertex(current, vertex)
        active_vertices = set().union(*current) if current else set()
    return len(active_vertices) == 1


class Theorem57Status(Enum):
    """Certainty for the Theorem 5.7 sufficient-hypothesis check.

    ``COUNTEREXAMPLE`` means a closed h-image violates the sufficient
    contractibility hypothesis. It does not assert failure of
    ``Delta(K^2(H(G))) ~= Delta(G)``.
    """

    PROVED = "proved"
    COUNTEREXAMPLE = "counterexample"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Theorem57Result:
    """Status and replayable certificates from a Theorem 5.7 check.

    A ``COUNTEREXAMPLE`` status is specifically a counterexample to the
    sufficient contractibility hypothesis, not to the underlying homotopy
    equivalence conclusion.
    """

    status: Theorem57Status
    checked_h_images: int
    certificates: dict[frozenset, tuple[tuple[Hashable, Hashable], ...]]
    witness: frozenset | None = None
    reason: str = ""


def theorem57_hypothesis_holds(
    graph: nx.Graph,
    *,
    max_faces: int | None = None,
    max_candidates: int | None = None,
    max_edges: int | None = None,
    max_maximal_cliques: int | None = None,
    max_closed_sets: int | None = None,
) -> Theorem57Result:
    """Check Theorem 5.7 on the full ``DB(P(G))`` with replayable evidence.

    Each ``h``-image of the clique graph of ``DB(P(G))`` is tested for a
    strong-collapse certificate. Failure to find a certificate is
    inconclusive unless exact rational homology proves noncontractibility.
    All face/closure bounds produce an inconclusive status rather than a
    positive or negative theorem verdict.
    """
    for name, value in (
        ("max_faces", max_faces),
        ("max_candidates", max_candidates),
        ("max_edges", max_edges),
        ("max_maximal_cliques", max_maximal_cliques),
        ("max_closed_sets", max_closed_sets),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be nonnegative or None")

    double_bound = double_bound_graph(
        graph,
        max_faces=max_faces,
        max_candidates=max_candidates,
        max_edges=max_edges,
    )
    if double_bound is None:
        return Theorem57Result(
            Theorem57Status.INCONCLUSIVE,
            0,
            {},
            reason="double-bound construction exceeded a configured limit",
        )
    clique_graph_of_double_bound = clique_graph(double_bound)
    if clique_graph_of_double_bound is None:
        raise RuntimeError("unbounded clique graph construction returned no graph")
    intersections = closed_clique_intersections(
        clique_graph_of_double_bound,
        max_maximal_cliques=max_maximal_cliques,
        max_closed_sets=max_closed_sets,
    )
    if not intersections.is_exact:
        return Theorem57Result(
            Theorem57Status.INCONCLUSIVE,
            0,
            {},
            reason="closed h-image enumeration exceeded a configured limit",
        )

    from pycombtop.simplex import relative_betti_numbers

    certificates: dict[frozenset, tuple[tuple[Hashable, Hashable], ...]] = {}
    first_unknown: frozenset | None = None
    for checked_count, h_image in enumerate(intersections.closed_sets, start=1):
        facets = tuple(frozenset(clique) for clique in h_image)
        certificate = strong_collapse_certificate(facets)
        if certificate is not None and verify_strong_collapse_certificate(
            facets, certificate
        ):
            certificates[h_image] = certificate
            continue

        betti_numbers = relative_betti_numbers(
            facets,
            lambda _: False,
            method="exact",
        )
        noncontractible = betti_numbers.get(0, 0) != 1 or any(
            betti_number > 0
            for dimension, betti_number in betti_numbers.items()
            if dimension > 0
        )
        if noncontractible:
            return Theorem57Result(
                Theorem57Status.COUNTEREXAMPLE,
                checked_count,
                certificates,
                witness=h_image,
                reason="exact rational homology is nonzero in reduced degree",
            )
        if first_unknown is None:
            first_unknown = h_image

    if first_unknown is not None:
        return Theorem57Result(
            Theorem57Status.INCONCLUSIVE,
            len(intersections.closed_sets),
            certificates,
            witness=first_unknown,
            reason="no strong-collapse certificate; vanishing homology is not proof",
        )

    return Theorem57Result(
        Theorem57Status.PROVED,
        len(intersections.closed_sets),
        certificates,
        reason="every closed h-image has a verified strong-collapse certificate",
    )

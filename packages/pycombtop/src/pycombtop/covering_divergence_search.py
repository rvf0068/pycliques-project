"""Opt-in bounded discovery over triangular double covers and clockwork maps."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from math import inf

import networkx as nx
from pycliques.cliques import Clique, clique_graph
from pycliques.clockwork_pairs import clockwork_coaffine_pair
from pycliques.clockwork_search import CertificateVerificationStatus
from pycliques.coaffinations import CoaffinePair
from pycliques.small import Certificate

from .covering_divergence import (
    TriangularCoverDivergenceCertificate,
    verify_triangular_cover_divergence_certificate,
)


class CoveringSearchStatus(Enum):
    """Outcome within the configured double-cover and clockwork search family."""

    FOUND = "FOUND"
    EXHAUSTED = "EXHAUSTED"
    LIMIT_REACHED = "LIMIT_REACHED"


@dataclass(frozen=True)
class CoveringSearchResult:
    """Bounded search result with separate cover-class and morphism completion."""

    status: CoveringSearchStatus
    certificate: TriangularCoverDivergenceCertificate | None
    cover_classes_total: int
    cover_classes_checked: int
    cover_candidates_checked: int
    morphism_searches_exhausted: int
    morphism_states: int
    morphism_assignments: int
    cover_classes_complete: bool
    morphisms_complete: bool
    configured_source_parameters: tuple[tuple[int, int], ...]
    limits: tuple[tuple[str, int], ...]
    reasons: tuple[str, ...]


def _triangles(graph: nx.Graph) -> tuple[tuple[Hashable, Hashable, Hashable], ...]:
    node_order = {vertex: index for index, vertex in enumerate(graph)}
    result: dict[frozenset[Hashable], tuple[Hashable, Hashable, Hashable]] = {}
    for first in graph:
        for second in graph.neighbors(first):
            if second == first:
                continue
            for third in graph.neighbors(first):
                if third in (first, second) or not graph.has_edge(second, third):
                    continue
                triangle = tuple(
                    sorted((first, second, third), key=node_order.__getitem__)
                )
                result[frozenset(triangle)] = triangle
    return tuple(
        result[key]
        for key in sorted(
            result,
            key=lambda simplex: tuple(sorted(node_order[v] for v in simplex)),
        )
    )


def _nullspace_basis(column_count: int, equations: list[int]) -> tuple[int, ...]:
    """Return bit-mask basis vectors for a homogeneous GF(2) kernel."""
    rows = [row for row in equations if row]
    pivots: list[int] = []
    pivot_row = 0
    for column in range(column_count):
        selected = next(
            (
                index
                for index in range(pivot_row, len(rows))
                if rows[index] >> column & 1
            ),
            None,
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        for index in range(len(rows)):
            if index != pivot_row and rows[index] >> column & 1:
                rows[index] ^= rows[pivot_row]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    free_columns = [column for column in range(column_count) if column not in pivots]
    basis = []
    for free in free_columns:
        vector = 1 << free
        for row_index, pivot in enumerate(pivots):
            if rows[row_index] >> free & 1:
                vector |= 1 << pivot
        basis.append(vector)
    return tuple(basis)


def _normalized_cocycle_basis(
    graph: nx.Graph,
) -> tuple[
    tuple[tuple[Hashable, Hashable], ...],
    tuple[tuple[Hashable, Hashable], ...],
    tuple[int, ...],
]:
    nodes = tuple(graph)
    node_index = {vertex: index for index, vertex in enumerate(nodes)}
    edges = tuple(
        sorted(
            (
                (first, second)
                if node_index[first] < node_index[second]
                else (second, first)
                for first, second in graph.edges()
            ),
            key=lambda edge: (node_index[edge[0]], node_index[edge[1]]),
        )
    )
    tree_edges = tuple(
        (first, second) if node_index[first] < node_index[second] else (second, first)
        for first, second in nx.dfs_edges(graph, source=nodes[0])
    )
    tree_edge_set = {frozenset(edge) for edge in tree_edges}
    cotree_edges = tuple(edge for edge in edges if frozenset(edge) not in tree_edge_set)
    cotree_index = {frozenset(edge): index for index, edge in enumerate(cotree_edges)}
    equations: list[int] = []
    for triangle in _triangles(graph):
        row = 0
        for edge in combinations(triangle, 2):
            index = cotree_index.get(frozenset(edge))
            if index is not None:
                row ^= 1 << index
        equations.append(row)
    return edges, tree_edges, _nullspace_basis(len(cotree_edges), equations)


def _all_nonzero_cocycles(
    graph: nx.Graph, *, max_classes: int
) -> tuple[tuple[tuple[Hashable, Hashable, int], ...], ...]:
    edges, tree_edges, basis = _normalized_cocycle_basis(graph)
    tree = {frozenset(edge) for edge in tree_edges}
    cotree = tuple(edge for edge in edges if frozenset(edge) not in tree)
    dimension = len(basis)
    cotree_index = {edge: index for index, edge in enumerate(cotree)}
    classes = []
    for coordinates in range(1, min(1 << dimension, max_classes + 1)):
        mask = 0
        for bit, vector in enumerate(basis):
            if coordinates >> bit & 1:
                mask ^= vector
        classes.append(
            tuple(
                (
                    first,
                    second,
                    0
                    if frozenset((first, second)) in tree
                    else (mask >> cotree_index[(first, second)]) & 1,
                )
                for first, second in edges
            )
        )
    return tuple(classes)


def _double_cover(
    graph: nx.Graph,
    cocycle: tuple[tuple[Hashable, Hashable, int], ...],
) -> tuple[nx.Graph, dict[Hashable, Hashable]]:
    voltage = {frozenset((first, second)): value for first, second, value in cocycle}
    cover = nx.Graph()
    projection: dict[Hashable, Hashable] = {}
    for vertex in graph:
        for sheet in (0, 1):
            lifted = (vertex, sheet)
            cover.add_node(lifted)
            projection[lifted] = vertex
    for first, second in graph.edges():
        change = voltage[frozenset((first, second))]
        for sheet in (0, 1):
            cover.add_edge((first, sheet), (second, sheet ^ change))
    return cover, projection


def _bounded_pair_morphism(
    source_pair: CoaffinePair,
    target_pair: CoaffinePair,
    *,
    max_states: int,
    max_assignments: int,
) -> tuple[dict[Hashable, Hashable] | None, int, int, bool]:
    """Search coaffination orbits with explicit state and assignment budgets."""
    source = source_pair.graph
    sigma = source_pair.coaffination
    target = target_pair.graph
    tau = target_pair.coaffination
    target_nodes = tuple(target)
    fixed_targets = tuple(vertex for vertex in target if tau[vertex] == vertex)

    unprocessed = set(source)
    orbits: list[tuple[Hashable, ...]] = []
    while unprocessed:
        vertex = next(node for node in source if node in unprocessed)
        partner = sigma[vertex]
        orbit = (vertex,) if partner == vertex else (vertex, partner)
        unprocessed.difference_update(orbit)
        orbits.append(orbit)
    node_order = {vertex: index for index, vertex in enumerate(source)}
    orbits.sort(
        key=lambda orbit: (
            -sum(source.degree(vertex) for vertex in orbit),
            min(node_order[vertex] for vertex in orbit),
        )
    )

    assignment: dict[Hashable, Hashable] = {}
    states = 0
    assignments_tried = 0
    limit_reached = False

    def consistent(additions: dict[Hashable, Hashable]) -> bool:
        partial = assignment | additions
        for vertex, image in additions.items():
            for neighbor in source.neighbors(vertex):
                if neighbor in partial:
                    other_image = partial[neighbor]
                    if image != other_image and not target.has_edge(image, other_image):
                        return False
        return True

    def backtrack(index: int) -> dict[Hashable, Hashable] | None:
        nonlocal states, assignments_tried, limit_reached
        if states >= max_states:
            limit_reached = True
            return None
        states += 1
        if index == len(orbits):
            return dict(assignment)
        orbit = orbits[index]
        candidates = fixed_targets if len(orbit) == 1 else target_nodes
        for image in candidates:
            if assignments_tried >= max_assignments:
                limit_reached = True
                return None
            assignments_tried += 1
            additions = (
                {orbit[0]: image}
                if len(orbit) == 1
                else {orbit[0]: image, orbit[1]: tau[image]}
            )
            if not consistent(additions):
                continue
            assignment.update(additions)
            found = backtrack(index + 1)
            if found is not None:
                return found
            for vertex in additions:
                del assignment[vertex]
            if limit_reached:
                return None
        return None

    return backtrack(0), states, assignments_tried, limit_reached


def search_covering_divergence(
    graph: nx.Graph,
    *,
    cover_degree: int = 2,
    max_cover_classes: int = 32,
    max_cover_vertices: int = 128,
    max_cover_edges: int = 512,
    max_clique_vertices: int = 100,
    max_clique_edges: int = 2000,
    max_source_order: int = 40,
    max_m: int = 3,
    max_n: int = 3,
    max_morphism_states: int = 100000,
    max_morphism_assignments: int = 1000000,
) -> CoveringSearchResult:
    """Search a bounded family of flat connected double covers for divergence.

    Double covers are represented by tree-normalized triangle-flat GF(2)
    cocycles, so distinct nonzero cohomology classes are enumerated once.
    ``EXHAUSTED`` applies only to these classes and the configured clockwork
    parameter ranges. No miss proves clique convergence.
    """
    if (
        not isinstance(graph, nx.Graph)
        or graph.is_directed()
        or graph.is_multigraph()
        or nx.number_of_selfloops(graph)
    ):
        raise ValueError("graph must be simple and undirected")
    if not graph or not nx.is_connected(graph):
        raise ValueError("graph must be nonempty and connected")
    if cover_degree != 2:
        raise ValueError("only degree-two covers are currently supported")
    limits = {
        "max_cover_classes": max_cover_classes,
        "max_cover_vertices": max_cover_vertices,
        "max_cover_edges": max_cover_edges,
        "max_clique_vertices": max_clique_vertices,
        "max_clique_edges": max_clique_edges,
        "max_source_order": max_source_order,
        "max_m": max_m,
        "max_n": max_n,
        "max_morphism_states": max_morphism_states,
        "max_morphism_assignments": max_morphism_assignments,
    }
    if any(value < 0 for value in limits.values()):
        raise ValueError("all search limits must be nonnegative")
    if max_cover_classes < 1:
        raise ValueError("max_cover_classes must be at least one")

    _, _, basis = _normalized_cocycle_basis(graph)
    total_classes = (1 << len(basis)) - 1
    classes_complete = total_classes <= max_cover_classes
    cocycles = _all_nonzero_cocycles(graph, max_classes=max_cover_classes)
    checked_classes = 0
    checked_covers = 0
    exhausted_morphisms = 0
    total_states = 0
    total_assignments = 0
    reasons: list[str] = []
    morphisms_complete = True
    morphism_budget_exhausted = False
    if not classes_complete:
        reasons.append(
            "nonzero H^1 classes exceed max_cover_classes; enumeration was truncated"
        )
    source_parameters = tuple(
        (m, n) for m in range(2, max_m + 1) for n in range(max_n + 1)
    )

    for cocycle in cocycles:
        checked_classes += 1
        cover, projection = _double_cover(graph, cocycle)
        if (
            cover.number_of_nodes() > max_cover_vertices
            or cover.number_of_edges() > max_cover_edges
        ):
            reasons.append("cover construction exceeded a configured size limit")
            morphisms_complete = False
            continue
        if not nx.is_connected(cover):
            raise AssertionError("nonzero normalized cocycle made a disconnected cover")
        checked_covers += 1
        deck = {vertex: (vertex[0], 1 - vertex[1]) for vertex in cover}
        distances = dict(nx.all_pairs_shortest_path_length(cover))
        deck_radius = min(distances[vertex].get(deck[vertex], inf) for vertex in cover)
        eligible = [
            (m, n)
            for m, n in source_parameters
            if 2 * m * (n + 3) <= max_source_order and m + 1 <= deck_radius
        ]
        if not eligible:
            continue

        clique_cover = clique_graph(
            cover,
            bound=max_clique_vertices,
            max_edges=max_clique_edges,
        )
        if clique_cover is None:
            reasons.append("clique-graph construction exceeded a configured limit")
            morphisms_complete = False
            continue
        target_deck: dict[Hashable, Hashable] = {}
        for clique in clique_cover:
            target_deck[clique] = Clique(deck[vertex] for vertex in clique)
        target_pair = CoaffinePair(clique_cover, target_deck)
        for m, n in eligible:
            source_pair = clockwork_coaffine_pair(m, n)
            morphism, states, assignments, limited = _bounded_pair_morphism(
                source_pair,
                target_pair,
                max_states=max_morphism_states - total_states,
                max_assignments=max_morphism_assignments - total_assignments,
            )
            total_states += states
            total_assignments += assignments
            if limited:
                reasons.append(
                    f"morphism search for R_{{{2 * m}}}^{n} reached a configured limit"
                )
                morphisms_complete = False
                morphism_budget_exhausted = True
                break
            if morphism is None:
                exhausted_morphisms += 1
                continue

            theorem_certificate = Certificate(
                rule="theorem_2_6_clockwork_pair_map",
                target=clique_cover,
                target_status="proven_divergent",
                map=morphism,
                m=m,
                n=n,
                radius=m + 1,
                source=source_pair.graph,
                source_coaffination=source_pair.coaffination,
                target_coaffination=target_deck,
                theorem_base_graph=clique_cover,
            )
            certificate = TriangularCoverDivergenceCertificate(
                base_graph=graph,
                cover_graph=cover,
                projection=projection,
                deck_transformation=deck,
                theorem_certificate=theorem_certificate,
                voltage_cocycle=cocycle,
                clique_bound=max_clique_vertices,
                clique_max_edges=max_clique_edges,
                deck_radius=m + 1,
            )
            verified = verify_triangular_cover_divergence_certificate(certificate)
            if verified.status is not CertificateVerificationStatus.VERIFIED:
                raise AssertionError(
                    "discovery returned a candidate that failed certificate replay: "
                    + verified.reason
                )
            return CoveringSearchResult(
                status=CoveringSearchStatus.FOUND,
                certificate=certificate,
                cover_classes_total=total_classes,
                cover_classes_checked=checked_classes,
                cover_candidates_checked=checked_covers,
                morphism_searches_exhausted=exhausted_morphisms,
                morphism_states=total_states,
                morphism_assignments=total_assignments,
                cover_classes_complete=classes_complete,
                morphisms_complete=False,
                configured_source_parameters=source_parameters,
                limits=tuple(sorted(limits.items())),
                reasons=tuple(reasons),
            )
        else:
            continue
        if morphism_budget_exhausted:
            break

    complete = classes_complete and morphisms_complete
    return CoveringSearchResult(
        status=(
            CoveringSearchStatus.EXHAUSTED
            if complete
            else CoveringSearchStatus.LIMIT_REACHED
        ),
        certificate=None,
        cover_classes_total=total_classes,
        cover_classes_checked=checked_classes,
        cover_candidates_checked=checked_covers,
        morphism_searches_exhausted=exhausted_morphisms,
        morphism_states=total_states,
        morphism_assignments=total_assignments,
        cover_classes_complete=classes_complete,
        morphisms_complete=morphisms_complete,
        configured_source_parameters=source_parameters,
        limits=tuple(sorted(limits.items())),
        reasons=tuple(reasons),
    )

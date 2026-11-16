"""Opt-in bounded discovery over triangular double covers and clockwork maps."""

from __future__ import annotations

from collections import deque
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


@dataclass
class _PairMorphismSearchFrame:
    """One reversible orbit-assignment level of the constraint search."""

    orbit_index: int
    remaining: int
    checkpoint: int


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
    """Find a coaffine graph morphism by a bounded, deterministic CSP.

    The variables are involution orbits in the source. Each variable's
    domain is a bit set of possible images of its first vertex; the
    coaffination determines the image of its partner. Compatibility masks
    encode *all* source edges between two orbits. AC-3 arc consistency and
    a minimum-remaining-values (MRV) branch order eliminate unsupported
    orbit images before trying them individually.

    This remains a search for graph morphisms, not induced embeddings:
    adjacent source vertices may have equal target images. There is no
    injectivity requirement. All complete assignments are verified before
    return. Budgets count attempted orbit images and successfully propagated
    partial-assignment states, and stop without asserting nonexistence.
    An exhaustive failure (including AC-3 contradiction) is not a
    certificate about clique convergence.
    """
    source = source_pair.graph
    sigma = source_pair.coaffination
    target = target_pair.graph
    tau = target_pair.coaffination
    source_nodes = tuple(source)
    target_nodes = tuple(target)
    node_to_target = {node: i for i, node in enumerate(target_nodes)}
    target_count = len(target_nodes)

    states = 0
    attempts = 0
    if max_states < 1:
        return None, states, attempts, True
    states = 1

    # A coaffination for this double-cover procedure is an involution.
    # Reject malformed inputs rather than silently omit a longer orbit.
    if any(
        sigma.get(vertex) not in source or sigma.get(sigma[vertex]) != vertex
        for vertex in source_nodes
    ):
        raise ValueError("Source coaffination must be an involution")
    if any(
        tau.get(vertex) not in target or tau.get(tau[vertex]) != vertex
        for vertex in target_nodes
    ):
        raise ValueError("Target coaffination must be an involution")

    seen: set[Hashable] = set()
    orbits: list[tuple[Hashable, ...]] = []
    orbit_of: dict[Hashable, tuple[int, int]] = {}
    for vertex in source_nodes:
        if vertex in seen:
            continue
        mate = sigma[vertex]
        orbit = (vertex,) if mate == vertex else (vertex, mate)
        orbit_index = len(orbits)
        orbits.append(orbit)
        for role, member in enumerate(orbit):
            seen.add(member)
            orbit_of[member] = orbit_index, role
    if not orbits:
        return {}, states, attempts, False

    tau_indices = [node_to_target[tau[v]] for v in target_nodes]
    full_domain = (1 << target_count) - 1
    fixed_domain = sum(
        1 << index for index, partner in enumerate(tau_indices) if partner == index
    )
    closed = [
        (1 << index)
        | sum(1 << node_to_target[neighbor] for neighbor in target.neighbors(vertex))
        for index, vertex in enumerate(target_nodes)
    ]
    # When the second endpoint of a source edge is a mate, the target
    # representative must lie in tau^{-1}(closed_neighborhood[image]).
    inverse_closed = [
        sum(1 << index for index, mate in enumerate(tau_indices) if mask >> mate & 1)
        for mask in closed
    ]

    domains = []
    for orbit in orbits:
        candidates = fixed_domain if len(orbit) == 1 else full_domain
        if len(orbit) == 2 and source.has_edge(orbit[0], orbit[1]):
            # An internal orbit edge must also map to an edge or equality.
            candidates &= sum(
                1 << index
                for index, partner in enumerate(tau_indices)
                if closed[index] >> partner & 1
            )
        domains.append(candidates)
    if any(mask == 0 for mask in domains):
        return None, states, attempts, False

    # Each source edge imposes a binary relation on its two orbit variables.
    # The roles identify whether a source endpoint uses the representative
    # target image (0) or its coaffine mate (1).
    constraints: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for u, v in source.edges():
        first, u_role = orbit_of[u]
        second, v_role = orbit_of[v]
        if first == second:
            continue  # The internal edge was handled by the unary domain.
        constraints.setdefault((first, second), []).append((u_role, v_role))
        constraints.setdefault((second, first), []).append((v_role, u_role))

    neighbors = [set() for _ in orbits]
    compatibility: dict[tuple[int, int], tuple[int, ...]] = {}
    for (i, j), edges in constraints.items():
        neighbors[i].add(j)
        support_masks = []
        for index in range(target_count):
            supported = full_domain
            for source_role, target_role in edges:
                image = index if source_role == 0 else tau_indices[index]
                supported &= (
                    closed[image] if target_role == 0 else inverse_closed[image]
                )
            support_masks.append(supported)
        compatibility[i, j] = tuple(support_masks)
    linked = [tuple(sorted(around)) for around in neighbors]
    degrees = [sum(source.degree(member) for member in orbit) for orbit in orbits]

    # Store only changed domains on a trail, so backtracking is O(changes),
    # not O(number of source orbits) per state. This is also safe for 1100+
    # orbits, where a recursive Python search would exceed its stack limit.
    trail: list[tuple[int, int]] = []

    def bits(mask: int):
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    def revise(i: int, j: int) -> bool:
        """Remove target images of orbit i lacking support in orbit j."""
        current = domains[i]
        support = compatibility[i, j]
        filtered = sum(
            1 << image for image in bits(current) if support[image] & domains[j]
        )
        if filtered == current:
            return False
        trail.append((i, current))
        domains[i] = filtered
        return True

    def propagate(arcs: deque[tuple[int, int]]) -> bool:
        while arcs:
            i, j = arcs.popleft()
            if revise(i, j):
                if domains[i] == 0:
                    return False
                for k in linked[i]:
                    if k != j:
                        arcs.append((k, i))
        return True

    # Initial AC-3, even before the first attempted assignment, finds
    # contradictions from pairs of orbits with disjoint support.
    if not propagate(deque(sorted(compatibility))):
        return None, states, attempts, False
    trail.clear()

    def choose_orbit(assigned: dict[int, int]) -> int:
        return min(
            (i for i in range(len(orbits)) if i not in assigned),
            key=lambda i: (
                domains[i].bit_count(),
                -sum(j in assigned for j in linked[i]),
                -degrees[i],
                i,
            ),
        )

    def restore(checkpoint: int) -> None:
        while len(trail) > checkpoint:
            orbit, old_domain = trail.pop()
            domains[orbit] = old_domain

    assignment: dict[int, int] = {}
    first = choose_orbit(assignment)
    frames = [_PairMorphismSearchFrame(first, domains[first], len(trail))]
    while frames:
        frame = frames[-1]
        restore(frame.checkpoint)
        assignment.pop(frame.orbit_index, None)
        if frame.remaining == 0:
            frames.pop()
            continue
        if attempts >= max_assignments:
            return None, states, attempts, True

        candidate_bit = frame.remaining & -frame.remaining
        frame.remaining ^= candidate_bit
        candidate = candidate_bit.bit_length() - 1
        attempts += 1
        i = frame.orbit_index
        if domains[i] != candidate_bit:
            trail.append((i, domains[i]))
            domains[i] = candidate_bit
        if not propagate(deque((j, i) for j in linked[i])):
            continue

        if states >= max_states:
            return None, states, attempts, True
        states += 1
        assignment[i] = candidate
        if len(assignment) == len(orbits):
            result: dict[Hashable, Hashable] = {}
            for orbit_index, orbit in enumerate(orbits):
                image = assignment[orbit_index]
                result[orbit[0]] = target_nodes[image]
                if len(orbit) == 2:
                    result[orbit[1]] = target_nodes[tau_indices[image]]
            if any(
                result[u] != result[v] and not target.has_edge(result[u], result[v])
                for u, v in source.edges()
            ):
                raise AssertionError("AC-3 returned a map that fails adjacency")
            if any(result[sigma[u]] != tau[result[u]] for u in source):
                raise AssertionError("AC-3 returned a map that fails equivariance")
            return result, states, attempts, False
        next_i = choose_orbit(assignment)
        frames.append(_PairMorphismSearchFrame(next_i, domains[next_i], len(trail)))
    return None, states, attempts, False


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

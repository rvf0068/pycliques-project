"""Bounded homology-guided retractions onto named graph families.

Independent of pycombtop. Betti input consists of exact reduced ranks or
certified upper bounds over GF(2). Missing means unknown.
The built-in target families search only targets with established divergence.
Verification of older retractions remains valid, including nondivergent targets.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import networkx as nx

from .homological_retractions import _validate_betti_bounds
from .retractions import (
    RetractionSearchStatus,
    bounded_retraction,
    verify_retraction_witness,
)


class TargetSearchStatus(Enum):
    FOUND = auto()
    EXHAUSTED = auto()
    LIMIT_REACHED = auto()
    HOMOLOGY_EXCLUDED = auto()


@dataclass(frozen=True)
class TargetRetractionResult:
    family: str
    status: TargetSearchStatus
    patterns_checked: int = 0
    induced_targets_checked: int = 0
    states_explored: int = 0
    assignments_tried: int = 0
    target_kind: str | None = None
    target_vertices: tuple[Hashable, ...] | None = None
    projection: dict[Hashable, Hashable] | None = None
    reason: str = ""


FAMILIES = ("octahedra", "suspensions", "cycle_complements")


def _target_betti(family: str, parameter: int) -> tuple[int, int]:
    """Positive-degree homology rank of a named target clique complex."""
    if family == "octahedra":
        return parameter - 1, 1
    if family == "suspensions":
        return 2, 1
    q, r = divmod(parameter, 3)
    if r == 0:
        return q - 1, 2
    return (q - 1, 1) if r == 1 else (q, 1)


def _parameters(family: str, n: int, limit: int) -> range:
    order = min(n, limit)
    if family == "octahedra":
        return range(3, order // 2 + 1)
    if family == "suspensions":
        return range(4, order - 1)
    # Complements of cycles are K-divergent only from C_8 onward.
    # Searching C_5, C_6 or C_7 can produce an irrelevant first witness,
    # prematurely ending the search before an induced divergent target.
    return range(8, order + 1)


def _octahedra(graph: nx.Graph, accepted: set[int]) -> Iterable[tuple[int, frozenset]]:
    """Enumerate all induced O_d using induced matchings in the complement."""
    complement = nx.complement(graph)
    edges = tuple(complement.edges())
    forbidden = {v: set(complement[v]) | {v} for v in complement}
    limit = max(accepted)

    def walk(start: int, chosen: tuple[int, ...], excluded: frozenset):
        dimension = len(chosen)
        if dimension in accepted:
            yield dimension, frozenset(v for i in chosen for v in edges[i])
        if dimension == limit:
            return
        for index in range(start, len(edges)):
            a, b = edges[index]
            if a in excluded or b in excluded:
                continue
            yield from walk(
                index + 1,
                chosen + (index,),
                excluded | forbidden[a] | forbidden[b],
            )

    yield from walk(0, (), frozenset())


def _cycle_complements(
    graph: nx.Graph, accepted: set[int]
) -> Iterable[tuple[int, frozenset]]:
    complement = nx.complement(graph)
    for cycle in nx.chordless_cycles(complement, length_bound=max(accepted)):
        if len(cycle) in accepted:
            yield len(cycle), frozenset(cycle)


def _suspensions(
    graph: nx.Graph, accepted: set[int]
) -> Iterable[tuple[int, frozenset]]:
    for a, b in nx.non_edges(graph):
        common = set(graph[a]) & set(graph[b])
        if len(common) < min(accepted):
            continue
        for cycle in nx.chordless_cycles(
            graph.subgraph(common), length_bound=max(accepted)
        ):
            if len(cycle) in accepted:
                yield len(cycle), frozenset(cycle) | {a, b}


_GENERATORS = {
    "octahedra": _octahedra,
    "suspensions": _suspensions,
    "cycle_complements": _cycle_complements,
}


def verify_targeted_retraction(graph: nx.Graph, result: TargetRetractionResult) -> bool:
    """Independently verify both the graph retraction and named target shape."""
    if (
        result.status is not TargetSearchStatus.FOUND
        or result.target_vertices is None
        or result.projection is None
        or result.target_kind is None
    ):
        return False
    target = graph.subgraph(result.target_vertices)
    if not verify_retraction_witness(
        graph, target, (result.projection, {v: v for v in target})
    ):
        return False
    try:
        family, dimension = result.target_kind.split(":")
        parameter = int(dimension)
    except (ValueError, AttributeError):
        return False
    if family == "octahedra":
        complement = nx.complement(target)
        return len(target) == 2 * parameter and all(
            degree == 1 for _, degree in complement.degree()
        )
    if family == "cycle_complements":
        complement = nx.complement(target)
        return (
            len(target) == parameter
            and parameter >= 5
            and (
                nx.is_connected(complement)
                and all(degree == 2 for _, degree in complement.degree())
            )
        )
    if family == "suspensions":
        if len(target) != parameter + 2 or parameter < 4:
            return False
        for a, b in nx.non_edges(target):
            cycle = target.copy()
            cycle.remove_nodes_from((a, b))
            if (
                nx.is_connected(cycle)
                and all(degree == 2 for _, degree in cycle.degree())
                and all(target.has_edge(a, v) and target.has_edge(b, v) for v in cycle)
            ):
                return True
    return False


def search_targeted_retractions(
    graph: nx.Graph,
    *,
    families: Sequence[str] = FAMILIES,
    betti_numbers: Mapping[int, int] | None = None,
    betti_upper_bounds: Mapping[int, int] | None = None,
    octahedron_dimensions: Sequence[int] | None = None,
    max_target_order: int = 16,
    max_patterns: int = 50_000,
    max_states: int = 100_000,
    max_assignments: int = 500_000,
) -> dict[str, TargetRetractionResult]:
    """Find ordinary graph retractions to known divergent target families.

    Cycle complements are searched only for C_n with n >= 8; smaller
    complements are not K-divergent. The independent witness verifier
    deliberately remains valid for previously saved C_5--C_7 retractions.

    The supplied reduced GF(2) Betti numbers are trusted to be EXACT,
    may cover only selected dimensions, and are never computed here.
    Certified upper bounds may also be supplied via betti_upper_bounds.
    Exact ranks take precedence; inconsistent bounds raise ValueError.
    Missing dimensions NEVER imply vanishing. A HOMOLOGY_EXCLUDED result
    is relative to these supplied facts, not an independent certificate.
    Budgets apply separately to each family and LIMIT_REACHED is never
    reported as exhaustive absence. Positive witnesses are replayed.
    ``octahedron_dimensions`` optionally restricts the octahedral family
    to exactly the requested dimensions; None preserves the historical
    all-dimensions search. This is especially useful for K^3 -> O_4.
    """
    if (
        not isinstance(graph, nx.Graph)
        or graph.is_multigraph()
        or graph.is_directed()
        or nx.number_of_selfloops(graph)
    ):
        raise ValueError("Require an undirected simple loopless graph")
    if len(set(families)) != len(families) or any(f not in FAMILIES for f in families):
        raise ValueError("Unknown or duplicated target family")
    if min(max_target_order, max_patterns, max_states, max_assignments) < 1:
        raise ValueError("Search bounds must be positive")
    if octahedron_dimensions is not None and (
        not octahedron_dimensions
        or any(type(d) is not int or d < 3 for d in octahedron_dimensions)
        or len(set(octahedron_dimensions)) != len(octahedron_dimensions)
    ):
        raise ValueError("Octahedron dimensions must be distinct integers >= 3")
    exact = {} if betti_numbers is None else dict(betti_numbers)
    known = {} if betti_upper_bounds is None else dict(betti_upper_bounds)
    _validate_betti_bounds(exact)
    _validate_betti_bounds(known)
    if any(d in known and rank > known[d] for d, rank in exact.items()):
        raise ValueError("Exact Betti rank exceeds supplied upper bound")
    known.update(exact)

    results = {}
    for family in families:
        all_params = set(_parameters(family, len(graph), max_target_order))
        if family == "octahedra" and octahedron_dimensions is not None:
            all_params.intersection_update(octahedron_dimensions)
        allowed = {
            p
            for p in all_params
            if (dimension := _target_betti(family, p)[0]) not in known
            or known[dimension] >= _target_betti(family, p)[1]
        }
        if not allowed:
            status = (
                TargetSearchStatus.HOMOLOGY_EXCLUDED
                if all_params
                else TargetSearchStatus.EXHAUSTED
            )
            results[family] = TargetRetractionResult(
                family,
                status,
                reason="No admissible target given bounds/known homology",
            )
            continue
        patterns = targets = states = assignments = 0
        seen: set[frozenset] = set()
        status = TargetSearchStatus.EXHAUSTED
        reason = "All requested target patterns were exhausted"
        hit = None
        for parameter, vertices in _GENERATORS[family](graph, allowed):
            if patterns >= max_patterns:
                status = TargetSearchStatus.LIMIT_REACHED
                reason = "Exceeded candidate-pattern budget"
                break
            patterns += 1
            if vertices in seen:
                continue
            seen.add(vertices)
            targets += 1
            if states >= max_states or assignments >= max_assignments:
                status = TargetSearchStatus.LIMIT_REACHED
                reason = "Exceeded total retraction-search budget"
                break
            check = bounded_retraction(
                graph,
                graph.subgraph(vertices),
                max_states=max_states - states,
                max_assignments=max_assignments - assignments,
            )
            states += check.states_explored
            assignments += check.assignments_tried
            if check.status is RetractionSearchStatus.LIMIT_REACHED:
                status = TargetSearchStatus.LIMIT_REACHED
                reason = "An induced target has an incomplete CSP search"
                break
            if check.status is RetractionSearchStatus.FOUND:
                hit = TargetRetractionResult(
                    family,
                    TargetSearchStatus.FOUND,
                    patterns,
                    targets,
                    states,
                    assignments,
                    f"{family}:{parameter}",
                    tuple(v for v in graph if v in vertices),
                    check.retraction,
                    "Verified induced target and graph retraction",
                )
                if not verify_targeted_retraction(graph, hit):
                    raise AssertionError("Positive retraction witness failed replay")
                break
        results[family] = hit or TargetRetractionResult(
            family,
            status,
            patterns,
            targets,
            states,
            assignments,
            reason=reason,
        )
    return results

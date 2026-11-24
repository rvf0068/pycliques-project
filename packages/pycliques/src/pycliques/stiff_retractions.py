"""Bounded search over proper nontrivial stiff induced retracts."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from itertools import combinations

import networkx as nx

from .coaffine_endomorphisms import _validate_graph
from .retractions import (
    RetractionSearchStatus,
    bounded_retraction,
    verify_retraction_witness,
)


@dataclass(frozen=True)
class StiffRetractionResult:
    """Result relative to the requested target-order window.

    EXHAUSTED means every eligible subset was checked; LIMIT_REACHED
    never excludes a retract. A FOUND witness fixes its image pointwise.
    """

    status: RetractionSearchStatus
    target_vertices: tuple[Hashable, ...] | None
    retraction: dict[Hashable, Hashable] | None
    subsets_checked: int
    stiff_candidates: int
    neighborhood_candidates: int
    states_explored: int
    assignments_tried: int


def search_proper_stiff_retraction(
    graph: nx.Graph,
    *,
    min_target_order: int = 2,
    max_target_order: int | None = None,
    max_subsets: int = 100_000,
    max_states: int = 100_000,
    max_assignments: int = 1_000_000,
) -> StiffRetractionResult:
    """Search proper induced stiff targets in increasing order.

    A target is stiff if no two distinct vertices have nested closed
    neighborhoods. Reject it also if an external vertex cannot map
    to any target vertex consistently with its fixed target neighbors.
    The remaining fixed-target CSP uses the existing bounded solver.

    The three budgets are global, not per target. In particular, subsets
    rejected by inexpensive tests still count against max_subsets.
    """
    _validate_graph(graph)
    if type(min_target_order) is not int or min_target_order < 2:
        raise ValueError("min_target_order must be an integer >= 2")
    if max_target_order is not None and (
        type(max_target_order) is not int or max_target_order < min_target_order
    ):
        raise ValueError("max_target_order must be >= min_target_order")
    if any(
        type(value) is not int or value < 1
        for value in (max_subsets, max_states, max_assignments)
    ):
        raise ValueError("budgets must be positive integers")

    nodes = list(graph)
    n = len(nodes)
    upper = n - 1 if max_target_order is None else min(n - 1, max_target_order)
    positions = {v: i for i, v in enumerate(nodes)}
    closed = [
        (1 << i) | sum(1 << positions[w] for w in graph[v])
        for i, v in enumerate(nodes)
    ]
    subsets = stiff = eligible = states = assignments = 0

    def result(
        status: RetractionSearchStatus,
        target: tuple[Hashable, ...] | None = None,
        projection: dict[Hashable, Hashable] | None = None,
    ) -> StiffRetractionResult:
        return StiffRetractionResult(
            status, target, projection, subsets, stiff, eligible, states, assignments
        )

    for order in range(min_target_order, upper + 1):
        for indices in combinations(range(n), order):
            if subsets >= max_subsets:
                return result(RetractionSearchStatus.LIMIT_REACHED)
            subsets += 1
            mask = sum(1 << i for i in indices)
            restricted = [closed[i] & mask for i in indices]
            if any(
                i != j and a & b == a
                for i, a in enumerate(restricted)
                for j, b in enumerate(restricted)
            ):
                continue
            stiff += 1
            if any(
                not any((closed[i] & mask) & ~row == 0 for row in restricted)
                for i in range(n)
                if not mask & (1 << i)
            ):
                continue
            eligible += 1
            if states >= max_states or assignments >= max_assignments:
                return result(RetractionSearchStatus.LIMIT_REACHED)
            target = tuple(nodes[i] for i in indices)
            subgraph = graph.subgraph(target)
            check = bounded_retraction(
                graph,
                subgraph,
                max_states=max_states - states,
                max_assignments=max_assignments - assignments,
            )
            states += check.states_explored
            assignments += check.assignments_tried
            if check.status is RetractionSearchStatus.FOUND:
                if (
                    check.retraction is None
                    or check.inclusion is None
                    or not verify_retraction_witness(
                        graph, subgraph, (check.retraction, check.inclusion)
                    )
                ):
                    raise AssertionError("invalid stiff retraction witness")
                return result(check.status, target, check.retraction)
            if check.status is RetractionSearchStatus.LIMIT_REACHED:
                return result(check.status)

    return result(RetractionSearchStatus.EXHAUSTED)

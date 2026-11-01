"""Exact bounded selection search over a pruned Helly-cover candidate universe."""

from __future__ import annotations

import json
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any

import networkx as nx
from pycliques.helly_cover import (
    HellyCoverPruningResult,
    HellyCoverStatus,
    HellyCoverVerificationStatus,
    HellyEdgeCoverVerification,
    is_helly_family,
    verify_helly_edge_cover,
)


class SelectedHellyCoverStatus(Enum):
    """Outcome of bounded exact search over a fixed candidate universe."""

    FOUND_VERIFIED_COVER = "FOUND_VERIFIED_COVER"
    NO_COVER_CERTIFIED = "NO_COVER_CERTIFIED"
    INCONCLUSIVE_BUDGET = "INCONCLUSIVE_BUDGET"


@dataclass(frozen=True)
class SelectedHellyCoverSearchResult:
    """Search outcome, including the fixed-point universe and replay metadata."""

    status: SelectedHellyCoverStatus
    selected_family: tuple[frozenset[Hashable], ...] | None
    verification: HellyEdgeCoverVerification | None
    states_explored: int
    max_states: int | None
    max_depth: int | None
    branch_policy: str
    graph_edges: tuple[tuple[int, int], ...]
    pruning_result: HellyCoverPruningResult
    diagnostics: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic replay metadata with graph-internal vertex ids."""
        label_to_index = {
            label: index
            for index, label in enumerate(self.pruning_result.index_to_label)
        }

        def encode_family(
            family: tuple[frozenset[Hashable], ...] | None,
        ) -> list[list[int]] | None:
            if family is None:
                return None
            return [
                sorted(label_to_index[label] for label in member) for member in family
            ]

        return {
            "status": self.status.value,
            "selected_family": encode_family(self.selected_family),
            "verification": (
                None
                if self.verification is None
                else {
                    "status": self.verification.status.value,
                    "uncovered_edge": (
                        None
                        if self.verification.uncovered_edge is None
                        else sorted(
                            label_to_index[label]
                            for label in self.verification.uncovered_edge
                        )
                    ),
                    "diagnostics": self.verification.diagnostics,
                }
            ),
            "states_explored": self.states_explored,
            "resource_bounds": {
                "max_states": self.max_states,
                "max_depth": self.max_depth,
            },
            "branch_policy": self.branch_policy,
            "graph_edges": [list(edge) for edge in self.graph_edges],
            "candidate_universe": self.pruning_result.to_dict(),
            "diagnostics": self.diagnostics,
        }

    def to_json(self) -> str:
        """Serialize replay metadata with stable JSON key ordering."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


_BRANCH_POLICY = (
    "Select an uncovered edge with the fewest remaining candidates (ties by "
    "graph edge index); branch over those candidates in vertex-index bitmask order."
)


def _vertex_mask(vertices: frozenset[Hashable], vertex_ids: dict[Hashable, int]) -> int:
    return sum(1 << vertex_ids[vertex] for vertex in vertices)


def search_selected_helly_cover(
    graph: nx.Graph,
    pruning_result: HellyCoverPruningResult,
    *,
    max_states: int | None = None,
    max_depth: int | None = None,
) -> SelectedHellyCoverSearchResult:
    """Search exactly for a selected cover within ``pruning_result`` survivors.

    A non-Helly partial selection is pruned because adding sets cannot restore
    the Helly property. Branching on an uncovered edge is complete within the
    fixed candidate universe: every edge cover must select a remaining
    candidate containing that edge. A negative result is scoped to the
    pruner's surviving candidates and recorded normalization assumptions.
    """
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise ValueError("graph must be a simple undirected NetworkX graph")
    if nx.number_of_selfloops(graph):
        raise ValueError("graph must not contain self-loops")
    if max_states is not None and max_states < 0:
        raise ValueError("max_states must be nonnegative or None")
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be nonnegative or None")

    vertices = tuple(graph.nodes())
    if vertices != pruning_result.index_to_label:
        raise ValueError("graph node order must match the pruning result")
    vertex_ids = {vertex: index for index, vertex in enumerate(vertices)}
    graph_edges: tuple[tuple[int, int], ...] = tuple(
        sorted(
            (
                min(vertex_ids[first], vertex_ids[second]),
                max(vertex_ids[first], vertex_ids[second]),
            )
            for first, second in graph.edges()
        )
    )
    vertex_set = set(vertices)
    candidates = pruning_result.surviving_completes
    candidate_masks: list[int] = []
    for candidate in candidates:
        if not candidate or not candidate <= vertex_set:
            raise ValueError("pruning result contains an invalid candidate")
        if any(
            not graph.has_edge(first, second)
            for first, second in combinations(candidate, 2)
        ):
            raise ValueError("pruning result contains a non-complete candidate")
        candidate_masks.append(_vertex_mask(candidate, vertex_ids))
    if candidate_masks != sorted(set(candidate_masks)):
        raise ValueError("pruning result candidates are not uniquely mask-ordered")

    if pruning_result.status is HellyCoverStatus.INCONCLUSIVE_BUDGET:
        return SelectedHellyCoverSearchResult(
            status=SelectedHellyCoverStatus.INCONCLUSIVE_BUDGET,
            selected_family=None,
            verification=None,
            states_explored=0,
            max_states=max_states,
            max_depth=max_depth,
            branch_policy=_BRANCH_POLICY,
            graph_edges=graph_edges,
            pruning_result=pruning_result,
            diagnostics="candidate pruning stopped before completing its universe",
        )

    edge_masks = tuple((1 << first) | (1 << second) for first, second in graph_edges)
    candidate_edge_masks = tuple(
        sum(
            1 << edge_index
            for edge_index, edge_mask in enumerate(edge_masks)
            if candidate_mask & edge_mask == edge_mask
        )
        for candidate_mask in candidate_masks
    )
    full_edge_mask = (1 << len(graph_edges)) - 1
    required_masks = tuple(
        sorted(
            _vertex_mask(facet, vertex_ids) for facet in pruning_result.required_facets
        )
    )
    candidate_index_by_mask = {
        candidate_mask: index for index, candidate_mask in enumerate(candidate_masks)
    }

    explored: set[tuple[int, ...]] = set()
    selected_solution: tuple[int, ...] | None = None
    budget_hit = False

    def candidate_masks_for(selected: tuple[int, ...]) -> set[int]:
        return {candidate_masks[index] for index in selected}

    def visit(selected: tuple[int, ...], covered_edges: int) -> None:
        nonlocal budget_hit, selected_solution
        if budget_hit or selected_solution is not None or selected in explored:
            return
        if max_states is not None and len(explored) >= max_states:
            budget_hit = True
            return
        explored.add(selected)
        if not is_helly_family(candidates[index] for index in selected):
            return

        if covered_edges == full_edge_mask:
            missing_required = tuple(
                mask
                for mask in required_masks
                if mask not in candidate_masks_for(selected)
            )
            if missing_required:
                required_candidate = candidate_index_by_mask.get(missing_required[0])
                if required_candidate is None:
                    return
                if max_depth is not None and len(selected) >= max_depth:
                    budget_hit = True
                    return
                visit(
                    tuple(sorted((*selected, required_candidate))),
                    covered_edges | candidate_edge_masks[required_candidate],
                )
                return

            family = tuple(candidates[index] for index in selected)
            verification = verify_helly_edge_cover(graph, family)
            if verification.status is not HellyCoverVerificationStatus.VERIFIED:
                raise AssertionError(
                    "complete Helly selection failed independent verification"
                )
            selected_solution = selected
            return

        if max_depth is not None and len(selected) >= max_depth:
            budget_hit = True
            return

        uncovered = [
            edge_index
            for edge_index in range(len(graph_edges))
            if not covered_edges & (1 << edge_index)
        ]
        supports_by_edge = {
            edge_index: tuple(
                candidate_index
                for candidate_index, edge_coverage in enumerate(candidate_edge_masks)
                if candidate_index not in selected and edge_coverage & (1 << edge_index)
            )
            for edge_index in uncovered
        }
        if any(not supports for supports in supports_by_edge.values()):
            return
        branch_edge = min(
            uncovered,
            key=lambda edge_index: (len(supports_by_edge[edge_index]), edge_index),
        )
        for candidate_index in supports_by_edge[branch_edge]:
            visit(
                tuple(sorted((*selected, candidate_index))),
                covered_edges | candidate_edge_masks[candidate_index],
            )
            if budget_hit or selected_solution is not None:
                return

    visit((), 0)
    states_explored = len(explored)
    if selected_solution is not None:
        family = tuple(candidates[index] for index in selected_solution)
        verification = verify_helly_edge_cover(graph, family)
        return SelectedHellyCoverSearchResult(
            status=SelectedHellyCoverStatus.FOUND_VERIFIED_COVER,
            selected_family=family,
            verification=verification,
            states_explored=states_explored,
            max_states=max_states,
            max_depth=max_depth,
            branch_policy=_BRANCH_POLICY,
            graph_edges=graph_edges,
            pruning_result=pruning_result,
            diagnostics="selected family independently verified as a Helly edge cover",
        )
    if budget_hit:
        return SelectedHellyCoverSearchResult(
            status=SelectedHellyCoverStatus.INCONCLUSIVE_BUDGET,
            selected_family=None,
            verification=None,
            states_explored=states_explored,
            max_states=max_states,
            max_depth=max_depth,
            branch_policy=_BRANCH_POLICY,
            graph_edges=graph_edges,
            pruning_result=pruning_result,
            diagnostics="exact search stopped at a configured state or depth limit",
        )
    return SelectedHellyCoverSearchResult(
        status=SelectedHellyCoverStatus.NO_COVER_CERTIFIED,
        selected_family=None,
        verification=None,
        states_explored=states_explored,
        max_states=max_states,
        max_depth=max_depth,
        branch_policy=_BRANCH_POLICY,
        graph_edges=graph_edges,
        pruning_result=pruning_result,
        diagnostics=(
            "all branches were exhausted in the recorded surviving candidate "
            "universe and normalization scope"
        ),
    )

"""Sound local propagation for Helly edge covers by graph completes.

The pruning operation is a necessary-condition filter, not a cover solver.
Surviving candidates do not imply that a Helly edge cover exists.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any

import networkx as nx


class HellyCoverStatus(Enum):
    """Outcome of candidate-universe propagation."""

    PRUNED_EXHAUSTIVE = "PRUNED_EXHAUSTIVE"
    NO_COVER_CERTIFIED = "NO_COVER_CERTIFIED"
    INCONCLUSIVE_BUDGET = "INCONCLUSIVE_BUDGET"


class HellyCoverVerificationStatus(Enum):
    """Outcome of independently checking a selected family."""

    VERIFIED = "VERIFIED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class HellyCoverPruningResult:
    """Candidate pruning result; ``PRUNED_EXHAUSTIVE`` is not a cover proof."""

    status: HellyCoverStatus
    surviving_completes: tuple[frozenset[Hashable], ...]
    initial_candidate_count: int
    rounds_completed: int
    removed_per_round: tuple[int, ...]
    uncovered_edge: tuple[Hashable, Hashable] | None
    required_facet_missing: bool
    diagnostics: str
    support_checks: int
    normalization_removed: int
    index_to_label: tuple[Hashable, ...]
    candidates_examined: int
    required_facets: tuple[frozenset[Hashable], ...]
    private_edges: tuple[tuple[frozenset[Hashable], frozenset[Hashable]], ...]
    fixed_gates: tuple[tuple[frozenset[Hashable], Hashable, Hashable], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready data using internal vertex indices."""
        label_to_index = {
            label: index for index, label in enumerate(self.index_to_label)
        }
        surviving = [
            sorted(label_to_index[label] for label in candidate)
            for candidate in self.surviving_completes
        ]
        return {
            "status": self.status.value,
            "surviving_completes": surviving,
            "initial_candidate_count": self.initial_candidate_count,
            "rounds_completed": self.rounds_completed,
            "removed_per_round": list(self.removed_per_round),
            "uncovered_edge": (
                None
                if self.uncovered_edge is None
                else sorted(label_to_index[label] for label in self.uncovered_edge)
            ),
            "required_facet_missing": self.required_facet_missing,
            "diagnostics": self.diagnostics,
            "support_checks": self.support_checks,
            "normalization_removed": self.normalization_removed,
            "required_facets": [
                sorted(label_to_index[label] for label in facet)
                for facet in self.required_facets
            ],
            "private_edges": [
                {
                    "facet": sorted(label_to_index[label] for label in facet),
                    "edge": sorted(label_to_index[label] for label in edge),
                }
                for facet, edge in self.private_edges
            ],
            "fixed_gates": [
                {
                    "facet": sorted(label_to_index[label] for label in facet),
                    "point": label_to_index[point],
                    "gate": label_to_index[gate],
                }
                for facet, point, gate in self.fixed_gates
            ],
            "index_to_label": [
                {
                    "index": index,
                    "type": f"{type(label).__module__}.{type(label).__qualname__}",
                    "repr": repr(label),
                }
                for index, label in enumerate(self.index_to_label)
            ],
            "candidates_examined": self.candidates_examined,
        }

    def to_json(self) -> str:
        """Serialize this result with stable key and vertex ordering."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class HellyEdgeCoverVerification:
    """Independent result for a selected family of completes."""

    status: HellyCoverVerificationStatus
    uncovered_edge: tuple[Hashable, Hashable] | None
    non_helly_subfamily: tuple[frozenset[Hashable], ...] | None
    diagnostics: str


def _validate_simple_graph(graph: nx.Graph) -> tuple[Hashable, ...]:
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise ValueError("graph must be a simple undirected NetworkX graph")
    if nx.number_of_selfloops(graph):
        raise ValueError("graph must not contain self-loops")
    return tuple(graph.nodes())


def _frozen_family(
    family: Iterable[Iterable[Hashable]], *, allow_empty_member: bool
) -> tuple[frozenset[Hashable], ...]:
    result: list[frozenset[Hashable]] = []
    seen: set[frozenset[Hashable]] = set()
    for member in family:
        frozen = frozenset(member)
        if not allow_empty_member and not frozen:
            raise ValueError("family members must be nonempty")
        if frozen not in seen:
            seen.add(frozen)
            result.append(frozen)
    return tuple(result)


def _is_complete(graph: nx.Graph, member: frozenset[Hashable]) -> bool:
    return all(
        graph.has_edge(first, second) for first, second in combinations(member, 2)
    )


def _find_non_helly_subfamily(
    members: tuple[frozenset[Hashable], ...],
    *,
    point_order: Iterable[Hashable] = (),
) -> tuple[frozenset[Hashable], ...] | None:
    """Return a bounded gate obstruction, preserving input member order."""
    ordered_points: list[Hashable] = []
    seen_points: set[Hashable] = set()
    for point in point_order:
        if point not in seen_points:
            seen_points.add(point)
            ordered_points.append(point)
    for member in members:
        for point in member:
            if point not in seen_points:
                seen_points.add(point)
                ordered_points.append(point)

    incidence: dict[Hashable, list[int]] = defaultdict(list)
    for member_index, member in enumerate(members):
        for point in member:
            incidence[point].append(member_index)

    for member_index, member in enumerate(members):
        for point in ordered_points:
            if point in member:
                continue
            touching = tuple(
                other_index
                for other_index in incidence[point]
                if member.intersection(members[other_index])
            )
            if not touching:
                continue

            common = set(member)
            for other_index in touching:
                common.intersection_update(members[other_index])
                if not common:
                    break
            if common:
                continue

            witness_indices = {member_index}
            for gate in ordered_points:
                if gate in member and not any(
                    gate not in members[other_index] for other_index in touching
                ):
                    raise AssertionError("failed gate has no excluding witness")
                if gate in member:
                    witness_indices.add(
                        next(
                            other_index
                            for other_index in touching
                            if gate not in members[other_index]
                        )
                    )
            return tuple(members[index] for index in sorted(witness_indices))
    return None


def is_helly_family(family: Iterable[Iterable[Hashable]]) -> bool:
    """Return whether a finite family of nonempty sets has the Helly property.

    Duplicate sets are treated as one family member, as in ordinary set-family
    semantics. The empty family is Helly.
    """
    members = _frozen_family(family, allow_empty_member=False)
    return _find_non_helly_subfamily(members) is None


def satisfies_helly_gate_condition(
    family: Iterable[Iterable[Hashable]],
) -> bool:
    """Check the gate characterization of Hellyness for finite set families."""
    members = _frozen_family(family, allow_empty_member=False)
    return _find_non_helly_subfamily(members) is None


def verify_helly_edge_cover(
    graph: nx.Graph, family: Iterable[Iterable[Hashable]]
) -> HellyEdgeCoverVerification:
    """Independently verify completeness, edge coverage, and Hellyness.

    Duplicate family members are ignored. A failed result includes either an
    uncovered edge or a pairwise-intersecting subfamily with empty intersection.
    """
    vertices = _validate_simple_graph(graph)
    members = _frozen_family(family, allow_empty_member=False)
    vertex_set = set(vertices)
    for member in members:
        if not member <= vertex_set:
            raise ValueError("family member contains a vertex absent from graph")
        if not _is_complete(graph, member):
            raise ValueError("family member is not a complete of graph")

    vertex_ids = {vertex: index for index, vertex in enumerate(vertices)}
    for first, second in sorted(
        graph.edges(), key=lambda edge: tuple(sorted(vertex_ids[v] for v in edge))
    ):
        if not any(first in member and second in member for member in members):
            return HellyEdgeCoverVerification(
                HellyCoverVerificationStatus.INVALID,
                (first, second),
                None,
                "the selected family leaves an edge uncovered",
            )

    ordered_members = tuple(
        sorted(members, key=lambda member: tuple(sorted(vertex_ids[v] for v in member)))
    )
    obstruction = _find_non_helly_subfamily(ordered_members, point_order=vertices)
    if obstruction is not None:
        return HellyEdgeCoverVerification(
            HellyCoverVerificationStatus.INVALID,
            None,
            obstruction,
            "the selected family has a pairwise-intersecting subfamily "
            "with empty intersection",
        )
    return HellyEdgeCoverVerification(
        HellyCoverVerificationStatus.VERIFIED,
        None,
        None,
        "the family is a Helly edge cover",
    )


def _bits(mask: int) -> Iterable[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _mask_to_complete(mask: int, vertices: tuple[Hashable, ...]) -> frozenset[Hashable]:
    return frozenset(vertices[index] for index in _bits(mask))


def _vertex_mask(
    vertices: Iterable[Hashable], vertex_ids: Mapping[Hashable, int]
) -> int:
    return sum(1 << vertex_ids[vertex] for vertex in vertices)


def prune_helly_cover_candidates(
    graph: nx.Graph,
    candidates: Iterable[Iterable[Hashable]],
    *,
    required_facets: Iterable[Iterable[Hashable]] = (),
    private_edges: Mapping[frozenset[Hashable], frozenset[Hashable]] | None = None,
    fixed_gates: Mapping[tuple[frozenset[Hashable], Hashable], Hashable] | None = None,
    max_candidates: int | None = None,
    max_rounds: int | None = None,
    max_support_checks: int | None = None,
) -> HellyCoverPruningResult:
    """Soundly prune a finite universe of candidate completes.

    The local gate condition is necessary for any Helly edge cover contained in
    the supplied universe. A fixed point is only a surviving candidate
    universe, not a selected cover. ``NO_COVER_CERTIFIED`` is returned only
    when sound propagation removes a required facet or leaves an edge
    uncovered. Work limits return ``INCONCLUSIVE_BUDGET``.

    Labels receive internal indices in graph iteration order, which is recorded
    in ``index_to_label`` and used by :meth:`HellyCoverPruningResult.to_dict`.
    ``private_edges`` and ``fixed_gates`` are explicit normalization
    assumptions; their designated facets must also be required facets.
    """
    vertices = _validate_simple_graph(graph)
    vertex_ids = {vertex: index for index, vertex in enumerate(vertices)}
    vertex_set = set(vertices)
    if max_candidates is not None and max_candidates < 0:
        raise ValueError("max_candidates must be nonnegative or None")
    if max_rounds is not None and max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative or None")
    if max_support_checks is not None and max_support_checks < 0:
        raise ValueError("max_support_checks must be nonnegative or None")

    required = _frozen_family(required_facets, allow_empty_member=False)
    for facet in required:
        if not facet <= vertex_set or not _is_complete(graph, facet):
            raise ValueError("every required facet must be a complete of graph")
    required_masks = {
        sum(1 << vertex_ids[vertex] for vertex in facet) for facet in required
    }

    private_masks: dict[int, int] = {}
    private_assumptions = []
    for raw_facet, raw_edge in (private_edges or {}).items():
        facet = frozenset(raw_facet)
        edge = frozenset(raw_edge)
        if facet not in required:
            raise ValueError("a private-edge facet must also be a required facet")
        if not facet <= vertex_set or not _is_complete(graph, facet):
            raise ValueError("private-edge facet must be a complete of graph")
        if len(edge) != 2 or not edge <= facet or not _is_complete(graph, edge):
            raise ValueError("private edge must be a graph edge contained in its facet")
        facet_mask = _vertex_mask(facet, vertex_ids)
        edge_mask = _vertex_mask(edge, vertex_ids)
        private_masks[facet_mask] = edge_mask
        private_assumptions.append((facet_mask, facet, edge))
    private_assumptions.sort(
        key=lambda item: (item[0], _vertex_mask(item[2], vertex_ids))
    )

    normalized_gates: list[tuple[int, int, int]] = []
    fixed_gate_assumptions = []
    for (raw_facet, point), raw_gate in (fixed_gates or {}).items():
        facet = frozenset(raw_facet)
        if facet not in required:
            raise ValueError("a fixed gate must be assigned around a required facet")
        if point not in vertex_set or point in facet:
            raise ValueError(
                "fixed-gate point must be a graph vertex outside its facet"
            )
        if raw_gate not in facet:
            raise ValueError("fixed gate must belong to its required facet")
        facet_mask = _vertex_mask(facet, vertex_ids)
        point_id = vertex_ids[point]
        gate_id = vertex_ids[raw_gate]
        normalized_gates.append((facet_mask, point_id, gate_id))
        fixed_gate_assumptions.append((facet_mask, facet, point, raw_gate))
    fixed_gate_assumptions.sort(
        key=lambda item: (item[0], vertex_ids[item[2]], vertex_ids[item[3]])
    )

    candidate_masks: list[int] = []
    candidate_set: set[int] = set()
    candidate_count = 0
    candidate_budget_hit = False
    for candidate in candidates:
        frozen = frozenset(candidate)
        if not frozen:
            raise ValueError("candidate completes must be nonempty")
        if not frozen <= vertex_set:
            raise ValueError("candidate contains a vertex absent from graph")
        if not _is_complete(graph, frozen):
            raise ValueError("candidate is not a complete of graph")
        mask = sum(1 << vertex_ids[vertex] for vertex in frozen)
        if mask in candidate_set:
            raise ValueError("candidate universe must not contain duplicate completes")
        candidate_set.add(mask)
        candidate_masks.append(mask)
        candidate_count += 1
        if max_candidates is not None and candidate_count > max_candidates:
            candidate_budget_hit = True
            break
    candidate_masks.sort()

    def make_result(
        status: HellyCoverStatus,
        active: list[int],
        rounds: int,
        removed: list[int],
        uncovered: tuple[Hashable, Hashable] | None,
        required_missing: bool,
        diagnostics: str,
        support_checks: int,
        normalization_removed: int,
    ) -> HellyCoverPruningResult:
        return HellyCoverPruningResult(
            status=status,
            surviving_completes=tuple(
                _mask_to_complete(mask, vertices) for mask in active
            ),
            initial_candidate_count=candidate_count,
            rounds_completed=rounds,
            removed_per_round=tuple(removed),
            uncovered_edge=uncovered,
            required_facet_missing=required_missing,
            diagnostics=diagnostics,
            support_checks=support_checks,
            normalization_removed=normalization_removed,
            index_to_label=vertices,
            candidates_examined=candidate_count,
            required_facets=tuple(
                sorted(required, key=lambda facet: _vertex_mask(facet, vertex_ids))
            ),
            private_edges=tuple(
                (facet, edge) for _, facet, edge in private_assumptions
            ),
            fixed_gates=tuple(
                (facet, point, gate) for _, facet, point, gate in fixed_gate_assumptions
            ),
        )

    if candidate_budget_hit:
        return make_result(
            HellyCoverStatus.INCONCLUSIVE_BUDGET,
            candidate_masks,
            0,
            [],
            None,
            False,
            f"candidate limit max_candidates={max_candidates} was exceeded "
            "before the universe was fully read",
            0,
            0,
        )

    if not required_masks <= candidate_set:
        raise ValueError("every required facet must appear in the candidate universe")

    active: list[int] = []
    for mask in candidate_masks:
        violates_private_edge = any(
            mask != facet_mask and mask & edge_mask == edge_mask
            for facet_mask, edge_mask in private_masks.items()
        )
        violates_gate = any(
            mask & (1 << point) and mask & facet_mask and not mask & (1 << gate)
            for facet_mask, point, gate in normalized_gates
        )
        if not violates_private_edge and not violates_gate:
            active.append(mask)
    normalization_removed = len(candidate_masks) - len(active)
    removed_per_round: list[int] = []
    support_checks = 0
    rounds_completed = 0

    adjacency_masks = [0] * len(vertices)
    for first, second in graph.edges():
        first_id = vertex_ids[first]
        second_id = vertex_ids[second]
        adjacency_masks[first_id] |= 1 << second_id
        adjacency_masks[second_id] |= 1 << first_id

    while True:
        if max_rounds is not None and rounds_completed >= max_rounds:
            return make_result(
                HellyCoverStatus.INCONCLUSIVE_BUDGET,
                active,
                rounds_completed,
                removed_per_round,
                None,
                False,
                f"round limit max_rounds={max_rounds} was reached before "
                "fixed-point verification",
                support_checks,
                normalization_removed,
            )

        supported_vertices: dict[int, int] = {}
        for candidate_mask in active:
            candidate_bits = tuple(_bits(candidate_mask))
            for point in candidate_bits:
                for gate in candidate_bits:
                    if gate != point:
                        key = point * len(vertices) + gate
                        supported_vertices[key] = (
                            supported_vertices.get(key, 0) | candidate_mask
                        )

        removed_this_round: list[int] = []
        budget_hit = False
        for facet_mask in active:
            facet_bits = tuple(_bits(facet_mask))
            facet_removed = False
            for point in range(len(vertices)):
                if facet_mask & (1 << point):
                    continue
                boundary = adjacency_masks[point] & facet_mask
                if not boundary:
                    continue
                gate_found = False
                for gate in facet_bits:
                    if (
                        max_support_checks is not None
                        and support_checks >= max_support_checks
                    ):
                        budget_hit = True
                        break
                    support_checks += 1
                    support = supported_vertices.get(point * len(vertices) + gate, 0)
                    if boundary & ~support == 0:
                        gate_found = True
                        break
                if budget_hit:
                    break
                if not gate_found:
                    facet_removed = True
                    break
            if budget_hit:
                break
            if facet_removed:
                removed_this_round.append(facet_mask)

        if budget_hit:
            return make_result(
                HellyCoverStatus.INCONCLUSIVE_BUDGET,
                active,
                rounds_completed,
                removed_per_round,
                None,
                False,
                f"support-check limit max_support_checks={max_support_checks} "
                "was reached before completing a propagation round",
                support_checks,
                normalization_removed,
            )

        removed_set = set(removed_this_round)
        active = [mask for mask in active if mask not in removed_set]
        removed_per_round.append(len(removed_this_round))
        rounds_completed += 1
        if not removed_this_round:
            break

    required_missing = not required_masks <= set(active)
    for first, second in sorted(
        graph.edges(), key=lambda edge: tuple(sorted(vertex_ids[v] for v in edge))
    ):
        edge_mask = (1 << vertex_ids[first]) | (1 << vertex_ids[second])
        if not any(mask & edge_mask == edge_mask for mask in active):
            return make_result(
                HellyCoverStatus.NO_COVER_CERTIFIED,
                active,
                rounds_completed,
                removed_per_round,
                (first, second),
                required_missing,
                "an edge is uncovered by every surviving candidate",
                support_checks,
                normalization_removed,
            )

    if required_missing:
        return make_result(
            HellyCoverStatus.NO_COVER_CERTIFIED,
            active,
            rounds_completed,
            removed_per_round,
            None,
            True,
            "sound pruning removed a required facet",
            support_checks,
            normalization_removed,
        )

    return make_result(
        HellyCoverStatus.PRUNED_EXHAUSTIVE,
        active,
        rounds_completed,
        removed_per_round,
        None,
        False,
        "candidate universe reached a sound propagation fixed point; "
        "no cover existence claim is made",
        support_checks,
        normalization_removed,
    )

"""Generate and independently check normalized O8 Helly-cover certificates."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import networkx as nx
from pycliques.helly_cover import (
    HellyCoverStatus,
    prune_helly_cover_candidates,
)

SCHEMA_VERSION = 1
F0 = frozenset({0, 2, 4, 6, 8})
PRIVATE_EDGE = frozenset({6, 8})
FOUR_UNIFORM_PRIVATE_EDGE = frozenset({4, 6})
BASE_GATE_DOMAINS = (
    (1, (2, 4)),
    (3, (0, 4)),
    (5, (0, 2)),
    (7, (0, 2, 4, 8)),
    (9, (0, 2, 4, 6)),
)
UNUSED_GATE_VALUES = (0, 2, 4)
UNUSED_PAIR_TYPES = tuple(
    itertools.combinations_with_replacement(UNUSED_GATE_VALUES, 2)
)
VERTEX_COUNT = 16
O8_EDGES = 112
PRE_NORMALIZATION_CANDIDATE_COUNT = 2912
PRIVATE_EDGE_CANDIDATE_COUNT = 2693


def _octahedral_graph(pair_count: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(2 * pair_count))
    graph.add_edges_from(
        (first, second)
        for first in range(2 * pair_count)
        for second in range(first + 1, 2 * pair_count)
        if first // 2 != second // 2
    )
    return graph


def _rank_candidates(max_rank: int) -> tuple[frozenset[int], ...]:
    graph = _octahedral_graph(8)
    candidates = []
    for rank in range(4, max_rank + 1):
        for pair_indices in itertools.combinations(range(8), rank):
            for choices in itertools.product((0, 1), repeat=rank):
                candidate = frozenset(
                    2 * pair_index + choice
                    for pair_index, choice in zip(pair_indices, choices)
                )
                if all(
                    graph.has_edge(first, second)
                    for first, second in itertools.combinations(candidate, 2)
                ):
                    candidates.append(candidate)
    return tuple(sorted(candidates, key=lambda member: _set_mask(member)))


def _normalizing_vertex_map(
    facet: frozenset[int],
    private_edge: frozenset[int],
) -> tuple[int, ...]:
    rank = len(facet)
    if rank not in (4, 5) or len(private_edge) != 2 or not private_edge <= facet:
        raise ValueError("normalization requires a rank-four/five facet and its edge")
    selected_by_pair = {vertex // 2: vertex for vertex in facet}
    edge_pairs = sorted(vertex // 2 for vertex in private_edge)
    if len(selected_by_pair) != rank or len(set(edge_pairs)) != 2:
        raise ValueError("facet and private edge must use distinct antipodal pairs")
    other_pairs = sorted(set(selected_by_pair) - set(edge_pairs))
    target_by_source: dict[int, int] = {}
    for source, target in zip(other_pairs, range(rank - 2)):
        target_by_source[source] = target
    for source, target in zip(edge_pairs, range(rank - 2, rank)):
        target_by_source[source] = target
    unused_sources = sorted(set(range(8)) - set(selected_by_pair))
    for source, target in zip(unused_sources, range(rank, 8)):
        target_by_source[source] = target
    swaps = {pair: vertex % 2 for pair, vertex in selected_by_pair.items()}
    return tuple(
        2 * target_by_source[vertex // 2] + ((vertex % 2) ^ swaps.get(vertex // 2, 0))
        for vertex in range(VERTEX_COUNT)
    )


def _verify_private_edge_normalization(max_rank: int) -> dict[str, Any]:
    candidates = tuple(
        candidate
        for candidate in _rank_candidates(max_rank)
        if len(candidate) == max_rank
    )
    checks = 0
    for facet in candidates:
        rank = len(facet)
        target_facet = frozenset(2 * pair for pair in range(rank))
        target_edge = frozenset((2 * (rank - 2), 2 * (rank - 1)))
        for edge in itertools.combinations(sorted(facet), 2):
            vertex_map = _normalizing_vertex_map(facet, frozenset(edge))
            if {vertex_map[vertex] for vertex in facet} != target_facet:
                raise AssertionError("private-facet normalization failed")
            if {vertex_map[vertex] for vertex in edge} != target_edge:
                raise AssertionError("private-edge normalization failed")
            if len(set(vertex_map)) != VERTEX_COUNT:
                raise AssertionError("normalization map is not a vertex permutation")
            for first, second in itertools.combinations(range(VERTEX_COUNT), 2):
                if (first // 2 != second // 2) != (
                    vertex_map[first] // 2 != vertex_map[second] // 2
                ):
                    raise AssertionError("normalization map is not an O8 automorphism")
            checks += 1
    return {
        "automorphism": (
            "Permute antipodal pairs and independently swap endpoints; send the "
            "selected facet vertices to the even endpoints and the private-edge "
            "pairs to the final two selected pair positions."
        ),
        "facet_edge_cases_checked": checks,
        "all_cases_map_to_normal_form": True,
    }


def _set_mask(vertices: frozenset[int]) -> int:
    return sum(1 << vertex for vertex in vertices)


def _base_assignments() -> tuple[tuple[int, ...], ...]:
    domains = tuple(domain for _, domain in BASE_GATE_DOMAINS)
    return tuple(itertools.product(*domains))


def _unused_pair_assignments() -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(itertools.combinations_with_replacement(UNUSED_PAIR_TYPES, 3))


def _representative_gates() -> tuple[dict[int, int], ...]:
    representatives: list[dict[int, int]] = []
    outside_pairs = _unused_pair_assignments()
    for base_values in _base_assignments():
        for pair_types in outside_pairs:
            gates = dict(zip((vertex for vertex, _ in BASE_GATE_DOMAINS), base_values))
            for pair_index, (first_gate, second_gate) in enumerate(pair_types, 5):
                gates[2 * pair_index] = first_gate
                gates[2 * pair_index + 1] = second_gate
            representatives.append(gates)
    return tuple(representatives)


def _canonical_gate_key(gates: dict[int, int]) -> tuple[int, ...]:
    base = tuple(gates[vertex] for vertex, _ in BASE_GATE_DOMAINS)
    pair_types = tuple(
        sorted(
            tuple(sorted((gates[2 * pair], gates[2 * pair + 1])))
            for pair in range(5, 8)
        )
    )
    return base + tuple(value for pair_type in pair_types for value in pair_type)


def _literal_gate_assignments():
    base_vertices = tuple(vertex for vertex, _ in BASE_GATE_DOMAINS)
    base_domains = tuple(domain for _, domain in BASE_GATE_DOMAINS)
    for base_values in itertools.product(*base_domains):
        for pair_values in itertools.product(UNUSED_GATE_VALUES, repeat=6):
            gates = dict(zip(base_vertices, base_values))
            for vertex, value in zip(range(10, 16), pair_values):
                gates[vertex] = value
            yield gates


def _verify_representative_coverage(
    representatives: tuple[dict[int, int], ...],
) -> dict[str, Any]:
    representative_keys = {_canonical_gate_key(gates) for gates in representatives}
    literal_keys = {_canonical_gate_key(gates) for gates in _literal_gate_assignments()}
    if len(representatives) != 7168 or len(representative_keys) != 7168:
        raise AssertionError("representative generator did not produce 7168 cases")
    if literal_keys != representative_keys:
        raise AssertionError("representatives do not cover all literal assignments")
    unused_pair_permutations = tuple(itertools.permutations(range(5, 8)))
    endpoint_swaps = tuple(itertools.product((0, 1), repeat=3))
    unused_pair_actions = tuple(
        (pair_permutation, swaps)
        for pair_permutation in unused_pair_permutations
        for swaps in endpoint_swaps
    )
    for pair_permutation, swaps in unused_pair_actions:
        for first, second in itertools.combinations(range(VERTEX_COUNT), 2):
            first_pair = first // 2
            second_pair = second // 2
            mapped_first = first
            mapped_second = second
            if first_pair >= 5:
                mapped_first = 2 * pair_permutation[first_pair - 5] + (
                    (first % 2) ^ swaps[first_pair - 5]
                )
            if second_pair >= 5:
                mapped_second = 2 * pair_permutation[second_pair - 5] + (
                    (second % 2) ^ swaps[second_pair - 5]
                )
            if (first_pair != second_pair) != (mapped_first // 2 != mapped_second // 2):
                raise AssertionError("unused-pair action is not an O8 automorphism")
    for gates in representatives:
        expected_key = _canonical_gate_key(gates)
        for pair_permutation, swaps in unused_pair_actions:
            transformed = {}
            for outside, gate in gates.items():
                if outside < 10:
                    transformed[outside] = gate
                    continue
                pair, side = divmod(outside, 2)
                target_pair = pair_permutation[pair - 5]
                target_side = side ^ swaps[pair - 5]
                transformed[2 * target_pair + target_side] = gate
            if _canonical_gate_key(transformed) != expected_key:
                raise AssertionError(
                    "unused-pair symmetry does not preserve gate representatives"
                )
    transformed_key_set = {
        _canonical_gate_key(_transform_gate_map(gates, permutation))
        for gates in representatives
        for permutation in itertools.permutations(range(3))
    }
    if not transformed_key_set <= representative_keys:
        raise AssertionError("selected-pair symmetry leaves representative space")
    return {
        "literal_assignment_count": 93_312,
        "representative_count": len(representatives),
        "canonicalization": (
            "Sort gate pairs within each unused antipodal pair, then sort the "
            "three pair types; retain the five specified gate values."
        ),
        "coverage_verified_by_exhaustive_canonicalization": True,
    }


def _case_gates(
    gates: dict[int, int], required_facet: frozenset[int] = F0
) -> dict[tuple[frozenset[Hashable], Hashable], Hashable]:
    return {(required_facet, outside): gate for outside, gate in gates.items()}


def _digest_masks(candidates: tuple[frozenset[int], ...]) -> str:
    encoded = "".join(f"{_set_mask(candidate)}\n" for candidate in candidates)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _run_engine_case(
    graph: nx.Graph,
    universe: tuple[frozenset[int], ...],
    gates: dict[int, int],
    required_facet: frozenset[int] = F0,
    private_edge: frozenset[int] = PRIVATE_EDGE,
):
    return prune_helly_cover_candidates(
        graph,
        universe,
        required_facets=(required_facet,),
        private_edges={required_facet: private_edge},
        fixed_gates=_case_gates(gates, required_facet),
    )


def _set_based_case(
    graph: nx.Graph,
    universe: tuple[frozenset[int], ...],
    gates: dict[int, int],
    required_facet: frozenset[int] = F0,
    private_edge: frozenset[int] = PRIVATE_EDGE,
) -> dict[str, Any]:
    """Independent set-based replay of normalization and local propagation."""
    active = set(universe)
    active = {
        candidate
        for candidate in active
        if candidate == required_facet or not private_edge <= candidate
    }
    for outside, gate in gates.items():
        active = {
            candidate
            for candidate in active
            if not (
                outside in candidate
                and candidate.intersection(required_facet)
                and gate not in candidate
            )
        }
    normalization_removed = len(universe) - len(active)
    removed_per_round: list[int] = []
    support_checks = 0
    while True:
        supported: dict[tuple[int, int], set[int]] = {}
        for candidate in active:
            for point in candidate:
                for gate in candidate:
                    if gate != point:
                        supported.setdefault((point, gate), set()).update(candidate)
        removed: set[frozenset[int]] = set()
        for facet in sorted(active, key=_set_mask):
            for point in sorted(set(graph) - facet):
                boundary = {
                    neighbor for neighbor in graph.neighbors(point) if neighbor in facet
                }
                if not boundary:
                    continue
                has_gate = False
                for gate in sorted(facet):
                    support_checks += 1
                    if boundary <= supported.get((point, gate), set()):
                        has_gate = True
                        break
                if not has_gate:
                    removed.add(facet)
                    break
        active.difference_update(removed)
        removed_per_round.append(len(removed))
        if not removed:
            break

    required_missing = required_facet not in active
    uncovered_edge = next(
        (
            tuple(sorted(edge))
            for edge in graph.edges()
            if not any(set(edge) <= candidate for candidate in active)
        ),
        None,
    )
    status = (
        "NO_COVER_CERTIFIED"
        if required_missing or uncovered_edge is not None
        else "PRUNED_EXHAUSTIVE"
    )
    return {
        "survivors": tuple(sorted(active, key=_set_mask)),
        "normalization_removed": normalization_removed,
        "removed_per_round": tuple(removed_per_round),
        "support_checks": support_checks,
        "uncovered_edge": uncovered_edge,
        "required_facet_missing": required_missing,
        "status": status,
    }


def _vertex_under_pair_permutation(vertex: int, permutation: tuple[int, ...]) -> int:
    if vertex not in range(6):
        return vertex
    pair, side = divmod(vertex, 2)
    return 2 * permutation[pair] + side


def _transform_gate_map(
    gates: dict[int, int], permutation: tuple[int, ...]
) -> dict[int, int]:
    return {
        _vertex_under_pair_permutation(
            outside, permutation
        ): _vertex_under_pair_permutation(gate, permutation)
        for outside, gate in gates.items()
    }


def _gate_orbit_key(gates: dict[int, int]) -> tuple[int, ...]:
    return min(
        _canonical_gate_key(_transform_gate_map(gates, permutation))
        for permutation in itertools.permutations(range(3))
    )


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def generate_certificate(max_rank: int = 5) -> dict[str, Any]:
    if max_rank != 5:
        raise ValueError("the certified normalization currently requires max_rank=5")
    graph = _octahedral_graph(8)
    universe = _rank_candidates(max_rank)
    if len(universe) != PRE_NORMALIZATION_CANDIDATE_COUNT:
        raise AssertionError(f"unexpected candidate count: {len(universe)}")
    normalized_count = sum(
        candidate == F0 or not PRIVATE_EDGE <= candidate for candidate in universe
    )
    if normalized_count != PRIVATE_EDGE_CANDIDATE_COUNT:
        raise AssertionError(
            f"unexpected private-edge candidate count: {normalized_count}"
        )

    representatives = _representative_gates()
    symmetry_coverage = _verify_representative_coverage(representatives)
    cases = []
    for case_id, gates in enumerate(representatives):
        result = _run_engine_case(graph, universe, gates)
        if result.status is HellyCoverStatus.INCONCLUSIVE_BUDGET:
            raise RuntimeError(f"case {case_id} unexpectedly hit a work budget")
        if (
            result.status is HellyCoverStatus.NO_COVER_CERTIFIED
            and result.uncovered_edge is None
        ):
            raise RuntimeError(
                f"case {case_id} lacks the required uncovered-edge certificate"
            )
        survivors = tuple(sorted(result.surviving_completes, key=_set_mask))
        case = {
            "case_id": case_id,
            "gates": [[outside, gates[outside]] for outside in sorted(gates)],
            "initial_candidate_count": len(universe),
            "after_private_edge_count": normalized_count,
            "normalization_removed": result.normalization_removed,
            "removed_per_round": list(result.removed_per_round),
            "rounds_completed": result.rounds_completed,
            "fixed_point": True,
            "status": result.status.value,
            "survivor_count": len(survivors),
            "survivor_masks_sha256": _digest_masks(survivors),
            "uncovered_edge": (
                None if result.uncovered_edge is None else sorted(result.uncovered_edge)
            ),
            "required_facet_missing": result.required_facet_missing,
            "support_checks": result.support_checks,
            "resource_bounds": {
                "max_candidates": None,
                "max_rounds": None,
                "max_support_checks": None,
            },
        }
        cases.append(case)
    groups: dict[tuple[int, ...], list[int]] = {}
    for case in cases:
        if case["status"] == HellyCoverStatus.PRUNED_EXHAUSTIVE.value:
            gate_map = dict(case["gates"])
            groups.setdefault(_gate_orbit_key(gate_map), []).append(case["case_id"])

    graph6 = nx.to_graph6_bytes(graph, nodes=list(range(VERTEX_COUNT)), header=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_repository_revision": _source_revision(),
        "graph6": graph6.decode("ascii").strip(),
        "vertex_ordering": list(range(VERTEX_COUNT)),
        "antipodal_pair_convention": "vertices {2i,2i+1} are nonadjacent",
        "edge_count": O8_EDGES,
        "candidate_ranks": [4, 5],
        "candidate_generation": (
            "Choose 4 or 5 antipodal pairs and one endpoint from each."
        ),
        "initial_candidate_count": len(universe),
        "required_facet": sorted(F0),
        "private_edge": sorted(PRIVATE_EDGE),
        "after_private_edge_count": normalized_count,
        "symmetry_coverage": symmetry_coverage,
        "survivor_orbit_types": [
            {"orbit_key": list(key), "case_ids": sorted(case_ids)}
            for key, case_ids in sorted(groups.items())
        ],
        "case_count": len(cases),
        "cases": cases,
    }


def _four_uniform_gate_domains(
    graph: nx.Graph,
    universe: tuple[frozenset[int], ...],
    required_facet: frozenset[int],
    private_edge: frozenset[int],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    domains = []
    for outside in sorted(set(graph) - required_facet):
        boundary = {
            vertex for vertex in required_facet if graph.has_edge(outside, vertex)
        }
        supported_gates = tuple(
            gate
            for gate in sorted(required_facet)
            if all(
                any(
                    {outside, neighbor, gate} <= candidate
                    for candidate in universe
                    if candidate == required_facet or not private_edge <= candidate
                )
                for neighbor in boundary
            )
        )
        domains.append((outside, supported_gates))
    return tuple(domains)


def _four_uniform_assignments(
    domains: tuple[tuple[int, tuple[int, ...]], ...],
) -> tuple[dict[int, int], ...]:
    return tuple(
        dict(zip((outside for outside, _ in domains), values))
        for values in itertools.product(*(choices for _, choices in domains))
    )


def generate_four_uniform_certificate() -> dict[str, Any]:
    """Exhaustively analyze every locally possible gate assignment at rank 4."""
    graph = _octahedral_graph(8)
    required_facet = frozenset({0, 2, 4, 6})
    private_edge = FOUR_UNIFORM_PRIVATE_EDGE
    universe = _rank_candidates(4)
    if len(universe) != 1120:
        raise AssertionError("unexpected rank-four candidate count")
    domains = _four_uniform_gate_domains(graph, universe, required_facet, private_edge)
    assignments = _four_uniform_assignments(domains)
    normalized_count = sum(
        candidate == required_facet or not private_edge <= candidate
        for candidate in universe
    )
    cases = []
    for case_id, gates in enumerate(assignments):
        result = _run_engine_case(graph, universe, gates, required_facet, private_edge)
        if result.status is HellyCoverStatus.INCONCLUSIVE_BUDGET:
            raise RuntimeError(f"rank-four case {case_id} hit a work budget")
        if result.uncovered_edge is None:
            raise RuntimeError(
                f"rank-four case {case_id} lacks an uncovered-edge certificate"
            )
        survivors = tuple(sorted(result.surviving_completes, key=_set_mask))
        cases.append(
            {
                "case_id": case_id,
                "gates": [[outside, gates[outside]] for outside in sorted(gates)],
                "initial_candidate_count": len(universe),
                "after_private_edge_count": normalized_count,
                "normalization_removed": result.normalization_removed,
                "removed_per_round": list(result.removed_per_round),
                "rounds_completed": result.rounds_completed,
                "fixed_point": True,
                "status": result.status.value,
                "survivor_count": len(survivors),
                "survivor_masks_sha256": _digest_masks(survivors),
                "uncovered_edge": (
                    None
                    if result.uncovered_edge is None
                    else sorted(result.uncovered_edge)
                ),
                "required_facet_missing": result.required_facet_missing,
                "support_checks": result.support_checks,
                "resource_bounds": {
                    "max_candidates": None,
                    "max_rounds": None,
                    "max_support_checks": None,
                },
            }
        )
    graph6 = nx.to_graph6_bytes(graph, nodes=list(range(VERTEX_COUNT)), header=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_kind": "four_uniform_private_edge_baseline",
        "source_repository_revision": _source_revision(),
        "graph6": graph6.decode("ascii").strip(),
        "vertex_ordering": list(range(VERTEX_COUNT)),
        "antipodal_pair_convention": "vertices {2i,2i+1} are nonadjacent",
        "candidate_ranks": [4],
        "candidate_generation": (
            "Choose four antipodal pairs and one endpoint from each."
        ),
        "initial_candidate_count": len(universe),
        "required_facet": sorted(required_facet),
        "private_edge": sorted(private_edge),
        "after_private_edge_count": normalized_count,
        "gate_domains": [[outside, list(choices)] for outside, choices in domains],
        "gate_assignment_count": len(assignments),
        "symmetry_reduction": "none; every locally possible assignment is enumerated",
        "case_count": len(cases),
        "cases": cases,
    }


def verify_four_uniform_certificate(certificate: dict[str, Any]) -> dict[str, int]:
    if certificate.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported certificate schema version")
    if certificate.get("analysis_kind") != "four_uniform_private_edge_baseline":
        raise ValueError("certificate is not a rank-four baseline")
    graph = _octahedral_graph(8)
    encoded = nx.to_graph6_bytes(graph, nodes=list(range(VERTEX_COUNT)), header=False)
    if certificate.get("graph6") != encoded.decode("ascii").strip():
        raise ValueError("certificate graph6 does not match the fixed O8 labeling")
    if certificate.get("vertex_ordering") != list(range(VERTEX_COUNT)):
        raise ValueError("rank-four vertex ordering is incorrect")
    if certificate.get("antipodal_pair_convention") != (
        "vertices {2i,2i+1} are nonadjacent"
    ):
        raise ValueError("rank-four antipodal-pair convention is incorrect")
    if certificate.get("candidate_ranks") != [4]:
        raise ValueError("rank-four candidate ranks are incorrect")
    required_facet = frozenset({0, 2, 4, 6})
    universe = _rank_candidates(4)
    if certificate.get("initial_candidate_count") != len(universe):
        raise ValueError("rank-four candidate-universe count is incorrect")
    if certificate.get("required_facet") != sorted(required_facet):
        raise ValueError("rank-four required facet is incorrect")
    if certificate.get("private_edge") != sorted(FOUR_UNIFORM_PRIVATE_EDGE):
        raise ValueError("rank-four private edge is incorrect")
    if not _verify_private_edge_normalization(4)["all_cases_map_to_normal_form"]:
        raise ValueError("rank-four private-edge normalization is not exhaustive")
    domains = _four_uniform_gate_domains(
        graph, universe, required_facet, FOUR_UNIFORM_PRIVATE_EDGE
    )
    assignments = _four_uniform_assignments(domains)
    if certificate.get("gate_domains") != [
        [outside, list(choices)] for outside, choices in domains
    ]:
        raise ValueError("rank-four gate domains do not match reconstructed support")
    if certificate.get("gate_assignment_count") != len(assignments):
        raise ValueError("rank-four gate assignment count is incorrect")
    private_count = sum(
        candidate == required_facet or not FOUR_UNIFORM_PRIVATE_EDGE <= candidate
        for candidate in universe
    )
    if certificate.get("after_private_edge_count") != private_count:
        raise ValueError("rank-four private-edge candidate count is incorrect")
    if certificate.get("case_count") != len(assignments):
        raise ValueError("rank-four case count is incorrect")
    cases = certificate.get("cases")
    if not isinstance(cases, list) or len(cases) != len(assignments):
        raise ValueError("certificate omits locally possible gate assignments")
    excluded = 0
    for expected_id, (case, expected_gates) in enumerate(zip(cases, assignments)):
        if case.get("case_id") != expected_id:
            raise ValueError("rank-four case ids are not in canonical order")
        gates = {int(outside): int(gate) for outside, gate in case["gates"]}
        if gates != expected_gates:
            raise ValueError("rank-four gate assignment is not the expected case")
        replay = _set_based_case(
            graph,
            universe,
            gates,
            required_facet,
            FOUR_UNIFORM_PRIVATE_EDGE,
        )
        survivors = replay["survivors"]
        if case.get("initial_candidate_count") != len(universe):
            raise ValueError("rank-four initial candidate count is incorrect")
        if case.get("after_private_edge_count") != private_count:
            raise ValueError("rank-four private-edge count is incorrect")
        if case.get("normalization_removed") != replay["normalization_removed"]:
            raise ValueError("rank-four normalization count does not replay")
        if case.get("removed_per_round") != list(replay["removed_per_round"]):
            raise ValueError("rank-four propagation rounds do not replay")
        if case.get("rounds_completed") != len(replay["removed_per_round"]):
            raise ValueError("rank-four round count is incorrect")
        if case.get("status") != replay["status"]:
            raise ValueError("rank-four status does not replay")
        if case.get("survivor_count") != len(survivors):
            raise ValueError("rank-four survivor count is incorrect")
        if case.get("survivor_masks_sha256") != _digest_masks(survivors):
            raise ValueError("rank-four survivor digest does not replay")
        actual_edge = next(
            (
                list(sorted(edge))
                for edge in graph.edges()
                if not any(set(edge) <= candidate for candidate in survivors)
            ),
            None,
        )
        reported_edge = case.get("uncovered_edge")
        if (reported_edge is not None and reported_edge != actual_edge) or (
            not replay["required_facet_missing"] and reported_edge != actual_edge
        ):
            raise ValueError("rank-four uncovered edge does not replay")
        if case.get("required_facet_missing") != replay["required_facet_missing"]:
            raise ValueError("rank-four required facet result does not replay")
        if case.get("support_checks") != replay["support_checks"]:
            raise ValueError("rank-four support-check count does not replay")
        if case.get("fixed_point") is not True:
            raise ValueError("rank-four case does not claim a fixed point")
        if case.get("resource_bounds") != {
            "max_candidates": None,
            "max_rounds": None,
            "max_support_checks": None,
        }:
            raise ValueError("rank-four resource bounds are incorrect")
        if replay["status"] != HellyCoverStatus.NO_COVER_CERTIFIED.value:
            raise ValueError(f"rank-four case {expected_id} was not excluded")
        if actual_edge is None:
            raise ValueError(f"rank-four case {expected_id} has no exclusion witness")
        excluded += 1
    return {"cases_checked": len(cases), "excluded": excluded}


def verify_certificate(certificate: dict[str, Any]) -> dict[str, int]:
    """Reconstruct every case with a separate set-based implementation."""
    if certificate.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported certificate schema version")
    graph = _octahedral_graph(8)
    encoded = nx.to_graph6_bytes(graph, nodes=list(range(VERTEX_COUNT)), header=False)
    if certificate.get("graph6") != encoded.decode("ascii").strip():
        raise ValueError("certificate graph6 does not match the fixed O8 labeling")
    if certificate.get("vertex_ordering") != list(range(VERTEX_COUNT)):
        raise ValueError("vertex ordering is incorrect")
    if certificate.get("antipodal_pair_convention") != (
        "vertices {2i,2i+1} are nonadjacent"
    ):
        raise ValueError("antipodal-pair convention is incorrect")
    if certificate.get("edge_count") != O8_EDGES:
        raise ValueError("O8 edge count is incorrect")
    if certificate.get("candidate_ranks") != [4, 5]:
        raise ValueError("candidate ranks are incorrect")
    universe = _rank_candidates(5)
    if certificate.get("initial_candidate_count") != len(universe):
        raise ValueError("candidate-universe count is incorrect")
    if certificate.get("required_facet") != sorted(F0):
        raise ValueError("required facet is incorrect")
    if certificate.get("private_edge") != sorted(PRIVATE_EDGE):
        raise ValueError("private edge is incorrect")
    if not _verify_private_edge_normalization(5)["all_cases_map_to_normal_form"]:
        raise ValueError("private-edge normalization is not exhaustive")
    representatives = _representative_gates()
    symmetry_coverage = _verify_representative_coverage(representatives)
    if certificate.get("symmetry_coverage") != symmetry_coverage:
        raise ValueError("symmetry canonicalization coverage does not replay")
    normalized_count = sum(
        candidate == F0 or not PRIVATE_EDGE <= candidate for candidate in universe
    )
    if certificate.get("after_private_edge_count") != normalized_count:
        raise ValueError("private-edge candidate count is incorrect")
    cases = certificate.get("cases")
    if not isinstance(cases, list) or len(cases) != len(representatives):
        raise ValueError("certificate does not contain every normalized gate case")
    if certificate.get("case_count") != len(cases):
        raise ValueError("case count is incorrect")
    excluded = 0
    surviving = 0
    expected_groups: dict[tuple[int, ...], list[int]] = {}
    for expected_id, (case, expected_gates) in enumerate(zip(cases, representatives)):
        if case.get("case_id") != expected_id:
            raise ValueError("case ids are not in canonical representative order")
        gates = {int(outside): int(gate) for outside, gate in case["gates"]}
        if gates != expected_gates:
            raise ValueError("case gate assignment is not the expected representative")
        replay = _set_based_case(graph, universe, gates)
        actual_survivors = replay["survivors"]
        if case.get("initial_candidate_count") != len(universe):
            raise ValueError(f"case {expected_id} has an incorrect initial count")
        if case.get("after_private_edge_count") != PRIVATE_EDGE_CANDIDATE_COUNT:
            raise ValueError(f"case {expected_id} has an incorrect normalization count")
        if case.get("normalization_removed") != replay["normalization_removed"]:
            raise ValueError(f"case {expected_id} has an incorrect filter count")
        if case.get("removed_per_round") != list(replay["removed_per_round"]):
            raise ValueError(f"case {expected_id} has incorrect round counts")
        if case.get("rounds_completed") != len(replay["removed_per_round"]):
            raise ValueError(f"case {expected_id} has an incorrect round count")
        if case.get("after_private_edge_count") != normalized_count:
            raise ValueError(f"case {expected_id} has an incorrect edge normalization")
        if case.get("status") != replay["status"]:
            raise ValueError(f"case {expected_id} has an incorrect status")
        if case.get("survivor_count") != len(actual_survivors):
            raise ValueError(f"case {expected_id} has an incorrect survivor count")
        if case.get("survivor_masks_sha256") != _digest_masks(actual_survivors):
            raise ValueError(f"case {expected_id} survivor digest does not replay")
        actual_edge = next(
            (
                list(sorted(edge))
                for edge in graph.edges()
                if not any(set(edge) <= candidate for candidate in actual_survivors)
            ),
            None,
        )
        reported_edge = case.get("uncovered_edge")
        if (reported_edge is not None and reported_edge != actual_edge) or (
            not replay["required_facet_missing"] and reported_edge != actual_edge
        ):
            raise ValueError(f"case {expected_id} has an incorrect uncovered edge")
        if case.get("required_facet_missing") != replay["required_facet_missing"]:
            raise ValueError(
                f"case {expected_id} has an incorrect required-facet result"
            )
        if case.get("support_checks") != replay["support_checks"]:
            raise ValueError(f"case {expected_id} has an incorrect support-check count")
        if case.get("fixed_point") is not True:
            raise ValueError(f"case {expected_id} does not claim a fixed point")
        if case.get("resource_bounds") != {
            "max_candidates": None,
            "max_rounds": None,
            "max_support_checks": None,
        }:
            raise ValueError(f"case {expected_id} reports unexpected resource bounds")
        if replay["status"] == HellyCoverStatus.NO_COVER_CERTIFIED.value:
            if actual_edge is None:
                raise ValueError("negative case has no independently replayed witness")
            excluded += 1
        else:
            surviving += 1
            expected_groups.setdefault(_gate_orbit_key(gates), []).append(expected_id)
    serialized_groups = [
        {"orbit_key": list(key), "case_ids": sorted(case_ids)}
        for key, case_ids in sorted(expected_groups.items())
    ]
    if certificate.get("survivor_orbit_types") != serialized_groups:
        raise ValueError("surviving symmetry-orbit classification does not replay")
    return {"cases_checked": len(cases), "excluded": excluded, "surviving": surviving}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rank", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--check", type=Path)
    parser.add_argument("--four-uniform", action="store_true")
    args = parser.parse_args()
    if args.check is not None:
        data = json.loads(args.check.read_text(encoding="utf-8"))
        verification = (
            verify_four_uniform_certificate(data)
            if data.get("analysis_kind") == "four_uniform_private_edge_baseline"
            else verify_certificate(data)
        )
        print(json.dumps(verification, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required when generating a certificate")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing file: {args.output}")
    certificate = (
        generate_four_uniform_certificate()
        if args.four_uniform
        else generate_certificate(args.max_rank)
    )
    if args.verify:
        verification = (
            verify_four_uniform_certificate(certificate)
            if args.four_uniform
            else verify_certificate(certificate)
        )
        print(json.dumps(verification, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(certificate, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    _main()

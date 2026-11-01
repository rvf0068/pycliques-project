"""Reproducibility checks for normalized O8 Helly-cover research cases."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from research.o8_helly_cover_certificates import (
    FOUR_UNIFORM_PRIVATE_EDGE,
    PRIVATE_EDGE,
    PRIVATE_EDGE_CANDIDATE_COUNT,
    _four_uniform_assignments,
    _four_uniform_gate_domains,
    _octahedral_graph,
    _rank_candidates,
    _representative_gates,
    _run_engine_case,
    _set_based_case,
    _verify_private_edge_normalization,
    _verify_representative_coverage,
    verify_four_uniform_certificate,
)

_FOUR_UNIFORM_CERTIFICATE_PATH = (
    Path(__file__).parent / "results" / "o8_four_uniform_helly_certificate.json"
)


def test_o8_candidate_counts_and_gate_representative_coverage() -> None:
    candidates = _rank_candidates(5)
    assert sum(len(candidate) == 4 for candidate in candidates) == 1120
    assert sum(len(candidate) == 5 for candidate in candidates) == 1792
    assert len(candidates) == 2912
    assert (
        sum(
            candidate == frozenset({0, 2, 4, 6, 8}) or not PRIVATE_EDGE <= candidate
            for candidate in candidates
        )
        == PRIVATE_EDGE_CANDIDATE_COUNT
    )

    representatives = _representative_gates()
    coverage = _verify_representative_coverage(representatives)
    assert coverage["literal_assignment_count"] == 93_312
    assert coverage["representative_count"] == 7168
    assert coverage["coverage_verified_by_exhaustive_canonicalization"]


@pytest.mark.parametrize("case_id", [0, 100, 5000])
def test_set_based_checker_matches_bitmask_engine_on_sample_cases(
    case_id: int,
) -> None:
    graph = _octahedral_graph(8)
    universe = _rank_candidates(5)
    gates = _representative_gates()[case_id]
    fast = _run_engine_case(graph, universe, gates)
    checked = _set_based_case(graph, universe, gates)

    assert fast.status.value == checked["status"]
    assert (
        tuple(
            sorted(
                fast.surviving_completes,
                key=lambda candidate: sum(1 << vertex for vertex in candidate),
            )
        )
        == checked["survivors"]
    )
    assert fast.removed_per_round == checked["removed_per_round"]
    assert fast.normalization_removed == checked["normalization_removed"]
    assert fast.support_checks == checked["support_checks"]
    assert (
        None if fast.uncovered_edge is None else tuple(sorted(fast.uncovered_edge))
    ) == checked["uncovered_edge"]


def test_four_uniform_gate_domains_cover_the_full_local_assignment_space() -> None:
    graph = _octahedral_graph(8)
    required = frozenset({0, 2, 4, 6})
    universe = _rank_candidates(4)
    domains = _four_uniform_gate_domains(
        graph, universe, required, FOUR_UNIFORM_PRIVATE_EDGE
    )
    assert tuple((vertex, choices) for vertex, choices in domains) == (
        (1, (2,)),
        (3, (0,)),
        (5, (0, 2, 6)),
        (7, (0, 2, 4)),
        (8, (0, 2)),
        (9, (0, 2)),
        (10, (0, 2)),
        (11, (0, 2)),
        (12, (0, 2)),
        (13, (0, 2)),
        (14, (0, 2)),
        (15, (0, 2)),
    )
    assert len(_four_uniform_assignments(domains)) == 2304


@pytest.mark.parametrize(("rank", "expected_cases"), [(4, 6720), (5, 17920)])
def test_private_facet_and_edge_normalization_is_transitive(
    rank: int, expected_cases: int
) -> None:
    result = _verify_private_edge_normalization(rank)
    assert result["facet_edge_cases_checked"] == expected_cases
    assert result["all_cases_map_to_normal_form"]


@pytest.fixture(scope="module")
def four_uniform_certificate() -> dict[str, object]:
    return json.loads(_FOUR_UNIFORM_CERTIFICATE_PATH.read_text())


def test_certificate_checker_rejects_missing_gate_assignment(
    four_uniform_certificate: dict[str, object],
) -> None:
    certificate = deepcopy(four_uniform_certificate)
    cases = certificate["cases"]
    assert isinstance(cases, list)
    cases.pop()

    with pytest.raises(ValueError, match="omits locally possible gate assignments"):
        verify_four_uniform_certificate(certificate)


def test_certificate_checker_replays_recorded_exclusion_witness(
    four_uniform_certificate: dict[str, object],
) -> None:
    certificate = deepcopy(four_uniform_certificate)
    cases = certificate["cases"]
    assert isinstance(cases, list)
    first_case = cases[0]
    assert isinstance(first_case, dict)
    first_case["uncovered_edge"] = (
        [0, 4] if first_case["uncovered_edge"] == [0, 2] else [0, 2]
    )

    with pytest.raises(ValueError, match="uncovered edge does not replay"):
        verify_four_uniform_certificate(certificate)

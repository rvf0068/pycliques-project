import json

import networkx as nx
import pytest
from pycliques import clique_graph, homotopy_clique_graph
from pycliques.named import octahedron

from research.h_operator_census import (
    INCONCLUSIVE_BUDGET,
    PROVED_BY_3_HELLY,
    PROVED_BY_COLLAPSE,
    PROVED_BY_SPHERE_CERTIFICATE,
    _betti_numbers_mod2,
    analyze_h_operator_graph,
    atlas_graphs,
    double_bound_clique_graph,
    graph6_graphs,
    run_h_operator_census,
    verify_s_reduction_certificate,
)


def _reference_middle(graph):
    return clique_graph(homotopy_clique_graph(graph))


def test_double_bound_realization_matches_reference_for_atlas_through_six():
    """Cross-check Proposition 7.5 on all 208 nonempty atlas graphs <= 6."""
    graphs = list(atlas_graphs(6))
    assert len(graphs) == 208
    for graph in graphs:
        reference = _reference_middle(graph)
        compact = double_bound_clique_graph(graph)
        assert reference is not None
        assert compact is not None
        assert nx.is_isomorphic(reference, compact)


@pytest.mark.parametrize(
    "graph",
    [
        nx.octahedral_graph(),
        nx.petersen_graph(),
        nx.cycle_graph(5),
        nx.complete_graph(4),
        nx.disjoint_union(nx.path_graph(3), nx.empty_graph(2)),
    ],
)
def test_double_bound_realization_matches_named_graphs(graph):
    reference = _reference_middle(graph)
    compact = double_bound_clique_graph(graph)
    assert reference is not None
    assert compact is not None
    assert nx.is_isomorphic(reference, compact)


def test_double_bound_realization_supports_noninteger_labels():
    graph = nx.relabel_nodes(nx.cycle_graph(5), {i: ("v", i) for i in range(5)})
    reference = _reference_middle(graph)
    compact = double_bound_clique_graph(graph)
    assert reference is not None
    assert compact is not None
    assert nx.is_isomorphic(reference, compact)


def test_double_bound_budgets_are_inconclusive():
    graph = nx.complete_graph(3)
    assert double_bound_clique_graph(graph, max_faces=0) is None
    assert double_bound_clique_graph(graph, max_candidates=0) is None
    assert double_bound_clique_graph(nx.path_graph(3), max_edges=0) is None


def test_experiment_uses_h_and_k_and_replays_collapse_for_single_vertex():
    result = analyze_h_operator_graph(
        nx.empty_graph(1), use_3_helly_certificate=False
    )
    assert result["h_order"] == 1
    assert result["middle_order"] == 1
    assert result["k2_order"] == 1
    assert result["status"] == PROVED_BY_COLLAPSE
    assert result["collapse_certificate_replayed"] is True
    assert result["betti_g"] == result["betti_k2"]


def test_experiment_budget_exit_is_inconclusive():
    result = analyze_h_operator_graph(
        nx.empty_graph(1), max_cliques=0, use_3_helly_certificate=False
    )
    assert result["status"] == INCONCLUSIVE_BUDGET
    assert result["middle_order"] is None
    assert result["resource_diagnostics"]


def test_independent_mod2_betti_calculation():
    assert _betti_numbers_mod2(nx.path_graph(3), 100) == []
    assert _betti_numbers_mod2(nx.cycle_graph(4), 100) == [0, 1]


def test_compact_and_reference_experiments_agree_on_sizes_and_homology():
    graph = nx.path_graph(4)
    reference = analyze_h_operator_graph(graph, use_3_helly_certificate=False)
    compact = analyze_h_operator_graph(
        graph, use_compact_realization=True, use_3_helly_certificate=False
    )
    for key in (
        "h_order",
        "h_size",
        "middle_order",
        "middle_size",
        "k2_order",
        "k2_size",
    ):
        assert compact[key] == reference[key]
    assert compact["betti_g"] == reference["betti_g"]
    assert compact["betti_k2"] == reference["betti_k2"]


def test_graph6_reader_accepts_header_and_multiple_graphs(tmp_path):
    path = tmp_path / "input.g6"
    first = nx.to_graph6_bytes(nx.path_graph(3), header=False).decode().strip()
    second = nx.to_graph6_bytes(nx.complete_graph(2), header=False).decode().strip()
    path.write_text(f">>graph6<<\n{first}\n{second}\n", encoding="ascii")
    graphs = list(graph6_graphs(path))
    assert len(graphs) == 2
    assert nx.is_isomorphic(graphs[0], nx.path_graph(3))
    assert nx.is_isomorphic(graphs[1], nx.complete_graph(2))


def test_census_jsonl_is_resumable_and_records_limits(tmp_path):
    path = tmp_path / "census.jsonl"
    graph = nx.empty_graph(1)
    first = run_h_operator_census([graph], output_jsonl=path)
    resumed = run_h_operator_census([graph], output_jsonl=path, resume=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0])
    record = json.loads(lines[1])
    assert first["graphs_processed"] == resumed["graphs_processed"] == 1
    assert metadata["limits"]["max_cliques"] == 10_000
    assert record["result"]["status"] == PROVED_BY_3_HELLY


def test_three_helly_shortcut_returns_before_constructing_h_or_k(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("the affirmative shortcut should avoid H and K")

    monkeypatch.setattr("research.h_operator_census.homotopy_clique_graph", unexpected)
    monkeypatch.setattr("research.h_operator_census.clique_graph", unexpected)

    result = analyze_h_operator_graph(nx.complete_graph(4))

    assert result["status"] == PROVED_BY_3_HELLY
    assert result["h_order"] is None
    assert result["middle_order"] is None
    audit = result["three_helly_audit"]
    assert audit["exact_test"] is True
    assert audit["maximal_cliques_complete"] is True
    assert audit["maximal_cliques_count"] == 1
    assert audit["four_sets_checked"] == 1
    assert "Theorem 4.2" in audit["theorem"]
    assert audit["witness"] is None


@pytest.mark.parametrize("graph", [nx.complete_graph(3), nx.empty_graph(3)])
def test_three_helly_audit_does_not_claim_skipped_clique_enumeration_is_complete(
    graph,
):
    audit = analyze_h_operator_graph(graph)["three_helly_audit"]

    assert audit["status"] == "proved"
    assert audit["maximal_cliques_count"] == 0
    assert audit["maximal_cliques_complete"] is False


def test_failed_three_helly_test_is_not_a_conjecture_counterexample():
    result = analyze_h_operator_graph(octahedron(4))

    assert result["three_helly_audit"]["status"] == "obstruction"
    assert result["three_helly_audit"]["witness"] is not None
    assert result["status"] not in (
        PROVED_BY_3_HELLY,
        "HOMOLOGY_MISMATCH_COUNTEREXAMPLE",
    )
    assert result["h_order"] is not None


def test_three_helly_inconclusive_budget_does_not_claim_proof():
    result = analyze_h_operator_graph(nx.path_graph(4), max_3helly_4sets=0)

    assert result["three_helly_audit"]["status"] == "inconclusive_budget"
    assert result["three_helly_audit"]["exact_test"] is False
    assert result["status"] != PROVED_BY_3_HELLY
    assert result["h_order"] is not None


def test_three_helly_shortcut_can_be_disabled():
    result = analyze_h_operator_graph(
        nx.complete_graph(4), use_3_helly_certificate=False
    )

    assert result["three_helly_audit"] is None
    assert result["h_order"] is not None
    assert result["status"] != PROVED_BY_3_HELLY


def test_o4_double_bound_and_second_clique_graph_regression():
    graph = octahedron(4)
    homotopy_graph = homotopy_clique_graph(graph)
    middle = clique_graph(homotopy_graph)
    compact = double_bound_clique_graph(graph)
    assert middle is not None
    assert compact is not None
    assert (middle.number_of_nodes(), middle.number_of_edges()) == (80, 848)
    assert nx.is_isomorphic(middle, compact)
    second = clique_graph(middle)
    assert second is not None
    assert (second.number_of_nodes(), second.number_of_edges()) == (192, 6688)


def test_o4_sphere_certificate_reduces_and_replays():
    result = analyze_h_operator_graph(
        octahedron(4),
        use_compact_realization=True,
        enable_sphere_certificate=True,
        max_simplices=2_500,
    )
    assert result["s_reduction_certificate_replayed"] is True
    assert (result["reduced_core_order"], result["reduced_core_size"]) == (40, 214)
    assert result["reduced_core_f_vector"] == [40, 214, 348, 174]
    assert result["sphere_certificate_replayed"] is True
    assert len(result["s_reduction_certificate"]) == 216
    assert sum(
        move["kind"] == "vertex" for move in result["s_reduction_certificate"]
    ) == 152
    assert sum(
        move["kind"] == "edge" for move in result["s_reduction_certificate"]
    ) == 64
    assert len(result["sphere_certificate"]["collapse_pairs"]) == 387
    assert len(result["sphere_certificate"]["deleted_facet"]) == 4
    assert result["status"] == PROVED_BY_SPHERE_CERTIFICATE
    json.dumps(result)
    compact = double_bound_clique_graph(octahedron(4))
    assert compact is not None
    second = clique_graph(compact)
    assert second is not None
    assert verify_s_reduction_certificate(
        second, result["s_reduction_certificate"]
    )


def test_o4_certificate_replay_is_invariant_under_input_relabeling():
    graph = nx.relabel_nodes(octahedron(4), {i: ("vertex", i) for i in range(8)})
    result = analyze_h_operator_graph(
        graph,
        use_compact_realization=True,
        enable_sphere_certificate=True,
    )
    assert result["status"] == PROVED_BY_SPHERE_CERTIFICATE
    assert result["s_reduction_certificate_replayed"] is True
    assert result["sphere_certificate_replayed"] is True
    assert (
        result["sphere_certificate"]["dimension"]
        == result["input_sphere_certificate"]["dimension"]
        == 3
    )


def test_sphere_certificate_budget_is_inconclusive():
    result = analyze_h_operator_graph(
        octahedron(4),
        use_compact_realization=True,
        enable_sphere_certificate=True,
        max_s_reduction_moves=0,
    )
    assert result["status"] != PROVED_BY_SPHERE_CERTIFICATE
    assert result["s_reduction_certificate_replayed"] is False
    assert result["status"] == INCONCLUSIVE_BUDGET
    assert "max_s_reduction_moves=0" in result["resource_diagnostics"]

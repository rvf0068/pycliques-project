import json

import networkx as nx
import pytest
from pycliques import clique_graph, homotopy_clique_graph

from research.h_operator_census import (
    INCONCLUSIVE_BUDGET,
    PROVED_BY_COLLAPSE,
    _betti_numbers_mod2,
    analyze_h_operator_graph,
    atlas_graphs,
    double_bound_clique_graph,
    graph6_graphs,
    run_h_operator_census,
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
    result = analyze_h_operator_graph(nx.empty_graph(1))
    assert result["h_order"] == 1
    assert result["middle_order"] == 1
    assert result["k2_order"] == 1
    assert result["status"] == PROVED_BY_COLLAPSE
    assert result["collapse_certificate_replayed"] is True
    assert result["betti_g"] == result["betti_k2"]


def test_experiment_budget_exit_is_inconclusive():
    result = analyze_h_operator_graph(nx.empty_graph(1), max_cliques=0)
    assert result["status"] == INCONCLUSIVE_BUDGET
    assert result["middle_order"] is None
    assert result["resource_diagnostics"]


def test_independent_mod2_betti_calculation():
    assert _betti_numbers_mod2(nx.path_graph(3), 100) == []
    assert _betti_numbers_mod2(nx.cycle_graph(4), 100) == [0, 1]


def test_compact_and_reference_experiments_agree_on_sizes_and_homology():
    graph = nx.path_graph(4)
    reference = analyze_h_operator_graph(graph)
    compact = analyze_h_operator_graph(graph, use_compact_realization=True)
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
    assert record["result"]["status"] == PROVED_BY_COLLAPSE

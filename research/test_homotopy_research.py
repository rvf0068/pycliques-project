"""Reproducible low-order checks for clique-graph homotopy research."""

import json
from dataclasses import replace
from itertools import combinations

import networkx as nx
import pytest
from pycliques import (
    EnumerationStatus,
    clique_graph,
    homotopy_structure_certificates,
    verify_homotopy_structure_certificates,
)
from pycombtop import (
    SimplicialComplex,
    clique_complex,
    nerve_of_cliques,
    relative_betti_numbers,
    relative_collapse,
    verify_relative_collapse,
)
from pyg6data import graph_generator

from research.homotopy_census import (
    analyze_graph,
    matches_research_filter,
    run_census,
    run_census_jsonl,
)


def _defect(graph: nx.Graph) -> int:
    kg = clique_graph(graph)
    assert kg is not None
    return len(
        clique_complex(kg).all_simplices() - nerve_of_cliques(graph).all_simplices()
    )


def test_defect_one_and_three_absent_in_connected_orders_six_and_seven():
    for order in (6, 7):
        defects = {_defect(graph) for graph in graph_generator(order, connected=True)}
        assert 1 not in defects
        assert 3 not in defects


def test_defect_two_has_a_single_verified_elementary_pair():
    graph = nx.from_graph6_bytes(b"EElw")
    kg = clique_graph(graph)
    assert kg is not None
    source = clique_complex(kg)
    target = nerve_of_cliques(graph)

    assert len(source.all_simplices() - target.all_simplices()) == 2
    result = relative_collapse(source, target)

    assert result.status.value == "PROVED"
    assert len(result.pairs) == 1
    assert verify_relative_collapse(source, target, result.pairs)


def test_paper_candidate_filter_count_order_seven():
    count = 0
    for graph in graph_generator(7, connected=True):
        kg = clique_graph(graph)
        assert kg is not None
        count += matches_research_filter(graph, kg)
    assert count == 2


@pytest.mark.slow
def test_paper_candidate_filter_count_order_eight():
    count = 0
    for graph in graph_generator(8, connected=True):
        kg = clique_graph(graph)
        assert kg is not None
        count += matches_research_filter(graph, kg)
    assert count == 34


def test_large_source_is_stopped_before_unbounded_simplex_materialization(
    monkeypatch,
):
    from research import homotopy_census

    def forbidden_all_simplices(self):
        raise AssertionError("all_simplices must not run after the budget is hit")

    monkeypatch.setattr(SimplicialComplex, "all_simplices", forbidden_all_simplices)
    result = homotopy_census.analyze_graph(nx.star_graph(19), max_simplices=20)

    assert result["collapse_status"] == "INCONCLUSIVE"
    assert result["defect"] is None
    assert "max_simplices=20" in result["collapse_diagnostics"]
    assert "empty simplex" in result["collapse_diagnostics"]


def test_structure_verifier_checks_complete_minimum_and_canonical_ids():
    graph = nx.from_graph6_bytes(b"GN]~l[")
    result = homotopy_structure_certificates(graph)
    assert result.smallest_violation_ids is not None

    assert not verify_homotopy_structure_certificates(
        graph, replace(result, smallest_violation_ids=None)
    )
    larger_minimal_face = next(
        face.clique_ids
        for face in result.missing_faces
        if len(face.clique_ids) > len(result.smallest_violation_ids)
    )
    assert not verify_homotopy_structure_certificates(
        graph, replace(result, smallest_violation_ids=larger_minimal_face)
    )

    corrupted_face = replace(
        result.missing_faces[0],
        clique_ids=(result.missing_faces[0].clique_ids[0],)
        * len(result.missing_faces[0].clique_ids),
    )
    assert not verify_homotopy_structure_certificates(
        graph,
        replace(result, missing_faces=(corrupted_face, *result.missing_faces[1:])),
    )


def test_structure_certificates_match_brute_force_on_small_graph6_input():
    graph = nx.from_graph6_bytes(b"EElw")
    kg = clique_graph(graph)
    assert kg is not None
    result = homotopy_structure_certificates(graph)
    nodes = list(kg.nodes())
    source_ids = {vertex: index for index, vertex in enumerate(graph.nodes())}
    ordered_nodes = sorted(
        nodes,
        key=lambda clique: tuple(sorted(source_ids[vertex] for vertex in clique)),
    )
    canonical_ids = {clique: index for index, clique in enumerate(ordered_nodes)}
    expected = set()

    for size in range(1, len(nodes) + 1):
        for candidate in combinations(nodes, size):
            if not all(
                kg.has_edge(first, second)
                for first, second in combinations(candidate, 2)
            ):
                continue
            if set.intersection(*(set(clique) for clique in candidate)):
                continue
            if all(
                (
                    set.intersection(
                        *(
                            set(clique)
                            for index, clique in enumerate(candidate)
                            if index != omitted
                        )
                    )
                    if len(candidate) > 1
                    else set(graph)
                )
                for omitted in range(len(candidate))
            ):
                expected.add(
                    tuple(sorted(canonical_ids[clique] for clique in candidate))
                )

    actual = {certificate.clique_ids for certificate in result.missing_faces}
    assert result.status is EnumerationStatus.COMPLETE
    assert actual == expected


def test_jsonl_census_resume_matches_in_memory_records(tmp_path, monkeypatch):
    from research import homotopy_census

    graphs = [nx.path_graph(3), nx.cycle_graph(4)]
    monkeypatch.setattr(
        homotopy_census,
        "graph_generator",
        lambda order, connected: iter(graphs),
    )
    output = tmp_path / "census.jsonl"
    analyze = homotopy_census.analyze_graph
    calls = 0

    def interrupt_second_graph(graph, **budgets):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return analyze(graph, **budgets)

    monkeypatch.setattr(homotopy_census, "analyze_graph", interrupt_second_graph)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_census_jsonl(
            3,
            output,
            max_states=20,
            max_simplices=100,
            checkpoint_interval=1,
        )
    monkeypatch.setattr(homotopy_census, "analyze_graph", analyze)
    resumed = run_census_jsonl(
        3,
        output,
        max_states=20,
        max_simplices=100,
        checkpoint_interval=1,
        resume=True,
    )
    no_op_resume = run_census_jsonl(
        3,
        output,
        max_states=20,
        max_simplices=100,
        checkpoint_interval=1,
        resume=True,
    )
    in_memory = run_census(3, max_states=20, max_simplices=100)
    lines = output.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines[1:]]

    assert resumed["graphs_processed_this_run"] == 1
    assert no_op_resume["graphs_processed_this_run"] == 0
    assert resumed["candidate_filter_complete"]
    assert [row["index"] for row in rows] == [0, 1]
    assert [row["graph6"] for row in rows] == [
        record["graph6"] for record in in_memory["records"]
    ]
    assert [row["result"]["collapse_status"] for row in rows] == [
        record["collapse_status"] for record in in_memory["records"]
    ]
    assert resumed["candidate_graph6"] == in_memory["candidate_graph6"]
    assert output.with_suffix(".jsonl.summary.json").exists()

    with pytest.raises(ValueError, match="metadata"):
        run_census_jsonl(
            3,
            output,
            max_states=21,
            max_simplices=100,
            checkpoint_interval=1,
            resume=True,
        )


def test_jsonl_resume_preserves_inconclusive_candidate_filters(tmp_path, monkeypatch):
    from research import homotopy_census

    graph = nx.star_graph(5)
    monkeypatch.setattr(
        homotopy_census,
        "graph_generator",
        lambda order, connected: iter([graph]),
    )
    output = tmp_path / "inconclusive.jsonl"

    summary = run_census_jsonl(6, output, max_simplices=0)
    resumed = run_census_jsonl(6, output, max_simplices=0, resume=True)

    assert summary["candidate_filter_complete"] is False
    assert summary["candidate_filter_inconclusive_graphs"] == 1
    assert resumed["graphs_processed_this_run"] == 0
    assert resumed["candidate_filter_inconclusive_graphs"] == 1


def test_small_census_analysis_matches_direct_complex_computation():
    graph = nx.from_graph6_bytes(b"EElw")
    kg = clique_graph(graph)
    assert kg is not None
    source = clique_complex(kg)
    target = nerve_of_cliques(graph)
    direct_collapse = relative_collapse(source, target)
    direct_betti = relative_betti_numbers(
        source.facet_set, lambda face: target.function(face)
    )

    result = analyze_graph(graph, max_states=1_000, max_simplices=2_500)

    assert result["defect"] == len(source.all_simplices() - target.all_simplices())
    assert result["collapse_status"] == direct_collapse.status.value
    assert result["relative_betti_q"] == direct_betti
    ordered_cliques = sorted(kg, key=lambda clique: tuple(sorted(clique)))
    clique_ids = {clique: index for index, clique in enumerate(ordered_cliques)}
    assert result["collapse_pairs"] == [
        [
            sorted(clique_ids[vertex] for vertex in free_face),
            sorted(clique_ids[vertex] for vertex in coface),
        ]
        for free_face, coface in direct_collapse.pairs
    ]

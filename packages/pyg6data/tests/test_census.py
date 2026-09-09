from pathlib import Path

from pyg6data.census import run_graph_census


def _accept_all(graph):
    return True


def _edge_count(graph):
    return graph.number_of_edges()


def test_census_streams_results_and_filters():
    """The harness records every index and filters before evaluation."""
    records = run_graph_census(
        6,
        lambda graph: graph.number_of_edges() == 0,
        lambda graph: graph.number_of_nodes(),
    )
    assert len(records) == 112
    assert all(record["status"] == "filtered" for record in records)


def test_census_captures_evaluator_exceptions():
    """One evaluator failure is recorded without aborting the census."""
    records = run_graph_census(6, lambda graph: True, lambda graph: 1 / 0)
    assert len(records) == 112
    assert all(record["status"] == "error" for record in records)
    assert all("ZeroDivisionError" in record["exception"] for record in records)


def test_census_checkpoint_resume(tmp_path: Path):
    """A completed checkpoint can be resumed without duplicating records."""
    checkpoint = tmp_path / "census.json"
    first = run_graph_census(
        6,
        lambda graph: graph.number_of_nodes() == 6,
        lambda graph: graph.number_of_edges(),
        checkpoint=checkpoint,
        checkpoint_interval=20,
    )
    resumed = run_graph_census(
        6,
        lambda graph: graph.number_of_nodes() == 6,
        lambda graph: graph.number_of_edges(),
        checkpoint=checkpoint,
        checkpoint_interval=20,
    )
    assert len(first) == len(resumed) == 112
    assert resumed == first


def test_census_process_workers_match_sequential():
    """Process workers produce the same ordered records as sequential mode."""
    sequential = run_graph_census(6, _accept_all, _edge_count)
    parallel = run_graph_census(6, _accept_all, _edge_count, workers=2)
    assert parallel == sequential

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
        run_id="edge-count-v1",
        checkpoint_interval=20,
    )
    resumed = run_graph_census(
        6,
        lambda graph: graph.number_of_nodes() == 6,
        lambda graph: graph.number_of_edges(),
        checkpoint=checkpoint,
        run_id="edge-count-v1",
        checkpoint_interval=20,
    )
    assert len(first) == len(resumed) == 112
    assert resumed == first


def test_census_process_workers_match_sequential():
    """Process workers produce the same ordered records as sequential mode."""
    sequential = run_graph_census(6, _accept_all, _edge_count)
    parallel = run_graph_census(6, _accept_all, _edge_count, workers=2)
    assert parallel == sequential


def test_checkpoint_identity_and_partial_stream(tmp_path):
    import sqlite3
    from contextlib import closing

    import pytest
    from pyg6data import iter_graph6_records, iter_graph_census

    db = tmp_path / "partial.sqlite"
    with closing(
        iter_graph_census(
            6,
            _accept_all,
            _edge_count,
            checkpoint=db,
            run_id="edges",
            checkpoint_interval=1,
        )
    ) as stream:
        row = next(stream)
        assert (row["index"], row["graph6"]) == next(iter_graph6_records(6))
    with sqlite3.connect(db) as inspected:
        assert inspected.execute("SELECT count(*) FROM records").fetchone() == (1,)
    for order, key in [(7, "edges"), (6, "vertices"), (6, None)]:
        with pytest.raises(ValueError, match="source|run_id"):
            list(
                iter_graph_census(
                    order, _accept_all, _edge_count, checkpoint=db, run_id=key
                )
            )
    assert run_graph_census(
        6, _accept_all, _edge_count, checkpoint=db, run_id="edges"
    ) == run_graph_census(6, _accept_all, _edge_count)
    with sqlite3.connect(db) as inspected:
        inspected.execute("UPDATE records SET record='{}' WHERE idx=0")
    with pytest.raises(ValueError, match="disagrees"):
        next(
            iter_graph_census(
                6, _accept_all, _edge_count, checkpoint=db, run_id="edges"
            )
        )


def test_explicit_sparse_legacy_checkpoint_import(tmp_path):
    import json

    import pytest
    from pyg6data import import_legacy_checkpoint

    source, target = tmp_path / "old.json", tmp_path / "new.sqlite"
    records = run_graph_census(6, _accept_all, _edge_count)
    payload = {"completed": [4, 0], "records": [records[4], records[0]]}
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="import_legacy_checkpoint"):
        run_graph_census(6, _accept_all, _edge_count, checkpoint=source, run_id="edges")
    import_legacy_checkpoint(source, target, order=6, run_id="edges")
    assert (
        run_graph_census(6, _accept_all, _edge_count, checkpoint=target, run_id="edges")
        == records
    )
    for bad in ([4, 4], [1, 0]):
        source.write_text(json.dumps({**payload, "completed": bad}))
        with pytest.raises(ValueError, match="disagree"):
            import_legacy_checkpoint(source, tmp_path / "bad", order=6, run_id="edges")
        assert not (tmp_path / "bad").exists()
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="disagrees"):
        import_legacy_checkpoint(source, tmp_path / "bad", order=7, run_id="edges")

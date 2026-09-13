from types import SimpleNamespace

import networkx as nx

from pycliques.small import Verdict


def _graph6(graph: nx.Graph) -> str:
    return nx.to_graph6_bytes(graph, header=False).decode("ascii").strip()


def test_parse_args_defaults():
    from pycliques.behavior import _parse_args

    args = _parse_args(["A_"])

    assert args.input == "A_"
    assert args.bound == 30
    assert args.progress is True


def test_parse_args_accepts_bound_and_no_progress():
    from pycliques.behavior import _parse_args

    args = _parse_args(["--bound", "12", "--no-progress", "A_"])

    assert args.bound == 12
    assert args.progress is False


def test_main_classifies_graph6_string(capsys, monkeypatch):
    from pycliques import behavior

    calls = []

    def classify(graph, *, bound, on_iterate):
        calls.append((graph, bound))
        return SimpleNamespace(verdict=Verdict.CONVERGENT, reason="test reason")

    monkeypatch.setattr(behavior, "classify_clique_behavior", classify)
    behavior._main(["--no-progress", "A_"])

    captured = capsys.readouterr()
    assert captured.out == "1\tCONVERGENT\ttest reason\n"
    assert captured.err == ""
    assert len(calls) == 1
    assert calls[0][0].order() == 2
    assert calls[0][1] == 30


def test_main_classifies_graph6_file_with_progress(
    tmp_path, capsys, caplog, monkeypatch
):
    from pycliques import behavior

    caplog.set_level("INFO", logger=behavior.__name__)
    graph_file = tmp_path / "graphs.g6"
    graph_file.write_text(f"{_graph6(nx.path_graph(2))}\n{_graph6(nx.path_graph(3))}\n")
    bounds = []

    def classify(graph, *, bound, on_iterate):
        bounds.append(bound)
        on_iterate(1)
        on_iterate(2)
        return SimpleNamespace(
            verdict=Verdict.INDETERMINATE,
            reason="test reason",
        )

    monkeypatch.setattr(behavior, "classify_clique_behavior", classify)
    behavior._main(["--bound", "7", str(graph_file)])

    captured = capsys.readouterr()
    assert captured.out == (
        "1\tINDETERMINATE\ttest reason\n"
        "2\tINDETERMINATE\ttest reason\n"
    )
    assert [record.message for record in caplog.records[:-1]] == [
        "Classifying graph 1...",
        "Working with iterate 1",
        "Working with iterate 2",
        "Classifying graph 2...",
        "Working with iterate 1",
        "Working with iterate 2",
    ]
    assert caplog.records[-1].message.startswith("Done at ")
    assert bounds == [7, 7]

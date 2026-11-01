"""Regression tests for the fast pass / Theorem 2.6 pass separation."""

import logging

import networkx as nx
import pycliques.small as small
import pytest
from pycliques.cliques import clique_graph
from pycliques.small import (
    Certificate,
    Verdict,
    _load_indeterminate_file_entries,
    _load_indeterminate_file_entries_with_metadata,
    _main_theorem26,
    _parse_theorem26_args,
    _run_fast_pass_indeterminate,
    _save_indeterminate,
    classify_clique_behavior,
    classify_clique_behavior_with_theorem_2_6,
)

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_theorem26_parse_args_defaults():
    args = _parse_theorem26_args(["9"])

    assert args.n == 9
    assert args.max_m == 3
    assert args.max_n == 3
    assert args.max_coaffinations == 20
    assert args.max_source_order == 40
    assert args.bound == 30
    assert args.from_indeterminate_file is False
    assert args.loglevel == logging.INFO


def test_theorem26_parse_args_custom():
    args = _parse_theorem26_args(
        [
            "9",
            "--max-m",
            "4",
            "--max-n",
            "1",
            "--max-coaffinations",
            "5",
            "--max-source-order",
            "50",
            "--bound",
            "100",
            "--from-indeterminate-file",
        ]
    )

    assert args.max_m == 4
    assert args.max_n == 1
    assert args.max_coaffinations == 5
    assert args.max_source_order == 50
    assert args.bound == 100
    assert args.from_indeterminate_file is True


# ---------------------------------------------------------------------------
# Fast pass must not invoke Theorem 2.6
# ---------------------------------------------------------------------------


def test_fast_pass_does_not_call_theorem_2_6(monkeypatch):
    """The fast pass must not call the expensive clockwork-pair search."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Theorem 2.6 search must not run during the fast pass")

    monkeypatch.setattr(small, "_classify_clockwork_pair_map", fail_if_called)

    # A graph only resolvable by Theorem 2.6: its clique graph search fails,
    # and its own clique graph resolves via Theorem 2.6 in the old pipeline.
    kg = clique_graph(nx.icosahedral_graph())
    result = classify_clique_behavior(kg)

    assert result.verdict is Verdict.INDETERMINATE


def test_theorem_2_6_pass_does_invoke_the_search(monkeypatch):
    """The combined pass must call the Theorem 2.6 search when unresolved."""
    called = []
    real = small._classify_clockwork_pair_map

    def spy(*args, **kwargs):
        called.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(small, "_classify_clockwork_pair_map", spy)

    kg = clique_graph(nx.icosahedral_graph())
    result = classify_clique_behavior_with_theorem_2_6(kg)

    assert called
    assert result.verdict is Verdict.DIVERGENT


def test_combined_pass_skips_theorem_2_6_when_already_resolved(monkeypatch):
    """A graph resolved by the fast pass must not reach Theorem 2.6."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Theorem 2.6 must not run for an already-resolved graph")

    monkeypatch.setattr(small, "classify_clockwork_pair_map", fail_if_called)

    result = classify_clique_behavior_with_theorem_2_6(nx.cycle_graph(4))

    assert result.verdict is Verdict.CONVERGENT


# ---------------------------------------------------------------------------
# Unresolved filtering
# ---------------------------------------------------------------------------


def test_run_fast_pass_indeterminate_filters_to_indeterminate_only(
    monkeypatch, tmp_path
):
    """Only fast-pass INDETERMINATE graphs are returned as candidates."""
    from pycliques.small import CliqueBehavior

    def fake_classify(graph, *, bound):
        # Deterministic verdicts keyed on edge count, to distinguish graphs
        # (every graph in the order-6 dataset has the same vertex count).
        m = graph.number_of_edges()
        if m % 3 == 0:
            return CliqueBehavior(Verdict.CONVERGENT, "fake", 1, False, graph)
        if m % 3 == 1:
            return CliqueBehavior(Verdict.DIVERGENT, "fake", 1, False, graph)
        return CliqueBehavior(Verdict.INDETERMINATE, "fake", 1, False, graph)

    monkeypatch.setattr(small, "classify_clique_behavior", fake_classify)

    convergent, divergent, indeterminate, processed = _run_fast_pass_indeterminate(
        6,
        data_dir=tmp_path,
        start=None,
        end=20,
        skip_dominated=False,
        bound=30,
    )

    assert convergent + divergent + len(indeterminate) == 21
    assert processed == set(range(21))
    assert convergent > 0
    assert divergent > 0
    assert len(indeterminate) > 0


# ---------------------------------------------------------------------------
# Progressive radius ordering: r = 3 before r = 4 before r = 5, ...
# ---------------------------------------------------------------------------


def test_theorem_2_6_search_tries_radii_in_increasing_order(monkeypatch):
    seen_radii = []

    def fake_candidates(graph, radius):
        seen_radii.append(radius)
        return iter(())

    monkeypatch.setattr(small, "candidate_target_coaffinations", fake_candidates)

    small._classify_clockwork_pair_map(nx.petersen_graph(), max_m=4, max_n=0, bound=5)

    # m = 2, 3, 4 <=> radius = 3, 4, 5, tried in increasing order.
    assert seen_radii == sorted(seen_radii)
    assert seen_radii[:3] == [3, 4, 5]


# ---------------------------------------------------------------------------
# --from-indeterminate-file loading
# ---------------------------------------------------------------------------


def test_load_indeterminate_file_entries_round_trips(tmp_path):
    g = nx.cycle_graph(5)
    _save_indeterminate(7, [(3, g, None)], tmp_path)

    entries = _load_indeterminate_file_entries(7, tmp_path)

    assert len(entries) == 1
    idx, loaded = entries[0]
    assert idx == 3
    assert nx.is_isomorphic(loaded, g)


def test_load_indeterminate_file_entries_missing_file_returns_empty(tmp_path):
    assert _load_indeterminate_file_entries(42, tmp_path) == []


# ---------------------------------------------------------------------------
# CLI wiring end-to-end
# ---------------------------------------------------------------------------


def test_main_theorem26_runs_fast_pass_then_theorem_2_6(monkeypatch, tmp_path):
    """_main_theorem26 collects fast-pass INDETERMINATE graphs, then classifies."""
    from pycliques.small import CliqueBehavior

    def fake_classify(graph, *, bound):
        m = graph.number_of_edges()
        if m % 3 == 0:
            return CliqueBehavior(Verdict.CONVERGENT, "fake", 1, False, graph)
        if m % 3 == 1:
            return CliqueBehavior(Verdict.DIVERGENT, "fake", 1, False, graph)
        return CliqueBehavior(Verdict.INDETERMINATE, "fake", 1, False, graph)

    monkeypatch.setattr(small, "classify_clique_behavior", fake_classify)

    calls = []

    def fake_clockwork(graph, **kwargs):
        calls.append(graph.order())
        return None

    monkeypatch.setattr(small, "classify_clockwork_pair_map", fake_clockwork)

    _main_theorem26(["6", "--data-dir", str(tmp_path), "--end", "20"])

    # Theorem 2.6 was invoked only for graphs the fast pass left indeterminate.
    assert calls, "expected at least one candidate to reach the Theorem 2.6 search"


def test_main_theorem26_from_indeterminate_file(monkeypatch, tmp_path):
    """--from-indeterminate-file reads candidates without rerunning the fast pass."""
    g = nx.icosahedral_graph()
    kg = clique_graph(g)
    _save_indeterminate(9, [(0, kg, None)], tmp_path)

    calls = []
    real = small.classify_clockwork_pair_map

    def spy(graph, **kwargs):
        calls.append(True)
        return real(graph, **kwargs)

    monkeypatch.setattr(small, "classify_clockwork_pair_map", spy)

    _main_theorem26(["9", "--data-dir", str(tmp_path), "--from-indeterminate-file"])

    assert calls


def _data_rows(path):
    return [
        line
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    ]


def test_theorem26_saved_cohort_removes_resolved_and_preserves_metadata(
    monkeypatch, tmp_path
):
    unresolved = nx.cycle_graph(5)
    resolved = nx.path_graph(4)
    conjecture = Certificate(
        rule="retracts",
        target_status="conjectured_divergent",
        target_label="survivor",
    )
    _save_indeterminate(
        9,
        [(3, resolved, None), (7, unresolved, conjecture)],
        tmp_path,
    )

    def fake_clockwork(graph, **kwargs):
        if graph.order() == 4:
            return ("DIVERGENT", "proved", Certificate(rule="theorem_2_6"))
        return None

    monkeypatch.setattr(small, "classify_clockwork_pair_map", fake_clockwork)
    _main_theorem26(["9", "--from-indeterminate-file", "--data-dir", str(tmp_path)])

    rows = _data_rows(tmp_path / "indeterminate_order_9.txt")
    assert len(rows) == 1
    assert rows[0].split()[0] == "7"
    assert "retracts conjectured_divergent survivor" in rows[0]


def test_theorem26_all_resolved_writes_empty_cohort_and_output(monkeypatch, tmp_path):
    _save_indeterminate(
        9,
        [(3, nx.path_graph(4), None), (7, nx.cycle_graph(5), None)],
        tmp_path,
    )
    output = tmp_path / "verdicts.tsv"
    monkeypatch.setattr(
        small,
        "classify_clockwork_pair_map",
        lambda graph, **kwargs: ("DIVERGENT", "proved", Certificate(rule="proof")),
    )

    _main_theorem26(
        [
            "9",
            "--from-indeterminate-file",
            "--data-dir",
            str(tmp_path),
            "--output-file",
            str(output),
        ]
    )

    assert _data_rows(tmp_path / "indeterminate_order_9.txt") == []
    assert len(_data_rows(output)) == 2
    assert all("\tDIVERGENT\tproved" in row for row in _data_rows(output))


def test_theorem26_no_save_preserves_input_bytes(monkeypatch, tmp_path):
    _save_indeterminate(
        9,
        [(3, nx.path_graph(4), None)],
        tmp_path,
    )
    path = tmp_path / "indeterminate_order_9.txt"
    original = path.read_bytes()
    monkeypatch.setattr(
        small,
        "classify_clockwork_pair_map",
        lambda graph, **kwargs: ("DIVERGENT", "proved", Certificate(rule="proof")),
    )

    _main_theorem26(
        [
            "9",
            "--from-indeterminate-file",
            "--no-save",
            "--data-dir",
            str(tmp_path),
        ]
    )

    assert path.read_bytes() == original


def test_theorem26_rejects_ranges_for_saved_cohort(tmp_path):
    with pytest.raises(SystemExit):
        _main_theorem26(
            [
                "9",
                "--from-indeterminate-file",
                "--start",
                "0",
                "--data-dir",
                str(tmp_path),
            ]
        )


def test_theorem26_range_update_preserves_outside_rows_and_adds_survivor(
    monkeypatch, tmp_path
):
    _save_indeterminate(
        9,
        [
            (1, nx.path_graph(4), None),
            (4, nx.cycle_graph(5), None),
            (9, nx.complete_graph(4), None),
        ],
        tmp_path,
    )
    new_unresolved = nx.path_graph(5)
    monkeypatch.setattr(
        small,
        "_run_fast_pass_indeterminate",
        lambda *args, **kwargs: (
            0,
            0,
            [(4, nx.cycle_graph(5), None), (5, new_unresolved, None)],
            {4, 5},
        ),
    )
    monkeypatch.setattr(
        small,
        "classify_clockwork_pair_map",
        lambda graph, **kwargs: (
            ("DIVERGENT", "proved", Certificate(rule="proof"))
            if graph.order() == 5 and graph.number_of_edges() == 5
            else None
        ),
    )

    _main_theorem26(["9", "--start", "4", "--end", "5", "--data-dir", str(tmp_path)])

    rows = _data_rows(tmp_path / "indeterminate_order_9.txt")
    assert {int(row.split()[0]) for row in rows} == {1, 5, 9}


def test_theorem26_full_fresh_pass_replaces_the_unresolved_cohort(
    monkeypatch, tmp_path
):
    _save_indeterminate(
        6,
        [
            (0, nx.cycle_graph(5), None),
            (99, nx.complete_graph(4), None),
        ],
        tmp_path,
    )
    monkeypatch.setattr(
        small,
        "_run_fast_pass_indeterminate",
        lambda *args, **kwargs: (1, 0, [], {0}),
    )

    _main_theorem26(["6", "--data-dir", str(tmp_path)])

    assert _data_rows(tmp_path / "indeterminate_order_6.txt") == []


def test_theorem26_saved_unresolved_metadata_survives(monkeypatch, tmp_path):
    prior = Certificate(
        rule="retracts",
        target_status="conjectured_divergent",
        target_label="survivor",
    )
    _save_indeterminate(9, [(7, nx.cycle_graph(5), prior)], tmp_path)
    monkeypatch.setattr(small, "classify_clockwork_pair_map", lambda *a, **k: None)

    _main_theorem26(["9", "--from-indeterminate-file", "--data-dir", str(tmp_path)])

    entry = _load_indeterminate_file_entries_with_metadata(9, tmp_path)[0]
    assert entry[0] == 7
    assert entry[2] is not None
    assert entry[2].rule == "retracts"
    assert entry[2].target_status == "conjectured_divergent"
    assert entry[2].target_label == "survivor"


def test_theorem26_rechecking_is_idempotent(monkeypatch, tmp_path):
    _save_indeterminate(9, [(7, nx.cycle_graph(5), None)], tmp_path)
    monkeypatch.setattr(small, "classify_clockwork_pair_map", lambda *a, **k: None)
    arguments = ["9", "--from-indeterminate-file", "--data-dir", str(tmp_path)]

    _main_theorem26(arguments)
    first_contents = (tmp_path / "indeterminate_order_9.txt").read_bytes()
    _main_theorem26(arguments)

    assert (tmp_path / "indeterminate_order_9.txt").read_bytes() == first_contents
    assert len(_data_rows(tmp_path / "indeterminate_order_9.txt")) == 1


def test_theorem26_empty_input_writes_empty_cohort(tmp_path):
    _main_theorem26(["9", "--from-indeterminate-file", "--data-dir", str(tmp_path)])

    assert _data_rows(tmp_path / "indeterminate_order_9.txt") == []

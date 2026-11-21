"""Regression tests for reference-status provenance in census rechecks.

A certified divergent reference must retain its provenance in saved entries;
an undecided result must preserve the previously recorded certificate.
"""

import networkx as nx
from pycliques.small import (
    Certificate,
    Verdict,
    _load_indeterminate_file_entries_with_metadata,
    _recheck_indeterminate_file,
    _save_indeterminate,
)


def test_recheck_preserves_saved_certificate_when_rule_is_inconclusive() -> None:
    graph = nx.path_graph(3)
    saved = Certificate(
        rule="reference", target_status="conjectured_divergent", target_label="example"
    )
    entries = [(7, graph, saved)]
    assert _recheck_indeterminate_file(entries, 12, classifier=lambda seq: None) == entries


def test_recheck_discards_classified_graph() -> None:
    graph = nx.path_graph(3)
    entries = [(9, graph, None)]
    assert (
        _recheck_indeterminate_file(
            entries,
            12,
            classifier=lambda seq: (
                Verdict.DIVERGENT,
                "certified",
                Certificate(rule="reference", target_status="proven_divergent"),
            ),
        )
        == []
    )


def test_saved_reference_status_metadata_roundtrip_and_merge(tmp_path) -> None:
    first = nx.path_graph(3)
    second = nx.cycle_graph(4)
    conjectured = Certificate(
        rule="reference", target_status="conjectured_divergent", target_label="example"
    )
    proven = Certificate(
        rule="reference", target_status="proven_divergent", target_label="snub"
    )
    _save_indeterminate(8, [(7, first, conjectured)], tmp_path)
    entries = _load_indeterminate_file_entries_with_metadata(8, tmp_path)
    assert len(entries) == 1
    assert entries[0][2] == conjectured

    _save_indeterminate(8, [(2, second, proven), (7, first, proven)], tmp_path)
    entries = _load_indeterminate_file_entries_with_metadata(8, tmp_path)
    assert [idx for idx, _, _ in entries] == [2, 7]
    assert [c.target_status for _, _, c in entries if c is not None] == [
        "proven_divergent",
        "proven_divergent",
    ]

    _save_indeterminate(8, [(11, second, None)], tmp_path, replace=True)
    entries = _load_indeterminate_file_entries_with_metadata(8, tmp_path)
    assert len(entries) == 1
    assert entries[0][0] == 11
    assert entries[0][2] is None

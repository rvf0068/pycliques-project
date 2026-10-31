"""Tests for processing indeterminate graph files."""

import csv

import networkx as nx
import pytest
from pycombtop.homotopy_file import analyze_indeterminate_file


def _graph6(graph: nx.Graph) -> str:
    encoded: bytes = nx.to_graph6_bytes(graph, header=False)
    return encoded.decode("ascii").strip()


def test_analyze_indeterminate_file_writes_exact_types(tmp_path):
    graph = nx.cycle_graph(5)
    input_path = tmp_path / "indeterminate_order_5.txt"
    output_path = tmp_path / "homotopy.txt"
    input_path.write_text(
        "# Format: original_index pared_order graph6 metadata\n"
        f"42 5 {_graph6(graph)} -\n",
        encoding="utf-8",
    )

    count = analyze_indeterminate_file(input_path, output_path)

    with output_path.open(encoding="utf-8", newline="") as result_file:
        assert result_file.readline().startswith("# Clique-complex")
        rows = list(csv.DictReader(result_file, delimiter="\t"))
    assert count == 1
    assert rows == [
        {
            "original_index": "42",
            "order": "5",
            "graph6": _graph6(graph),
            "homotopy_type": r"\(S^{1}\)",
            "status": "exact",
            "reason": rows[0]["reason"],
        }
    ]


def test_analyze_indeterminate_file_marks_size_limited_result(tmp_path):
    graph = nx.cycle_graph(5)
    input_path = tmp_path / "indeterminate_order_5.txt"
    output_path = tmp_path / "homotopy.txt"
    input_path.write_text(f"42 5 {_graph6(graph)} - - -\n", encoding="utf-8")

    analyze_indeterminate_file(input_path, output_path, max_vertices=4)

    with output_path.open(encoding="utf-8", newline="") as result_file:
        result_file.readline()
        row = next(csv.DictReader(result_file, delimiter="\t"))
    assert row["homotopy_type"] == "Too large to check"
    assert row["status"] == "limited"


def test_analyze_indeterminate_file_rejects_order_mismatch(tmp_path):
    input_path = tmp_path / "indeterminate_order_5.txt"
    output_path = tmp_path / "homotopy.txt"
    input_path.write_text(f"42 6 {_graph6(nx.cycle_graph(5))}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match graph6 order"):
        analyze_indeterminate_file(input_path, output_path)


def test_analyze_indeterminate_file_rejects_same_input_and_output(tmp_path):
    input_path = tmp_path / "indeterminate_order_5.txt"
    input_path.write_text(f"42 5 {_graph6(nx.cycle_graph(5))}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be different"):
        analyze_indeterminate_file(input_path, input_path)

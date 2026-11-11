"""Smoke tests for the opt-in order-10 representative-core benchmark."""

import csv

import networkx as nx
import pytest

from research import benchmark_order10_homotopy_cores as benchmark


def test_input_identity_and_unique_representatives(tmp_path):
    path = tmp_path / "representatives.tsv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["representative_index", "graph6", "core_graph6", "members"],
            delimiter="\t",
        )
        writer.writeheader()
        core = nx.to_graph6_bytes(nx.cycle_graph(5), header=False).decode().strip()
        for index in range(60):
            writer.writerow({
                "representative_index": str(index),
                "graph6": "source",
                "core_graph6": core,
                "members": str(index),
            })
    rows = benchmark.load_representatives(path)
    assert len(rows) == 60
    assert rows[-1]["representative_index"] == "59"
    with path.open("a", encoding="utf-8") as output:
        output.write("59\tsource\tDhc\t59\n")
    with pytest.raises(ValueError, match="60 unique"):
        benchmark.load_representatives(path)


def test_child_exact_status_and_certificate_metadata():
    core = nx.to_graph6_bytes(nx.cycle_graph(5), header=False).decode().strip()
    result = benchmark.classify_child(core, 64, 25000, 13)
    assert result["status"] == "EXACT"
    assert result["is_exact"] and not result["is_limited"]
    assert result["certified_spheres"] == {"1": 1}


def test_child_limited_is_not_reinterpreted_as_contractible():
    core = nx.to_graph6_bytes(nx.cycle_graph(5), header=False).decode().strip()
    result = benchmark.classify_child(core, 4, 25000, 13)
    assert result["status"] == "LIMITED"
    assert result["certified_spheres"] is None

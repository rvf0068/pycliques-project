"""Regressions for the order-10 benchmark journal and certified residual types."""

import csv
import gzip
import json
from importlib.resources import files
from itertools import islice

import networkx as nx
import pytest

from research import benchmark_order10_homotopy_cores as benchmark
from research.verify_order10_remaining_types import load_rows, replay_row


def write_representatives(path, indices):
    core = nx.to_graph6_bytes(nx.cycle_graph(5), header=False).decode().strip()
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "representative_index", "graph6", "core_order",
                "core_edges", "core_graph6", "members",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for index in indices:
            writer.writerow({
                "representative_index": str(index),
                "graph6": core,
                "core_order": "5",
                "core_edges": "5",
                "core_graph6": core,
                "members": str(index),
            })


@pytest.fixture
def representatives(tmp_path, monkeypatch):
    path = tmp_path / "representatives.tsv"
    write_representatives(path, sorted(benchmark.EXPECTED_REPRESENTATIVE_INDICES))
    # All synthetic fixtures intentionally replace the pinned blob identity.
    monkeypatch.setattr(
        benchmark, "PHASE1_REPRESENTATIVES_GIT_SHA1",
        benchmark.git_blob_sha1(path.read_bytes()),
    )
    rows = benchmark.load_representatives(path)
    return path, rows


def test_input_requires_exact_phase1_source_indices_and_committed_blob(
    representatives, monkeypatch,
):
    path, rows = representatives
    assert len(rows) == 60
    assert int(rows[0]["representative_index"]) == 1561859
    assert int(rows[-1]["representative_index"]) == 11716425

    # Copilot suggested 0..59, which would be incorrect: these are original
    # McKay census indices, not ordinal positions in representatives.tsv.
    write_representatives(path, range(60))
    monkeypatch.setattr(
        benchmark, "PHASE1_REPRESENTATIVES_GIT_SHA1",
        benchmark.git_blob_sha1(path.read_bytes()),
    )
    with pytest.raises(ValueError, match="exact 60 Phase-1"):
        benchmark.load_representatives(path)

    # Even the right set must match the exact committed Phase-1 input.
    write_representatives(path, sorted(benchmark.EXPECTED_REPRESENTATIVE_INDICES))
    with pytest.raises(ValueError, match="differs from committed"):
        benchmark.load_representatives(path)


def test_input_rejects_duplicate_and_missing_representatives(representatives, monkeypatch):
    path, _rows = representatives
    ids = sorted(benchmark.EXPECTED_REPRESENTATIVE_INDICES)
    write_representatives(path, ids[:-1] + [ids[0]])
    monkeypatch.setattr(
        benchmark, "PHASE1_REPRESENTATIVES_GIT_SHA1",
        benchmark.git_blob_sha1(path.read_bytes()),
    )
    with pytest.raises(ValueError, match="exact 60 Phase-1"):
        benchmark.load_representatives(path)


def exact_record(row):
    core = benchmark.classify_child(row["core_graph6"], 64, 25000, 13)
    return {
        **core,
        "representative_index": int(row["representative_index"]),
        "source_graph6": row["graph6"],
        "core_graph6": row["core_graph6"],
        "members": row["members"].split(";"),
        "elapsed_seconds": 0.02,
    }


def test_resume_validates_full_record_not_just_allowed_index(representatives, tmp_path):
    _path, rows = representatives
    output = tmp_path / "resume.jsonl"
    record = exact_record(rows[0])
    benchmark.write_jsonl_line(output, record)
    assert benchmark.read_completed_records(output, rows) == {
        int(rows[0]["representative_index"])
    }
    assert json.loads(output.read_text()) == record


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_graph6", "different graph"),
        ("core_graph6", "different core"),
        ("members", ["mismatched"]),
        ("status", "UNKNOWN"),
        ("core_order", 999),
        ("core_edges", 999),
        ("is_exact", False),
        ("is_limited", True),
        ("certified_spheres", None),
        ("reason", ""),
        ("elapsed_seconds", -1),
    ],
)
def test_resume_rejects_tampered_record(representatives, tmp_path, field, value):
    _path, rows = representatives
    record = exact_record(rows[0])
    record[field] = value
    output = tmp_path / "tampered.jsonl"
    output.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid benchmark record at line 1"):
        benchmark.read_completed_records(output, rows)


def test_resume_rejects_duplicate_unknown_and_invalid_json(representatives, tmp_path):
    _path, rows = representatives
    record = exact_record(rows[0])
    output = tmp_path / "bad.jsonl"
    benchmark.write_jsonl_line(output, record)
    benchmark.write_jsonl_line(output, record)
    with pytest.raises(RuntimeError, match="line 2"):
        benchmark.read_completed_records(output, rows)

    record["representative_index"] = 123
    output.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unexpected or repeated"):
        benchmark.read_completed_records(output, rows)

    output.write_text("{invalid\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="line 1"):
        benchmark.read_completed_records(output, rows)


@pytest.mark.parametrize("status", ["LIMITED", "BETTI_ONLY", "TIMEOUT", "ERROR"])
def test_resume_accepts_well_formed_inexact_or_failed_records(
    representatives, tmp_path, status,
):
    _path, rows = representatives
    record = exact_record(rows[0])
    record["status"] = status
    if status in {"LIMITED", "BETTI_ONLY"}:
        record["is_exact"] = False
        record["is_limited"] = status == "LIMITED"
        record["certified_spheres"] = None
    else:
        record = {
            key: record[key] for key in (
                "status", "representative_index", "source_graph6",
                "core_graph6", "members", "elapsed_seconds",
            )
        }
        if status == "ERROR":
            record["diagnostic"] = "child process failed"
    output = tmp_path / "inexact.jsonl"
    output.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert benchmark.read_completed_records(output, rows) == {
        int(rows[0]["representative_index"])
    }


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


@pytest.mark.parametrize("row", load_rows(), ids=lambda row: str(row["index"]))
def test_order10_exact_type_and_method(row):
    replay_row(row)


def test_order10_residual_graph6_at_recorded_indices():
    by_index = {row["index"]: row["graph6"] for row in load_rows()}
    dataset = files("pyg6data.data").joinpath("graph10c.g6.gz")
    with (
        dataset.open("rb") as compressed,
        gzip.open(compressed, "rt", encoding="ascii") as source,
    ):
        actual = {
            index: line.rstrip()
            for index, line in enumerate(islice(source, max(by_index) + 1))
            if index in by_index
        }
    assert actual == by_index

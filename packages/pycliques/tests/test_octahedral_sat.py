"""Exact SAT encoding, bounded execution, and independent witness checks."""

from __future__ import annotations

import ctypes.util
import json
import subprocess
import sys

import networkx as nx
import pytest

import pycliques as pc
from pycliques import _octahedral_sat as sat
from pycliques.coaffine_endomorphisms import CoaffineEndomorphismResult
from pycliques.retractions import RetractionSearchStatus as Status

requires_z3 = pytest.mark.skipif(
    not sys.platform.startswith("linux") or ctypes.util.find_library("z3") is None,
    reason="optional Linux system Z3 library unavailable",
)


@requires_z3
def test_sat_matches_portable_search():
    # Exhaustive portable negative answers are independent of the CNF.
    graphs = [g for g in nx.graph_atlas_g() if len(g) == 6]
    graphs += [
        nx.complete_multipartite_graph(2, 2, 2, 2),
        nx.complete_multipartite_graph(3, 2, 2),
        nx.disjoint_union(nx.octahedral_graph(), nx.path_graph(2)),
    ]
    # Exercise the encoding in-process on this small corpus rather than
    # starting a subprocess for each of 159 atlas graphs.
    from pycliques._octahedral_sat_worker import solve

    library = ctypes.util.find_library("z3")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "problem.cnf"
        for graph in graphs:
            complement = nx.complement(graph)
            rows = [sorted(complement[v]) for v in graph]
            expected = pc.find_octahedral_retraction_coaffine(
                graph, backend="python", max_states=100_000
            )
            x = sat._write_cnf(rows, path, 3, 100_000, float("inf"))
            data = solve(library, path, 100_000, 1024, sum(map(len, x)))
            assert data["status"] == expected.status.name
            if expected.status is Status.FOUND:
                selected = set(data["selected"])
                f = {
                    u: next(v for v, literal in row.items() if literal in selected)
                    for u, row in enumerate(x)
                }
                assert pc.verify_coaffine_endomorphism(graph, f)
                assert all(f[f[f[u]]] == f[u] for u in graph)
                assert pc.verify_induced_octahedron(
                    graph, tuple(nx.complement(graph.subgraph(set(f.values()))).edges())
                )


@requires_z3
def test_public_sat_witness_arbitrary_labels_and_dimension():
    graph = nx.relabel_nodes(
        nx.complete_multipartite_graph(2, 2, 2, 2),
        {i: ("v", i) for i in range(8)},
    )
    result = pc.find_octahedral_retraction_coaffine(
        graph, backend="sat", min_dimension=4, time_limit=15
    )
    assert result.status is Status.FOUND
    assert result.dimension == 4
    assert result.retraction == {v: v for v in graph}
    assert result.inclusion == result.retraction
    assert result.endomorphism is not None
    assert result.arc_revisions == result.assignments_tried == 0
    assert pc.verify_coaffine_endomorphism(graph, result.endomorphism)
    assert pc.find_octahedral_retraction_coaffine(
        graph, backend="sat", min_dimension=5
    ).status is Status.EXHAUSTED


@requires_z3
def test_public_sat_negative_and_formula_budget():
    for graph in (nx.complete_graph(6), nx.cycle_graph(6)):
        result = pc.find_octahedral_retraction_coaffine(graph, backend="sat")
        assert result.status is Status.EXHAUSTED
        assert result.retraction is None
    graph = nx.octahedral_graph()
    for kwargs in (
        {"sat_max_clauses": 1},
        {"sat_max_clauses": 10},
        {"sat_max_clauses": 25},
        {"time_limit": 1e-12},
    ):
        result = pc.find_octahedral_retraction_coaffine(
            graph, backend="sat", **kwargs
        )
        assert result.status is Status.LIMIT_REACHED
        assert result.retraction is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "wrong"},
        {"sat_max_clauses": 0},
        {"sat_memory_limit_mb": 0},
        {"sat_memory_limit_mb": 2**32},
        {"value_order": "fiber"},
        {"max_states": 2**32},
    ],
)
def test_sat_argument_validation(kwargs):
    options = {"backend": "sat", **kwargs}
    with pytest.raises(ValueError):
        pc.find_octahedral_retraction_coaffine(nx.octahedral_graph(), **options)


def test_sat_missing_dependency(monkeypatch):
    monkeypatch.setattr(sat.ctypes.util, "find_library", lambda _: None)
    with pytest.raises(RuntimeError, match="Z3 shared library"):
        pc.find_octahedral_retraction_coaffine(nx.octahedral_graph(), backend="sat")


@pytest.mark.parametrize("mode", ["killed", "timeout", "unknown", "oserror"])
def test_worker_limits_are_inconclusive(monkeypatch, mode):
    monkeypatch.setattr(sat.ctypes.util, "find_library", lambda _: "libz3.so.4")
    monkeypatch.setattr(sat.sys, "platform", "linux")

    def run(*args, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(args[0], 1)
        if mode == "oserror":
            raise OSError("cannot start worker")
        if mode == "killed":
            return subprocess.CompletedProcess(args[0], -9, "", "")
        return subprocess.CompletedProcess(
            args[0], 0, json.dumps({"status": "LIMIT_REACHED", "conflicts": 1}), ""
        )

    monkeypatch.setattr(sat.subprocess, "run", run)
    result = pc.find_octahedral_retraction_coaffine(
        nx.octahedral_graph(), backend="sat"
    )
    assert result.status is Status.LIMIT_REACHED
    assert result.retraction is None


def test_reject_incomplete_worker_assignment(monkeypatch):
    monkeypatch.setattr(sat.ctypes.util, "find_library", lambda _: "libz3.so.4")
    monkeypatch.setattr(sat.sys, "platform", "linux")
    monkeypatch.setattr(
        sat.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            json.dumps({"status": "FOUND", "conflicts": 0, "selected": []}),
            "",
        ),
    )
    with pytest.raises(RuntimeError, match="incomplete assignment"):
        pc.find_octahedral_retraction_coaffine(nx.octahedral_graph(), backend="sat")


def test_reject_invalid_sat_certificate(monkeypatch):
    graph = nx.octahedral_graph()
    monkeypatch.setattr(
        sat,
        "search_sat",
        lambda *args, **kwargs: CoaffineEndomorphismResult(
            Status.FOUND, {v: v for v in graph}, 0, 0, 0
        ),
    )
    with pytest.raises(RuntimeError, match="invalid certificate"):
        pc.find_octahedral_retraction_coaffine(graph, backend="sat")


@requires_z3
def test_worker_main(tmp_path, monkeypatch, capsys):
    from pycliques import _octahedral_sat_worker as worker

    path = tmp_path / "small.cnf"
    path.write_text("p cnf 1 1\n1 0\n")
    library = ctypes.util.find_library("z3")
    calls = []
    monkeypatch.setattr(
        worker.resource, "setrlimit", lambda key, limits: calls.append((key, limits))
    )
    monkeypatch.setattr(
        worker.sys, "argv", ["worker", library, str(path), "1000", "1024", "1"]
    )
    worker.main()
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "FOUND"
    assert data["selected"] == [1]
    assert calls == [(worker.resource.RLIMIT_AS, (1024**3, 1024**3))]


def test_encoding_impossible_threshold(tmp_path):
    # Exercise the counter's impossible threshold without using the public
    # size precheck. An empty clause is an immediate contradiction.
    path = tmp_path / "small.cnf"
    sat._write_cnf([[1], [0]], path, 3, 1000, float("inf"))
    assert path.read_text().splitlines()[-1].strip() == "0"

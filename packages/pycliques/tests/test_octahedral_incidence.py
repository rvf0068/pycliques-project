"""Incidence retraction encoding, native walk, and certificate scope."""

from __future__ import annotations

import ctypes.util
import shutil
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pycliques as pc
import pytest
from pycliques import _octahedral_sat as sat
from pycliques import octahedral_incidence as inc
from pycliques.retractions import RetractionSearchStatus as Status

requires_z3 = pytest.mark.skipif(
    not sys.platform.startswith("linux") or ctypes.util.find_library("z3") is None,
    reason="optional Linux Z3 backend unavailable",
)


def cube(d=3):
    return pc.ImplicitCliqueGraph.from_family(
        [
            {v for v in range(2**d) if (v >> i) & 1 == bit}
            for i in range(d)
            for bit in (0, 1)
        ]
    )


def graph_host(graph):
    # Every graph has an intersection representation, with edge atoms and
    # a private atom at each vertex. The private atoms preserve distinctness.
    return pc.ImplicitCliqueGraph.from_family(
        [{("v", u)} | {("e", min(u, v), max(u, v)) for v in graph[u]} for u in graph]
    )


def verify(host, result):
    assert result.status is Status.FOUND
    assert result.antipodal_pairs and result.retraction and result.inclusion
    assert result.endomorphism
    g = host.to_networkx()
    target = g.subgraph(result.retraction.values())
    assert pc.verify_induced_octahedron(g, result.antipodal_pairs)
    assert pc.verify_retraction_witness(
        g, target, (result.retraction, result.inclusion)
    )
    assert pc.verify_coaffine_endomorphism(g, result.endomorphism)


@requires_z3
def test_incidence_exact_and_regional():
    host = cube(4)
    result = pc.find_octahedral_retraction_incidence(host, backend="sat")
    verify(host, result)
    assert result.dimension == 4
    regional = pc.find_octahedral_retraction_incidence(
        host, backend="sat", retained_pairs=[(0, 1)]
    )
    verify(host, regional)
    assert regional.endomorphism[0] == 1
    assert regional.retained_pairs == ((0, 1),)
    # The CNF cannot choose only three pairs on this host.
    assert (
        pc.find_octahedral_retraction_incidence(
            host, backend="sat", min_dimension=5
        ).status
        is Status.EXHAUSTED
    )
    assert (
        pc.find_octahedral_retraction_incidence(
            graph_host(nx.cycle_graph(6)), backend="sat"
        ).status
        is Status.EXHAUSTED
    )


@requires_z3
def test_incidence_encoding_against_portable_atlas(tmp_path):
    from pycliques._octahedral_sat_worker import solve

    path = tmp_path / "instance.cnf"
    library = ctypes.util.find_library("z3")
    for g in nx.graph_atlas_g():
        if len(g) != 6:
            continue
        host = graph_host(g)
        support = host.clique_masks
        rows = [[j for j, mask in enumerate(support) if not s & mask] for s in support]
        x = sat._write_cnf(rows, path, 3, 100_000, float("inf"), supports=support)
        data = solve(library, path, 100_000, 1024, sum(map(len, x)))
        expected = pc.find_octahedral_retraction_coaffine(
            g, backend="python", max_states=100_000
        )
        assert data["status"] == expected.status.name
        if data["status"] == "FOUND":
            selected = set(data["selected"])
            f = {
                u: next(v for v, literal in row.items() if literal in selected)
                for u, row in enumerate(x)
            }
            assert pc.verify_coaffine_endomorphism(g, f)
            assert all(f[f[f[v]]] == f[v] for v in g)
            assert pc.verify_induced_octahedron(
                g, tuple(nx.complement(g.subgraph(set(f.values()))).edges())
            )


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    if shutil.which("g++") is None or not sys.platform.startswith("linux"):
        pytest.skip("GNU C++17 compiler unavailable")
    return pc.build_incidence_native(cache_dir=tmp_path_factory.mktemp("walk"))


def test_walk_blind_and_seeded(executable):
    host = cube()
    for starts in ((), (((0, 1), (2, 3)),)):
        result = pc.find_octahedral_retraction_incidence(
            host,
            executable=executable,
            max_steps=20,
            time_limit=10,
            starting_matchings=starts,
        )
        verify(host, result)
        assert result.uncovered == 0
        assert len(result.partial_cuts) == 3
        assert 1 <= result.states_explored <= 20


def test_sparse_active_labels(executable):
    raw = pc.ImplicitCliqueGraph.from_family(
        [{99}] + [set(cube().clique(i)) for i in cube()]
    )
    # The first support is deliberately inactive, giving noncompact IDs.
    host = raw._with_active(raw.active_mask & ~1)
    result = pc.find_octahedral_retraction_incidence(
        host, executable=executable, max_steps=20, time_limit=10
    )
    verify(host, result)
    assert set(result.retraction) == set(range(1, 7))


def test_walk_limit_and_hybrid(monkeypatch):
    host = cube()
    monkeypatch.setattr(inc, "_walk", lambda *a, **k: ((), 6, 0))
    result = pc.find_octahedral_retraction_incidence(host, executable=Path("dummy"))
    assert result.status is Status.LIMIT_REACHED
    assert result.uncovered == 6

    from pycliques.coaffine_endomorphisms import CoaffineEndomorphismResult

    monkeypatch.setattr(
        sat,
        "search_sat",
        lambda *a, **k: CoaffineEndomorphismResult(Status.EXHAUSTED, None, 4, 0, 0),
    )
    result = pc.find_octahedral_retraction_incidence(
        host, executable=Path("dummy"), backend="hybrid"
    )
    assert result.status is Status.LIMIT_REACHED
    assert result.states_explored == 4


def test_timeout_checkpoint(monkeypatch):
    host = cube()

    def timed_out(*args, **kwargs):
        checkpoint = kwargs["cwd"] / "best.txt"
        checkpoint.write_text(f"4 1 1\n0 1 {host.clique_masks[0]}\n")
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(inc.subprocess, "run", timed_out)
    result = pc.find_octahedral_retraction_incidence(host, executable=Path("dummy"))
    assert result.status is Status.LIMIT_REACHED
    assert result.uncovered == 4
    assert len(result.partial_cuts) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "wrong"},
        {"min_dimension": 2},
        {"max_steps": 0},
        {"max_states": 2**32},
        {"seed": -1},
        {"time_limit": 0},
        {"retained_pairs": [(0, 1)]},
        {"backend": "sat", "starting_matchings": [[(0, 1)]]},
        {"backend": "sat", "retained_pairs": [(0, 2)]},
        {"backend": "sat", "retained_pairs": [(0, 1), (1, 0)]},
        {"backend": "sat", "retained_pairs": [(0, 99)]},
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        pc.find_octahedral_retraction_incidence(cube(), **kwargs)


def test_invalid_cuts_and_checkpoint(monkeypatch):
    host = cube()
    with pytest.raises(ValueError):
        inc._check_cuts(host, ((0, 1, 0),))

    def invalid(*args, **kwargs):
        (kwargs["cwd"] / "best.txt").write_text("0 1 1\n0 99 1\n")
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(inc.subprocess, "run", invalid)
    with pytest.raises(RuntimeError, match="Invalid incidence walk checkpoint"):
        pc.find_octahedral_retraction_incidence(host, executable=Path("dummy"))


def test_empty_and_universal_hosts():
    assert (
        pc.find_octahedral_retraction_incidence(
            pc.ImplicitCliqueGraph.from_family([])
        ).status
        is Status.EXHAUSTED
    )
    host = pc.ImplicitCliqueGraph.from_family(
        [set(cube().clique(i)) for i in cube()] + [set(range(8))]
    )
    assert pc.find_octahedral_retraction_incidence(host).status is Status.EXHAUSTED


def test_width_limit_and_formula_budget():
    host = graph_host(nx.empty_graph(70))
    with pytest.raises(ValueError, match="64 ground atoms"):
        pc.find_octahedral_retraction_incidence(host)
    if ctypes.util.find_library("z3") and sys.platform.startswith("linux"):
        result = pc.find_octahedral_retraction_incidence(
            cube(), backend="sat", sat_max_clauses=1
        )
        assert result.status is Status.LIMIT_REACHED


@requires_z3
def test_regional_exhaustion_is_not_global_exhaustion():
    core = cube()
    host = pc.ImplicitCliqueGraph.from_family([core.clique(i) for i in core] + [{0}])
    verify(host, pc.find_octahedral_retraction_incidence(host, backend="sat"))
    # The singleton has only three neighbors, so it cannot belong to O_3.
    result = pc.find_octahedral_retraction_incidence(
        host, backend="sat", retained_pairs=[(1, 6)]
    )
    assert result.status is Status.EXHAUSTED
    assert result.retained_pairs == ((1, 6),)


@requires_z3
def test_sat_sparse_active_ids_and_wide_ground():
    core = cube()
    raw = pc.ImplicitCliqueGraph.from_family(
        [set(range(100, 170))] + [core.clique(i) for i in core]
    )
    host = raw._with_active(raw.active_mask & ~1)
    result = pc.find_octahedral_retraction_incidence(
        host, backend="sat", retained_pairs=[(1, 2)]
    )
    verify(host, result)
    assert set(result.retraction) == set(range(1, 7))


def test_walk_failure_and_missing_checkpoint(monkeypatch):
    host = cube()

    def killed(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], -9, "", "killed")

    monkeypatch.setattr(inc.subprocess, "run", killed)
    with pytest.raises(RuntimeError, match="Incidence walk failed"):
        pc.find_octahedral_retraction_incidence(host, executable=Path("dummy"))
    monkeypatch.setattr(
        inc.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""),
    )
    result = pc.find_octahedral_retraction_incidence(host, executable=Path("dummy"))
    assert result.status is Status.LIMIT_REACHED
    assert result.uncovered == len(host)


def test_native_build_errors(monkeypatch, tmp_path):
    from pycliques import octahedral_incidence_native as native

    monkeypatch.setattr(native.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="compiler"):
        native.build_incidence_native(cache_dir=tmp_path)
    monkeypatch.setattr(native.shutil, "which", lambda _: "/bin/false")
    monkeypatch.setattr(
        native.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "bad"),
    )
    with pytest.raises(RuntimeError, match="build failed"):
        native.build_incidence_native(cache_dir=tmp_path)
    monkeypatch.setattr(native, "_cached", lambda: tmp_path / "missing")
    with pytest.raises(RuntimeError, match="Build the incidence walk"):
        native.find_incidence_native()


def test_cached_native_discovery(executable, monkeypatch):
    from pycliques import octahedral_incidence_native as native

    monkeypatch.setattr(native, "_cached", lambda: executable)
    assert native.find_incidence_native() == executable
    host = cube()
    result = pc.find_octahedral_retraction_incidence(host, max_steps=20)
    verify(host, result)


def test_reject_nonimplicit_host_and_bad_certificate():
    with pytest.raises(TypeError, match="ImplicitCliqueGraph"):
        pc.find_octahedral_retraction_incidence(nx.octahedral_graph())
    host = cube()
    identity = {v: v for v in host}
    with pytest.raises(RuntimeError, match="invalid certificate"):
        inc._found(host, ((0, 1), (2, 3), (4, 5)), identity, identity, 0, (), (), 3)


def test_checkpoint_incorrect_coverage(monkeypatch):
    host = cube()

    def wrong_count(*args, **kwargs):
        checkpoint = kwargs["cwd"] / "best.txt"
        checkpoint.write_text(f"0 1 1\n0 1 {host.clique_masks[0]}\n")
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(inc.subprocess, "run", wrong_count)
    with pytest.raises(RuntimeError, match="Invalid incidence walk checkpoint"):
        pc.find_octahedral_retraction_incidence(host, executable=Path("dummy"))

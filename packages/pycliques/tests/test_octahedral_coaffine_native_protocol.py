"""Failure-path coverage for the optional coaffine native subprocess protocol."""

from __future__ import annotations

import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pycliques.coaffine_endomorphisms as general
import pycliques.octahedral_coaffine as coaffine
import pycliques.octahedral_coaffine_native as native
import pytest
from pycliques.octahedral_coaffine_native import (
    _cached_executable,
    build_coaffine_native,
    find_coaffine_native_executable,
    run_coaffine_native,
)


def _run_fake(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    *,
    returncode: int = 0,
    stderr: str = "",
    time_limit: float | None = 3.0,
    min_dimension: int = 3,
    value_order: str = "label",
):
    """Exercise the real parser while replacing only native process execution."""
    monkeypatch.setattr(
        native.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        ),
    )
    return run_coaffine_native(
        nx.octahedral_graph(),
        min_dimension=min_dimension,
        max_states=10,
        time_limit=time_limit,
        executable=Path("/not-executed"),
        value_order=value_order,
    )


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("", "no output"),
        ("BAD FOUND 1 0 0", "Malformed native result header"),
        ("RESULT FOUND unknown 0 0", "Non-integer native search counters"),
        ("RESULT FOUND 11 0 0", "Invalid native search counters"),
        ("RESULT FOUND 1 -1 0", "Invalid native search counters"),
        (
            "RESULT EXHAUSTED 1 0 0\nEXTRA",
            "Unexpected witness in non-FOUND native output",
        ),
        ("RESULT FOUND 1 0 0", "Missing native retraction witness fields"),
        (
            "RESULT FOUND 1 0 0\nBAD 0,3\nPROJECTION 0\nENDOMORPHISM 0",
            "Incorrect native witness field names",
        ),
        (
            "RESULT FOUND 1 0 0\nPAIRS 0,broken\n"
            "PROJECTION 0 1 2 3 4 5\nENDOMORPHISM 0 1 2 3 4 5",
            "Malformed native witness indices",
        ),
        (
            "RESULT FOUND 1 0 0\nPAIRS 0,3\n"
            "PROJECTION 0 1 2 3 4 5\nENDOMORPHISM 0 1 2 3 4 5",
            "Native witness has invalid dimensions or labels",
        ),
        (
            "RESULT FOUND 1 0 0\nPAIRS 0,3 1,4 2,5\n"
            "PROJECTION 0 1\nENDOMORPHISM 0 1 2 3 4 5",
            "Native witness has invalid dimensions or labels",
        ),
        (
            "RESULT FOUND 1 0 0\nPAIRS 0,3 1,4 2,9\n"
            "PROJECTION 0 1 2 3 4 5\nENDOMORPHISM 0 1 2 3 4 5",
            "Native witness has invalid dimensions or labels",
        ),
    ],
)
def test_native_protocol_rejects_malformed_output(
    monkeypatch: pytest.MonkeyPatch, output: str, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _run_fake(monkeypatch, output)


def test_native_protocol_accepts_valid_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (
        "RESULT FOUND 1 2 3\n"
        "PAIRS 0,3 1,4 2,5\n"
        "PROJECTION 0 1 2 3 4 5\n"
        "ENDOMORPHISM 3 4 5 0 1 2"
    )
    result = _run_fake(monkeypatch, output)
    assert result.status == "FOUND"
    assert result.states_explored == 1
    assert result.arc_revisions == 2
    assert result.assignments_tried == 3
    assert result.pairs == ((0, 3), (1, 4), (2, 5))
    assert result.projection == (0, 1, 2, 3, 4, 5)
    assert result.endomorphism == (3, 4, 5, 0, 1, 2)


def test_native_protocol_handles_limit_and_no_time_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_fake(monkeypatch, "RESULT LIMIT_REACHED 5 12 9", time_limit=None)
    assert result.status == "LIMIT_REACHED"
    assert result.projection is None
    assert result.states_explored == 5


def test_native_process_failure_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="Native coaffine backend failed"):
        _run_fake(monkeypatch, "", returncode=2, stderr="bad graph data")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="native", timeout=1)

    monkeypatch.setattr(native.subprocess, "run", timeout)
    result = run_coaffine_native(
        nx.octahedral_graph(),
        min_dimension=3,
        max_states=10,
        time_limit=0.01,
        executable=Path("/not-executed"),
    )
    assert result.status == "LIMIT_REACHED"
    assert result.states_explored == 0


def test_native_rejects_graphs_over_capacity() -> None:
    with pytest.raises(ValueError, match="at most 2048"):
        run_coaffine_native(
            nx.empty_graph(2049),
            min_dimension=3,
            max_states=10,
            time_limit=1,
            executable=Path("/not-executed"),
        )


def test_configured_native_executable_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv("PYCLIQUES_COAFFINE_NATIVE", str(missing))
    with pytest.raises(RuntimeError, match="does not identify an executable"):
        find_coaffine_native_executable()

    candidate = tmp_path / "candidate"
    candidate.write_bytes(b"dummy binary")
    monkeypatch.setenv("PYCLIQUES_COAFFINE_NATIVE", str(candidate))
    with pytest.raises(RuntimeError, match="does not identify an executable"):
        find_coaffine_native_executable()
    candidate.chmod(0o755)
    monkeypatch.setenv("PYCLIQUES_COAFFINE_NATIVE", str(candidate))
    assert find_coaffine_native_executable() == candidate


def test_cached_native_executable_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PYCLIQUES_COAFFINE_NATIVE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert find_coaffine_native_executable() is None

    cache = _cached_executable()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"cached binary")
    cache.chmod(0o755)
    assert find_coaffine_native_executable() == cache


def test_native_build_requires_existing_compiler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(native.shutil, "which", lambda compiler: None)
    with pytest.raises(RuntimeError, match="not found"):
        build_coaffine_native(compiler="missing-cxx", cache_dir=tmp_path)


def test_native_build_surfaces_compiler_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(native.shutil, "which", lambda compiler: "/fake/cxx")
    monkeypatch.setattr(
        native.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stderr="intentional test compiler error"
        ),
    )
    with pytest.raises(RuntimeError, match="intentional test compiler error"):
        build_coaffine_native(compiler="c++", cache_dir=tmp_path)


def test_native_found_without_witness_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        coaffine,
        "find_coaffine_native_executable",
        lambda: Path("/not-executed"),
    )
    monkeypatch.setattr(
        coaffine,
        "run_coaffine_native",
        lambda *args, **kwargs: native.NativeCoaffineRawResult("FOUND", 1, 0, 0),
    )
    with pytest.raises(RuntimeError, match="missing a witness"):
        coaffine.find_octahedral_retraction_coaffine(
            nx.octahedral_graph(), backend="native"
        )


def test_native_cli_requires_explicit_build_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the command-line guard without needing a C++ compiler."""
    entrypoint = str(Path(native.__file__))
    monkeypatch.setattr(sys, "argv", ["octahedral_coaffine_native.py"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(entrypoint, run_name="__main__")
    assert caught.value.code == 2


def test_native_cli_builds_real_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the installed CLI exactly as a user invokes --build."""
    if shutil.which("c++") is None:
        pytest.skip("Optional C++17 compiler is not installed")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["octahedral_coaffine_native.py", "--build"])
    runpy.run_path(str(Path(native.__file__)), run_name="__main__")
    built = Path(capsys.readouterr().out.strip())
    assert built.is_file()
    assert built.parent == tmp_path / "pycliques"
    assert built.name.startswith("coaffine-native-")


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("RESULT FOUND 1 0 0", "Missing native endomorphism witness"),
        (
            "RESULT FOUND 1 0 0\nPROJECTION 3 4 5 0 1 2",
            "Missing native endomorphism witness",
        ),
        (
            "RESULT FOUND 1 0 0\nENDOMORPHISM 3 4 bad 0 1 2",
            "Malformed native endomorphism indices",
        ),
        (
            "RESULT FOUND 1 0 0\nENDOMORPHISM 3 4 5",
            "Native endomorphism has invalid labels",
        ),
        (
            "RESULT FOUND 1 0 0\nENDOMORPHISM 3 4 5 0 1 6",
            "Native endomorphism has invalid labels",
        ),
    ],
)
def test_general_protocol_rejects_bad_witnesses(
    monkeypatch: pytest.MonkeyPatch, output: str, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _run_fake(monkeypatch, output, min_dimension=0, value_order="fiber")


def test_general_protocol_accepts_witness(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_fake(
        monkeypatch,
        "RESULT FOUND 2 3 4\nENDOMORPHISM 3 4 5 0 1 2",
        min_dimension=0,
        value_order="fiber",
    )
    assert result.endomorphism == (3, 4, 5, 0, 1, 2)
    assert result.pairs is result.projection is None


def test_native_protocol_invalid_order(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="value_order"):
        _run_fake(monkeypatch, "", value_order="unknown")


def test_general_backend_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(general, "find_coaffine_native_executable", lambda: None)
    graph = nx.cycle_graph(5)
    with pytest.raises(ValueError, match="backend"):
        general.find_coaffine_endomorphism(graph, backend="unknown")
    with pytest.raises(RuntimeError, match="not built"):
        general.find_coaffine_endomorphism(graph, backend="native")
    assert general.find_coaffine_endomorphism(graph).endomorphism is not None

    monkeypatch.setattr(
        general, "find_coaffine_native_executable", lambda: Path("/not-executed")
    )
    with pytest.raises(ValueError, match="at most 2048"):
        general.find_coaffine_endomorphism(nx.empty_graph(2049), backend="native")
    with pytest.raises(ValueError, match="64-bit"):
        general.find_coaffine_endomorphism(
            graph, backend="native", max_states=1 << 65
        )
    assert general.find_coaffine_endomorphism(
        graph, backend="auto", max_states=1 << 65
    ).endomorphism is not None


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        (None, "missing an endomorphism"),
        ((0, 1, 2, 3, 4), "invalid endomorphism"),
    ],
)
def test_general_native_rejects_unverified_witness(
    monkeypatch: pytest.MonkeyPatch,
    mapping: tuple[int, ...] | None,
    message: str,
) -> None:
    monkeypatch.setattr(
        general, "find_coaffine_native_executable", lambda: Path("/not-executed")
    )
    monkeypatch.setattr(
        general,
        "run_coaffine_native",
        lambda *a, **k: NativeCoaffineRawResult(
            "FOUND", 1, 0, 0, endomorphism=mapping
        ),
    )
    with pytest.raises(RuntimeError, match=message):
        general.find_coaffine_endomorphism(nx.cycle_graph(5), backend="native")


def test_general_native_limit_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        general, "find_coaffine_native_executable", lambda: Path("/not-executed")
    )
    monkeypatch.setattr(
        general,
        "run_coaffine_native",
        lambda *a, **k: NativeCoaffineRawResult("LIMIT_REACHED", 2, 3, 4),
    )
    result = general.find_coaffine_endomorphism(nx.cycle_graph(5), backend="native")
    assert result.status is general.RetractionSearchStatus.LIMIT_REACHED
    assert result.endomorphism is None
    assert (result.states_explored, result.arc_revisions, result.assignments_tried) == (
        2,
        3,
        4,
    )

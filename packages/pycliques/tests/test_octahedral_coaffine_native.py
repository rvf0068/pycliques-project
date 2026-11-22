"""Native bitset coaffine search regression tests, independent positive replay."""

from __future__ import annotations

import shutil
from pathlib import Path

import networkx as nx
import pycliques
import pytest
from pycliques import RetractionSearchStatus
from pycliques.octahedral_coaffine_native import (
    NativeCoaffineRawResult,
    build_coaffine_native,
    find_coaffine_native_executable,
)


@pytest.fixture(scope="module")
def native_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("c++") is None:
        pytest.skip("Optional C++17 compiler is not installed")
    return build_coaffine_native(cache_dir=tmp_path_factory.mktemp("coaffine-native"))


def _native(monkeypatch: pytest.MonkeyPatch, binary: Path):
    monkeypatch.setenv("PYCLIQUES_COAFFINE_NATIVE", str(binary))


def test_native_binary_build_and_discovery(
    native_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert native_binary.is_file()
    _native(monkeypatch, native_binary)
    assert find_coaffine_native_executable() == native_binary


def test_native_octahedron_arbitrary_labels(
    native_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _native(monkeypatch, native_binary)
    graph = nx.relabel_nodes(nx.octahedral_graph(), {i: ("v", i) for i in range(6)})
    for backend in ("native", "auto"):
        result = pycliques.find_octahedral_retraction_coaffine(
            graph, backend=backend, time_limit=5
        )
        assert result.status is RetractionSearchStatus.FOUND
        assert result.dimension == 3
        assert result.retraction == {v: v for v in graph}
        assert result.antipodal_pairs is not None
        assert result.endomorphism is not None
        assert all(
            result.retraction[v] == result.endomorphism[result.endomorphism[v]]
            for v in graph
        )
        assert pycliques.verify_induced_octahedron(graph, result.antipodal_pairs)


@pytest.mark.parametrize(
    "graph",
    [
        nx.empty_graph(0),
        nx.complete_graph(8),
        nx.cycle_graph(6),
        nx.path_graph(6),
    ],
)
def test_native_exhaustive_negative_small_cases(
    graph: nx.Graph, native_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _native(monkeypatch, native_binary)
    native = pycliques.find_octahedral_retraction_coaffine(
        graph, backend="native", max_states=5000, time_limit=5
    )
    python = pycliques.find_octahedral_retraction_coaffine(
        graph, backend="python", max_states=5000, time_limit=5
    )
    assert native.status is python.status is RetractionSearchStatus.EXHAUSTED
    assert native.retraction is None


def test_native_min_dimension_restricts_oct(
    native_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _native(monkeypatch, native_binary)
    result = pycliques.find_octahedral_retraction_coaffine(
        nx.octahedral_graph(),
        backend="native",
        min_dimension=4,
        time_limit=5,
    )
    assert result.status is RetractionSearchStatus.EXHAUSTED


def test_native_vs_reference_small_graph_atlas(
    native_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _native(monkeypatch, native_binary)
    for graph in nx.graph_atlas_g()[:210:3]:
        native = pycliques.find_octahedral_retraction_coaffine(
            graph, backend="native", max_states=10_000, time_limit=5
        )
        python = pycliques.find_octahedral_retraction_coaffine(
            graph, backend="python", max_states=10_000, time_limit=5
        )
        assert native.status is python.status, (
            nx.to_graph6_bytes(graph, header=False),
            native,
            python,
        )


def test_native_early_budget_is_not_negative(
    native_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _native(monkeypatch, native_binary)
    graph = nx.disjoint_union(nx.octahedral_graph(), nx.path_graph(2))
    result = pycliques.find_octahedral_retraction_coaffine(
        graph, backend="native", max_states=1, time_limit=5
    )
    assert result.status in (
        RetractionSearchStatus.LIMIT_REACHED,
        RetractionSearchStatus.FOUND,
    )
    assert result.status is not RetractionSearchStatus.EXHAUSTED


def test_invalid_native_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    import pycliques.octahedral_coaffine as module

    monkeypatch.setattr(module, "find_coaffine_native_executable", lambda: None)
    with pytest.raises(RuntimeError, match="not built"):
        module.find_octahedral_retraction_coaffine(
            nx.octahedral_graph(), backend="native"
        )
    with pytest.raises(ValueError, match="backend"):
        module.find_octahedral_retraction_coaffine(
            nx.octahedral_graph(), backend="unknown"
        )
    with pytest.raises(ValueError, match="at most 2048"):
        monkeypatch.setattr(
            module, "find_coaffine_native_executable", lambda: Path("/tmp/fake")
        )
        module.find_octahedral_retraction_coaffine(
            nx.empty_graph(2049), backend="native"
        )


def test_oversized_native_counters_fall_back_under_auto(
    native_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _native(monkeypatch, native_binary)
    graph = nx.octahedral_graph()
    huge_state_limit = 1 << 65
    result = pycliques.find_octahedral_retraction_coaffine(
        graph, backend="auto", max_states=huge_state_limit, time_limit=5
    )
    assert result.status is RetractionSearchStatus.FOUND
    with pytest.raises(ValueError, match="64-bit"):
        pycliques.find_octahedral_retraction_coaffine(
            graph, backend="native", max_states=huge_state_limit
        )
    huge_dimension = 1 << 40
    result = pycliques.find_octahedral_retraction_coaffine(
        graph, backend="auto", min_dimension=huge_dimension
    )
    assert result.status is RetractionSearchStatus.EXHAUSTED
    with pytest.raises(ValueError, match="32-bit"):
        pycliques.find_octahedral_retraction_coaffine(
            graph, backend="native", min_dimension=huge_dimension
        )


def test_reject_corrupt_native_witness(monkeypatch: pytest.MonkeyPatch) -> None:
    import pycliques.octahedral_coaffine as module

    monkeypatch.setattr(
        module, "find_coaffine_native_executable", lambda: Path("/tmp/fake")
    )
    fake = NativeCoaffineRawResult(
        "FOUND",
        1,
        0,
        0,
        ((0, 3), (1, 4), (2, 5)),
        (0, 0, 0, 0, 0, 0),
        (3, 4, 5, 0, 1, 2),
    )
    monkeypatch.setattr(module, "run_coaffine_native", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="invalid certificate"):
        module.find_octahedral_retraction_coaffine(
            nx.octahedral_graph(), backend="native"
        )


@pytest.mark.slow
def test_native_rediscovers_mckay_253650_o12(
    native_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unseeded proof: no antipodal pairs or saved images are provided."""
    _native(monkeypatch, native_binary)
    graph = nx.from_graph6_bytes(b"HEjer~{")
    pared = pycliques.completely_pared_graph(graph)
    for _ in range(4):
        k = pycliques.clique_graph(pared)
        assert k is not None
        k = nx.relabel_nodes(k, {vertex: i for i, vertex in enumerate(k)}, copy=True)
        pared = pycliques.completely_pared_graph(k)
    assert (len(pared), pared.number_of_edges()) == (259, 28890)

    result = pycliques.find_octahedral_retraction_coaffine(
        pared, backend="native", max_states=10_000, time_limit=30
    )
    assert result.status is RetractionSearchStatus.FOUND
    assert result.dimension == 12
    assert result.states_explored <= 10_000
    assert result.retraction is not None
    assert result.antipodal_pairs is not None
    assert result.endomorphism is not None
    assert result.inclusion is not None
    target = pared.subgraph(v for pair in result.antipodal_pairs for v in pair)
    assert pycliques.verify_retraction_witness(
        pared, target, (result.retraction, result.inclusion)
    )
    assert all(
        result.retraction[v] == result.endomorphism[result.endomorphism[v]]
        for v in pared
    )

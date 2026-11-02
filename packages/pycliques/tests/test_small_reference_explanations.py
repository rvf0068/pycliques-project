from collections.abc import Iterator

import networkx as nx
import pytest
from pycliques.named import snub_disphenoid
from pycliques.small import (
    ReferenceStatus,
    Verdict,
    _classify_reference_dependency,
    _classify_reference_graph,
    _load_indeterminate_file_entries_with_metadata,
    _reference_identifier,
    _save_indeterminate,
    _serialize_certificate,
    clear_reference_graphs,
    register_reference_graph,
)


def _restore_builtins() -> None:
    clear_reference_graphs()
    register_reference_graph(
        snub_disphenoid(),
        status=ReferenceStatus.CONJECTURED,
        label="snub_disphenoid",
    )
    register_reference_graph(
        nx.octahedral_graph(),
        status=ReferenceStatus.PROVEN,
        label="octahedral_graph",
    )


@pytest.fixture
def isolated_registry() -> Iterator[None]:
    clear_reference_graphs()
    try:
        yield
    finally:
        _restore_builtins()


def _relabel(graph: nx.Graph) -> nx.Graph:
    return nx.relabel_nodes(graph, {v: f"v{v}" for v in graph})


def _decoded(identifier: str) -> nx.Graph:
    assert identifier.startswith("graph6:")
    assert not any(c.isspace() for c in identifier)
    return nx.from_graph6_bytes(identifier.removeprefix("graph6:").encode("ascii"))


def test_named_proven_direct_reference():
    result = _classify_reference_graph(nx.octahedral_graph())

    assert result is not None
    verdict, reason, cert = result
    assert verdict is Verdict.DIVERGENT
    assert "'octahedral_graph'" in reason
    assert "proven clique divergent" in reason
    assert cert is not None
    assert cert.rule == "reference"
    assert cert.target_status == "proven_divergent"
    assert cert.target_label == "octahedral_graph"


def test_named_conjectured_direct_reference():
    result = _classify_reference_graph(snub_disphenoid())

    assert result is not None
    verdict, reason, cert = result
    assert verdict is Verdict.INDETERMINATE
    assert "'snub_disphenoid'" in reason
    assert "conjectured" in reason
    assert "proven" not in reason
    assert cert is not None
    assert cert.target_status == "conjectured_divergent"
    assert cert.target_label == "snub_disphenoid"


def test_relabeled_input_uses_registered_name():
    result = _classify_reference_graph(_relabel(nx.octahedral_graph()))

    assert result is not None
    assert result[0] is Verdict.DIVERGENT
    assert "'octahedral_graph'" in result[1]


def test_unnamed_proven_direct_reference(isolated_registry):
    ref = nx.petersen_graph()
    register_reference_graph(ref, status=ReferenceStatus.PROVEN)

    result = _classify_reference_graph(ref)
    relabeled = _classify_reference_graph(_relabel(ref))

    assert result is not None and relabeled is not None
    verdict, reason, cert = result
    assert verdict is Verdict.DIVERGENT
    assert cert is not None
    assert cert.target_label is not None
    assert f"'{cert.target_label}'" in reason
    assert "proven clique divergent" in reason
    assert nx.is_isomorphic(_decoded(cert.target_label), ref)
    assert relabeled[1] == reason
    assert relabeled[2] is not None
    assert relabeled[2].target_label == cert.target_label


def test_unnamed_conjectured_direct_reference(isolated_registry):
    ref = nx.petersen_graph()
    register_reference_graph(ref, status=ReferenceStatus.CONJECTURED)

    result = _classify_reference_graph(_relabel(ref))

    assert result is not None
    verdict, reason, cert = result
    assert verdict is Verdict.INDETERMINATE
    assert cert is not None
    assert cert.target_status == "conjectured_divergent"
    assert cert.target_label is not None
    assert cert.target_label.startswith("graph6:")
    assert "conjectured" in reason
    assert "proven" not in reason


def test_reference_identifier_prefers_label():
    assert _reference_identifier(nx.path_graph(3), "name") == "name"
    assert _reference_identifier(nx.path_graph(3), None).startswith("graph6:")


def test_named_retraction_reasons(isolated_registry):
    for ref, status, label, word in [
        (nx.octahedral_graph(), ReferenceStatus.PROVEN, "octahedral_graph", "proven"),
        (
            snub_disphenoid(),
            ReferenceStatus.CONJECTURED,
            "snub_disphenoid",
            "conditional",
        ),
    ]:
        clear_reference_graphs()
        register_reference_graph(ref, status=status, label=label)

        result = _classify_reference_dependency(ref)

        assert result is not None
        _, reason, cert = result
        assert reason.startswith(f"retracts to registered reference '{label}'")
        assert word in reason
        assert "isomorphic" not in reason
        assert cert is not None
        assert cert.rule == "retracts"
        assert cert.map is not None
        assert cert.target_label == label


def test_unnamed_retraction_reason(isolated_registry):
    ref = nx.octahedral_graph()
    register_reference_graph(ref, status=ReferenceStatus.PROVEN)

    result = _classify_reference_dependency(_relabel(ref))

    assert result is not None
    verdict, reason, cert = result
    assert verdict is Verdict.DIVERGENT
    assert cert is not None
    assert cert.rule == "retracts"
    assert cert.map is not None
    assert cert.target_label is not None
    assert cert.target_label.startswith("graph6:")
    assert f"'{cert.target_label}'" in reason
    assert nx.is_isomorphic(_decoded(cert.target_label), ref)


def test_graph6_label_roundtrips_in_indeterminate_file(isolated_registry, tmp_path):
    ref = nx.petersen_graph()
    register_reference_graph(ref, status=ReferenceStatus.CONJECTURED)
    result = _classify_reference_graph(ref)
    assert result is not None
    cert = result[2]
    assert cert is not None

    assert len(_serialize_certificate(cert).split()) == 3

    _save_indeterminate(10, [(3, ref, cert)], tmp_path)
    entries = _load_indeterminate_file_entries_with_metadata(10, tmp_path)

    assert len(entries) == 1
    index, graph, loaded = entries[0]
    assert index == 3
    assert nx.is_isomorphic(graph, ref)
    assert loaded is not None
    assert loaded.rule == "reference"
    assert loaded.target_status == "conjectured_divergent"
    assert loaded.target_label == cert.target_label

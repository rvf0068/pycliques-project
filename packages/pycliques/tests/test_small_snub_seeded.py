"""Optional seeded snub O24 retraction search through the normal classifier."""

import networkx as nx
import pycliques
import pytest
from pycliques.known_octahedral_witnesses import (
    snub_o24_matches_source,
    snub_o24_pairs_for_iterates,
)
from pycliques.small import (
    Verdict,
    _test_iterate_certificates,
    classify_clique_behavior,
)


@pytest.mark.parametrize(
    "presets",
    [
        ("not_registered",),
        ("snub_o24", "snub_o24"),
    ],
)
def test_rejects_unknown_or_duplicate_octahedral_preset(
    presets: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="Unknown or duplicated"):
        classify_clique_behavior(
            nx.from_graph6_bytes(b"GQyuzw"),
            octahedron_retract_presets=presets,
        )


@pytest.mark.parametrize(
    "tries,bound",
    [(4, 1200), (5, 1075)],
)
def test_snub_seed_requires_sufficient_iterates_and_bound(
    tries: int, bound: int
) -> None:
    with pytest.raises(ValueError, match="tries >= 5 and bound >= 1076"):
        classify_clique_behavior(
            nx.from_graph6_bytes(b"GQyuzw"),
            tries=tries,
            bound=bound,
            octahedron_retract_presets=("snub_o24",),
        )


def test_snub_reference_is_now_proven() -> None:
    graph = nx.from_graph6_bytes(b"GQyuzw")
    result = classify_clique_behavior(graph)
    assert result.verdict is Verdict.DIVERGENT
    assert result.certificate is not None
    assert result.certificate.rule == "reference"
    assert result.certificate.target_status == "proven_divergent"


@pytest.mark.slow
def test_snub_seeded_classifier_discovers_new_retraction() -> None:
    """Do not read the saved projection; use only the known O24 embedding."""
    graph = nx.from_graph6_bytes(b"GQyuzw")
    result = classify_clique_behavior(
        graph,
        tries=5,
        bound=1200,
        clockwork_depth=0,
        retraction_depth=0,
        octahedron_retract_presets=("snub_o24",),
        octahedron_max_states=5000,
    )
    assert result.verdict is Verdict.DIVERGENT
    assert result.certificate is not None
    assert result.certificate.rule == "octahedron_retraction_seeded"
    assert result.certificate.iterate_index == 4
    assert result.certificate.target_label == "O_24"
    assert result.certificate.target is not None
    assert len(result.certificate.target) == 48
    assert result.certificate.target_status == "proven_divergent"
    assert result.certificate.map is not None


def test_transport_snub_target_through_nested_clique_labels() -> None:
    """Test the O24 embedding independently of its saved projection."""
    source = nx.from_graph6_bytes(b"GQyuzw")
    assert snub_o24_matches_source(source)
    assert not snub_o24_matches_source(nx.cycle_graph(8))
    iterates = [source]
    for _ in range(4):
        next_graph = pycliques.clique_graph(iterates[-1], bound=1200)
        assert next_graph is not None
        iterates.append(next_graph)
    assert [len(g) for g in iterates] == [8, 12, 20, 56, 1076]
    assert snub_o24_pairs_for_iterates(iterates[:4]) is None
    pairs = snub_o24_pairs_for_iterates(iterates)
    assert pairs is not None
    assert len(pairs) == 24
    vertices = [v for pair in pairs for v in pair]
    assert nx.is_isomorphic(
        iterates[-1].subgraph(vertices), pycliques.octahedron(24)
    )


def test_seeded_classifier_rule_returns_verified_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the seeded classification rule without a huge graph in CI."""
    source = nx.from_graph6_bytes(b"GQyuzw")
    target = pycliques.octahedron(24)
    pairs = [
        (a, b)
        for a, b in nx.non_edges(target)
    ]
    assert len(pairs) == 24
    host = target.copy()
    host.add_edge(48, 0)

    # Keep the public solver real; mock only the expensive iterate sequence
    # and its embedding transport. The separate test above checks that
    # transport against the actual K4(snub).
    monkeypatch.setattr(
        "pycliques.small.snub_o24_pairs_for_iterates",
        lambda _iterates: tuple(pairs),
    )
    sequence = [source, nx.path_graph(1), nx.path_graph(1), nx.path_graph(1), host]
    verdict = _test_iterate_certificates(
        sequence,
        tries=5,
        retraction_depth=0,
        targets=(),
        octahedron_retract_presets=("snub_o24",),
        octahedron_max_states=5000,
    )
    assert verdict is not None
    status, reason, certificate = verdict
    assert status is Verdict.DIVERGENT
    assert "seeded O_24" in reason
    assert certificate is not None
    assert certificate.rule == "octahedron_retraction_seeded"
    assert certificate.target is not None
    assert certificate.map is not None
    assert pycliques.verify_retraction_witness(
        host, certificate.target, certificate.map
    )

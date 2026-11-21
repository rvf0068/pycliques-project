"""Optional seeded snub O24 retraction search through the normal classifier."""

import networkx as nx
import pytest
from pycliques.small import Verdict, classify_clique_behavior


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

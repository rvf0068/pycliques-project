"""Reject inconsistent solver witnesses in opt-in snub searches.

The production solver checks witnesses itself; these tests check that the
classifier also rejects a violated solver contract rather than claiming
a mathematical proof from an invalid certificate.
"""

from collections.abc import Hashable

import networkx as nx
import pytest
from pycliques.known_octahedral_witnesses import _labeled_f44
from pycliques.retractions import BoundedRetractionResult, RetractionSearchStatus
from pycliques.small import CliqueSequence, _test_iterate_certificates


class _SyntheticSnubSequence(CliqueSequence):
    def __init__(self) -> None:
        self.source = _labeled_f44()
        self.host = nx.octahedral_graph()
        super().__init__(self.source)

    def __getitem__(self, index: int) -> nx.Graph:
        return self.source if index == 0 else self.host


@pytest.mark.parametrize("missing", ["projection", "inclusion"])
def test_seeded_retraction_rejects_incomplete_found_result(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    seq = _SyntheticSnubSequence()
    pairs = tuple(nx.non_edges(seq.host))
    monkeypatch.setattr("pycliques.small.snub_o24_pairs_for_iterates", lambda _: pairs)
    projection = None if missing == "projection" else {v: v for v in seq.host}
    inclusion = None if missing == "inclusion" else {v: v for v in seq.host}
    monkeypatch.setattr(
        "pycliques.small.find_octahedral_retraction",
        lambda *_args, **_kwargs: BoundedRetractionResult(
            RetractionSearchStatus.FOUND, projection, inclusion, 1, 0
        ),
    )
    with pytest.raises(AssertionError, match="Missing seeded retraction witness"):
        _test_iterate_certificates(
            seq,
            tries=0,
            retraction_depth=0,
            targets=(),
            octahedron_retract_presets=("snub_o24",),
        )


def test_seeded_retraction_rejects_invalid_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seq = _SyntheticSnubSequence()
    pairs = tuple(nx.non_edges(seq.host))
    monkeypatch.setattr("pycliques.small.snub_o24_pairs_for_iterates", lambda _: pairs)
    projection: dict[Hashable, Hashable] = {v: 0 for v in seq.host}
    inclusion = {v: v for v in seq.host}
    monkeypatch.setattr(
        "pycliques.small.find_octahedral_retraction",
        lambda *_args, **_kwargs: BoundedRetractionResult(
            RetractionSearchStatus.FOUND, projection, inclusion, 1, 0
        ),
    )
    with pytest.raises(AssertionError, match="Seeded O24 witness failed replay"):
        _test_iterate_certificates(
            seq,
            tries=0,
            retraction_depth=0,
            targets=(),
            octahedron_retract_presets=("snub_o24",),
        )

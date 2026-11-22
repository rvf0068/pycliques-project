"""New snub O24 witness is kept separate from blind discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pycliques
import pytest

from research.snub_blind_o24_from_api import reproduce, snub_fourth_iterate

DATA = Path(__file__).parent / "data" / "snub_o24_blind_retraction.json"


def test_new_snub_certificate_is_independently_valid() -> None:
    """Fast certificate replay; the algorithm and search are not invoked."""
    g = snub_fourth_iterate()
    data = json.loads(DATA.read_text())
    pairs = tuple(tuple(pair) for pair in data["target_pairs"])
    projection = dict(zip(data["vertex_order"], data["images"]))
    assert pycliques.verify_induced_octahedron(g, pairs, require_dominating=True)
    target = g.subgraph([v for pair in pairs for v in pair])
    assert pycliques.verify_retraction_witness(
        g, target, (projection, {v: v for pair in pairs for v in pair})
    )


@pytest.mark.slow
def test_snub_o24_is_discovered_without_coordinates(tmp_path: Path) -> None:
    """Expensive full discovery from the graph alone; opt-in research test."""
    certificate = reproduce(output=tmp_path / "retraction.json")
    assert len(certificate["target_pairs"]) == 24
    assert len(certificate["images"]) == 1076

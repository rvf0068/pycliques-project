"""Recheck order-10 graph 2318241 after excluding nondivergent cycle targets.

This reproduces its original pared clique iterate, computes exact mod-2
homology when within the face budget, and searches only cycle complements
with n >= 8. The saved v1 result and environment are not modified.
"""

from __future__ import annotations

import json

import networkx as nx

from pycliques.small import CliqueSequence
from pycliques.targeted_retractions import (
    TargetSearchStatus,
    search_targeted_retractions,
    verify_targeted_retraction,
)
from pycombtop.bounded_betti import bounded_reduced_betti_mod2

ORIGINAL = "I?`ESZv^o"
SAVED_PARED_G1 = "O~~~xepAWJqNO~W~yCuCN"


def main() -> None:
    original = nx.from_graph6_bytes(ORIGINAL.encode("ascii"))
    saved = nx.from_graph6_bytes(SAVED_PARED_G1.encode("ascii"))
    assert len(original) == 10
    assert (len(saved), saved.number_of_edges()) == (16, 70)

    # The saved encoding and independently reconstructed iterate need not
    # have identical vertex labels, but must describe isomorphic graphs.
    calculated = CliqueSequence(original, bound=120)[1]
    if calculated is None or not nx.is_isomorphic(calculated, saved):
        raise AssertionError("Saved first pared iterate disagrees with input graph")

    known = bounded_reduced_betti_mod2(saved, max_faces=150_000)
    mode = "full"
    if known is None:
        known = bounded_reduced_betti_mod2(
            saved, max_faces=150_000, max_degree=3
        )
        mode = "partial" if known is not None else "unknown"
    if known is not None and known.get(1) is not None:
        assert known[1] == 2, "Saved first Betti number is inconsistent"

    result = search_targeted_retractions(
        saved,
        families=("cycle_complements",),
        betti_numbers=known,
        max_target_order=16,
        max_patterns=20_000,
        max_states=100_000,
        max_assignments=250_000,
    )["cycle_complements"]

    found = result.status is TargetSearchStatus.FOUND
    details = {
        "index": 2318241,
        "original_graph6": ORIGINAL,
        "pared_iterate": 1,
        "pared_graph6": SAVED_PARED_G1,
        "exact_homology_mode": mode,
        "betti_known": known,
        "family": result.family,
        "status": result.status.name,
        "reason": result.reason,
        "patterns_checked": result.patterns_checked,
        "induced_targets_checked": result.induced_targets_checked,
        "states_explored": result.states_explored,
        "assignments_tried": result.assignments_tried,
    }
    if found:
        if not verify_targeted_retraction(saved, result):
            raise AssertionError("New positive retraction witness failed replay")
        if result.target_kind is None:
            raise AssertionError("Missing target kind")
        length = int(result.target_kind.split(":")[1])
        if length < 8:
            raise AssertionError("Nondivergent cycle-complement target found")
        vertices = list(saved)
        position = {v: i for i, v in enumerate(vertices)}
        details.update(
            {
                "target_kind": result.target_kind,
                "target_indices": [position[v] for v in result.target_vertices],
                "projection_indices": [
                    position[result.projection[v]] for v in vertices
                ],
            }
        )
    print(json.dumps(details, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

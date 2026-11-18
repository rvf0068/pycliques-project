"""Exact finite-field simplicial homology via GUDHI.

Only mathematical results from a COMPLETE simplex tree are returned. A face
budget overflow raises HomologyResourceLimit, never an invented zero.
Insert FACETS, not an expansion of the one-skeleton; thus nonflag
simplicial complexes are represented correctly.

The separate bounded_betti module independently computes GF(2) homology
from integer-bitset boundary matrices for flag complexes of graphs.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable

import gudhi

DEFAULT_MAX_FACES = 250_000
HOMOLOGY_COEFFICIENT_FIELD = 2


class HomologyResourceLimit(RuntimeError):
    """Exact homology could not complete within the explicit face budget."""


def reduced_betti_facets_mod2(
    facets: Iterable[Iterable[Hashable]],
    *,
    max_faces: int = DEFAULT_MAX_FACES,
) -> list[int]:
    """Return all reduced Betti numbers of a finite facet complex over F_2.

    On an empty complex return an empty profile (the project's convention).
    A zero in the returned profile is certified. No partial profile is ever
    returned when the configured face budget is exceeded.
    """
    if type(max_faces) is not int or max_faces < 1:
        raise ValueError("max_faces must be a positive integer")
    tree = gudhi.SimplexTree()
    indices: dict[Hashable, int] = {}
    top = -1
    for facet in facets:
        vertices = tuple(dict.fromkeys(facet))
        if not vertices:
            continue
        if (1 << len(vertices)) - 1 > max_faces:
            raise HomologyResourceLimit(
                "An individual facet exceeds the exact homology face budget"
            )
        numbered = []
        for vertex in vertices:
            if vertex not in indices:
                indices[vertex] = len(indices)
            numbered.append(indices[vertex])
        tree.insert(numbered)
        if tree.num_simplices() > max_faces:
            raise HomologyResourceLimit(
                "Exact homology simplex tree exceeds the face budget"
            )
        top = max(top, len(vertices) - 1)
    if top == -1:
        return []
    tree.compute_persistence(
        homology_coeff_field=HOMOLOGY_COEFFICIENT_FIELD,
        persistence_dim_max=True,
    )
    ordinary = tree.betti_numbers()
    result = [
        int(ordinary[i]) if i < len(ordinary) else 0
        for i in range(top + 1)
    ]
    result[0] -= 1
    if any(n < 0 for n in result):
        raise AssertionError("Invalid negative reduced Betti number from GUDHI")
    return result

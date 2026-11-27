"""Optional GUDHI backend for exact finite-field simplicial homology.

GUDHI is imported lazily so the pure-Python homology backend can be used
without importing the accelerated library. Face-budget overflow raises
HomologyResourceLimit; no partial profile or invented zero is returned.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable

DEFAULT_MAX_FACES = 250_000
HOMOLOGY_COEFFICIENT_FIELD = 2
GUDHI_MAX_FIELD = 46_337


class HomologyResourceLimit(RuntimeError):
    """Exact homology could not complete within the explicit face budget."""


def _is_prime(number: int) -> bool:
    """Return whether number is prime."""
    if number < 2:
        return False
    if number % 2 == 0:
        return number == 2
    divisor = 3
    while divisor * divisor <= number:
        if number % divisor == 0:
            return False
        divisor += 2
    return True


def gudhi_betti_facets(
    facets: Iterable[Iterable[Hashable]],
    *,
    field: int = HOMOLOGY_COEFFICIENT_FIELD,
    reduced: bool = True,
    max_faces: int = DEFAULT_MAX_FACES,
) -> list[int]:
    """Return exact Betti numbers over a prime field using GUDHI.

    The input consists of maximal faces, not a one-skeleton. The result
    includes every dimension through the dimension of the complex, including
    certified zeros. The face budget counts distinct nonempty simplices.

    GUDHI supports prime coefficient fields at most 46337. Relative homology
    and rational coefficients are handled by the separate pure-Python
    backend.
    """
    if type(max_faces) is not int or max_faces < 1:
        raise ValueError("max_faces must be a positive integer")
    if type(field) is not int or not _is_prime(field):
        raise ValueError("field must be a prime integer")
    if field > GUDHI_MAX_FIELD:
        raise ValueError("GUDHI supports prime coefficient fields at most 46337")
    if type(reduced) is not bool:
        raise TypeError("reduced must be a bool")

    try:
        import gudhi
    except ImportError as error:
        raise ImportError(
            "The GUDHI backend is unavailable; use backend='python' instead"
        ) from error

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
        homology_coeff_field=field,
        persistence_dim_max=True,
    )
    ordinary = tree.betti_numbers()
    result = [int(ordinary[i]) if i < len(ordinary) else 0 for i in range(top + 1)]
    if reduced:
        result[0] -= 1
    if any(number < 0 for number in result):
        raise AssertionError("Invalid negative Betti number returned by GUDHI")
    return result


def reduced_betti_facets_mod2(
    facets: Iterable[Iterable[Hashable]],
    *,
    max_faces: int = DEFAULT_MAX_FACES,
) -> list[int]:
    """Return all reduced Betti numbers over GF(2), using GUDHI.

    This compatibility helper retains the original module-level function.
    New code should use pycombtop.simplicial_homology.
    """
    return gudhi_betti_facets(facets, field=2, reduced=True, max_faces=max_faces)

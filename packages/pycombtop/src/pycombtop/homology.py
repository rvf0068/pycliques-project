"""Unified exact simplicial homology over fields.

The portable backend uses sparse exact column reduction over prime fields or
the rationals. GUDHI is an accelerated backend for absolute prime-field
homology.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Literal

from .homology_backend import (
    DEFAULT_MAX_FACES,
    GUDHI_MAX_FIELD,
    HomologyResourceLimit,
    gudhi_betti_facets,
)
from .simplex import SimplicialComplex
from .simplicial_maps import _ordered_vertices

CoefficientField = int | Literal["Q"]
HomologyBackend = Literal["auto", "python", "gudhi"]


@dataclass(frozen=True, slots=True)
class HomologyResult:
    """Exact simplicial homology data.

    betti_numbers[d] is the dimension of homology in degree d. The tuple
    includes certified zeros through the top nonempty chain dimension; it is
    empty when there are no nonempty simplices.
    """

    betti_numbers: tuple[int, ...]
    field: CoefficientField
    reduced: bool
    relative: bool
    backend: Literal["python", "gudhi"]

    @property
    def max_dimension(self) -> int:
        """Return the highest degree represented, or -1 for no chains."""
        return len(self.betti_numbers) - 1

    def betti_number(self, degree: int) -> int:
        """Return the Betti number in a nonnegative degree."""
        if type(degree) is not int or degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        if degree >= len(self.betti_numbers):
            return 0
        return self.betti_numbers[degree]


def _is_prime(number: int) -> bool:
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


def _normalize_field(field: CoefficientField) -> CoefficientField:
    if isinstance(field, str) and field == "Q":
        return "Q"
    if type(field) is not int or not _is_prime(field):
        raise ValueError("field must be 'Q' or a prime integer")
    return field


def _enumerate_faces(
    facets: Iterable[Iterable[Hashable]],
    vertex_ranks: dict[Hashable, int],
    *,
    max_faces: int | None,
) -> dict[int, set[frozenset[Hashable]]]:
    """Enumerate distinct nonempty faces, grouped by dimension."""
    by_dimension: dict[int, set[frozenset[Hashable]]] = {}
    count = 0
    for facet in facets:
        vertices = tuple(sorted(facet, key=vertex_ranks.__getitem__))
        for size in range(1, len(vertices) + 1):
            faces = by_dimension.setdefault(size - 1, set())
            for face in combinations(vertices, size):
                signature = frozenset(face)
                if signature in faces:
                    continue
                faces.add(signature)
                count += 1
                if max_faces is not None and count > max_faces:
                    raise HomologyResourceLimit(
                        "Exact homology face enumeration exceeds the face budget"
                    )
    return by_dimension


def _rank_mod_prime(columns: Iterable[dict[int, int]], prime: int) -> int:
    """Compute sparse column rank over GF(prime)."""
    pivots: dict[int, dict[int, int]] = {}
    for column in columns:
        vector = {row: value % prime for row, value in column.items() if value % prime}
        while vector:
            pivot = max(vector)
            coefficient = vector[pivot]
            pivot_column = pivots.get(pivot)
            if pivot_column is None:
                inverse = pow(coefficient, -1, prime)
                pivots[pivot] = {
                    row: value * inverse % prime for row, value in vector.items()
                }
                break
            for row, value in pivot_column.items():
                reduced = (vector.get(row, 0) - coefficient * value) % prime
                if reduced:
                    vector[row] = reduced
                else:
                    vector.pop(row, None)
    return len(pivots)


def _rank_over_rationals(columns: Iterable[dict[int, int]]) -> int:
    """Compute sparse column rank over Q using exact Fraction arithmetic."""
    pivots: dict[int, dict[int, Fraction]] = {}
    zero = Fraction(0)
    for column in columns:
        vector = {row: Fraction(value) for row, value in column.items() if value}
        while vector:
            pivot = max(vector)
            coefficient = vector[pivot]
            pivot_column = pivots.get(pivot)
            if pivot_column is None:
                pivots[pivot] = {
                    row: value / coefficient for row, value in vector.items()
                }
                break
            for row, value in pivot_column.items():
                reduced = vector.get(row, zero) - coefficient * value
                if reduced:
                    vector[row] = reduced
                else:
                    vector.pop(row, None)
    return len(pivots)


def _boundary_columns(
    basis: tuple[tuple[Hashable, ...], ...],
    lower_positions: dict[frozenset[Hashable], int],
) -> Iterable[dict[int, int]]:
    """Yield oriented boundary columns, projecting relative faces to zero."""
    for simplex in basis:
        column: dict[int, int] = {}
        for position in range(len(simplex)):
            face = frozenset(simplex[:position] + simplex[position + 1 :])
            row = lower_positions.get(face)
            if row is not None:
                column[row] = 1 if position % 2 == 0 else -1
        yield column


def _python_betti_numbers(
    complex_: SimplicialComplex,
    *,
    field: CoefficientField,
    reduced: bool,
    relative_to: SimplicialComplex | None,
    vertex_order: tuple[Hashable, ...],
    max_faces: int,
) -> tuple[int, ...]:
    """Compute exact Betti numbers from the simplicial boundary complex."""
    vertex_ranks = {vertex: index for index, vertex in enumerate(vertex_order)}
    ambient_faces = _enumerate_faces(
        complex_.facet_set, vertex_ranks, max_faces=max_faces
    )

    relative_faces: dict[int, set[frozenset[Hashable]]] = {}
    if relative_to is not None:
        for facet in relative_to.facet_set:
            if not any(facet <= ambient for ambient in complex_.facet_set):
                raise ValueError("relative_to must be a subcomplex of complex_")
        relative_faces = _enumerate_faces(
            relative_to.facet_set, vertex_ranks, max_faces=None
        )

    basis: dict[int, tuple[tuple[Hashable, ...], ...]] = {}
    for degree, faces in ambient_faces.items():
        quotient_faces = faces - relative_faces.get(degree, set())
        if quotient_faces:
            basis[degree] = tuple(
                tuple(sorted(face, key=vertex_ranks.__getitem__))
                for face in sorted(
                    quotient_faces,
                    key=lambda simplex: tuple(
                        sorted(vertex_ranks[vertex] for vertex in simplex)
                    ),
                )
            )

    if not basis:
        return ()

    top = max(basis)
    boundary_ranks = [0] * (top + 2)
    has_augmentation = reduced and (relative_to is None or not relative_to.facet_set)
    if has_augmentation and basis.get(0):
        boundary_ranks[0] = 1

    for degree in range(1, top + 1):
        lower_basis = basis.get(degree - 1, ())
        lower_positions = {
            frozenset(simplex): index for index, simplex in enumerate(lower_basis)
        }
        columns = _boundary_columns(basis.get(degree, ()), lower_positions)
        if field == "Q":
            boundary_ranks[degree] = _rank_over_rationals(columns)
        else:
            assert type(field) is int
            boundary_ranks[degree] = _rank_mod_prime(columns, field)

    betti_numbers = tuple(
        len(basis.get(degree, ())) - boundary_ranks[degree] - boundary_ranks[degree + 1]
        for degree in range(top + 1)
    )
    if any(number < 0 for number in betti_numbers):
        raise AssertionError("Invalid negative Betti number from boundary reduction")
    return betti_numbers


def simplicial_homology(
    complex_: SimplicialComplex,
    *,
    field: CoefficientField = 2,
    reduced: bool = True,
    relative_to: SimplicialComplex | None = None,
    backend: HomologyBackend = "auto",
    vertex_order: Iterable[Hashable] | None = None,
    max_faces: int = DEFAULT_MAX_FACES,
) -> HomologyResult:
    """Compute exact simplicial homology of a finite complex or pair.

    Parameters
    ----------
    complex_:
        The finite abstract simplicial complex, represented by its facets.
        Ambient ghost labels do not contribute simplices or chains.
    field:
        2 (the default) computes over GF(2); any prime integer p computes
        over GF(p); "Q" computes over the rationals. These are field
        coefficients, not integral homology, so torsion is not reported.
    reduced:
        Compute reduced homology (default) rather than ordinary homology.
        For a relative pair this is the reduced relative chain complex; it
        agrees with ordinary relative homology when the subcomplex contains
        the empty face, as every non-void simplicial complex does.
    relative_to:
        Optional subcomplex A; when supplied, compute the homology of
        (X, A). Every facet of A must be a face of X. The void
        complex (no faces) is a valid subcomplex.
    backend:
        "auto" uses GUDHI for absolute prime-field calculations when it is
        available and supports the field, and otherwise uses "python".
        "python" selects the portable exact sparse boundary reducer.
        "gudhi" explicitly requests the accelerated backend, which supports
        absolute homology over prime fields only.
    vertex_order:
        Optional ordering of the actual vertices. The default uses the same
        stable-label convention as the simplicial chain-basis API. Supplying
        an order is useful for labels whose stable representations collide.
    max_faces:
        Maximum number of distinct nonempty simplices in the ambient complex.
        Exceeding the budget raises HomologyResourceLimit; no partial result
        is returned.

    Returns
    -------
    HomologyResult
        Betti numbers, coefficient field, reduced/relative flags, and the
        backend used. Certified zeros are retained through the top chain
        dimension. The void and empty-face complexes both have an empty
        nonnegative-dimensional profile, matching the package convention.

    Notes
    -----
    The pure-Python backend uses exact arithmetic over GF(p) or Q and requires
    no numerical linear algebra. Rational Betti numbers do not detect torsion
    in integral homology. The current face budget is intended for explicit
    finite complexes, not persistence-scale filtrations.
    """
    if not isinstance(complex_, SimplicialComplex):
        raise TypeError("complex_ must be a SimplicialComplex")
    if relative_to is not None and not isinstance(relative_to, SimplicialComplex):
        raise TypeError("relative_to must be a SimplicialComplex or None")
    if type(reduced) is not bool:
        raise TypeError("reduced must be a bool")
    if type(max_faces) is not int or max_faces < 1:
        raise ValueError("max_faces must be a positive integer")
    if backend not in ("auto", "python", "gudhi"):
        raise ValueError("backend must be 'auto', 'python', or 'gudhi'")

    field = _normalize_field(field)
    explicit_order = None if vertex_order is None else tuple(vertex_order)
    ordered_vertices = _ordered_vertices(complex_.vertex_set, explicit_order)

    can_use_gudhi = (
        relative_to is None
        and type(field) is int
        and field <= GUDHI_MAX_FIELD
        and backend in ("auto", "gudhi")
    )
    if backend == "gudhi" and not can_use_gudhi:
        raise ValueError(
            "the GUDHI backend supports absolute homology over prime fields "
            "at most 46337"
        )

    if can_use_gudhi:
        assert type(field) is int
        try:
            betti_numbers = gudhi_betti_facets(
                complex_.facet_set,
                field=field,
                reduced=reduced,
                max_faces=max_faces,
            )
        except ImportError:
            if backend == "gudhi":
                raise
        else:
            return HomologyResult(
                tuple(betti_numbers),
                field,
                reduced,
                False,
                "gudhi",
            )

    betti_numbers = _python_betti_numbers(
        complex_,
        field=field,
        reduced=reduced,
        relative_to=relative_to,
        vertex_order=ordered_vertices,
        max_faces=max_faces,
    )
    return HomologyResult(
        betti_numbers,
        field,
        reduced,
        relative_to is not None,
        "python",
    )

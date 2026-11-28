"""Exact homology characters for finite permutation actions.

The acting group is represented by a pycombtop.PermutationAction.
Only the requested group element is evaluated; the group is never enumerated.
All chain and image-subspace calculations use exact rational linear algebra.
"""

from __future__ import annotations

from sympy import Matrix, SparseMatrix
from sympy.combinatorics import Permutation

from pycombtop import PermutationAction, SimplicialComplex


def _orientation_sign(values: tuple[int, ...]) -> int:
    inversions = sum(
        values[i] > values[j]
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )
    return -1 if inversions % 2 else 1


class EquivariantHomology:
    """Cache exact chain data and evaluate characters on rational homology.

    Parameters
    ----------
    complex_:
        A finite simplicial complex.
    action:
        A permutation action on exactly the vertices of complex_.
    reduced:
        If true, use augmented chains and compute reduced homology.

    Notes
    -----
    For each degree, the character is computed from the exact identity
    tr(H_p) = tr(C_p) - tr(im(d_p)) - tr(im(d_{p+1})).
    The trace on an image is calculated in an exact basis of that invariant
    subspace. No floating-point SVD or group enumeration is used.
    """

    def __init__(
        self,
        complex_: SimplicialComplex,
        action: PermutationAction,
        *,
        reduced: bool = False,
    ) -> None:
        action.validate_complex(complex_)
        self.complex = complex_
        self.action = action
        self.reduced = reduced
        self._faces_cache: dict[int, tuple[tuple[int, ...], ...]] = {}
        self._face_index_cache: dict[int, dict[tuple[int, ...], int]] = {}
        self._boundary_cache: dict[int, SparseMatrix] = {}
        self._chain_cache: dict[tuple[tuple[int, ...], int], SparseMatrix] = {}
        self._image_trace_cache: dict[tuple[tuple[int, ...], int], int] = {}

    def _faces(self, degree: int) -> tuple[tuple[int, ...], ...]:
        if degree not in self._faces_cache:
            if degree == -1 and self.reduced:
                faces = ((),)
            elif degree < 0 or degree > self.complex.dimension():
                faces = ()
            else:
                faces = tuple(
                    sorted(
                        tuple(sorted(self.action.index[v] for v in face))
                        for face in self.complex.faces_of_dimension(degree)
                    )
                )
            self._faces_cache[degree] = faces
            self._face_index_cache[degree] = {
                face: i for i, face in enumerate(faces)
            }
        return self._faces_cache[degree]

    def _normalize(self, permutation: Permutation) -> Permutation:
        if not isinstance(permutation, Permutation):
            raise TypeError("permutation must be a SymPy Permutation")
        if permutation.size > len(self.action):
            raise ValueError("permutation degree exceeds the action-set size")
        normalized = Permutation(permutation.array_form, size=len(self.action))
        if not self.action.group.contains(normalized):
            raise ValueError("permutation is not in the acting group")
        return normalized

    def _chain_matrix(
        self, permutation: Permutation, degree: int
    ) -> SparseMatrix:
        key = (tuple(permutation.array_form), degree)
        if key not in self._chain_cache:
            faces = self._faces(degree)
            if degree == -1 and self.reduced:
                matrix = SparseMatrix(1, 1, {(0, 0): 1})
            else:
                index = self._face_index_cache[degree]
                entries: dict[tuple[int, int], int] = {}
                for column, face in enumerate(faces):
                    raw = tuple(permutation(i) for i in face)
                    image = tuple(sorted(raw))
                    row = index[image]
                    entries[row, column] = _orientation_sign(raw)
                matrix = SparseMatrix(len(faces), len(faces), entries)
            self._chain_cache[key] = matrix
        return self._chain_cache[key]

    def _boundary_matrix(self, degree: int) -> SparseMatrix:
        """Boundary C_degree -> C_(degree-1), including augmentation."""
        if degree not in self._boundary_cache:
            source = self._faces(degree)
            if degree == 0:
                if self.reduced:
                    matrix = SparseMatrix(
                        1, len(source), {(0, j): 1 for j in range(len(source))}
                    )
                else:
                    matrix = SparseMatrix(0, len(source), {})
            elif degree < 0:
                matrix = SparseMatrix(0, 0, {})
            else:
                self._faces(degree - 1)
                target_index = self._face_index_cache[degree - 1]
                entries = {}
                for column, face in enumerate(source):
                    for i in range(len(face)):
                        boundary_face = face[:i] + face[i + 1 :]
                        entries[target_index[boundary_face], column] = (-1) ** i
                matrix = SparseMatrix(
                    len(self._faces(degree - 1)), len(source), entries
                )
            self._boundary_cache[degree] = matrix
        return self._boundary_cache[degree]

    def _image_character(
        self, permutation: Permutation, degree: int
    ) -> int:
        """Trace of the group element on im(d_degree)."""
        key = (tuple(permutation.array_form), degree)
        if key in self._image_trace_cache:
            return self._image_trace_cache[key]

        boundary = self._boundary_matrix(degree)
        basis_vectors = boundary.columnspace()
        if not basis_vectors:
            result = 0
        else:
            basis = Matrix.hstack(*basis_vectors)
            rank = basis.cols
            action_matrix = self._chain_matrix(permutation, degree - 1)
            transformed = action_matrix * basis
            pivot_rows = basis.T.rref()[1]
            square = basis.extract(pivot_rows, range(rank))
            coordinates = square.inv() * transformed.extract(pivot_rows, range(rank))
            result = int(coordinates.trace())
        self._image_trace_cache[key] = result
        return result

    def character(self, permutation: Permutation, degree: int) -> int:
        """Return the exact character on H_degree (or reduced homology).

        permutation must belong to the acting SymPy permutation group.
        Degrees outside the homological range return zero.
        """
        if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        element = self._normalize(permutation)
        if degree > self.complex.dimension():
            return 0
        chain_trace = self._chain_matrix(element, degree).trace()
        value = (
            chain_trace
            - self._image_character(element, degree)
            - self._image_character(element, degree + 1)
        )
        integer_value = int(value)
        if value != integer_value:
            raise ArithmeticError("finite-group homology character was not integral")
        return integer_value

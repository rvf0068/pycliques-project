"""Exact homology characters for finite permutation actions.

The acting group is represented by a pycombtop.PermutationAction.
Only the requested group element is evaluated; the group is never enumerated.
All chain and image-subspace calculations use exact rational linear algebra.
"""

from __future__ import annotations

from pycombtop import (
    PermutationAction,
    SimplicialComplex,
    SimplicialMap,
    induced_chain_map,
)
from sympy import Matrix, SparseMatrix
from sympy.combinatorics import Permutation


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
        A permutation action on exactly the vertices of the given complex.
    reduced:
        If true, use augmented chains and compute reduced homology. For a
        nonempty relative subcomplex, relative chains are not augmented.
    relative_to:
        Optional invariant subcomplex A; when supplied, compute homology of
        the pair (X, A).

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
        relative_to: SimplicialComplex | None = None,
    ) -> None:
        if relative_to is not None and not isinstance(relative_to, SimplicialComplex):
            raise TypeError("relative_to must be a SimplicialComplex or None")
        if relative_to is None:
            action.validate_complex(complex_)
        else:
            action.validate_invariant_subcomplex(complex_, relative_to)
        if type(reduced) is not bool:
            raise TypeError("reduced must be a bool")
        self.complex = complex_
        self.action = action
        self.reduced = reduced
        self.relative_to = relative_to
        self._faces_cache: dict[int, tuple[tuple[int, ...], ...]] = {}
        self._face_index_cache: dict[int, dict[tuple[int, ...], int]] = {}
        self._boundary_cache: dict[int, SparseMatrix] = {}
        self._chain_cache: dict[tuple[tuple[int, ...], int], SparseMatrix] = {}
        self._image_trace_cache: dict[tuple[tuple[int, ...], int], int] = {}
        self._image_basis_cache: dict[
            int, tuple[Matrix, tuple[int, ...], Matrix] | None
        ] = {}
        self._homology_basis_cache: dict[
            int, tuple[Matrix, Matrix, tuple[int, ...], Matrix]
        ] = {}

    def _faces(self, degree: int) -> tuple[tuple[int, ...], ...]:
        if degree not in self._faces_cache:
            if (
                degree == -1
                and self.reduced
                and (self.relative_to is None or not self.relative_to.facet_set)
            ):
                faces = ((),)
            elif degree < 0 or degree > self.complex.dimension():
                faces = ()
            else:
                relative_faces = (
                    set()
                    if self.relative_to is None
                    else set(self.relative_to.faces_of_dimension(degree))
                )
                faces = tuple(
                    sorted(
                        tuple(sorted(self.action.index[v] for v in face))
                        for face in self.complex.faces_of_dimension(degree)
                        if face not in relative_faces
                    )
                )
            self._faces_cache[degree] = faces
            self._face_index_cache[degree] = {face: i for i, face in enumerate(faces)}
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

    def _chain_matrix(self, permutation: Permutation, degree: int) -> SparseMatrix:
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
                has_augmentation = self.reduced and (
                    self.relative_to is None or not self.relative_to.facet_set
                )
                if has_augmentation:
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
                        row = target_index.get(boundary_face)
                        if row is not None:
                            entries[row, column] = (-1) ** i
                matrix = SparseMatrix(
                    len(self._faces(degree - 1)), len(source), entries
                )
            self._boundary_cache[degree] = matrix
        return self._boundary_cache[degree]

    def _image_basis(
        self, degree: int
    ) -> tuple[Matrix, tuple[int, ...], Matrix] | None:
        """Cache a basis and coordinate extractor for im(d_degree)."""
        if degree not in self._image_basis_cache:
            basis_vectors = self._boundary_matrix(degree).columnspace()
            if not basis_vectors:
                self._image_basis_cache[degree] = None
            else:
                basis = Matrix.hstack(*basis_vectors)
                rank = basis.cols
                pivot_rows = tuple(basis.T.rref()[1])
                coordinate_extractor = basis.extract(pivot_rows, range(rank)).inv()
                self._image_basis_cache[degree] = (
                    basis,
                    pivot_rows,
                    coordinate_extractor,
                )
        return self._image_basis_cache[degree]

    def _image_character(self, permutation: Permutation, degree: int) -> int:
        """Trace of the group element on im(d_degree)."""
        key = (tuple(permutation.array_form), degree)
        if key not in self._image_trace_cache:
            image_basis = self._image_basis(degree)
            if image_basis is None:
                result = 0
            else:
                basis, pivot_rows, coordinate_extractor = image_basis
                transformed = self._chain_matrix(permutation, degree - 1) * basis
                coordinates = coordinate_extractor * transformed.extract(
                    pivot_rows, range(basis.cols)
                )
                result = int(coordinates.trace())
            self._image_trace_cache[key] = result
        return self._image_trace_cache[key]

    def _homology_basis(
        self, degree: int
    ) -> tuple[Matrix, Matrix, tuple[int, ...], Matrix]:
        """Return boundary and homology bases with an exact coordinate extractor.

        The returned tuple is (B, H, pivot_rows, inverse), where columns of B
        form a basis of boundaries, columns of H complete them to a basis of
        cycles, and inverse extracts coordinates in the combined [B | H]
        basis from the selected ambient coordinates.
        """
        if degree not in self._homology_basis_cache:
            boundary_vectors = self._boundary_matrix(degree + 1).columnspace()
            chain_dimension = len(self._faces(degree))
            boundaries = (
                Matrix.hstack(*boundary_vectors)
                if boundary_vectors
                else Matrix(chain_dimension, 0, [])
            )
            cycles = self._boundary_matrix(degree).nullspace()
            homology_columns: list[Matrix] = []
            combined = boundaries
            rank = combined.cols
            for cycle in cycles:
                candidate = combined.row_join(cycle)
                new_rank = candidate.rank()
                if new_rank > rank:
                    homology_columns.append(cycle)
                    combined = candidate
                    rank = new_rank
            homology = (
                Matrix.hstack(*homology_columns)
                if homology_columns
                else Matrix(chain_dimension, 0, [])
            )
            if combined.cols:
                pivot_rows = tuple(combined.T.rref()[1])
                inverse = combined.extract(pivot_rows, range(combined.cols)).inv()
            else:
                pivot_rows = ()
                inverse = Matrix(0, 0, [])
            self._homology_basis_cache[degree] = (
                boundaries,
                homology,
                pivot_rows,
                inverse,
            )
        return self._homology_basis_cache[degree]

    def representation_matrix(self, permutation: Permutation, degree: int) -> Matrix:
        """Return the exact matrix of a group element on rational homology.

        The basis is computed deterministically from the ordered simplex
        basis: a basis of boundaries is extended by cycle vectors to a basis
        of cycles. Coordinates are taken modulo boundaries. No group
        enumeration or numerical linear algebra is used.
        """
        if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        element = self._normalize(permutation)
        if degree > self.complex.dimension():
            return Matrix(0, 0, [])
        boundaries, homology, pivot_rows, inverse = self._homology_basis(degree)
        if homology.cols == 0:
            return Matrix(0, 0, [])
        transformed = self._chain_matrix(element, degree) * homology
        coordinates = inverse * transformed.extract(pivot_rows, range(homology.cols))
        result = coordinates[boundaries.cols :, :]
        if result.rows != homology.cols or result.cols != homology.cols:
            raise ArithmeticError("failed to obtain homology coordinates")
        return result

    def _validate_relative_map(
        self, simplicial_map: SimplicialMap, target: EquivariantHomology
    ) -> None:
        """Check that the simplicial map sends the source subcomplex into target."""
        if self.relative_to is None or not self.relative_to.facet_set:
            return
        if target.relative_to is None:
            raise ValueError(
                "simplicial_map must send the relative subcomplex into target"
            )
        target_faces = target.relative_to.all_simplices()
        if any(
            frozenset(simplicial_map(vertex) for vertex in facet) not in target_faces
            for facet in self.relative_to.facet_set
        ):
            raise ValueError(
                "simplicial_map must send the relative subcomplex into target"
            )

    def _induced_chain_map(
        self,
        target: EquivariantHomology,
        simplicial_map: SimplicialMap,
        degree: int,
    ) -> SparseMatrix:
        """Return the chain map projected to the relative quotient bases."""
        ambient = induced_chain_map(
            simplicial_map,
            degree,
            source_order=self.action.elements,
            target_order=target.action.elements,
        )
        source_basis = tuple(
            sorted(
                tuple(sorted(self.action.index[vertex] for vertex in face))
                for face in self.complex.faces_of_dimension(degree)
            )
        )
        target_basis = tuple(
            sorted(
                tuple(sorted(target.action.index[vertex] for vertex in face))
                for face in target.complex.faces_of_dimension(degree)
            )
        )
        source_positions = {face: index for index, face in enumerate(source_basis)}
        target_positions = {face: index for index, face in enumerate(target_basis)}
        source_rows = [source_positions[face] for face in self._faces(degree)]
        target_rows = [target_positions[face] for face in target._faces(degree)]
        return ambient.extract(target_rows, source_rows)

    def induced_homology_map(
        self, simplicial_map: SimplicialMap, degree: int
    ) -> Matrix:
        """Return the exact map induced on rational homology by an equivariant self-map.

        The simplicial map must have this complex as both source and target,
        and it must commute with every generator of the stored permutation
        action. The returned matrix uses the deterministic homology basis
        used by :meth:`representation_matrix`. For reduced homology, the
        augmented chain map is used in degree zero as well.

        This initial interface deliberately treats self-maps only: relating
        actions on different vertex sets requires an explicit identification
        of their acting groups, which PermutationAction does not currently
        encode.
        """
        if not isinstance(simplicial_map, SimplicialMap):
            raise TypeError("simplicial_map must be a SimplicialMap instance")
        if (
            simplicial_map.source.vertex_set != self.complex.vertex_set
            or simplicial_map.source.facet_set != self.complex.facet_set
            or simplicial_map.target.vertex_set != self.complex.vertex_set
            or simplicial_map.target.facet_set != self.complex.facet_set
        ):
            raise ValueError(
                "simplicial_map must be a self-map of the acted-on complex"
            )
        self._validate_relative_map(simplicial_map, self)
        if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        for generator in self.action.generators:
            for vertex in self.action.elements:
                image = self.action.elements[generator(self.action.index[vertex])]
                if (
                    simplicial_map(image)
                    != self.action.elements[
                        generator(self.action.index[simplicial_map(vertex)])
                    ]
                ):
                    raise ValueError(
                        "simplicial_map must be equivariant under each action generator"
                    )
        if degree > self.complex.dimension():
            return Matrix(0, 0, [])

        boundaries, homology, pivot_rows, inverse = self._homology_basis(degree)
        if homology.cols == 0:
            return Matrix(0, 0, [])
        chain_map = self._induced_chain_map(self, simplicial_map, degree)
        transformed = chain_map * homology
        coordinates = inverse * transformed.extract(pivot_rows, range(homology.cols))
        result = coordinates[boundaries.cols :, :]
        if result.rows != homology.cols or result.cols != homology.cols:
            raise ArithmeticError("failed to obtain induced homology coordinates")
        return result

    def induced_homology_map_to(
        self,
        target: EquivariantHomology,
        simplicial_map: SimplicialMap,
        degree: int,
    ) -> Matrix:
        """Return the exact map induced by an equivariant map to another complex.

        The source action must be a subgroup of the target action in the
        symmetric group on their ordered vertex positions. Thus the elements
        sequences specify an explicit position-by-position identification,
        and each source generator acts on the target through the same
        permutation of positions. This treats the target as a source-group
        space by restriction; no abstract group identification is inferred.

        The map is checked against the source generators. Both objects must
        use the same reduced/ordinary homology convention. The result is a
        rectangular matrix from the source homology basis to the target
        homology basis and intertwines the restricted source-group actions.
        """
        if not isinstance(target, EquivariantHomology):
            raise TypeError("target must be an EquivariantHomology instance")
        if not isinstance(simplicial_map, SimplicialMap):
            raise TypeError("simplicial_map must be a SimplicialMap instance")
        if self.reduced != target.reduced:
            raise ValueError("source and target must agree on reduced homology")
        if (
            simplicial_map.source.vertex_set != self.complex.vertex_set
            or simplicial_map.source.facet_set != self.complex.facet_set
            or simplicial_map.target.vertex_set != target.complex.vertex_set
            or simplicial_map.target.facet_set != target.complex.facet_set
        ):
            raise ValueError(
                "simplicial_map source and target must match the homology objects"
            )
        self._validate_relative_map(simplicial_map, target)
        if len(self.action) != len(target.action):
            raise ValueError("actions must have the same number of ordered positions")

        # Membership of every source generator proves G <= H, since the
        # source generators generate G. This avoids enumerating either group.
        if any(
            not target.action.group.contains(
                Permutation(generator.array_form, size=len(target.action))
            )
            for generator in self.action.generators
        ):
            raise ValueError(
                "source action must be a subgroup of the target action "
                "on ordered positions"
            )
        if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
            raise ValueError("degree must be a nonnegative integer")

        for generator in self.action.generators:
            for position, vertex in enumerate(self.action.elements):
                source_image = self.action.elements[generator(position)]
                target_image = target.action.elements[
                    generator(target.action.index[simplicial_map(vertex)])
                ]
                if simplicial_map(source_image) != target_image:
                    raise ValueError(
                        "simplicial_map must intertwine the corresponding "
                        "action generators"
                    )

        _, source_homology, _, _ = self._homology_basis(degree)
        target_boundaries, target_homology, target_pivots, target_inverse = (
            target._homology_basis(degree)
        )
        if source_homology.cols == 0 or target_homology.cols == 0:
            return Matrix(target_homology.cols, source_homology.cols, [])

        chain_map = self._induced_chain_map(target, simplicial_map, degree)
        transformed = chain_map * source_homology
        coordinates = target_inverse * transformed.extract(
            target_pivots, range(source_homology.cols)
        )
        result = coordinates[target_boundaries.cols :, :]
        if result.rows != target_homology.cols or result.cols != source_homology.cols:
            raise ArithmeticError("failed to obtain induced homology coordinates")
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

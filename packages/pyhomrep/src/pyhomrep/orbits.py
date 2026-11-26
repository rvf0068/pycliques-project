"""Exact characteristic-zero homology of signed simplex orbit complexes."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from pycombtop import SimplicialComplex
from sympy import SparseMatrix
from sympy.polys.matrices import DomainMatrix

VertexMap = Callable[[Any], Any]


def _sign(values: Sequence[int]) -> int:
    return (-1) ** sum(
        a > b for i, a in enumerate(values) for b in values[i + 1 :]
    )


def _rank(matrix) -> int:
    return int(DomainMatrix.from_Matrix(matrix).rank())


class OrbitChainComplex:
    """Signed coinvariant chains over Q for a finite simplicial group action.

    Supply generators as explicit maps on vertex labels. All faces in the
    requested degrees are enumerated, but only orientable orbits index the
    boundary matrices. An optional invariant subcomplex defines relative chains.
    The group itself is never enumerated.
    """

    def __init__(
        self,
        complex: SimplicialComplex,
        generators: Iterable[VertexMap] = (),
        *,
        reduced: bool = False,
        relative: SimplicialComplex | None = None,
    ):
        self.complex = complex
        self.reduced = reduced
        self.relative = relative
        self.vertices = tuple(
            next(iter(face)) for face in complex.faces_of_dimension(0)
        )
        self.index = {v: i for i, v in enumerate(self.vertices)}
        self.actions = tuple(
            tuple(self.index[g(v)] for v in self.vertices) for g in generators
        )
        expected = set(range(len(self.vertices)))
        if any(set(g) != expected for g in self.actions):
            raise ValueError("Each generator must permute the vertices")
        self._orbits: dict[int, tuple] = {}
        self._boundaries: dict[int, Any] = {}
        self._ranks: dict[int, int] = {}

    def _faces(self, sc, k):
        if k == -1:
            return {()} if self.reduced else set()
        if k < -1 or k > sc.dimension():
            return set()
        return {
            tuple(sorted(self.index[v] for v in face))
            for face in sc.faces_of_dimension(k)
        }

    def _data(self, k: int):
        if k in self._orbits:
            return self._orbits[k]
        all_faces = self._faces(self.complex, k)
        excluded = (
            self._faces(self.relative, k) if self.relative is not None else set()
        )
        if not excluded <= all_faces:
            raise ValueError("Relative complex must be a subcomplex")
        faces = all_faces - excluded
        unseen = set(faces)
        reps = []
        coords = {}
        for seed in sorted(faces):
            if seed not in unseen:
                continue
            signs = {seed: 1}
            todo = deque([seed])
            orientable = True
            while todo:
                face = todo.popleft()
                for g in self.actions:
                    raw = tuple(g[v] for v in face)
                    image = tuple(sorted(raw))
                    if image not in faces:
                        raise ValueError("Action must preserve the chain basis")
                    s = signs[face] * _sign(raw)
                    if image in signs:
                        if signs[image] != s:
                            orientable = False
                    else:
                        signs[image] = s
                        todo.append(image)
            unseen.difference_update(signs)
            if orientable:
                i = len(reps)
                reps.append(seed)
                coords.update({face: (i, s) for face, s in signs.items()})
        result = (reps, coords, all_faces)
        self._orbits[k] = result
        return result

    def representatives(self, k: int) -> tuple:
        """Oriented orbit representatives expressed in original vertex labels."""
        return tuple(
            tuple(self.vertices[i] for i in face) for face in self._data(k)[0]
        )

    def chain_dimension(self, k: int) -> int:
        return len(self._data(k)[0])

    def boundary_matrix(self, k: int):
        """Exact sparse boundary of the degree-k coinvariant chains."""
        if k not in self._boundaries:
            source = self._data(k)[0]
            target, coords, _ = self._data(k - 1)
            entries: dict[tuple[int, int], int] = {}
            for j, face in enumerate(source):
                for i in range(len(face)):
                    term = coords.get(face[:i] + face[i + 1 :])
                    if term is not None:
                        row, sign = term
                        key = (row, j)
                        entries[key] = entries.get(key, 0) + (-1) ** i * sign
            self._boundaries[k] = SparseMatrix(len(target), len(source), entries)
        return self._boundaries[k]

    def boundary_rank(self, k: int) -> int:
        if k not in self._ranks:
            self._ranks[k] = _rank(self.boundary_matrix(k))
        return self._ranks[k]

    def betti_number(self, k: int) -> int:
        """Dimension of invariant (equivalently coinvariant) rational homology."""
        return (
            self.chain_dimension(k)
            - self.boundary_rank(k)
            - self.boundary_rank(k + 1)
        )

    def map_matrix(self, target: OrbitChainComplex, f: VertexMap, k: int):
        """Chain matrix for an equivariant simplicial map, in degree k.

        Currently maps between relative complexes are not supported.
        Generator lists must correspond in the source and target.
        """
        if self.relative is not None or target.relative is not None:
            raise ValueError("Relative chain maps are not supported")
        if self.reduced != target.reduced:
            raise ValueError("Source and target must agree on reduced chains")
        image = tuple(target.index[f(v)] for v in self.vertices)
        if len(self.actions) != len(target.actions) or any(
            image[g[i]] != h[image[i]]
            for g, h in zip(self.actions, target.actions, strict=True)
            for i in range(len(image))
        ):
            raise ValueError("Map must intertwine the supplied generators")
        reps, _, all_faces = self._data(k)
        tgt, coords, target_faces = target._data(k)
        # Validate all faces, including those whose orbit is nonorientable.
        for face in all_faces:
            raw = tuple(image[i] for i in face)
            support = tuple(sorted(set(raw)))
            image_faces = (
                target_faces if len(support) == len(face)
                else target._data(len(support) - 1)[2]
            )
            if support not in image_faces:
                raise ValueError("Map must be simplicial")
        entries = {}
        for j, face in enumerate(reps):
            raw = tuple(image[i] for i in face)
            if len(set(raw)) < len(raw):
                continue
            term = coords.get(tuple(sorted(raw)))
            if term is not None:
                i, s = term
                entries[i, j] = s * _sign(raw)
        return SparseMatrix(len(tgt), len(reps), entries)


def homology_map_rank(
    source: OrbitChainComplex, target: OrbitChainComplex, f: VertexMap, k: int
) -> int:
    """Exact rank of the induced homology map on coinvariants."""
    matrix = source.map_matrix(target, f, k)
    # The rank of [D 0; F B] is rank(D) + dim(F ker(D) + im(B)).
    # This avoids constructing a potentially dense nullspace basis.
    source.map_matrix(target, f, k - 1)
    source.map_matrix(target, f, k + 1)
    d = source.boundary_matrix(k)
    b = target.boundary_matrix(k + 1)
    block = d.row_join(SparseMatrix(d.rows, b.cols, {})).col_join(
        matrix.row_join(b)
    )
    return _rank(block) - source.boundary_rank(k) - target.boundary_rank(k + 1)

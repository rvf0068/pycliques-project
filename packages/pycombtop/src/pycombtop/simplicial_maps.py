"""Simplicial maps, contiguity, and induced maps on simplicial chains."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from types import MappingProxyType

from sympy import SparseMatrix

from .simplex import Simplex, SimplicialComplex


def _is_face(complex_: SimplicialComplex, face: Iterable[Hashable]) -> bool:
    """Test face membership from the maximal-face representation."""
    candidate = frozenset(face)
    return any(candidate <= facet for facet in complex_.facet_set)


def _same_complex(first: SimplicialComplex, second: SimplicialComplex) -> bool:
    """Compare realized complexes, ignoring ambient ghost labels."""
    return first.vertex_set == second.vertex_set and first.facet_set == second.facet_set


def _stable_vertex_key(vertex: Hashable) -> tuple:
    """Build a recursively comparable key for common immutable labels."""
    if isinstance(vertex, (set, frozenset)):
        return ("set", tuple(sorted(_stable_vertex_key(v) for v in vertex)))
    if isinstance(vertex, tuple):
        return ("tuple", tuple(_stable_vertex_key(v) for v in vertex))
    return (
        "atom",
        type(vertex).__module__,
        type(vertex).__qualname__,
        repr(vertex),
    )


def _ordered_vertices(
    vertices: set[Hashable], vertex_order: Iterable[Hashable] | None
) -> tuple[Hashable, ...]:
    """Validate an explicit order or choose a stable order for vertex labels."""
    if vertex_order is not None:
        order = tuple(vertex_order)
        if len(order) != len(vertices) or len(set(order)) != len(order):
            raise ValueError("vertex_order must list every actual vertex exactly once")
        if set(order) != vertices:
            raise ValueError("vertex_order must list exactly the actual vertices")
        return order

    keyed = sorted(
        ((_stable_vertex_key(vertex), vertex) for vertex in vertices),
        key=lambda item: item[0],
    )
    if any(keyed[i - 1][0] == keyed[i][0] for i in range(1, len(keyed))):
        raise ValueError(
            "vertex labels have ambiguous stable keys; supply vertex_order explicitly"
        )
    return tuple(vertex for _, vertex in keyed)


def _vertex_ranks(
    complex_: SimplicialComplex, vertex_order: Iterable[Hashable] | None
) -> dict[Hashable, int]:
    return {
        vertex: index
        for index, vertex in enumerate(
            _ordered_vertices(complex_.vertex_set, vertex_order)
        )
    }


def _ordered_simplex(
    simplex: Iterable[Hashable], ranks: Mapping[Hashable, int]
) -> tuple[Hashable, ...]:
    return tuple(sorted(simplex, key=ranks.__getitem__))


def _simplex_basis_key(
    simplex: Iterable[Hashable], ranks: Mapping[Hashable, int]
) -> tuple[int, ...]:
    return tuple(sorted(ranks[vertex] for vertex in simplex))


def _permutation_sign(
    sequence: Iterable[Hashable], ranks: Mapping[Hashable, int]
) -> int:
    """Sign of a sequence relative to the order encoded by vertex ranks."""
    positions = [ranks[vertex] for vertex in sequence]
    inversions = sum(
        positions[i] > positions[j]
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    )
    return -1 if inversions % 2 else 1


def _validate_degree(degree: int) -> None:
    if type(degree) is not int or degree < 0:
        raise ValueError("degree must be a nonnegative integer")


class SimplicialMap:
    """A simplicial map between finite abstract simplicial complexes.

    The map is specified on the actual vertices (the 0-simplices), not on
    ambient base labels that are not vertices. It need not be injective:
    vertices of a simplex may have the same image, so simplices may collapse
    to lower-dimensional faces.

    The vertex mapping must be defined on exactly source.vertex_set, take
    values in target.vertex_set, and send every source facet to a target face.

    Examples
    --------
    A filled triangle can map simplicially onto an edge by identifying
    vertices::

        >>> from pycombtop import SimplicialComplex, SimplicialMap
        >>> X = SimplicialComplex({0, 1, 2}, [{0, 1, 2}])
        >>> Y = SimplicialComplex({"a", "b"}, [{"a", "b"}])
        >>> f = SimplicialMap(X, Y, {0: "a", 1: "a", 2: "b"})
        >>> f.image_simplex({0, 1, 2}) == {"a", "b"}
        True
    """

    __slots__ = ("source", "target", "_vertex_map")

    def __init__(
        self,
        source: SimplicialComplex,
        target: SimplicialComplex,
        vertex_map: Mapping[Hashable, Hashable],
    ) -> None:
        if not isinstance(source, SimplicialComplex) or not isinstance(
            target, SimplicialComplex
        ):
            raise TypeError("source and target must be SimplicialComplex instances")
        if not isinstance(vertex_map, Mapping):
            raise TypeError("vertex_map must be a mapping")

        mapping = dict(vertex_map)
        if set(mapping) != source.vertex_set:
            raise ValueError("vertex_map must be defined on exactly source.vertex_set")
        if any(vertex not in target.vertex_set for vertex in mapping.values()):
            raise ValueError("every vertex image must belong to target.vertex_set")

        for facet in source.facet_set:
            image = {mapping[vertex] for vertex in facet}
            if not _is_face(target, image):
                raise ValueError("the image of each source facet must be a target face")

        self.source = source
        self.target = target
        self._vertex_map: Mapping[Hashable, Hashable] = MappingProxyType(mapping)

    @property
    def vertex_map(self) -> Mapping[Hashable, Hashable]:
        """Read-only mapping on the actual vertices of the source."""
        return self._vertex_map

    def __call__(self, vertex: Hashable) -> Hashable:
        """Return the image of a source vertex."""
        return self._vertex_map[vertex]

    def image_simplex(self, simplex: Iterable[Hashable]) -> Simplex:
        """Return the vertex-set image of a source simplex."""
        face = Simplex(simplex)
        if not _is_face(self.source, face):
            raise ValueError("the argument must be a simplex of the source")
        return Simplex(self._vertex_map[vertex] for vertex in face)

    def is_contiguous_to(self, other: SimplicialMap) -> bool:
        """Return whether this map and other are contiguous."""
        return are_contiguous(self, other)

    def induced_chain_map(
        self,
        degree: int,
        *,
        source_order: Iterable[Hashable] | None = None,
        target_order: Iterable[Hashable] | None = None,
    ) -> SparseMatrix:
        """Return the induced matrix on simplicial chains in degree."""
        return induced_chain_map(
            self,
            degree,
            source_order=source_order,
            target_order=target_order,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SimplicialMap)
            and _same_complex(self.source, other.source)
            and _same_complex(self.target, other.target)
            and dict(self.vertex_map) == dict(other.vertex_map)
        )

    def __repr__(self) -> str:
        return f"SimplicialMap({dict(self.vertex_map)!r})"


def are_contiguous(first: SimplicialMap, second: SimplicialMap) -> bool:
    """Test whether two simplicial maps are contiguous.

    The maps must have the same domain and codomain. They are contiguous
    precisely when, for every source simplex sigma, the union of its two
    images is a simplex of the target. It is enough to check maximal source
    simplices.
    """
    if not isinstance(first, SimplicialMap) or not isinstance(second, SimplicialMap):
        raise TypeError("both arguments must be SimplicialMap instances")
    if not _same_complex(first.source, second.source) or not _same_complex(
        first.target, second.target
    ):
        raise ValueError("contiguous maps must have the same source and target")

    return all(
        _is_face(
            first.target,
            first.image_simplex(facet) | second.image_simplex(facet),
        )
        for facet in first.source.facet_set
    )


def simplicial_chain_basis(
    complex_: SimplicialComplex,
    degree: int,
    *,
    vertex_order: Iterable[Hashable] | None = None,
) -> tuple[Simplex, ...]:
    """Return the oriented simplex basis of the chain group in degree.

    Each basis simplex is oriented by listing its vertices in increasing
    vertex_order. The basis itself is lexicographically ordered by those
    vertex ranks. If no order is supplied, a stable order is chosen for
    common immutable labels; ambiguous labels require an explicit order.

    Only actual vertices are ordered. Ambient ghost labels do not occur in
    simplicial chain groups.
    """
    _validate_degree(degree)
    ranks = _vertex_ranks(complex_, vertex_order)
    faces = complex_.faces_of_dimension(degree)
    return tuple(sorted(faces, key=lambda face: _simplex_basis_key(face, ranks)))


def simplicial_boundary_matrix(
    complex_: SimplicialComplex,
    degree: int,
    *,
    vertex_order: Iterable[Hashable] | None = None,
) -> SparseMatrix:
    """Return the integer boundary matrix from C_degree to C_(degree-1).

    Rows and columns use simplicial_chain_basis with the same vertex_order.
    This is the ordinary (unaugmented) simplicial chain complex, so the
    degree-zero boundary is the zero map to the zero group. Matrices are
    exact sympy.SparseMatrix instances over the integers.
    """
    _validate_degree(degree)
    ranks = _vertex_ranks(complex_, vertex_order)
    columns = simplicial_chain_basis(complex_, degree, vertex_order=vertex_order)
    if degree == 0:
        return SparseMatrix(0, len(columns), {})

    rows = simplicial_chain_basis(complex_, degree - 1, vertex_order=vertex_order)
    row_positions = {simplex: index for index, simplex in enumerate(rows)}
    entries: dict[tuple[int, int], int] = {}
    for column, simplex in enumerate(columns):
        for position, vertex in enumerate(_ordered_simplex(simplex, ranks)):
            face = Simplex(simplex - {vertex})
            entries[row_positions[face], column] = (-1) ** position
    return SparseMatrix(len(rows), len(columns), entries)


def induced_chain_map(
    simplicial_map: SimplicialMap,
    degree: int,
    *,
    source_order: Iterable[Hashable] | None = None,
    target_order: Iterable[Hashable] | None = None,
) -> SparseMatrix:
    """Return the matrix induced by a simplicial map on C_degree.

    The rows and columns use the bases returned by simplicial_chain_basis
    on the target and source, respectively. A source simplex contributes
    zero if the vertex map identifies two of its vertices; otherwise its
    image contributes the orientation sign of the induced permutation.
    The returned matrix has integer entries.
    """
    if not isinstance(simplicial_map, SimplicialMap):
        raise TypeError("simplicial_map must be a SimplicialMap instance")
    _validate_degree(degree)

    source_ranks = _vertex_ranks(simplicial_map.source, source_order)
    target_ranks = _vertex_ranks(simplicial_map.target, target_order)
    source_basis = simplicial_chain_basis(
        simplicial_map.source, degree, vertex_order=source_order
    )
    target_basis = simplicial_chain_basis(
        simplicial_map.target, degree, vertex_order=target_order
    )
    target_positions = {simplex: index for index, simplex in enumerate(target_basis)}

    entries: dict[tuple[int, int], int] = {}
    for column, simplex in enumerate(source_basis):
        image_sequence = tuple(
            simplicial_map(vertex) for vertex in _ordered_simplex(simplex, source_ranks)
        )
        if len(set(image_sequence)) != len(image_sequence):
            continue
        image = Simplex(image_sequence)
        row = target_positions.get(image)
        if row is None:
            raise AssertionError("a simplicial map sent a simplex outside the target")
        entries[row, column] = _permutation_sign(image_sequence, target_ranks)

    return SparseMatrix(len(target_basis), len(source_basis), entries)


def prism_chain_homotopy(
    first: SimplicialMap,
    second: SimplicialMap,
    degree: int,
    *,
    source_order: Iterable[Hashable] | None = None,
    target_order: Iterable[Hashable] | None = None,
) -> SparseMatrix:
    """Return the standard prism operator for two contiguous maps.

    The matrix represents P_degree from C_degree(source) to
    C_(degree+1)(target). On an oriented simplex [v_0,...,v_d], it is

        sum_i (-1)^i
        [f(v_0),...,f(v_i),g(v_i),...,g(v_d)].

    Degenerate terms (those with repeated vertices) are zero. The matrices
    satisfy boundary * P_degree + P_(degree-1) * boundary = g_# - f_#.
    Rows and columns use simplicial_chain_basis.
    """
    if not are_contiguous(first, second):
        raise ValueError("the prism operator requires contiguous maps")
    _validate_degree(degree)

    source_ranks = _vertex_ranks(first.source, source_order)
    target_ranks = _vertex_ranks(first.target, target_order)
    source_basis = simplicial_chain_basis(
        first.source, degree, vertex_order=source_order
    )
    target_basis = simplicial_chain_basis(
        first.target, degree + 1, vertex_order=target_order
    )
    target_positions = {simplex: index for index, simplex in enumerate(target_basis)}

    entries: dict[tuple[int, int], int] = {}
    for column, simplex in enumerate(source_basis):
        vertices = _ordered_simplex(simplex, source_ranks)
        for split in range(degree + 1):
            sequence = (
                tuple(first(vertex) for vertex in vertices[: split + 1])
                + tuple(second(vertex) for vertex in vertices[split:])
            )
            if len(set(sequence)) != degree + 2:
                continue
            image = Simplex(sequence)
            row = target_positions.get(image)
            if row is None:
                raise AssertionError("a prism term is not a target simplex")
            key = (row, column)
            entries[key] = entries.get(key, 0) + (-1) ** split * _permutation_sign(
                sequence, target_ranks
            )

    return SparseMatrix(
        len(target_basis),
        len(source_basis),
        {key: value for key, value in entries.items() if value},
    )

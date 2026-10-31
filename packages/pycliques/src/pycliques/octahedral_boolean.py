"""Implicit Boolean model for the iterated clique graph of an octahedron."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from math import comb

import networkx as nx

BooleanVertex = tuple[int, int]


@dataclass(frozen=True)
class MaximalIntersectingFamilies:
    """Bounded enumeration result for maximal intersecting subset families."""

    n: int
    families: tuple[int, ...]
    is_exact: bool
    reason: str = ""


@dataclass(frozen=True)
class OctahedralModelResult:
    """A complete implicit model, or an explicit inconclusive enumeration."""

    model: OctahedralBooleanModel | None
    is_exact: bool
    reason: str = ""


def _family_bit(subset_mask: int) -> int:
    """Return the membership bit for a coordinate-subset mask."""
    return 1 << subset_mask


def enumerate_maximal_intersecting_families(
    n: int,
    *,
    max_families: int | None = None,
    max_subset_vertices: int | None = None,
) -> MaximalIntersectingFamilies:
    """Enumerate maximal intersecting families of nonempty subsets of ``[n]``.

    A family is encoded as an integer bitset whose bit at position ``S``
    records whether the nonempty subset mask ``S`` belongs to the family.
    If either configured bound is exceeded, ``is_exact`` is false and the
    returned tuple contains only families seen before the exit.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    if max_families is not None and max_families < 0:
        raise ValueError("max_families must be nonnegative or None")
    if max_subset_vertices is not None and max_subset_vertices < 0:
        raise ValueError("max_subset_vertices must be nonnegative or None")

    subset_count = (1 << n) - 1
    if max_subset_vertices is not None and subset_count > max_subset_vertices:
        return MaximalIntersectingFamilies(
            n,
            (),
            False,
            "nonempty subset-vertex limit exceeded",
        )

    intersection_graph = nx.Graph()
    intersection_graph.add_nodes_from(range(1, 1 << n))
    for first, second in combinations(range(1, 1 << n), 2):
        if first & second:
            intersection_graph.add_edge(first, second)

    families: list[int] = []
    for clique in nx.find_cliques(intersection_graph):
        if max_families is not None and len(families) >= max_families:
            return MaximalIntersectingFamilies(
                n,
                tuple(sorted(families)),
                False,
                "maximal-family limit exceeded",
            )
        family = 0
        for subset_mask in clique:
            family |= _family_bit(subset_mask)
        families.append(family)

    return MaximalIntersectingFamilies(
        n,
        tuple(sorted(families)),
        True,
    )


@dataclass(frozen=True)
class OctahedralBooleanModel:
    """Implicit model of ``K(K(H(O_n)))`` with no global edge allocation.

    A vertex ``(epsilon, family_index)`` identifies a sign word and a
    maximal intersecting family. Adjacency, degree, order, and size are
    computed from the family bitsets and containment counts.
    """

    n: int
    families: tuple[int, ...]
    containment_counts: tuple[int, ...]
    _family_indices_by_subset: tuple[tuple[int, ...], ...]
    _family_index_by_bitset: dict[int, int]

    @classmethod
    def from_families(cls, n: int, families: tuple[int, ...]) -> OctahedralBooleanModel:
        if n < 1:
            raise ValueError("n must be a positive integer")
        full = (1 << n) - 1
        if not families:
            raise ValueError("at least one maximal intersecting family is required")
        if len(set(families)) != len(families):
            raise ValueError("families must be distinct")
        if tuple(sorted(families)) != families:
            raise ValueError("families must be deterministically sorted")

        by_subset: list[list[int]] = [[] for _ in range(full + 1)]
        for family_index, family in enumerate(families):
            for subset_mask in range(1, full + 1):
                if family & _family_bit(subset_mask):
                    by_subset[subset_mask].append(family_index)
        counts = [0] * (n + 1)
        for subset_mask in range(1, full + 1):
            counts[subset_mask.bit_count()] += len(by_subset[subset_mask])
        for size in range(1, n + 1):
            counts[size] //= comb(n, size)

        return cls(
            n,
            families,
            tuple(counts),
            tuple(tuple(indices) for indices in by_subset),
            {family: index for index, family in enumerate(families)},
        )

    @property
    def family_count(self) -> int:
        return len(self.families)

    @property
    def order(self) -> int:
        """Return the number of vertices without enumerating them."""
        return (1 << self.n) * self.family_count

    @property
    def size(self) -> int:
        """Return the number of edges without enumerating them."""
        same_sign_edges = (1 << self.n) * comb(self.family_count, 2)
        cross_sign_edges = (1 << (self.n - 1)) * sum(
            comb(self.n, subset_size) * self.containment_counts[subset_size] ** 2
            for subset_size in range(1, self.n)
        )
        return same_sign_edges + cross_sign_edges

    @property
    def full_mask(self) -> int:
        return (1 << self.n) - 1

    def vertices(self) -> Iterator[BooleanVertex]:
        """Yield vertices lazily in deterministic sign/family order."""
        for epsilon in range(1 << self.n):
            for family_index in range(self.family_count):
                yield epsilon, family_index

    def _validate_vertex(self, vertex: BooleanVertex) -> None:
        epsilon, family_index = vertex
        if not 0 <= epsilon <= self.full_mask:
            raise ValueError("sign word is outside the n-dimensional cube")
        if not 0 <= family_index < self.family_count:
            raise ValueError("family index is out of range")

    def adjacent(self, first: BooleanVertex, second: BooleanVertex) -> bool:
        """Return whether two modeled vertices are adjacent."""
        self._validate_vertex(first)
        self._validate_vertex(second)
        epsilon, first_family_index = first
        delta, second_family_index = second
        if first == second:
            return False
        if epsilon == delta:
            return first_family_index != second_family_index
        agreement = self.full_mask ^ (epsilon ^ delta)
        if not agreement:
            return False
        return all(
            family & _family_bit(agreement)
            for family in (
                self.families[first_family_index],
                self.families[second_family_index],
            )
        )

    def neighbors(self, vertex: BooleanVertex) -> Iterator[BooleanVertex]:
        """Yield adjacent vertices without constructing a graph or edge set."""
        self._validate_vertex(vertex)
        epsilon, family_index = vertex
        yield from (
            (epsilon, other_index)
            for other_index in range(self.family_count)
            if other_index != family_index
        )

        family = self.families[family_index]
        for agreement in range(1, self.full_mask):
            if family & _family_bit(agreement):
                delta = epsilon ^ self.full_mask ^ agreement
                for other_index in self._family_indices_by_subset[agreement]:
                    yield delta, other_index

    def degree(self, vertex: BooleanVertex) -> int:
        """Return a vertex degree from the family containment counts."""
        self._validate_vertex(vertex)
        family = self.families[vertex[1]]
        cross_degree = sum(
            self.containment_counts[subset_mask.bit_count()]
            for subset_mask in range(1, self.full_mask)
            if family & _family_bit(subset_mask)
        )
        return self.family_count - 1 + cross_degree

    def transform_vertex(
        self,
        vertex: BooleanVertex,
        permutation: tuple[int, ...],
        sign_flip: int = 0,
    ) -> BooleanVertex:
        """Apply a coordinate permutation and sign flip to a modeled vertex."""
        self._validate_vertex(vertex)
        if sorted(permutation) != list(range(self.n)):
            raise ValueError("permutation must permute all n coordinates")
        if not 0 <= sign_flip <= self.full_mask:
            raise ValueError("sign_flip must be an n-bit mask")

        epsilon, family_index = vertex
        transformed_epsilon = 0
        transformed_family = 0
        original_family = self.families[family_index]
        for coordinate in range(self.n):
            if epsilon & (1 << coordinate):
                transformed_epsilon |= 1 << permutation[coordinate]
        transformed_epsilon ^= sign_flip
        for subset_mask in range(1, self.full_mask + 1):
            if not original_family & _family_bit(subset_mask):
                continue
            transformed_subset = 0
            for coordinate in range(self.n):
                if subset_mask & (1 << coordinate):
                    transformed_subset |= 1 << permutation[coordinate]
            transformed_family |= _family_bit(transformed_subset)
        transformed_index = self._family_index_by_bitset.get(transformed_family)
        if transformed_index is None:
            raise RuntimeError("coordinate transform left the family enumeration")
        return transformed_epsilon, transformed_index


def build_octahedral_boolean_model(
    n: int,
    *,
    max_families: int | None = None,
    max_subset_vertices: int | None = None,
) -> OctahedralModelResult:
    """Build the implicit ``K(K(H(O_n)))`` model or report an inconclusive limit."""
    enumeration = enumerate_maximal_intersecting_families(
        n,
        max_families=max_families,
        max_subset_vertices=max_subset_vertices,
    )
    if not enumeration.is_exact:
        return OctahedralModelResult(None, False, enumeration.reason)
    model = OctahedralBooleanModel.from_families(n, enumeration.families)
    return OctahedralModelResult(model, True)


def degree_histogram(model: OctahedralBooleanModel) -> dict[int, int]:
    """Return the analytic degree histogram by summing over model vertices."""
    counts = Counter(model.degree(vertex) for vertex in model.vertices())
    return dict(sorted(counts.items()))

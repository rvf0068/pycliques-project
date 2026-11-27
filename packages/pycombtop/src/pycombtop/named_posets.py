"""Standard finite posets used in combinatorial and equivariant topology."""

from collections.abc import Hashable, Iterable, Iterator
from itertools import combinations

import networkx as nx
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import multiset_partitions

from .posets import Poset

Partition = frozenset[frozenset[int]]


def _rank_bounds(
    n: int, lower: int, upper: int | None, proper: bool
) -> tuple[int, int]:
    """Validate a rank interval and optionally remove both extrema."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if upper is None:
        upper = n
    if any(not isinstance(x, int) or isinstance(x, bool) for x in (lower, upper)):
        raise ValueError("rank bounds must be integers")
    if not 0 <= lower <= upper <= n:
        raise ValueError("rank bounds must satisfy 0 <= min_rank <= max_rank <= n")
    if proper:
        return max(1, lower), min(n - 1, upper)
    return lower, upper


def boolean_lattice(
    ground: int | Iterable[Hashable],
    *,
    proper: bool = False,
    min_rank: int = 0,
    max_rank: int | None = None,
) -> Poset:
    """Boolean lattice ordered by inclusion, with optional rank truncation.

    An integer n denotes the ground set range(n); alternatively supply
    distinct hashable labels. Rank is subset cardinality. The proper part
    excludes the empty and full subsets, including in the degenerate cases.
    """
    if isinstance(ground, int):
        if isinstance(ground, bool) or ground < 0:
            raise ValueError("ground size must be a nonnegative integer")
        labels = tuple(range(ground))
    else:
        labels = tuple(ground)
        if len(set(labels)) != len(labels):
            raise ValueError("ground labels must be distinct")
    lower, upper = _rank_bounds(len(labels), min_rank, max_rank, proper)
    faces = [
        frozenset(face)
        for rank in range(lower, upper + 1)
        for face in combinations(labels, rank)
    ]
    return Poset(
        faces,
        (
            (face - {x}, face)
            for face in faces
            if len(face) > lower
            for x in face
        ),
    )


def _partitions(n: int) -> Iterator[Partition]:
    if n == 0:
        yield frozenset()
        return
    for part in multiset_partitions(list(range(n))):
        yield frozenset(frozenset(block) for block in part)


def nontrivial_partitions(n: int) -> list[Partition]:
    """Return proper partitions in the historical research enumeration order."""
    if n < 3:
        raise ValueError("n must be at least 3")
    return [part for part in _partitions(n) if 1 < len(part) < n]


def refines(a: Partition, b: Partition) -> bool:
    """Whether every block of a lies in a block of b (same ground set)."""
    return all(any(x <= y for y in b) for x in a)


def _partition_poset(parts: list[Partition]) -> Poset:
    allowed = set(parts)

    def edges() -> Iterator[tuple[Partition, Partition]]:
        for finer in parts:
            for a, b in combinations(finer, 2):
                coarser = (finer - {a, b}) | {a | b}
                if coarser in allowed:
                    yield finer, coarser

    return Poset(parts, edges())


def partition_lattice(
    n: int,
    *,
    proper: bool = False,
    min_rank: int = 0,
    max_rank: int | None = None,
) -> Poset:
    """Set partitions of range(n), ordered by refinement.

    The finest partition is the minimum; rank is n minus the block count.
    The maximum rank is max(0, n-1). Truncation selects a contiguous rank
    interval, and proper removes both extrema of the original lattice.
    Labels are frozensets of nonempty frozenset blocks.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    lower, upper = _rank_bounds(max(0, n - 1), min_rank, max_rank, proper)
    return _partition_poset(
        [part for part in _partitions(n) if lower <= n - len(part) <= upper]
    )


def partition_graph(n: int) -> tuple[nx.Graph, list[Partition]]:
    """Historical indexed comparability graph of the proper partition lattice."""
    parts = nontrivial_partitions(n)
    graph = _partition_poset(parts).comparability_graph()
    indexed = nx.relabel_nodes(graph, {part: i for i, part in enumerate(parts)})
    return indexed, parts


def induced_permutation(
    g: Permutation, n: int, parts: list[Partition], index: dict[Partition, int]
) -> Permutation:
    """Permutation of partition indices induced by a permutation of range(n)."""
    g = Permutation(g.array_form, size=n)
    img = [
        index[frozenset(frozenset(g(x) for x in block) for block in p)] for p in parts
    ]
    return Permutation(img, size=len(parts))

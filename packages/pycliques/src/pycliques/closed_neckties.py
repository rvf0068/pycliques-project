"""Bitset closure and necktie-center tests on an intersection graph.

The family must be the full maximal-clique family of G when applying the
homotopy criterion of Larrion--Neumann-Lara--Pizana (as stated in Theorem
2.4 of Larrion--Pizana--Villarroel-Flores, Discrete Math. 309 (2009)).
The index itself works for arbitrary distinct nonempty sets.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Generic, TypeVar, overload

from .implicit_clique_graph import ImplicitCliqueGraph


def _indices(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


_T = TypeVar("_T")


class _IndexedSequence(Sequence[_T], Generic[_T]):
    """Read-only, optionally cached sequence without materializing all items."""

    def __init__(self, size: int, get: Callable[[int], _T]) -> None:
        self._size = size
        self._get = get

    def __len__(self) -> int:
        return self._size

    @overload
    def __getitem__(self, index: int) -> _T: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[_T, ...]: ...

    def __getitem__(self, index: int | slice) -> _T | tuple[_T, ...]:
        if isinstance(index, slice):
            return tuple(self._get(i) for i in range(*index.indices(self._size)))
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError(index)
        return self._get(index)


class ClosedNecktieIndex:
    """Index a family in input order, using integer masks for its subfamilies.

    No NetworkX intersection graph is constructed. Incidence stars are
    indexed once; neighborhoods can be eager or lazily cached.
    """

    def __init__(
        self,
        family: Iterable[Iterable[Hashable]],
        *,
        lazy: bool = False,
        cache_size: int = 256,
    ) -> None:
        """Use lazy=True to keep only incidence and a bounded neighborhood cache."""
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self.family: Sequence[frozenset[Hashable]]
        self.adjacency: Sequence[int]
        if lazy:
            self._init_implicit(ImplicitCliqueGraph.from_family(family), cache_size)
            return
        self.family = tuple(frozenset(q) for q in family)
        if any(not q for q in self.family) or len(set(self.family)) != len(self.family):
            raise ValueError("family must consist of distinct nonempty sets")
        self.full = (1 << len(self.family)) - 1
        self.stars: dict[Hashable, int] = defaultdict(int)
        for i, q in enumerate(self.family):
            for v in q:
                self.stars[v] |= 1 << i
        self.adjacency = tuple(self._union(q) for q in self.family)

    @classmethod
    def from_implicit(
        cls, graph: ImplicitCliqueGraph, *, cache_size: int = 256
    ) -> ClosedNecktieIndex:
        """Share an implicit graph's incidence; decode members only as needed.

        A full, unpared implicit graph is required, to retain contiguous
        member IDs. The caller must supply the full maximal-clique family
        for applications of the homotopy criterion.
        """
        result = cls.__new__(cls)
        result._init_implicit(graph, cache_size)
        return result

    def _init_implicit(self, graph: ImplicitCliqueGraph, cache_size: int) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        n = len(graph.clique_masks)
        self.full = (1 << n) - 1
        if graph.active_mask != self.full:
            raise ValueError("expected a full unpared implicit family")
        self.stars = dict(zip(graph.base_vertices, graph.incidence_masks, strict=True))
        self.family = _IndexedSequence(n, graph.clique)
        lookup = lru_cache(maxsize=cache_size)(graph.closed_neighborhood_mask)
        self.adjacency = _IndexedSequence(n, lookup)

    def _union(self, vertices: Iterable[Hashable]) -> int:
        result = 0
        for v in vertices:
            result |= self.stars[v]
        return result

    def _validate(self, mask: int) -> None:
        if not 0 < mask <= self.full:
            raise ValueError("expected a nonempty subfamily mask")

    def common(self, mask: int) -> int:
        """Return the common closed neighborhood of a nonempty subfamily."""
        self._validate(mask)
        result = self.full
        for i in _indices(mask):
            result &= self.adjacency[i]
        return result

    def close(self, mask: int) -> int:
        """Intersect all maximal completes containing the input complete.

        The answer consists of universal vertices in the common closed
        neighborhood, so enumeration of maximal completes is unnecessary.
        """
        common = self.common(mask)
        if mask & common != mask:
            raise ValueError("expected a complete mask")
        return self._close_common(common, self.full, mask)

    def _close_common(self, common: int, within: int, known: int = 0) -> int:
        # A nonneighbor of any candidate in common cannot be universal.
        # Always test against the original common, never a shrinking subset.
        universal = known & within
        candidates = common & within & ~universal
        while candidates:
            bit = candidates & -candidates
            disjoint = common & ~self.adjacency[bit.bit_length() - 1]
            if not disjoint:
                universal |= bit
            candidates &= ~(bit | disjoint)
        return universal

    def is_bad(self, mask: int) -> bool:
        """Whether the nonempty subfamily has empty total intersection."""
        self._validate(mask)
        return not any(star & mask == mask for star in self.stars.values())

    def centers(self, mask: int) -> int:
        """Centers belonging to every maximal complete containing the input.

        The nonempty maximal incidence patterns of the family members suffice:
        a center must meet the intersection associated with each such pattern.
        """
        closed = self.close(mask)
        patterns: dict[int, int] = defaultdict(int)
        for star in self.stars.values():
            pattern = star & mask
            if pattern:
                patterns[pattern] |= star
        maximal: list[int] = []
        for pattern in sorted(patterns, key=int.bit_count, reverse=True):
            if any(pattern & other == pattern for other in maximal):
                continue
            maximal.append(pattern)
            closed &= patterns[pattern]
            if not closed:
                break
        return closed

    def members(self, mask: int) -> tuple[frozenset[Hashable], ...]:
        """Decode a nonempty subfamily mask to its original members."""
        self._validate(mask)
        return tuple(self.family[i] for i in _indices(mask))

    def nerve_facets(self, mask: int) -> tuple[int, ...]:
        """Maximal incidence stars restricted to the specified subfamily."""
        self._validate(mask)
        maximal: list[int] = []
        for pattern in sorted(
            {s & mask for s in self.stars.values()} - {0},
            key=lambda m: (-m.bit_count(), m),
        ):
            if not any(pattern & other == pattern for other in maximal):
                maximal.append(pattern)
        return tuple(maximal)

    def _closed_subsets(
        self,
        root: int,
        within: int,
        prune: Callable[[int, int], bool] | None = None,
    ) -> Iterator[int]:
        """Canonical augmentation with global closure and a restricted domain."""
        common = self.common(root) if root else self.full
        upper = common & within
        if root:
            if prune is not None and prune(root, upper):
                return
            yield root
        stack = [(root, common, upper & ~root)]
        while stack:
            current, common, candidates = stack.pop()
            if not candidates:
                continue
            bit = candidates & -candidates
            stack.append((current, common, candidates ^ bit))
            child_common = common & self.adjacency[bit.bit_length() - 1]
            child = self._close_common(child_common, within, current | bit)
            if (child & ~current) & (bit - 1):
                continue
            later = child_common & within & ~((bit << 1) - 1)
            upper = child | later
            if prune is not None and prune(child, upper):
                continue
            yield child
            stack.append((child, child_common, later & ~child))

    def closed_containing(
        self, anchor: int, *, prune: Callable[[int, int], bool] | None = None
    ) -> Iterator[int]:
        """Enumerate closed completes containing anchor, optionally pruning.

        The callback has the same interval semantics as closed_within.
        Here the upper family need not be a complete.
        """
        if not 0 <= anchor < len(self.family):
            raise ValueError("anchor out of range")
        yield from self._closed_subsets(self.close(1 << anchor), self.full, prune)

    def closed_within(
        self,
        maximal: int,
        *,
        prune: Callable[[int, int], bool] | None = None,
    ) -> Iterator[int]:
        """Enumerate nonempty closed completes within a maximal complete.

        Closure is computed in the whole intersection graph, not the induced
        complete. The optional callback receives (closed, upper), masks with
        closed contained in upper. Returning True skips this node and all
        its canonical descendants, each of which lies in that interval.
        The caller must justify any pruning; without it this is exhaustive.
        """
        if self.common(maximal) != maximal:
            raise ValueError("expected a maximal complete")
        root = self._close_common(self.full, maximal)
        yield from self._closed_subsets(root, maximal, prune)

    def _validated_generators(
        self, generators: Iterable[Iterable[int]]
    ) -> tuple[tuple[int, ...], ...]:
        n = len(self.family)
        perms = tuple(tuple(p) for p in generators)
        stars = set(self.stars.values())
        for perm in perms:
            if (
                sorted(perm) != list(range(n))
                or {sum(1 << perm[i] for i in _indices(s)) for s in stars} != stars
            ):
                raise ValueError("generators must preserve family incidence")
        return perms

    def _vertex_orbit_reps(self, perms: tuple[tuple[int, ...], ...]) -> Iterator[int]:
        remaining = set(range(len(self.family)))
        while remaining:
            anchor = min(remaining)
            orbit = {anchor}
            queue = [anchor]
            for v in queue:
                for perm in perms:
                    w = perm[v]
                    if w not in orbit:
                        orbit.add(w)
                        queue.append(w)
            remaining -= orbit
            yield anchor

    def vertex_orbit_representatives(
        self, generators: Iterable[Iterable[int]] = ()
    ) -> Iterator[int]:
        """Yield one member ID per orbit, after validating incidence symmetry."""
        yield from self._vertex_orbit_reps(self._validated_generators(generators))

    def closed_representatives(
        self,
        generators: Iterable[Iterable[int]] = (),
        *,
        prune: Callable[[int, int], bool] | None = None,
    ) -> Iterator[int]:
        """Cover all closed completes using representatives of vertex orbits.

        Each supplied permutation must preserve the set of incidence stars;
        graph automorphisms alone need not preserve the original nerve.
        Completes can recur across anchors. With no generators every vertex
        is an anchor.
        """
        perms = self._validated_generators(generators)
        for anchor in self._vertex_orbit_reps(perms):
            yield from self.closed_containing(anchor, prune=prune)

    def _maximal_completes(self, anchor: int | None = None) -> Iterator[int]:
        """Pivoted Bron-Kerbosch on the existing neighborhood bitsets."""
        def neighbors(i: int) -> int:
            return self.adjacency[i] & ~(1 << i)

        if anchor is None:
            stack = [(0, self.full, 0)]
        else:
            stack = [(1 << anchor, neighbors(anchor), 0)]
        while stack:
            chosen, candidates, excluded = stack.pop()
            if not candidates:
                if chosen and not excluded:
                    yield chosen
                continue
            pivots = _indices(candidates | excluded)
            pivot = max(pivots, key=lambda i: (candidates & neighbors(i)).bit_count())
            for i in _indices(candidates & ~neighbors(pivot)):
                bit = 1 << i
                near = neighbors(i)
                stack.append((chosen | bit, candidates & near, excluded & near))
                candidates &= ~bit
                excluded |= bit

    def maximal_complete_orbits(
        self, generators: Iterable[Iterable[int]] = ()
    ) -> Iterator[tuple[int, int]]:
        """Yield (maximal complete mask, orbit size) under incidence symmetries.

        Enumerate maximal completes through vertex-orbit representatives,
        then generate their full orbits by the supplied permutations. Every
        maximal complete belongs to exactly one yielded orbit. With no
        generators, yield all maximal completes with orbit size one.
        """
        perms = self._validated_generators(generators)
        if not perms:
            for maximal in self._maximal_completes():
                yield maximal, 1
            return
        remaining: set[int] = set()
        for anchor in self._vertex_orbit_reps(perms):
            remaining.update(self._maximal_completes(anchor))
        while remaining:
            maximal = remaining.pop()
            orbit = {maximal}
            queue = [maximal]
            for mask in queue:
                indices = tuple(_indices(mask))
                for perm in perms:
                    image = sum(1 << perm[i] for i in indices)
                    if image not in orbit:
                        orbit.add(image)
                        queue.append(image)
                        remaining.discard(image)
            yield min(orbit), len(orbit)


@dataclass(frozen=True)
class ClosedCenterResult:
    """True: criterion holds; False: witness; None: check limit reached."""

    holds: bool | None
    checked: int
    witness: int | None = None


def check_closed_centers(
    index: ClosedNecktieIndex,
    *,
    generators: Iterable[Iterable[int]] = (),
    max_checks: int | None = None,
) -> ClosedCenterResult:
    """Check the necktie-center hypothesis on closed completes.

    A center of h(A) is a center of A in every maximal complete containing
    A. Conversely a failure on closed A is itself a counterexample to the
    hypothesis, not to homotopy equivalence. The limit counts yielded
    completes, including repeats across anchors, not internal search steps.
    """
    if max_checks is not None and max_checks < 0:
        raise ValueError("max_checks must be nonnegative")
    checked = 0
    for closed in index.closed_representatives(generators):
        if max_checks is not None and checked >= max_checks:
            return ClosedCenterResult(None, checked)
        checked += 1
        if index.is_bad(closed) and not index.centers(closed):
            return ClosedCenterResult(False, checked, closed)
    return ClosedCenterResult(True, checked)

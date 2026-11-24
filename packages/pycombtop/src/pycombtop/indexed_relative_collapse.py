"""Indexed elementary collapses relative to a preserved subcomplex."""

from collections import defaultdict, deque
from collections.abc import Iterable, Iterator

from .relative_collapse import CollapseResult, CollapseStatus
from .simplex import Simplex, SimplicialComplex, _stable_vertex_key


def _bits(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit
        mask ^= bit


def _relative_masks(
    source: SimplicialComplex,
    target: SimplicialComplex,
    limit: int | None,
) -> tuple[tuple, set[int] | None]:
    """Enumerate only faces outside target, descending from source facets."""
    vertices = tuple(
        sorted(
            {v for f in source.facet_set for v in f},
            key=_stable_vertex_key,
        )
    )
    index = {v: 1 << i for i, v in enumerate(vertices)}
    source_masks = {sum(index[v] for v in f) for f in source.facet_set}
    if any(v not in index for f in target.facet_set for v in f):
        raise ValueError("target must be a subcomplex of source")
    target_masks = {sum(index[v] for v in f) for f in target.facet_set}
    if any(
        not any(t & s == t for s in source_masks) for t in target_masks
    ):
        raise ValueError("target must be a subcomplex of source")

    # Every face contained in a target facet has all its subfaces there too.
    # Index target facets by vertex to recognize such branches without ever
    # expanding them.
    stars: dict[int, int] = defaultdict(int)
    for i, facet in enumerate(target_masks):
        for bit in _bits(facet):
            stars[bit] |= 1 << i
    all_targets = (1 << len(target_masks)) - 1

    def preserved(face: int) -> bool:
        containing = all_targets
        for bit in _bits(face):
            containing &= stars.get(bit, 0)
            if not containing:
                return False
        return bool(containing)

    bad: set[int] = set()
    stack = sorted(source_masks)
    while stack:
        face = stack.pop()
        if face in bad or preserved(face):
            continue
        if limit is not None and len(bad) >= limit:
            return vertices, None
        bad.add(face)
        stack.extend(face ^ bit for bit in _bits(face))
    return vertices, bad


def _decode(vertices: tuple, mask: int) -> Simplex:
    return Simplex(vertices[b.bit_length() - 1] for b in _bits(mask))


def indexed_relative_collapse(
    source: SimplicialComplex,
    target: SimplicialComplex,
    *,
    max_bad_faces: int | None = 1_000_000,
) -> CollapseResult:
    """Greedily collapse source onto target using an immediate-coface index.

    All cofaces of a face outside target are also outside target. A nonempty
    such face with exactly one immediate coface is free, so no maximal-facet
    recomputation is necessary. The target's simplices are never expanded.

    A greedy dead end or a face-budget exit is inconclusive, not proof that
    the collapse is impossible. The existing backtracking routine can be
    used on small complexes when this algorithm stalls. The face budget is
    a cap on distinct relative faces, not a time or memory guarantee.
    """
    if max_bad_faces is not None and max_bad_faces < 0:
        raise ValueError("max_bad_faces must be nonnegative or None")
    vertices, live = _relative_masks(source, target, max_bad_faces)
    if live is None:
        return CollapseResult(
            CollapseStatus.INCONCLUSIVE, (), "Relative face budget exceeded.", 0
        )
    parents: dict[int, set[int]] = defaultdict(set)
    for face in live:
        for bit in _bits(face):
            lower = face ^ bit
            if lower in live:
                parents[lower].add(face)
    queue = deque(sorted(f for f in live if f and len(parents[f]) == 1))
    pairs: list[tuple[Simplex, Simplex]] = []
    while queue:
        lower = queue.popleft()
        if lower not in live or len(parents[lower]) != 1:
            continue
        upper = next(iter(parents[lower]))
        pairs.append((_decode(vertices, lower), _decode(vertices, upper)))
        for face in (lower, upper):
            live.remove(face)
            for bit in _bits(face):
                sub = face ^ bit
                if sub in live:
                    parents[sub].remove(face)
                    if sub and len(parents[sub]) == 1:
                        queue.append(sub)
    if live:
        return CollapseResult(
            CollapseStatus.INCONCLUSIVE,
            tuple(pairs),
            f"Greedy dead end with {len(live)} relative faces.",
            len(pairs) + 1,
        )
    if not verify_indexed_relative_collapse(
        source, target, pairs, max_bad_faces=max_bad_faces
    ):
        raise RuntimeError("indexed relative-collapse certificate failed replay")
    return CollapseResult(
        CollapseStatus.PROVED,
        tuple(pairs),
        "Certificate independently replayed on the relative faces.",
        len(pairs) + 1,
    )


def verify_indexed_relative_collapse(
    source: SimplicialComplex,
    target: SimplicialComplex,
    pairs: Iterable[tuple[Simplex, Simplex]],
    *,
    max_bad_faces: int | None = 1_000_000,
) -> bool:
    """Replay a certificate by direct immediate-coface membership tests.

    Rebuilds the relative face set from facets and does not use the search's
    coface index. A resource-limit exit returns False, not a certificate.
    """
    if max_bad_faces is not None and max_bad_faces < 0:
        raise ValueError("max_bad_faces must be nonnegative or None")
    try:
        vertices, live = _relative_masks(source, target, max_bad_faces)
        if live is None:
            return False
        index = {v: 1 << i for i, v in enumerate(vertices)}
        full = (1 << len(vertices)) - 1
        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                return False
            lower, upper = (
                sum(index[v] for v in frozenset(face)) for face in pair
            )
            if (
                not lower
                or lower not in live
                or upper not in live
                or lower & upper != lower
                or (lower ^ upper).bit_count() != 1
            ):
                return False
            if any(
                (lower | bit) != upper and (lower | bit) in live
                for bit in _bits(full ^ lower)
            ):
                return False
            live.remove(lower)
            live.remove(upper)
        return not live
    except (KeyError, TypeError, ValueError):
        return False

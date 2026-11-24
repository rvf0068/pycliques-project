"""Alternating nerve reduction without expanding simplices.

For a finite complex covered by its maximal simplices, the nerve theorem
identifies its homotopy type with that of its facet-incidence nerve. Repeat
that construction while the number of facets decreases. A one-facet
endpoint certifies contractibility; a larger endpoint is inconclusive.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


def _indices(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _maximal_masks(masks: Iterable[int]) -> tuple[int, ...]:
    maximal: list[int] = []
    for mask in sorted(set(masks) - {0}, key=lambda m: (-m.bit_count(), m)):
        if not any(mask & other == mask for other in maximal):
            maximal.append(mask)
    return tuple(maximal)


def _dual(facets: tuple[int, ...]) -> tuple[int, ...]:
    incidence: dict[int, int] = {}
    for i, facet in enumerate(facets):
        for v in _indices(facet):
            incidence[v] = incidence.get(v, 0) | (1 << i)
    return _maximal_masks(incidence.values())


@dataclass(frozen=True)
class NerveReduction:
    """Exact nerve-equivalent endpoint and its successive facet counts."""

    facets: tuple[int, ...]
    facet_counts: tuple[int, ...]

    @property
    def proved_contractible(self) -> bool:
        """A one-facet nonempty endpoint is a simplex."""
        return len(self.facets) == 1


def alternating_nerve_reduction(facets: Iterable[int]) -> NerveReduction:
    """Reduce a complex specified by nonnegative integer facet masks.

    The empty mask is ignored. Each nerve step relabels vertices by facet
    indices, and removes nonmaximal incidence stars. Two successive nerve
    steps cannot increase the number of maximal facets. Stop when that
    number ceases to decrease, or at a single facet. The larger endpoint
    remains homotopy equivalent to the input; it need not be a minimal core.
    """
    masks = tuple(facets)
    if any(not isinstance(m, int) or m < 0 for m in masks):
        raise ValueError("facets must be nonnegative integer masks")
    current = _maximal_masks(masks)
    counts = [len(current)]
    while len(current) > 1:
        old_count = len(current)
        current = _dual(current)
        counts.append(len(current))
        if len(current) <= 1:
            break
        current = _dual(current)
        counts.append(len(current))
        if len(current) >= old_count:
            break
    return NerveReduction(current, tuple(counts))


def verify_nerve_reduction(facets: Iterable[int], certificate: NerveReduction) -> bool:
    """Replay by explicit set incidences, independently of the bitset dual."""
    try:
        masks = tuple(facets)
        if any(not isinstance(m, int) or m < 0 for m in masks):
            return False

        def maximal(family: set[frozenset[int]]) -> list[frozenset[int]]:
            keep = [f for f in family if f and not any(f < g for g in family)]
            return sorted(keep, key=lambda f: (-len(f), sum(1 << v for v in f)))

        current = maximal({frozenset(_indices(m)) for m in masks})
        counts = [len(current)]
        while len(current) > 1:
            old_count = len(current)
            for _ in range(2):
                vertices: set[int] = set().union(*current)
                incidence = {
                    frozenset(i for i, f in enumerate(current) if v in f)
                    for v in vertices
                }
                current = maximal(incidence)
                counts.append(len(current))
                if len(current) <= 1:
                    break
            if len(current) <= 1 or len(current) >= old_count:
                break
        endpoint = tuple(sum(1 << v for v in f) for f in current)
        return certificate == NerveReduction(endpoint, tuple(counts))
    except (AttributeError, TypeError, ValueError):
        return False

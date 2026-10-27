"""Certificate-producing elementary collapses relative to a subcomplex."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from .simplex import Simplex, SimplicialComplex


class CollapseStatus(Enum):
    """Outcome of a bounded relative-collapse search."""

    PROVED = "PROVED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class CollapseResult:
    """A relative-collapse certificate and search diagnostics."""

    status: CollapseStatus
    pairs: tuple[tuple[Simplex, Simplex], ...]
    diagnostics: str
    explored_states: int

    def certificate_json(self) -> str:
        """Return the collapse pairs as compact JSON with stable face ordering.

        Vertices must be JSON-serializable. This is intended for reproducible
        census output and deliberately raises ``TypeError`` for other labels.
        """
        encoded_pairs = [
            [
                sorted(free_face, key=repr),
                sorted(coface, key=repr),
            ]
            for free_face, coface in self.pairs
        ]
        return json.dumps(encoded_pairs, separators=(",", ":"), sort_keys=True)


def _validate_pair(
    current: set[Simplex],
    target_simplices: set[Simplex],
    free_face: Simplex,
    coface: Simplex,
) -> bool:
    if free_face in target_simplices or coface in target_simplices:
        return False
    if free_face not in current or coface not in current:
        return False
    if len(coface) != len(free_face) + 1 or not free_face < coface:
        return False

    maximal_cofaces = [
        facet
        for facet in current
        if free_face <= facet
        and not any(facet < other for other in current if facet < other)
    ]
    return maximal_cofaces == [coface]


def verify_relative_collapse(
    source: SimplicialComplex,
    target: SimplicialComplex,
    pairs: tuple[tuple[Simplex, Simplex], ...] | list[tuple[Simplex, Simplex]],
) -> bool:
    """Independently replay and validate a relative elementary-collapse proof.

    Returns ``True`` exactly when the target is a subcomplex of ``source`` and
    the listed pairs reduce the source to precisely the target without
    deleting any target face.
    """
    current = source.all_simplices()
    target_simplices = target.all_simplices()
    if not target_simplices <= current:
        return False

    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return False
        free_face, coface = map(Simplex, pair)
        if not _validate_pair(current, target_simplices, free_face, coface):
            return False
        current.remove(free_face)
        current.remove(coface)

    return current == target_simplices


def _facets(simplices: set[Simplex]) -> set[Simplex]:
    return {face for face in simplices if not any(face < other for other in simplices)}


def _available_pairs(
    current: set[Simplex], target_simplices: set[Simplex]
) -> list[tuple[Simplex, Simplex]]:
    facets = _facets(current)
    pairs: list[tuple[Simplex, Simplex]] = []
    for free_face in current:
        if not free_face or free_face in target_simplices:
            continue
        cofaces = [facet for facet in facets if free_face <= facet]
        if len(cofaces) == 1 and len(cofaces[0]) == len(free_face) + 1:
            coface = cofaces[0]
            if coface not in target_simplices:
                pairs.append((free_face, coface))
    return sorted(
        pairs,
        key=lambda pair: (
            len(pair[0]),
            tuple(sorted(map(repr, pair[0]))),
            tuple(sorted(map(repr, pair[1]))),
        ),
    )


def relative_collapse(
    source: SimplicialComplex,
    target: SimplicialComplex,
    *,
    max_states: int | None = 10_000,
) -> CollapseResult:
    """Search for elementary collapses of ``source`` onto ``target``.

    ``target`` is immutable throughout the search. Competing free-face pairs
    are explored with depth-first bounded backtracking. Exhausting the state
    budget or finding no sequence returns ``INCONCLUSIVE``; neither outcome
    establishes that no relative collapse exists.
    """
    if max_states is not None and max_states < 1:
        raise ValueError("max_states must be positive or None")

    initial = source.all_simplices()
    target_simplices = target.all_simplices()
    if not target_simplices <= initial:
        raise ValueError("target must be a subcomplex of source")

    explored = 0
    visited: set[frozenset[Simplex]] = set()
    best_path: tuple[tuple[Simplex, Simplex], ...] = ()
    budget_exhausted = False
    initial_state = frozenset(initial)
    stack: list[tuple[set[Simplex], tuple[tuple[Simplex, Simplex], ...]]] = [
        (initial, ())
    ]
    queued = {initial_state}
    while stack:
        current, path = stack.pop()
        state = frozenset(current)
        queued.discard(state)
        if current == target_simplices:
            best_path = path
            break
        if state in visited:
            continue
        if max_states is not None and explored >= max_states:
            budget_exhausted = True
            break
        visited.add(state)
        explored += 1

        candidates = _available_pairs(current, target_simplices)
        for free_face, coface in reversed(candidates):
            next_current = current - {free_face, coface}
            next_state = frozenset(next_current)
            if next_state not in visited and next_state not in queued:
                stack.append((next_current, (*path, (free_face, coface))))
                queued.add(next_state)
    if best_path and verify_relative_collapse(source, target, best_path):
        return CollapseResult(
            CollapseStatus.PROVED,
            best_path,
            "Certificate independently replayed to the preserved target.",
            explored,
        )
    if initial == target_simplices:
        return CollapseResult(
            CollapseStatus.PROVED, (), "Source equals target.", explored
        )

    diagnostic = (
        f"Search stopped at max_states={max_states}."
        if budget_exhausted
        else (
            "No sequence found in the explored search; "
            "non-collapsibility is not established."
        )
    )
    return CollapseResult(CollapseStatus.INCONCLUSIVE, (), diagnostic, explored)

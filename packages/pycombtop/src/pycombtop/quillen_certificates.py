"""Weak-point reductions and verifiable Quillen-A fiber certificates.

The certificates use beat reductions as explicit, replayable witnesses of
contractibility. This is a sufficient certificate format, not a complete
decision procedure for contractibility of finite spaces.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Literal

from .posets import (
    BeatReduction,
    Poset,
    beat_core,
    lower_fibers,
    verify_beat_reduction,
)

WeakPointDirection = Literal["lower", "upper"]


@dataclass(frozen=True)
class WeakPointMove:
    """Deletion of a point whose strict lower or upper set is certified contractible."""

    point: Hashable
    direction: WeakPointDirection
    link_reduction: BeatReduction


@dataclass(frozen=True)
class WeakPointReduction:
    """A finite-poset core and a replayable sequence of weak-point deletions."""

    core: Poset
    moves: tuple[WeakPointMove, ...]


def _strict_link(poset: Poset, point: Hashable, direction: WeakPointDirection) -> Poset:
    """Return the strict lower or upper set of a point."""
    if direction == "lower":
        return poset.induced(poset.lower_set(point, strict=True))
    return poset.induced(poset.upper_set(point, strict=True))


def weak_point_moves(poset: Poset) -> tuple[WeakPointMove, ...]:
    """Find weak points with beat-reducible strict lower or upper sets.

    A finite-poset point is weak when its strict lower or upper set is
    contractible. Here contractibility is certified constructively by a beat
    reduction to a singleton. Other contractible links may not be detected.
    """
    moves: list[WeakPointMove] = []
    for point in poset:
        for direction in ("lower", "upper"):
            reduction = beat_core(_strict_link(poset, point, direction))
            if len(reduction.core) == 1:
                moves.append(WeakPointMove(point, direction, reduction))
    return tuple(moves)


def weak_point_core(poset: Poset) -> WeakPointReduction:
    """Delete certified weak points until none remain."""
    current = poset
    moves: list[WeakPointMove] = []
    while (available := weak_point_moves(current)):
        move = available[0]
        moves.append(move)
        current = current.induced(x for x in current if x != move.point)
    return WeakPointReduction(current, tuple(moves))


def verify_weak_point_reduction(
    poset: Poset, reduction: WeakPointReduction
) -> bool:
    """Replay all link-contractibility witnesses and check the terminal core."""
    if not isinstance(reduction, WeakPointReduction):
        return False
    current = poset
    for move in reduction.moves:
        if not isinstance(move, WeakPointMove) or move not in weak_point_moves(current):
            return False
        link = _strict_link(current, move.point, move.direction)
        if (
            not verify_beat_reduction(link, move.link_reduction)
            or len(move.link_reduction.core) != 1
        ):
            return False
        current = current.induced(x for x in current if x != move.point)
    return current == reduction.core and not weak_point_moves(current)


@dataclass(frozen=True)
class QuillenFiberCertificate:
    """Beat-reduction witnesses for all principal lower fibers of a poset map.

    By Quillen's Theorem A, if every lower fiber is contractible, the induced
    map of order complexes is a homotopy equivalence. This certificate proves
    that sufficient condition using beat reductions to singletons.
    """

    mapping: tuple[tuple[Hashable, Hashable], ...]
    fibers: tuple[tuple[Hashable, BeatReduction], ...]


def quillen_fiber_certificate(
    source: Poset,
    target: Poset,
    mapping: Mapping[Hashable, Hashable],
) -> QuillenFiberCertificate | None:
    """Construct a sufficient Quillen-A certificate, or return None.

    For each y in target, the fiber is f^{-1}(target_{<= y}). The function
    returns a certificate exactly when each such fiber beat-reduces to one
    point. Invalid or non-order-preserving maps raise ValueError, as in
    :func:`lower_fibers`.
    """
    fibers = lower_fibers(source, target, mapping)
    witnesses: list[tuple[Hashable, BeatReduction]] = []
    for point in target:
        reduction = beat_core(fibers[point])
        if len(reduction.core) != 1:
            return None
        witnesses.append((point, reduction))
    return QuillenFiberCertificate(
        tuple((point, mapping[point]) for point in source),
        tuple(witnesses),
    )


def verify_quillen_fiber_certificate(
    source: Poset,
    target: Poset,
    mapping: Mapping[Hashable, Hashable],
    certificate: QuillenFiberCertificate,
) -> bool:
    """Verify every lower-fiber witness without trusting its construction."""
    if not isinstance(certificate, QuillenFiberCertificate):
        return False
    try:
        fibers = lower_fibers(source, target, mapping)
    except (KeyError, TypeError, ValueError):
        return False
    expected_mapping = tuple((point, mapping[point]) for point in source)
    if certificate.mapping != expected_mapping:
        return False
    if tuple(point for point, _ in certificate.fibers) != tuple(target):
        return False
    for point, reduction in certificate.fibers:
        fiber = fibers[point]
        if (
            not verify_beat_reduction(fiber, reduction)
            or len(reduction.core) != 1
        ):
            return False
    return True

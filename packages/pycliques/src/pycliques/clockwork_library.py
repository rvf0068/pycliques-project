"""Generate theorem-certified divergent clockwork retraction targets."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from math import comb, factorial, prod

import networkx as nx

from .clockwork import (
    ClockworkDivergenceStatus,
    clockwork_divergence_result,
    clockwork_graph,
)


@dataclass(frozen=True)
class ClockworkTarget:
    """Immutable construction parameters of a certified divergent target.

    The graph property constructs a fresh graph on every access.
    """

    segment_sizes: tuple[int, ...]
    core_adjacencies: tuple[tuple[int, ...], ...]
    crown_size: int
    permutation: tuple[int, ...]

    @property
    def graph(self) -> nx.Graph:
        """Reconstruct the target with the original construction labels."""
        return clockwork_graph(
            list(self.segment_sizes),
            [list(row) for row in self.core_adjacencies],
            self.crown_size,
            list(self.permutation),
        )


@dataclass(frozen=True)
class ClockworkTargetLibrary:
    """Pairwise nonisomorphic certified targets and enumeration diagnostics.

    The counts are counts of parameter templates, not isomorphism classes.
    An inconclusive recognition is counted in unresolved_templates.
    """

    targets: tuple[ClockworkTarget, ...]
    templates_checked: int
    unresolved_templates: int


def generate_divergent_clockwork_library(
    segment_sizes: Iterable[Iterable[int]] = ((1, 1, 1), (2, 1, 1)),
    *,
    crown_size: int = 2,
    max_templates: int = 10_000,
    fallback_enabled: bool = False,
    fallback_max_order: int = 9,
    fallback_max_candidates: int = 5000,
) -> ClockworkTargetLibrary:
    """Enumerate a finite family of certified divergent clockwork graphs.

    For each supplied cyclic core-size tuple, enumerate the nondecreasing
    adjacency-count rows of :func:`pycliques.clockwork.core` and every
    wrap-around crown permutation. Retain only graphs whose divergence
    is established by :func:`pycliques.clockwork.clockwork_divergence_result`.
    Deduplicate by exact graph isomorphism, including across size tuples.

    Each size tuple must have at least three positive integer entries and
    the uniform crown size must be at least two. Duplicate size tuples
    are ignored. If the number of parameter templates would exceed
    max_templates, raise ValueError before constructing any graphs.

    Recognition limits are forwarded to the theorem checker. With the
    default fast-only recognition, the output need not contain every
    divergent graph even in this restricted template family. The library
    is not an enumeration of all divergent clockworks.

    Example
    -------
    To try the generated targets on a host graph, pass each target.graph
    to pycliques.retractions.retracts and independently verify any witness.
    A negative search only excludes the targets actually checked.
    """
    if (
        not isinstance(crown_size, int)
        or isinstance(crown_size, bool)
        or crown_size < 2
    ):
        raise ValueError("crown_size must be an integer at least two")
    if (
        not isinstance(max_templates, int)
        or isinstance(max_templates, bool)
        or max_templates < 1
    ):
        raise ValueError("max_templates must be a positive integer")
    if fallback_max_order < 0 or fallback_max_candidates < 1:
        raise ValueError("invalid recognition limits")

    sizes_list: list[tuple[int, ...]] = []
    expected = 0
    for raw_sizes in segment_sizes:
        sizes = tuple(raw_sizes)
        if len(sizes) < 3 or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in sizes
        ):
            raise ValueError("each core-size tuple needs >=3 positive integers")
        if sizes in sizes_list:
            continue
        sizes_list.append(sizes)
        # A row of length a with entries in 0..b has binomial(a+b,a)
        # nondecreasing possibilities (combinations with replacement).
        expected += factorial(crown_size) * prod(
            comb(size + sizes[(i + 1) % len(sizes)], size)
            for i, size in enumerate(sizes)
        )
        if expected > max_templates:
            raise ValueError("template family exceeds max_templates")

    representatives: list[tuple[nx.Graph, ClockworkTarget]] = []
    checked = unresolved = 0
    for sizes in sizes_list:
        tables = [
            tuple(
                itertools.combinations_with_replacement(
                    range(sizes[(i + 1) % len(sizes)] + 1), size
                )
            )
            for i, size in enumerate(sizes)
        ]
        for rows in itertools.product(*tables):
            for permutation in itertools.permutations(range(crown_size)):
                target = ClockworkTarget(sizes, rows, crown_size, permutation)
                graph = target.graph
                checked += 1
                result = clockwork_divergence_result(
                    graph,
                    fallback_enabled=fallback_enabled,
                    fallback_max_order=fallback_max_order,
                    fallback_max_candidates=fallback_max_candidates,
                )
                if result.status is ClockworkDivergenceStatus.LIMIT_REACHED:
                    unresolved += 1
                if result.status is not ClockworkDivergenceStatus.DIVERGENT:
                    continue
                if any(nx.is_isomorphic(graph, old) for old, _ in representatives):
                    continue
                representatives.append((graph, target))
    return ClockworkTargetLibrary(
        tuple(target for _, target in representatives), checked, unresolved
    )

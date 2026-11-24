"""Verify the local contractibility hypothesis of LPV08, Theorem 5.7."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from pycliques.closed_neckties import ClosedNecktieIndex

from .strong_collapses import (
    uniform_strong_collapse_certificate,
    verify_uniform_strong_collapse_certificate,
)


@dataclass(frozen=True)
class ClosedTraceResult:
    """Proof status and counts; a False result is only inconclusive."""

    proved: bool
    visited: int
    completed_orbits: int
    covered_maximal_completes: int
    pruned_intervals: int
    individual_certificates: int
    unresolved: int | None = None
    strategy: str = "maximal"


class _NodeLimit(Exception):
    pass


def check_closed_trace_contractibility(
    index: ClosedNecktieIndex,
    *,
    generators: Iterable[Iterable[int]] = (),
    max_nodes: int | None = None,
    strategy: str = "maximal",
) -> ClosedTraceResult:
    """Check all nonempty closed completes using uniform strong collapses.

    The index must represent the entire maximal-clique family of a graph
    for the conclusion of LPV08 Theorem 5.7 to apply to that graph.
    Every successful interval or individual certificate is independently
    replayed. A failed collapse or node-budget exit is inconclusive,
    never evidence of noncontractibility. The budget counts nonempty
    closed nodes visited, including repeats across maximal completes;
    orbit enumeration itself is not subject to that budget.
    Strategy 'vertex' starts from vertex orbits without enumerating maximal
    completes; covered_maximal_completes is then zero (not counted).
    """
    if max_nodes is not None and max_nodes < 0:
        raise ValueError("max_nodes must be nonnegative or None")
    if strategy not in {"maximal", "vertex"}:
        raise ValueError("strategy must be maximal or vertex")
    visited = completed = covered = pruned = individual = 0

    def result(proved: bool, unresolved: int | None = None) -> ClosedTraceResult:
        return ClosedTraceResult(
            proved,
            visited,
            completed,
            covered,
            pruned,
            individual,
            unresolved,
            strategy,
        )

    def certify(m: int, e: int) -> bool:
        fixed = index.members(m)
        possible = index.members(e)
        cert = uniform_strong_collapse_certificate(fixed, possible)
        if cert is None:
            return False
        if not verify_uniform_strong_collapse_certificate(fixed, possible, cert):
            raise RuntimeError("uniform strong-collapse certificate failed replay")
        return True

    def prune(m: int, e: int) -> bool:
        nonlocal visited, pruned
        if max_nodes is not None and visited >= max_nodes:
            raise _NodeLimit
        visited += 1
        if certify(m, e):
            pruned += 1
            return True
        return False

    def searches() -> Iterator[tuple[Iterator[int], int]]:
        if strategy == "vertex":
            for anchor in index.vertex_orbit_representatives(generators):
                yield index.closed_containing(anchor, prune=prune), 0
        else:
            for maximal, size in index.maximal_complete_orbits(generators):
                yield index.closed_within(maximal, prune=prune), size

    try:
        for search, size in searches():
            for closed in search:
                # A common vertex already supplies a cone point.
                if index.is_bad(closed):
                    if not certify(closed, closed):
                        return result(False, closed)
                    individual += 1
            completed += 1
            covered += size
    except _NodeLimit:
        return result(False)
    return result(True)

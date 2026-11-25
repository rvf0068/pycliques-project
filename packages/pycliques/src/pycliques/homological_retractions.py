"""Homological obstructions to arbitrary graph retractions.

No dependency on pycombtop: callers supply certified reduced mod-2 Betti
bounds. A retraction induces a retraction on every iterated clique graph.
Therefore a target lower bound exceeding a host upper bound at the same
iterate and degree obstructs a retraction. Missing entries are unknown.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class HomologicalRetractionObstruction:
    """The precise inequality contradicted by a hypothetical retraction."""

    iterate: int
    degree: int
    host_upper_bound: int
    target_lower_bound: int


def _validate_betti_bounds(bounds: Mapping[int, int]) -> None:
    if any(
        type(d) is not int or d < 0 or type(b) is not int or b < 0
        for d, b in bounds.items()
    ):
        raise ValueError("Betti degrees and bounds must be nonnegative integers")


def find_homological_retraction_obstruction(
    host_upper_bounds: Mapping[int, Mapping[int, int]],
    target_lower_bounds: Mapping[int, Mapping[int, int]],
) -> HomologicalRetractionObstruction | None:
    """Compare certified reduced GF(2) bounds indexed by iterate and degree.

    Exact Betti numbers may be supplied as either upper or lower bounds.
    Level 0 denotes the original graph (or starting pared iterate); level
    j denotes its j-th clique graph, or a homotopy-equivalent pared iterate.
    Both profiles must use the same relative level and coefficient field.
    Bounds are trusted inputs; no homology computation occurs here.
    None means no obstruction was established, NOT that a retraction exists.
    """
    for profile in (host_upper_bounds, target_lower_bounds):
        for level, bounds in profile.items():
            if type(level) is not int or level < 0:
                raise ValueError("Iterate indices must be nonnegative integers")
            _validate_betti_bounds(bounds)
    for level in sorted(host_upper_bounds.keys() & target_lower_bounds.keys()):
        upper = host_upper_bounds[level]
        lower = target_lower_bounds[level]
        for degree in sorted(upper.keys() & lower.keys()):
            if lower[degree] > upper[degree]:
                return HomologicalRetractionObstruction(
                    level, degree, upper[degree], lower[degree]
                )
    return None

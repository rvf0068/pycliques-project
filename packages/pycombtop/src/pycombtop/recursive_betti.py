"""Face-free upper bounds on reduced Betti numbers of flag complexes.

In the complement H, pick a vertex u and delete its neighbors v_i in
sequence. The final deletion is a cone. The deletion/link exact sequence
bounds beta_d by the sum of beta_(d-1) of the intermediate links.
The recurrence holds over every field. Zero therefore also proves integral
homology vanishing (by the universal coefficient theorem).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import networkx as nx


@dataclass(frozen=True)
class RecursiveBettiBounds:
    """Certified upper bounds, including only completed degrees.

    A positive upper bound is NOT an exact Betti number. A zero is exact.
    The bounds hold over every field. Limits do not imply vanishing.
    """

    upper_bounds: dict[int, int]
    states_explored: int
    is_limited: bool

    @property
    def known_zero_betti(self) -> dict[int, int]:
        """Exact zero entries, usable as mod-2 Betti metadata."""
        return {d: 0 for d, rank in self.upper_bounds.items() if rank == 0}


class _StateLimit(Exception):
    pass


def recursive_betti_upper_bounds(
    graph: nx.Graph,
    *,
    max_degree: int,
    max_states: int = 100_000,
    max_cache: int = 100_000,
) -> RecursiveBettiBounds:
    """Bound reduced homology in degrees 0 through max_degree.

    No faces are enumerated. In each induced complement, choose a
    minimum-degree pivot, breaking ties by input vertex order. Delete its
    neighbors in decreasing degree order (ties by input order). Compute
    each degree in ascending order, retaining completed bounds on limits.

    The augmented empty independence complex has beta_(-1)=1; a
    nonempty complex has beta_(-1)=0. Isolated complement vertices
    give cones. Cache size and uncached nontrivial recursion states
    are bounded; max_cache=0 disables caching. Recursion exhaustion
    is also inconclusive. Bounds are reproducible for fixed vertex order.
    """
    if (
        not isinstance(graph, nx.Graph)
        or graph.is_directed()
        or graph.is_multigraph()
        or nx.number_of_selfloops(graph)
    ):
        raise ValueError("Require a simple undirected loopless graph")
    if type(max_degree) is not int or max_degree < 0:
        raise ValueError("max_degree must be a nonnegative integer")
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if type(max_cache) is not int or max_cache < 0:
        raise ValueError("max_cache must be a nonnegative integer")
    nodes = list(graph)
    index = {v: i for i, v in enumerate(nodes)}
    full = (1 << len(nodes)) - 1
    complement = [
        full ^ ((1 << i) | sum(1 << index[w] for w in graph[v]))
        for i, v in enumerate(nodes)
    ]
    states = 0

    def indices(mask: int):
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    @lru_cache(maxsize=max_cache)
    def bound(mask: int, degree: int) -> int:
        nonlocal states
        if degree == -1:
            return int(mask == 0)
        if not mask or degree >= mask.bit_count():
            return 0
        if states >= max_states:
            raise _StateLimit
        states += 1
        pivot = -1
        smallest = len(nodes) + 1
        for v in indices(mask):
            count = (complement[v] & mask).bit_count()
            if count == 0:
                return 0
            if count < smallest:
                pivot, smallest = v, count
        neighbors = sorted(
            indices(complement[pivot] & mask),
            key=lambda v: (-(complement[v] & mask).bit_count(), v),
        )
        remaining = mask
        total = 0
        for v in neighbors:
            bit = 1 << v
            link = remaining & ~(complement[v] | bit)
            total += bound(link, degree - 1)
            remaining &= ~bit
        return total

    bounds: dict[int, int] = {}
    limited = False
    try:
        for degree in range(max_degree + 1):
            bounds[degree] = bound(full, degree)
    except (_StateLimit, RecursionError):
        limited = True
    finally:
        bound.cache_clear()
    return RecursiveBettiBounds(bounds, states, limited)

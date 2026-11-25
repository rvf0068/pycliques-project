"""Face-free upper bounds on reduced Betti numbers of flag complexes.

In the complement H, pick a vertex u and delete its neighbors v_i in
sequence. The final deletion is a cone. The deletion/link exact sequence
bounds beta_d by the sum of beta_(d-1) of the intermediate links.
The recurrence holds over every field. Zero therefore also proves integral
homology vanishing (by the universal coefficient theorem).
"""

from __future__ import annotations

from collections.abc import Iterator
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
    reduced_euler_characteristic: int | None = None

    @property
    def known_zero_betti(self) -> dict[int, int]:
        """Exact zero entries, usable as mod-2 Betti metadata."""
        return {d: 0 for d, rank in self.upper_bounds.items() if rank == 0}

    @property
    def euler_refined_upper_bounds(self) -> dict[int, int]:
        """Tighten bounds using the exact Euler characteristic, when available."""
        chi = self.reduced_euler_characteristic
        if chi is None:
            return dict(self.upper_bounds)
        even = sum(b for d, b in self.upper_bounds.items() if d % 2 == 0)
        odd = sum(b for d, b in self.upper_bounds.items() if d % 2 == 1)
        return {
            d: min(b, chi + odd if d % 2 == 0 else even - chi)
            for d, b in self.upper_bounds.items()
        }


class _StateLimit(Exception):
    pass


def recursive_betti_upper_bounds(
    graph: nx.Graph,
    *,
    max_degree: int | None = None,
    max_states: int = 100_000,
    max_cache: int = 100_000,
    neighbor_order: str = "descending",
) -> RecursiveBettiBounds:
    """Bound reduced homology in degrees 0 through max_degree, or all degrees.

    No faces are enumerated. In each induced complement, choose a
    minimum-degree pivot, breaking ties by input vertex order. Delete its
    neighbors in decreasing degree order (ties by input order). Compute
    each degree in ascending order, retaining completed bounds on limits.

    With max_degree=None, one cached polynomial recurrence computes all
    coefficients simultaneously. A completed result includes degrees 0
    through n-1 (explicit zeros), and its alternating sum gives the exact
    reduced Euler characteristic. A limited all-degree run returns no
    bounds or Euler characteristic. The finite-degree mode is unchanged.
    Alternate neighbor orders can give tighter bounds; use descending,
    ascending, or reverse_ties (descending degree, reverse input ties).

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
    if max_degree is not None and (type(max_degree) is not int or max_degree < 0):
        raise ValueError("max_degree must be a nonnegative integer or None")
    if neighbor_order not in ("descending", "ascending", "reverse_ties"):
        raise ValueError("Unknown neighbor_order")
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

    def indices(mask: int) -> Iterator[int]:
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    def branches(mask: int) -> Iterator[int]:
        nonlocal states
        if states >= max_states:
            raise _StateLimit
        states += 1
        pivot = -1
        smallest = len(nodes) + 1
        for v in indices(mask):
            count = (complement[v] & mask).bit_count()
            if count == 0:
                return
            if count < smallest:
                pivot, smallest = v, count
        sign = 1 if neighbor_order == "ascending" else -1
        neighbors = sorted(
            indices(complement[pivot] & mask),
            key=lambda v: (
                sign * (complement[v] & mask).bit_count(),
                -v if neighbor_order == "reverse_ties" else v,
            ),
        )
        remaining = mask
        for v in neighbors:
            bit = 1 << v
            yield remaining & ~(complement[v] | bit)
            remaining &= ~bit

    @lru_cache(maxsize=max_cache)
    def bound(mask: int, degree: int) -> int:
        if degree == -1:
            return int(mask == 0)
        if not mask or degree >= mask.bit_count():
            return 0
        return sum(bound(link, degree - 1) for link in branches(mask))

    @lru_cache(maxsize=max_cache)
    def polynomial(mask: int) -> tuple[int, ...]:
        # Index k represents degree k-1, including the augmented empty face.
        if not mask:
            return (1,)
        total: list[int] = []
        for link in branches(mask):
            child = polynomial(link)
            if not child:
                continue
            total.extend([0] * max(0, len(child) + 1 - len(total)))
            for k, coefficient in enumerate(child):
                total[k + 1] += coefficient
        return tuple(total)

    bounds: dict[int, int] = {}
    limited = False
    euler: int | None = None
    try:
        if max_degree is None:
            coefficients = polynomial(full)
            bounds = {
                d: coefficients[d + 1] if d + 1 < len(coefficients) else 0
                for d in range(len(nodes))
            }
            # The deletion-link identity is exact for Euler characteristic,
            # even though the individual coefficients are only upper bounds.
            euler = sum(
                coefficient if k % 2 else -coefficient
                for k, coefficient in enumerate(coefficients)
            )
        else:
            for degree in range(max_degree + 1):
                bounds[degree] = bound(full, degree)
    except (_StateLimit, RecursionError):
        limited = True
    finally:
        bound.cache_clear()
        polynomial.cache_clear()
    return RecursiveBettiBounds(bounds, states, limited, euler)

"""Bounded existence search for weak coaffine graph endomorphisms.

A retraction onto a graph admitting such an endomorphism lifts it to
the host by inclusion, endomorphism, retraction. Exhaustion therefore
rules out every such retract, not just octahedra.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import networkx as nx

from .retractions import RetractionSearchStatus


def _validate_graph(graph: nx.Graph) -> None:
    if (
        not isinstance(graph, nx.Graph)
        or graph.is_directed()
        or graph.is_multigraph()
        or nx.number_of_selfloops(graph)
    ):
        raise ValueError("require a simple undirected loopless graph")


def verify_coaffine_endomorphism(
    graph: nx.Graph, mapping: Mapping[Hashable, Hashable]
) -> bool:
    """Check all vertices and edges; edge images may coincide."""
    try:
        _validate_graph(graph)
        if set(mapping) != set(graph) or not set(mapping.values()) <= set(graph):
            return False
        return all(
            mapping[v] != v and not graph.has_edge(v, mapping[v]) for v in graph
        ) and all(
            mapping[u] == mapping[v] or graph.has_edge(mapping[u], mapping[v])
            for u, v in graph.edges()
        )
    except (TypeError, ValueError, AttributeError):
        return False


@dataclass(frozen=True)
class CoaffineEndomorphismResult:
    """FOUND carries a checked map; EXHAUSTED is an exact negative."""

    status: RetractionSearchStatus
    endomorphism: dict[Hashable, Hashable] | None
    states_explored: int
    assignments_tried: int
    arc_revisions: int


def find_coaffine_endomorphism(
    graph: nx.Graph,
    *,
    max_states: int = 100_000,
    time_limit: float | None = None,
) -> CoaffineEndomorphismResult:
    """Find any coaffine weak endomorphism, not necessarily bijective.

    Image domains are the open neighborhoods in the complement. Arc
    consistency and deterministic MRV branching preserve every solution.
    The search is opt-in and does not modify the octahedral search or its
    native backend. LIMIT_REACHED gives no negative conclusion.
    """
    _validate_graph(graph)
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if time_limit is not None and (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not math.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and positive")

    started = time.monotonic()
    nodes = list(graph)
    n = len(nodes)
    positions = {v: i for i, v in enumerate(nodes)}
    neighbors = [tuple(positions[w] for w in graph[v]) for v in nodes]
    closed = [(1 << i) | sum(1 << j for j in row) for i, row in enumerate(neighbors)]
    full = (1 << n) - 1
    domains = [full ^ mask for mask in closed]
    states = assignments = revisions = 0

    def result(
        status: RetractionSearchStatus,
        mapping: dict[Hashable, Hashable] | None = None,
    ) -> CoaffineEndomorphismResult:
        return CoaffineEndomorphismResult(
            status, mapping, states, assignments, revisions
        )

    class _Limit(Exception):
        pass

    def check_time() -> None:
        if time_limit is not None and time.monotonic() - started >= time_limit:
            raise _Limit

    def propagate(d: list[int], pending: deque[tuple[int, int]]) -> bool:
        nonlocal revisions
        while pending:
            check_time()
            i, j = pending.popleft()
            supported = 0
            choices = d[j]
            while choices and supported != full:
                bit = choices & -choices
                choices ^= bit
                supported |= closed[bit.bit_length() - 1]
            revised = d[i] & supported
            if revised == d[i]:
                continue
            revisions += 1
            if not revised:
                return False
            d[i] = revised
            pending.extend((k, i) for k in neighbors[i] if k != j)
        return True

    def search(d: list[int]) -> dict[Hashable, Hashable] | None:
        nonlocal states, assignments
        check_time()
        if states >= max_states:
            raise _Limit
        states += 1
        undecided = [i for i, mask in enumerate(d) if mask.bit_count() > 1]
        if not undecided:
            return {nodes[i]: nodes[mask.bit_length() - 1] for i, mask in enumerate(d)}
        x = min(undecided, key=lambda i: (d[i].bit_count(), -len(neighbors[i])))
        choices = d[x]
        while choices:
            check_time()
            bit = choices & -choices
            choices ^= bit
            assignments += 1
            candidate = d.copy()
            candidate[x] = bit
            if propagate(candidate, deque((i, x) for i in neighbors[x])):
                answer = search(candidate)
                if answer is not None:
                    return answer
        return None

    if any(not mask for mask in domains):
        return result(RetractionSearchStatus.EXHAUSTED)
    try:
        if not propagate(
            domains, deque((i, j) for i in range(n) for j in neighbors[i])
        ):
            return result(RetractionSearchStatus.EXHAUSTED)
        answer = search(domains)
    except (_Limit, RecursionError):
        return result(RetractionSearchStatus.LIMIT_REACHED)
    if answer is None:
        return result(RetractionSearchStatus.EXHAUSTED)
    if not verify_coaffine_endomorphism(graph, answer):
        raise AssertionError("coaffine solver returned an invalid witness")
    return result(RetractionSearchStatus.FOUND, answer)

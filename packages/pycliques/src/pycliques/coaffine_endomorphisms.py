"""Bounded existence search for weak coaffine graph endomorphisms.

A retraction onto a graph admitting such an endomorphism lifts it to
the host by inclusion, endomorphism, retraction. Exhaustion therefore
rules out every such retract, not just octahedra.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Hashable, Iterator, Mapping
from dataclasses import dataclass

import networkx as nx

from .octahedral_coaffine_native import (
    MAX_NATIVE_ORDER,
    find_coaffine_native_executable,
    run_coaffine_native,
)
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


def _validate_search_order(value_order: str) -> None:
    if value_order not in ("label", "fiber"):
        raise ValueError("value_order must be 'label' or 'fiber'")


def find_coaffine_endomorphism(
    graph: nx.Graph,
    *,
    max_states: int = 100_000,
    time_limit: float | None = None,
    backend: str = "auto",
    value_order: str = "label",
) -> CoaffineEndomorphismResult:
    """Find any coaffine weak endomorphism, not necessarily bijective.

    Both backends maintain the transpose of the image domains. Select an
    MRV pivot, preferring larger source degree on ties. The default label
    ordering preserves the historical branching policy; value_order="fiber"
    tries larger possible fibers first. Neither changes exhaustiveness.

    Native acceleration uses the same optional C++17 executable as
    find_octahedral_retraction_coaffine. LIMIT_REACHED is inconclusive.
    """
    _validate_graph(graph)
    _validate_search_order(value_order)
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if time_limit is not None and (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not math.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and positive")
    if backend not in ("auto", "python", "native"):
        raise ValueError("backend must be 'auto', 'python', or 'native'")
    executable = None if backend == "python" else find_coaffine_native_executable()
    if backend == "native":
        if executable is None:
            raise RuntimeError(
                "Native backend is not built; run "
                "'python -m pycliques.octahedral_coaffine_native --build'"
            )
        if len(graph) > MAX_NATIVE_ORDER:
            raise ValueError(
                f"Native coaffine backend supports at most {MAX_NATIVE_ORDER} vertices"
            )
        if max_states > (1 << 63) - 1:
            raise ValueError("Native max_states must fit in a signed 64-bit integer")
    if executable is not None and len(graph) <= MAX_NATIVE_ORDER and (
        max_states <= (1 << 63) - 1
    ):
        native = run_coaffine_native(
            graph,
            min_dimension=0,
            max_states=max_states,
            time_limit=time_limit,
            executable=executable,
            value_order=value_order,
        )
        status = RetractionSearchStatus[native.status]
        mapping = None
        if status is RetractionSearchStatus.FOUND:
            if native.endomorphism is None:
                raise RuntimeError("Native FOUND result is missing an endomorphism")
            nodes = list(graph)
            mapping = {
                v: nodes[i] for v, i in zip(nodes, native.endomorphism, strict=True)
            }
            if not verify_coaffine_endomorphism(graph, mapping):
                raise RuntimeError("Native backend returned an invalid endomorphism")
        return CoaffineEndomorphismResult(
            status,
            mapping,
            native.states_explored,
            native.assignments_tried,
            native.arc_revisions,
        )
    return _search_coaffine_python(
        graph, max_states=max_states, time_limit=time_limit, value_order=value_order
    )


def _search_coaffine_python(
    graph: nx.Graph,
    *,
    max_states: int,
    time_limit: float | None,
    value_order: str,
    accept: Callable[[dict[Hashable, Hashable]], bool] | None = None,
) -> CoaffineEndomorphismResult:
    """Shared exact CSP; optional leaf predicate for octahedral targets."""
    started = time.monotonic()
    nodes = list(graph)
    n = len(nodes)
    positions = {v: i for i, v in enumerate(nodes)}
    adjacency = [sum(1 << positions[w] for w in graph[v]) for v in nodes]
    full = (1 << n) - 1
    complement = [full ^ (row | (1 << i)) for i, row in enumerate(adjacency)]
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

    def indices(bits: int) -> Iterator[int]:
        while bits:
            bit = bits & -bits
            yield bit.bit_length() - 1
            bits ^= bit

    def propagate(
        rows: list[int], cols: list[int], processed: list[int], queue: list[int]
    ) -> bool:
        nonlocal revisions
        queued = sum(1 << v for v in queue)
        while queue:
            check_time()
            v = queue.pop()
            queued &= ~(1 << v)
            if not rows[v]:
                return False
            bad = full
            for t in indices(rows[v]):
                bad &= complement[t]
                if not bad:
                    break
            fresh = bad & ~processed[v]
            processed[v] = bad
            for t in indices(fresh):
                removed = cols[t] & adjacency[v]
                cols[t] &= ~removed
                for u in indices(removed):
                    revisions += 1
                    rows[u] &= ~(1 << t)
                    if not rows[u]:
                        return False
                    if not queued & (1 << u):
                        queued |= 1 << u
                        queue.append(u)
        return True

    def search(
        rows: list[int], cols: list[int], processed: list[int]
    ) -> dict[Hashable, Hashable] | None:
        nonlocal states, assignments
        check_time()
        if states >= max_states:
            raise _Limit
        states += 1
        undecided = [i for i, mask in enumerate(rows) if mask.bit_count() > 1]
        if not undecided:
            mapping = {
                nodes[i]: nodes[mask.bit_length() - 1] for i, mask in enumerate(rows)
            }
            if not verify_coaffine_endomorphism(graph, mapping):
                raise AssertionError("coaffine solver returned an invalid witness")
            return mapping if accept is None or accept(mapping) else None
        x = min(
            undecided, key=lambda i: (rows[i].bit_count(), -adjacency[i].bit_count())
        )
        choices = list(indices(rows[x]))
        if value_order == "fiber":
            choices.sort(key=lambda t: -cols[t].bit_count())
        for t in choices:
            check_time()
            assignments += 1
            next_rows, next_cols, next_processed = (
                rows.copy(),
                cols.copy(),
                processed.copy(),
            )
            for other in indices(rows[x] & ~(1 << t)):
                next_cols[other] &= ~(1 << x)
            next_rows[x] = 1 << t
            if propagate(next_rows, next_cols, next_processed, [x]):
                answer = search(next_rows, next_cols, next_processed)
                if answer is not None:
                    return answer
        return None

    rows, cols, processed = complement.copy(), complement.copy(), [0] * n
    if any(not mask for mask in rows):
        return result(RetractionSearchStatus.EXHAUSTED)
    try:
        if not propagate(rows, cols, processed, list(range(n))):
            return result(RetractionSearchStatus.EXHAUSTED)
        answer = search(rows, cols, processed)
    except (_Limit, RecursionError):
        return result(RetractionSearchStatus.LIMIT_REACHED)
    if answer is None:
        return result(RetractionSearchStatus.EXHAUSTED)
    return result(RetractionSearchStatus.FOUND, answer)

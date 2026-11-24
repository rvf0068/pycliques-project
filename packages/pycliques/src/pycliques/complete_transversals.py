"""Exact complete-transversal search without enumerating maximal cliques."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator
from dataclasses import dataclass

import networkx as nx

from .coaffine_endomorphisms import _validate_graph
from .retractions import RetractionSearchStatus


def _indices(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


@dataclass(frozen=True)
class CompleteTransversalResult:
    """FOUND carries a nonempty complete set meeting every supplied set.

    EXHAUSTED proves no such set exists; LIMIT_REACHED is inconclusive.
    """

    status: RetractionSearchStatus
    transversal: frozenset[Hashable] | None
    states_explored: int


def verify_complete_transversal(
    graph: nx.Graph,
    family: Iterable[Iterable[Hashable]],
    transversal: Iterable[Hashable],
) -> bool:
    """Independently check completeness and every required intersection."""
    try:
        _validate_graph(graph)
        chosen = frozenset(transversal)
        required = tuple(frozenset(s) for s in family)
        nodes = set(graph)
        return (
            bool(chosen)
            and chosen <= nodes
            and all(s <= nodes and bool(s & chosen) for s in required)
            and all(u == v or graph.has_edge(u, v) for u in chosen for v in chosen)
        )
    except (TypeError, ValueError):
        return False


def find_complete_transversal(
    graph: nx.Graph,
    family: Iterable[Iterable[Hashable]],
    *,
    max_states: int = 1_000_000,
) -> CompleteTransversalResult:
    """Search for a nonempty complete set intersecting every member of family.

    A complete transversal exists iff a maximal clique transversal exists
    (extend the complete set). Empty families are allowed; an empty member
    makes the instance impossible. Every member must be a subset of graph.

    Bitmask search branches on an unmet set, retaining only vertices
    adjacent to the partial solution. The first 128 unmet sets are scanned
    for a small branching domain; this affects order, not completeness.
    """
    _validate_graph(graph)
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    nodes = tuple(graph)
    positions = {v: i for i, v in enumerate(nodes)}
    required = tuple(frozenset(s) for s in family)
    if any(not s <= positions.keys() for s in required):
        raise ValueError("family members must be subsets of graph")
    if not nodes or any(not s for s in required):
        return CompleteTransversalResult(RetractionSearchStatus.EXHAUSTED, None, 0)
    if not required:
        return CompleteTransversalResult(
            RetractionSearchStatus.FOUND, frozenset((nodes[0],)), 0
        )

    clauses = sorted(
        {sum(1 << positions[v] for v in s) for s in required},
        key=lambda mask: (mask.bit_count(), mask),
    )
    n = len(nodes)
    # Construct incidence efficiently for large families.
    width = (len(clauses) + 7) // 8
    rows = [bytearray(width) for _ in nodes]
    for i, clause in enumerate(clauses):
        for v in _indices(clause):
            rows[v][i // 8] |= 1 << (i % 8)
    hits = [int.from_bytes(row, "little") for row in rows]
    adj = [sum(1 << positions[w] for w in graph[v]) for v in nodes]
    states = 0

    class _Limit(Exception):
        pass

    def search(available: int, remaining: int, selected: int) -> int | None:
        nonlocal states
        if states >= max_states:
            raise _Limit
        states += 1
        if not remaining:
            return selected
        possible = 0
        for v in _indices(available):
            possible |= hits[v]
        if remaining & ~possible:
            return None
        candidates = 0
        best = n + 1
        for k, i in enumerate(_indices(remaining)):
            options = available & clauses[i]
            size = options.bit_count()
            if size < best:
                candidates, best = options, size
            if best <= 1 or k == 127:
                break
        order = sorted(
            _indices(candidates), key=lambda v: -(hits[v] & remaining).bit_count()
        )
        for v in order:
            answer = search(
                available & adj[v], remaining & ~hits[v], selected | (1 << v)
            )
            if answer is not None:
                return answer
            # Partition the search according to the first chosen option.
            available &= ~(1 << v)
        return None

    try:
        answer = search((1 << n) - 1, (1 << len(clauses)) - 1, 0)
    except (_Limit, RecursionError):
        return CompleteTransversalResult(
            RetractionSearchStatus.LIMIT_REACHED, None, states
        )
    if answer is None:
        return CompleteTransversalResult(RetractionSearchStatus.EXHAUSTED, None, states)
    witness = frozenset(nodes[i] for i in _indices(answer))
    if not verify_complete_transversal(graph, required, witness):
        raise AssertionError("invalid complete transversal")
    return CompleteTransversalResult(RetractionSearchStatus.FOUND, witness, states)


def find_split_cover_counterexample(
    graph: nx.Graph,
    splits: Iterable[tuple[Iterable[Hashable], Iterable[Hashable]]],
    *,
    max_states: int = 1_000_000,
) -> CompleteTransversalResult:
    """Find a complete set contained in neither side of any supplied split.

    EXHAUSTED certifies every maximal clique is contained in at least one
    side of a supplied bipartition, without listing the maximal cliques.
    FOUND is a counterexample, extendible to a maximal clique. This checks
    coverage only, not the endpoint-containment constraints of a retraction.
    """
    _validate_graph(graph)
    nodes = set(graph)
    family: list[frozenset[Hashable]] = []
    for a, b in splits:
        left, right = frozenset(a), frozenset(b)
        if left & right or left | right != nodes:
            raise ValueError("each split must partition the vertex set")
        family.extend((left, right))
    return find_complete_transversal(graph, family, max_states=max_states)

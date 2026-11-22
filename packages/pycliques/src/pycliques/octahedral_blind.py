"""General necessary obstructions and exact completion of octahedral retracts.

The graph is arbitrary: neither snub coordinates nor saved witnesses occur here.
A ``LIMIT_REACHED`` result is never evidence of nonexistence.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import Enum

import networkx as nx

from .octahedral_nonlocal_search import verify_induced_octahedron


def _bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


class OctahedralBlindStatus(Enum):
    FOUND = "found"
    EXHAUSTED = "exhausted"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True)
class OctahedralScreen:
    possible: bool
    reason: str
    forced_vertices: int
    singleton_domains: int
    binary_domains: int
    clauses: int


@dataclass(frozen=True)
class OctahedralCompletion:
    status: OctahedralBlindStatus
    antipodal_pairs: tuple[tuple[Hashable, Hashable], ...] | None
    states_explored: int
    cached_states: int


@dataclass(frozen=True)
class OctahedralConflict:
    antipodal_pairs: tuple[tuple[Hashable, Hashable], ...]
    reason: str
    checks: int
    deletion_minimal: bool


@dataclass(frozen=True)
class OctahedralHittingSet:
    deleted_indices: tuple[int, ...]
    cost: int
    nodes_explored: int


class _OctahedralMasks:
    """Bit-mask encoding with positions determined by graph iteration order."""

    def __init__(self, graph: nx.Graph):
        if (
            not isinstance(graph, nx.Graph)
            or graph.is_directed()
            or graph.is_multigraph()
            or nx.number_of_selfloops(graph)
        ):
            raise ValueError("require a simple undirected loopless NetworkX graph")
        self.nodes = tuple(graph)
        self.index = {x: i for i, x in enumerate(self.nodes)}
        self.full = (1 << len(self.nodes)) - 1
        self.closed = [1 << i for i in range(len(self.nodes))]
        for u, v in graph.edges:
            a, b = self.index[u], self.index[v]
            self.closed[a] |= 1 << b
            self.closed[b] |= 1 << a
        self.nonedge = [self.full & ~x for x in self.closed]

    def pairs(self, graph: nx.Graph, pairs: Sequence[tuple[Hashable, Hashable]]):
        try:
            normalized = tuple(tuple(pair) for pair in pairs)
        except TypeError as error:
            raise ValueError("pairs must contain two vertices each") from error
        if normalized and not verify_induced_octahedron(graph, normalized):
            raise ValueError("pairs must induce an octahedron")
        return tuple((self.index[a], self.index[b]) for a, b in normalized)

    def common(self, pairs: Sequence[tuple[int, int]]) -> int:
        common = self.full
        for a, b in pairs:
            common &= self.closed[a] & self.closed[b]
        return common


def _screen_masks(
    m: _OctahedralMasks, pairs: Sequence[tuple[int, int]]
) -> OctahedralScreen:
    """2-SAT on vertices that cannot map to any future antipodal pair."""
    if not pairs:
        return OctahedralScreen(True, "NO_PAIRS", 0, 0, 0, 0)
    common = m.common(pairs)
    antipodes = tuple(x for a, b in pairs for x in (b, a))
    # Positions of anchor vertices in the target list (a_0,b_0,a_1,b_1,...).
    anchor_pos = {v: i for i, v in enumerate(x for pair in pairs for x in pair)}
    domains: dict[int, tuple[int, ...]] = {}
    for x in range(len(m.nodes)):
        complement_neighbors = m.nonedge[x]
        if complement_neighbors & common:
            continue  # A future target image has not been ruled out.
        if x in anchor_pos:
            choices = (anchor_pos[x],)
        else:
            choices = tuple(i for i, a in enumerate(antipodes)
                            if complement_neighbors & (1 << a))
        if not choices:
            return OctahedralScreen(False, "EMPTY_DOMAIN", len(domains) + 1, 0, 0, 0)
        domains[x] = choices

    singleton = {x for x in domains if len(domains[x]) == 1}
    binary = tuple(x for x in domains if len(domains[x]) == 2)
    binary_positions = {x: i for i, x in enumerate(binary)}
    eligible = singleton | set(binary)
    edges = [[] for _ in range(2 * len(binary))]
    reverse = [[] for _ in edges]
    clauses = 0

    def imply(u: int, v: int) -> None:
        edges[u].append(v)
        reverse[v].append(u)

    for x in eligible:
        for y in _bits(m.closed[x] & ~(1 << x)):
            if y <= x or y not in eligible:
                continue
            for ix, a in enumerate(domains[x]):
                for iy, b in enumerate(domains[y]):
                    if b != (a ^ 1):
                        continue
                    clauses += 1
                    if x in singleton and y in singleton:
                        return OctahedralScreen(
                            False, "FORCED_EDGE", len(domains), len(singleton),
                            len(binary), clauses
                        )
                    if x in singleton:
                        literal = 2 * binary_positions[y] + iy
                        imply(literal, literal ^ 1)
                    elif y in singleton:
                        literal = 2 * binary_positions[x] + ix
                        imply(literal, literal ^ 1)
                    else:
                        p = 2 * binary_positions[x] + ix
                        q = 2 * binary_positions[y] + iy
                        imply(p, q ^ 1)
                        imply(q, p ^ 1)

    # Iterative Kosaraju, so long implication paths do not exhaust recursion.
    visited = bytearray(len(edges))
    finishing: list[int] = []
    for root in range(len(edges)):
        if visited[root]:
            continue
        visited[root] = 1
        stack = [(root, 0)]
        while stack:
            u, i = stack[-1]
            if i == len(edges[u]):
                finishing.append(u)
                stack.pop()
            else:
                stack[-1] = (u, i + 1)
                v = edges[u][i]
                if not visited[v]:
                    visited[v] = 1
                    stack.append((v, 0))
    component = [-1] * len(edges)
    number = 0
    for root in reversed(finishing):
        if component[root] >= 0:
            continue
        component[root] = number
        stack = [root]
        while stack:
            u = stack.pop()
            for v in reverse[u]:
                if component[v] < 0:
                    component[v] = number
                    stack.append(v)
        number += 1
    possible = all(component[2 * i] != component[2 * i + 1]
                   for i in range(len(binary)))
    return OctahedralScreen(
        possible, "NOT_DISPROVED" if possible else "2SAT_CONFLICT",
        len(domains), len(singleton), len(binary), clauses
    )


def screen_partial_octahedral_retraction(
    graph: nx.Graph,
    antipodal_pairs: Sequence[tuple[Hashable, Hashable]],
) -> OctahedralScreen:
    """Sound necessary test for a retraction onto any octahedral extension.

    ``possible=False`` PROVES nonexistence of such an extension. A positive
    result only means this inexpensive 2-SAT relaxation did not refute it.
    """
    m = _OctahedralMasks(graph)
    return _screen_masks(m, m.pairs(graph, antipodal_pairs))


def minimize_octahedral_conflict(
    graph: nx.Graph,
    antipodal_pairs: Sequence[tuple[Hashable, Hashable]],
    *,
    max_checks: int | None = None,
) -> OctahedralConflict | None:
    """Delete pairs from an UNSAT selection while preserving sound failure.

    With a budget, the returned UNSAT core need not be deletion-minimal.
    """
    if max_checks is not None and (type(max_checks) is not int or max_checks < 1):
        raise ValueError("max_checks must be a positive integer or None")
    m = _OctahedralMasks(graph)
    indexed = list(m.pairs(graph, antipodal_pairs))
    checked: dict[tuple[tuple[int, int], ...], OctahedralScreen] = {}
    checks = 0

    def check(pairs: list[tuple[int, int]]):
        nonlocal checks
        key = tuple(pairs)
        if key not in checked:
            if max_checks is not None and checks >= max_checks:
                return None
            checked[key] = _screen_masks(m, pairs)
            checks += 1
        return checked[key]

    initial = check(indexed)
    if initial is None or initial.possible:
        return None
    i = 0
    finished = True
    while i < len(indexed):
        smaller = indexed[:i] + indexed[i + 1:]
        report = check(smaller)
        if report is None:
            finished = False
            break
        if not report.possible:
            indexed = smaller
            i = 0
        else:
            i += 1
    result = check(indexed)
    assert result is not None and not result.possible
    labels = tuple((m.nodes[a], m.nodes[b]) for a, b in indexed)
    return OctahedralConflict(labels, result.reason, checks, finished)


def minimum_octahedral_repair(
    conflicts: Sequence[int],
    number_of_pairs: int,
    *,
    weights: Sequence[int] | None = None,
) -> OctahedralHittingSet:
    """Solve the exact positive-weight hitting-set problem for conflict masks.

    Each conflict mask is a nonempty subset of pair indices. Its deletion
    is necessary, not sufficient, for a retracting extension.
    """
    k = number_of_pairs
    if type(k) is not int or k < 0:
        raise ValueError("number_of_pairs must be nonnegative")
    costs = (1,) * k if weights is None else tuple(weights)
    if len(costs) != k or any(type(w) is not int or w < 1 for w in costs):
        raise ValueError("weights must be positive integers")
    cores = tuple(sorted(set(conflicts)))
    if any(type(c) is not int or c <= 0 or c >= (1 << k) for c in cores):
        raise ValueError("conflicts must be nonempty subsets of pair indices")
    cores = tuple(c for c in cores
                  if not any(d != c and (d & c) == d for d in cores))
    if not cores:
        return OctahedralHittingSet((), 0, 1)
    best = (sum(costs) + 1, 0)
    explored = 0
    seen: dict[int, int] = {}

    def visit(deleted: int, cost: int, remaining: tuple[int, ...]):
        nonlocal best, explored
        explored += 1
        if cost >= best[0]:
            return
        if not remaining:
            best = (cost, deleted)
            return
        if seen.get(deleted, cost + 1) <= cost:
            return
        seen[deleted] = cost
        pivot = min(remaining, key=int.bit_count)
        for i in range(k):
            if pivot & (1 << i):
                visit(deleted | (1 << i), cost + costs[i],
                      tuple(c for c in remaining if not (c & (1 << i))))

    visit(0, 0, cores)
    return OctahedralHittingSet(
        tuple(i for i in range(k) if best[1] & (1 << i)),
        best[0], explored
    )


def complete_dominating_octahedron(
    graph: nx.Graph,
    antipodal_pairs: Sequence[tuple[Hashable, Hashable]],
    *,
    target_dimension: int,
    max_states: int = 200_000,
) -> OctahedralCompletion:
    """Exact bounded search for a dominating induced O_target_dimension.

    EXHAUSTED is a proof of nonexistence of an extension retaining all
    supplied antipodal pairs; LIMIT_REACHED makes no such claim.
    """
    if type(target_dimension) is not int or target_dimension < 2:
        raise ValueError("target_dimension must be an integer >= 2")
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    m = _OctahedralMasks(graph)
    partial = m.pairs(graph, antipodal_pairs)
    remaining = target_dimension - len(partial)
    if remaining < 0:
        raise ValueError("more initial pairs than the target dimension")
    memo: dict[tuple[int, int], tuple[tuple[int, int], ...] | bool] = {}
    states = 0

    def solve(common: int, r: int):
        nonlocal states
        key = (common, r)
        if key in memo:
            return memo[key]
        if states >= max_states:
            return None
        states += 1
        if r == 0:
            found: tuple[tuple[int, int], ...] | bool = () if common == 0 else False
            memo[key] = found
            return found
        if common.bit_count() < 2 * r:
            memo[key] = False
            return False
        pivot_neighborhood = 0
        min_degree = len(m.nodes) + 1
        number_of_nonedges = 0
        for v in _bits(common):
            n = m.nonedge[v] & common
            degree = n.bit_count()
            number_of_nonedges += degree
            if degree < min_degree:
                min_degree = degree
                pivot_neighborhood = n
        if min_degree == 0 or number_of_nonedges // 2 < r:
            memo[key] = False
            return False
        candidates = set()
        choices = []
        for a in _bits(pivot_neighborhood):
            for b in _bits(m.nonedge[a] & common):
                pair = (min(a, b), max(a, b))
                if pair in candidates:
                    continue
                candidates.add(pair)
                next_common = common & m.closed[a] & m.closed[b]
                if next_common.bit_count() < 2 * (r - 1):
                    continue
                residual_edges = sum(
                    (m.nonedge[v] & next_common).bit_count()
                    for v in _bits(next_common)
                ) // 2
                if residual_edges < r - 1:
                    continue
                choices.append((
                    next_common.bit_count(), -residual_edges, pair, next_common
                ))
        choices.sort()
        unknown = False
        for _, _, pair, next_common in choices:
            result = solve(next_common, r - 1)
            if isinstance(result, tuple):
                found = (pair, *result)
                memo[key] = found
                return found
            if result is None:
                unknown = True
        if unknown:
            return None
        memo[key] = False
        return False

    remainder = solve(m.common(partial), remaining)
    if remainder is None:
        return OctahedralCompletion(OctahedralBlindStatus.LIMIT_REACHED,
                                     None, states, len(memo))
    if remainder is False:
        return OctahedralCompletion(OctahedralBlindStatus.EXHAUSTED,
                                     None, states, len(memo))
    pairs = partial + remainder
    labels = tuple((m.nodes[a], m.nodes[b]) for a, b in pairs)
    if not verify_induced_octahedron(graph, labels, require_dominating=True):
        raise AssertionError("completion failed independent verification")
    return OctahedralCompletion(OctahedralBlindStatus.FOUND, labels,
                                 states, len(memo))

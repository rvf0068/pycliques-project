"""Bounded, nonlocal search for induced octahedra.

This module contains no saved snub embeddings. Given a simple graph, it
searches for an induced O_d, optionally starting from a smaller induced
octahedron. A target must separately admit a verified retraction before it
can certify clique divergence.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum

import networkx as nx

from ._bitsets import iter_indices as _bits
from ._graph_utils import is_simple_graph


class OctahedronSearchStatus(Enum):
    FOUND = "found"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True)
class OctahedronSearchResult:
    """A checked induced target or an inconclusive bounded search."""

    status: OctahedronSearchStatus
    antipodal_pairs: tuple[tuple[Hashable, Hashable], ...] | None
    restarts: int
    best_dimension: int
    best_common_neighbors: int | None


def verify_induced_octahedron(
    graph: nx.Graph,
    antipodal_pairs: Sequence[tuple[Hashable, Hashable]],
    *,
    require_dominating: bool = False,
) -> bool:
    """Check antipodal nonedges, all cross-edges, and optional domination."""
    pairs = [tuple(pair) for pair in antipodal_pairs]
    if not pairs or any(len(pair) != 2 for pair in pairs):
        return False
    vertices = [v for pair in pairs for v in pair]
    if len(set(vertices)) != len(vertices) or any(v not in graph for v in vertices):
        return False
    for i, (a, b) in enumerate(pairs):
        if graph.has_edge(a, b):
            return False
        for c, d in pairs[:i]:
            if not all(graph.has_edge(u, v) for u in (a, b) for v in (c, d)):
                return False
    if require_dominating:
        common = set(graph)
        for v in vertices:
            common.intersection_update(set(graph[v]) | {v})
            if not common:
                break
        if common:
            return False
    return True


def find_induced_octahedron_amplified(
    graph: nx.Graph,
    *,
    target_dimension: int,
    seed_pairs: Sequence[tuple[Hashable, Hashable]] = (),
    max_restarts: int = 20_000,
    pair_samples: int = 32,
    seed: int = 24_000,
    exploration_restarts: int = 10_000,
    require_dominating: bool = True,
) -> OctahedronSearchResult:
    """Search for an induced octahedron in any finite simple graph.

    Repeatedly retain a small, randomized subset of existing antipodal
    pairs and rebuild a larger octahedron in their common closed
    neighborhood. This nonlocal destroy-and-repair search can replace many
    pairs simultaneously. No labels or pairs of any particular graph are
    embedded in the algorithm.

    Seed pairs are optional, must induce an octahedron, and need not be
    complement-dominating. FOUND includes independently checked pairs;
    LIMIT_REACHED means only that the search budget was exhausted. Use
    find_octahedral_retraction separately to establish a retraction.
    """
    if not is_simple_graph(graph):
        raise ValueError("require a simple undirected graph")
    if type(target_dimension) is not int or target_dimension < 3:
        raise ValueError("target_dimension must be an integer >= 3")
    if type(max_restarts) is not int or max_restarts < 1:
        raise ValueError("max_restarts must be a positive integer")
    if type(pair_samples) is not int or pair_samples < 1:
        raise ValueError("pair_samples must be a positive integer")
    if type(exploration_restarts) is not int or exploration_restarts < 0:
        raise ValueError("exploration_restarts must be nonnegative")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if seed_pairs and not verify_induced_octahedron(graph, seed_pairs):
        raise ValueError("seed_pairs must induce an octahedron")

    initial_dim = len(seed_pairs)
    if 2 * target_dimension > len(graph):
        return OctahedronSearchResult(
            OctahedronSearchStatus.LIMIT_REACHED, None, 0, initial_dim, None
        )

    nodes = list(graph)
    index = {v: i for i, v in enumerate(nodes)}
    adjacency = [1 << i for i in range(len(nodes))]
    for u, v in graph.edges:
        adjacency[index[u]] |= 1 << index[v]
        adjacency[index[v]] |= 1 << index[u]
    full = (1 << len(nodes)) - 1
    initial = [(index[a], index[b]) for a, b in seed_pairs]
    remaining = full
    for u, v in initial:
        remaining &= adjacency[u] & adjacency[v]
    pool: list[tuple[list[tuple[int, int]], int]] = [(initial, remaining)]
    rng = random.Random(seed)

    def grow(
        retained: list[tuple[int, int]], samples: int, trial: int
    ) -> tuple[list[tuple[int, int]], int]:
        allowed = full
        for u, v in retained:
            allowed &= adjacency[u] & adjacency[v]
        pairs = retained.copy()
        while allowed:
            choices = list(_bits(allowed))
            if len(choices) < 2:
                break
            best: tuple[float, int, int, int] | None = None
            for _ in range(samples):
                u = rng.choice(choices)
                antipodes = list(_bits(allowed & ~adjacency[u]))
                if not antipodes:
                    continue
                v = antipodes[0] if len(antipodes) == 1 else rng.choice(antipodes)
                next_allowed = allowed & adjacency[u] & adjacency[v]
                pressure = next_allowed.bit_count()
                if len(pairs) >= target_dimension - 3:
                    score = pressure * (0.1 if trial % 4 == 0 else 1)
                    score += rng.random() * 40
                else:
                    score = pressure + rng.random() * 6
                if best is None or score > best[0]:
                    best = (score, u, v, next_allowed)
            if best is None:
                break
            _, u, v, allowed = best
            pairs.append((u, v))
        return pairs, allowed

    def rank(entry: tuple[list[tuple[int, int]], int]) -> tuple[int, int]:
        pairs, allowed = entry
        return (len(pairs) - (3 if allowed else 0), -allowed.bit_count())

    best_dimension = initial_dim
    best_common: int | None = None
    for trial in range(max_restarts):
        focused = trial >= exploration_restarts
        if focused:
            base, _ = rng.choice(pool[: min(20, len(pool))])
            keep = rng.choice((1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15))
            n_keep = min(keep, len(base))
            samples = pair_samples
        else:
            base, _ = rng.choice(pool)
            n_keep = rng.randrange(1, min(len(base), 14) + 1) if base else 0
            samples = max(1, pair_samples * 3 // 4)
        pairs, allowed = grow(rng.sample(base, n_keep), samples, trial)
        dimension, common = len(pairs), allowed.bit_count()
        if dimension > best_dimension:
            best_dimension, best_common = dimension, common
        if dimension >= target_dimension and (not require_dominating or common == 0):
            found = pairs if require_dominating else pairs[:target_dimension]
            labels = tuple((nodes[a], nodes[b]) for a, b in found)
            if verify_induced_octahedron(
                graph, labels, require_dominating=require_dominating
            ):
                return OctahedronSearchResult(
                    OctahedronSearchStatus.FOUND,
                    labels,
                    trial + 1,
                    best_dimension,
                    common,
                )
        threshold = max(3, initial_dim + (1 if focused else 0))
        if dimension >= threshold and (
            common == 0 or (dimension >= best_dimension and rng.random() < 0.06)
        ):
            pool.append((pairs, allowed))
            pool.sort(key=rank, reverse=True)
            del pool[60 if focused else 55 :]
    return OctahedronSearchResult(
        OctahedronSearchStatus.LIMIT_REACHED,
        None,
        max_restarts,
        best_dimension,
        best_common,
    )


class InducedOctahedronEnumeration:
    """One-shot bounded enumeration of induced octahedra of a fixed dimension.

    Exhaustion is certified only after consuming the iterator to completion
    and checking exhausted. Stopping early or reaching max_states is not
    evidence of nonexistence. Each matching is emitted exactly once.
    """

    def __init__(
        self, graph: nx.Graph, target_dimension: int, max_states: int
    ) -> None:
        if not is_simple_graph(graph):
            raise ValueError("require a simple undirected graph")
        if type(target_dimension) is not int or target_dimension < 1:
            raise ValueError("target_dimension must be a positive integer")
        if type(max_states) is not int or max_states < 1:
            raise ValueError("max_states must be a positive integer")
        self.states = 0
        self.limited = False
        self.exhausted = False
        nodes = tuple(graph)
        index = {v: i for i, v in enumerate(nodes)}
        full = (1 << len(nodes)) - 1
        nonedges = [
            full ^ ((1 << i) | sum(1 << index[w] for w in graph[v]))
            for i, v in enumerate(nodes)
        ]

        def visit(
            remaining: int, pairs: tuple[tuple[int, int], ...]
        ) -> Iterator[tuple[tuple[Hashable, Hashable], ...]]:
            if self.states >= max_states:
                self.limited = True
                return
            self.states += 1
            need = target_dimension - len(pairs)
            if not need:
                yield tuple((nodes[u], nodes[v]) for u, v in pairs)
                return
            if remaining.bit_count() < 2 * need:
                return
            degrees = {
                v: (nonedges[v] & remaining).bit_count() for v in _bits(remaining)
            }
            for v, d in degrees.items():
                if not d:
                    remaining ^= 1 << v
            if remaining.bit_count() < 2 * need:
                return
            u = min(_bits(remaining), key=lambda v: (degrees[v], v))
            others = remaining ^ (1 << u)
            # Every induced matching either contains exactly one edge
            # incident with u, or avoids u. Pair branches come first.
            for v in _bits(nonedges[u] & others):
                allowed = others & ~(nonedges[u] | nonedges[v] | (1 << v))
                yield from visit(allowed, (*pairs, (u, v)))
                if self.limited:
                    return
            yield from visit(others, pairs)

        def run() -> Iterator[tuple[tuple[Hashable, Hashable], ...]]:
            yield from visit(full, ())
            self.exhausted = not self.limited

        self._iterator = run()

    def __iter__(self) -> InducedOctahedronEnumeration:
        return self

    def __next__(self) -> tuple[tuple[Hashable, Hashable], ...]:
        return next(self._iterator)


def enumerate_induced_octahedra(
    graph: nx.Graph, *, target_dimension: int, max_states: int = 500_000
) -> InducedOctahedronEnumeration:
    """Enumerate fixed-size induced matchings in the complement, lazily."""
    return InducedOctahedronEnumeration(graph, target_dimension, max_states)

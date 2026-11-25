"""Blind induced-octahedron search using compatible nonedges.

An induced O_d corresponds to a complete set in the auxiliary graph whose
vertices are nonedges of G, with edges joining pairs of nonedges having
four cross-edges in G. We avoid explicitly constructing that potentially
huge auxiliary graph by intersecting closed-neighborhood bitmasks.

No graph-specific antipodal pairs or saved projections are used.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator

import networkx as nx

from ._graph_utils import is_simple_graph
from .octahedral_nonlocal_search import (
    OctahedronSearchResult,
    OctahedronSearchStatus,
    verify_induced_octahedron,
)


def find_induced_octahedron_compatibility(
    graph: nx.Graph,
    *,
    target_dimension: int,
    max_restarts: int = 2_000,
    local_steps: int = 150,
    pair_samples: int = 32,
    seed: int = 24_001,
    require_dominating: bool = True,
) -> OctahedronSearchResult:
    """Find a large induced octahedron without preselected antipodal pairs.

    Sample compatible nonedges greedily, then use fixed-dimension pair
    replacements to reduce the common closed-neighborhood intersection.
    A complement-dominating embedding has an empty such intersection,
    a necessary but NOT sufficient condition for an octahedral retraction.

    This is a bounded heuristic: LIMIT_REACHED is always inconclusive.
    Each restart uses up to local_steps replacement attempts and samples
    at most pair_samples candidate pairs per greedy decision.
    """
    if not is_simple_graph(graph):
        raise ValueError("require a simple undirected loopless NetworkX graph")
    if type(target_dimension) is not int or target_dimension < 3:
        raise ValueError("target_dimension must be an integer >= 3")
    if type(max_restarts) is not int or max_restarts < 1:
        raise ValueError("max_restarts must be positive")
    if type(local_steps) is not int or local_steps < 0:
        raise ValueError("local_steps must be nonnegative")
    if type(pair_samples) is not int or pair_samples < 1:
        raise ValueError("pair_samples must be positive")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    nodes = list(graph)
    n = len(nodes)
    if n < 2 * target_dimension:
        return OctahedronSearchResult(
            OctahedronSearchStatus.LIMIT_REACHED, None, 0, 0, None
        )

    indices = {v: i for i, v in enumerate(nodes)}
    closed = [1 << i for i in range(n)]
    for u, v in graph.edges:
        i, j = indices[u], indices[v]
        closed[i] |= 1 << j
        closed[j] |= 1 << i
    full = (1 << n) - 1
    rng = random.Random(seed)

    def bits(mask: int) -> Iterator[int]:
        while mask:
            low = mask & -mask
            yield low.bit_length() - 1
            mask ^= low

    def common(pairs: list[tuple[int, int]]) -> int:
        allowed = full
        for u, v in pairs:
            allowed &= closed[u] & closed[v]
        return allowed

    def extend(base: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int]:
        pairs = base.copy()
        allowed = common(pairs)
        while allowed.bit_count() >= 2:
            vertices = list(bits(allowed))
            best: tuple[float, tuple[int, int], int] | None = None
            for _ in range(pair_samples):
                u = rng.choice(vertices)
                antipodes = list(bits(allowed & ~closed[u]))
                if not antipodes:
                    continue
                v = rng.choice(antipodes)
                next_allowed = allowed & closed[u] & closed[v]
                pressure = next_allowed.bit_count()
                if len(pairs) >= target_dimension - 2:
                    # Stop preserving neighbors and try to eliminate them.
                    score = -pressure + rng.random() * (
                        15 if len(pairs) == target_dimension - 2 else 30
                    )
                else:
                    score = pressure + rng.random() * 15
                if best is None or score > best[0]:
                    best = (score, (u, v), next_allowed)
            if best is None:
                break
            pairs.append(best[1])
            allowed = best[2]
            if len(pairs) >= target_dimension and allowed == 0:
                break
        return pairs, allowed

    def repair(
        pairs: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], int]:
        work = pairs.copy()
        best_pairs = work.copy()
        best_common = common(work).bit_count()
        current_common = best_common
        for step in range(local_steps):
            dropped = rng.randrange(len(work))
            remaining = work[:dropped] + work[dropped + 1 :]
            allowed = common(remaining)
            vertices = list(bits(allowed))
            if len(vertices) < 2:
                continue
            candidates: list[tuple[int, tuple[int, int]]] = []
            for _ in range(min(pair_samples, 20)):
                u = rng.choice(vertices)
                antipodes = list(bits(allowed & ~closed[u]))
                if not antipodes:
                    continue
                v = rng.choice(antipodes)
                count = (allowed & closed[u] & closed[v]).bit_count()
                candidates.append((count, (u, v)))
            if not candidates:
                continue
            candidate_common, pair = min(candidates, key=lambda x: x[0])
            temperature = max(1.0, 10 * (1 - step / max(1, local_steps)))
            if candidate_common <= current_common or rng.random() < math.exp(
                (current_common - candidate_common) / temperature
            ):
                work[dropped] = pair
                current_common = candidate_common
                if current_common < best_common:
                    best_pairs = work.copy()
                    best_common = current_common
                    if best_common == 0:
                        break
        return best_pairs, best_common

    pool: list[list[tuple[int, int]]] = []
    best_dimension = 0
    best_common: int | None = None

    def accept(
        pairs: list[tuple[int, int]], common_count: int, trial: int
    ) -> OctahedronSearchResult | None:
        if len(pairs) < target_dimension or (require_dominating and common_count != 0):
            return None
        labels = tuple((nodes[u], nodes[v]) for u, v in pairs)
        if not verify_induced_octahedron(
            graph, labels, require_dominating=require_dominating
        ):
            raise AssertionError("internal octahedron failed verification")
        return OctahedronSearchResult(
            OctahedronSearchStatus.FOUND,
            labels,
            trial + 1,
            best_dimension,
            common_count,
        )

    for trial in range(max_restarts):
        if pool and rng.random() < 0.8:
            base = rng.choice(pool[: min(len(pool), 25)])
            keep = rng.randrange(1, min(len(base), target_dimension - 1) + 1)
            retained = rng.sample(base, keep)
        else:
            retained = []

        pairs, allowed = extend(retained)
        dimension, common_count = len(pairs), allowed.bit_count()
        if dimension > best_dimension or (
            dimension == best_dimension
            and (best_common is None or common_count < best_common)
        ):
            best_dimension, best_common = dimension, common_count

        found = accept(pairs, common_count, trial)
        if found is not None:
            return found
        if dimension >= target_dimension and common_count and local_steps:
            candidate = pairs[:target_dimension] if rng.random() < 0.5 else pairs
            repaired, repaired_common = repair(candidate)
            found = accept(repaired, repaired_common, trial)
            if found is not None:
                return found
        if dimension >= max(3, target_dimension - 5):
            pool.append(pairs)
            pool.sort(key=lambda p: (len(p), -common(p).bit_count()), reverse=True)
            del pool[60:]

    return OctahedronSearchResult(
        OctahedronSearchStatus.LIMIT_REACHED,
        None,
        max_restarts,
        best_dimension,
        best_common,
    )

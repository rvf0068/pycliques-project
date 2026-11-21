"""Constraint-propagation search for retractions onto a specified induced octahedron.

For an octahedral target, each target vertex has precisely one non-neighbor:
its antipode. Hence the only forbidden edge images are antipodal pairs. Domains
are represented by bit masks, with singleton-driven propagation and an MRV
backtracking search. This avoids enumerating host or target automorphisms.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import networkx as nx

from .retractions import (
    BoundedRetractionResult,
    RetractionSearchStatus,
    verify_retraction_witness,
)


@dataclass
class _SearchFrame:
    vertex: int
    choices: list[int]
    trail_start: int
    next_choice: int = 0


def find_octahedral_retraction(
    graph: nx.Graph,
    antipodal_pairs: Sequence[tuple[Hashable, Hashable]],
    *,
    max_states: int = 20_000,
    seed: int = 9883,
) -> BoundedRetractionResult:
    """Find a retraction onto a *specified induced* octahedron of ``graph``.

    ``antipodal_pairs`` gives the target as disjoint ordered pairs of host
    vertex labels; cross-pair edges must all be present, and each pair must
    be a nonedge. The first pair has target indices 0 and 1, etc. The
    returned ``retraction`` and ``inclusion`` use the original graph labels.

    ``FOUND`` carries a checked retraction. ``EXHAUSTED`` proves that no
    retraction fixes this particular induced target. ``LIMIT_REACHED`` gives
    no conclusion. The search is seeded for reproducible tie breaking and
    limits are imposed on explored states, not on wall-clock time.

    This method does *not* find an induced octahedron automatically; the
    embedding must already have been identified. Graph edges may collapse
    to a vertex, as in the standard clique-graph morphism convention.
    """
    if not isinstance(max_states, int) or isinstance(max_states, bool) or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    pairs = [tuple(p) for p in antipodal_pairs]
    if len(pairs) < 3 or any(len(p) != 2 for p in pairs):
        raise ValueError("the target must have at least three antipodal pairs")
    target_vertices = [v for a, b in pairs for v in (a, b)]
    if len(set(target_vertices)) != len(target_vertices):
        raise ValueError("antipodal pairs must be vertex-disjoint")
    if not set(target_vertices).issubset(graph):
        raise ValueError("all antipodal vertices must belong to the host")
    for i, (a, b) in enumerate(pairs):
        if graph.has_edge(a, b):
            raise ValueError("endpoints of an antipodal pair must be nonadjacent")
        for c, d in pairs[:i]:
            if not all(graph.has_edge(u, v) for u in (a, b) for v in (c, d)):
                raise ValueError("different antipodal pairs must be completely joined")

    nodes = list(graph)
    indices = {v: i for i, v in enumerate(nodes)}
    neighbors = [[indices[w] for w in graph[v]] for v in nodes]
    target_position = {v: i for i, v in enumerate(target_vertices)}
    target_set = set(target_vertices)
    outside = [i for i, v in enumerate(nodes) if v not in target_set]
    full_mask = (1 << len(target_vertices)) - 1
    domains = [full_mask] * len(nodes)
    # The only forbidden target image for an outside x adjacent to a target
    # vertex w is antipode(w). Intersect these constraints before search.
    for i, v in enumerate(nodes):
        if v in target_set:
            domains[i] = 1 << target_position[v]
        else:
            mask = full_mask
            for w in graph[v]:
                p = target_position.get(w)
                if p is not None:
                    mask &= ~(1 << (p ^ 1))
            domains[i] = mask

    target = graph.subgraph(target_vertices)
    inclusion = {v: v for v in target_vertices}
    if any(mask == 0 for mask in domains):
        return BoundedRetractionResult(RetractionSearchStatus.EXHAUSTED, None, None, 1, 0)

    rng = random.Random(seed)
    trail: list[tuple[int, int]] = []

    def rollback(size: int) -> None:
        while len(trail) > size:
            index, old = trail.pop()
            domains[index] = old

    def propagate(queue: list[int]) -> bool:
        """A singleton forbids its antipode at every neighbor."""
        while queue:
            index = queue.pop()
            mask = domains[index]
            if mask == 0:
                return False
            if mask.bit_count() != 1:
                continue
            forbidden = 1 << ((mask.bit_length() - 1) ^ 1)
            for j in neighbors[index]:
                old = domains[j]
                if old & forbidden:
                    new = old & ~forbidden
                    if new == 0:
                        return False
                    trail.append((j, old))
                    domains[j] = new
                    if new.bit_count() == 1:
                        queue.append(j)
        return True

    queue = [i for i, mask in enumerate(domains) if mask.bit_count() == 1]
    if not propagate(queue):
        return BoundedRetractionResult(RetractionSearchStatus.EXHAUSTED, None, None, 1, 0)

    def choose_frame() -> _SearchFrame | None:
        size = min((domains[i].bit_count() for i in outside if domains[i].bit_count() > 1), default=0)
        if size == 0:
            return None
        candidates = [i for i in outside if domains[i].bit_count() == size]
        if size <= 4 and len(candidates) > 1:
            sample = rng.sample(candidates, min(15, len(candidates)))

            def pressure(index: int) -> int:
                # Count options that would forbid an option in a neighbor.
                return sum(
                    bool(domains[j] & (1 << (position ^ 1)))
                    for position in range(len(target_vertices))
                    if (domains[index] >> position) & 1
                    for j in neighbors[index]
                    if nodes[j] not in target_set
                )

            index = max(sample, key=pressure)
        else:
            index = candidates[0]
        choices = [p for p in range(len(target_vertices)) if (domains[index] >> p) & 1]
        counts = {
            p: sum(bool(domains[j] & (1 << (p ^ 1))) for j in neighbors[index] if nodes[j] not in target_set)
            for p in choices
        }
        choices.sort(key=lambda p: (counts[p], rng.random()))
        return _SearchFrame(index, choices, len(trail))

    frames: list[_SearchFrame] = []
    states_explored = 1
    assignments_tried = 0

    def found() -> BoundedRetractionResult:
        projection = {
            nodes[i]: target_vertices[mask.bit_length() - 1]
            for i, mask in enumerate(domains)
        }
        witness = projection, inclusion
        if not verify_retraction_witness(graph, target, witness):
            raise AssertionError("octahedral propagation produced an invalid witness")
        return BoundedRetractionResult(
            RetractionSearchStatus.FOUND,
            projection,
            inclusion,
            states_explored,
            assignments_tried,
        )

    current = choose_frame()
    if current is None:
        return found()
    frames.append(current)
    limited = False
    while frames:
        frame = frames[-1]
        rollback(frame.trail_start)
        if frame.next_choice == len(frame.choices):
            frames.pop()
            continue
        if states_explored >= max_states:
            limited = True
            break
        choice = frame.choices[frame.next_choice]
        frame.next_choice += 1
        assignments_tried += 1
        trail.append((frame.vertex, domains[frame.vertex]))
        domains[frame.vertex] = 1 << choice
        if not propagate([frame.vertex]):
            continue
        states_explored += 1
        child = choose_frame()
        if child is None:
            return found()
        frames.append(child)
    return BoundedRetractionResult(
        RetractionSearchStatus.LIMIT_REACHED if limited else RetractionSearchStatus.EXHAUSTED,
        None,
        None,
        states_explored,
        assignments_tried,
    )

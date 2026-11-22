"""Exact bounded two-shore search for retractions onto a fixed induced octahedron.

Each antipodal pair is oriented (a_i,b_i). Non-target vertices can map
onto either shore. Images within one shore are always adjacent or equal.
Only opposite images a_i/b_i of adjacent source vertices are forbidden.

Assign images in the a-shore whenever possible. Branch exclusively on
non-target vertices forced to map to the b-shore; an assigned b_i only
deletes a_i from domains of adjacent still-unassigned vertices.

Every branch is mathematically complete for the fixed target, regardless
of orientation. Auto-orientation changes performance, never correctness.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import networkx as nx

from .octahedral_amplification import _check_pairs
from .retractions import (
    BoundedRetractionResult,
    RetractionSearchStatus,
    verify_retraction_witness,
)


@dataclass
class _ShoreFrame:
    vertex: int
    choices: tuple[int, ...]
    trail_start: int
    next_choice: int = 0


def find_octahedral_retraction_shores(
    graph: nx.Graph,
    antipodal_pairs: Sequence[tuple[Hashable, Hashable]],
    *,
    max_states: int = 20_000,
    orientation: int | None = None,
    max_orientation_samples: int = 65_536,
) -> BoundedRetractionResult:
    """Find a full ordinary retraction onto a specified induced O_d.

    The pairs need not be numbered or labeled canonically. On orientation
    bit i equal to 0, a_i is the pair's first endpoint and b_i its second;
    bit i equal to 1 swaps them. If orientation is None, minimize the
    number of initially forced b-shore vertices over at most
    max_orientation_samples deterministic orientations. Orientation
    selection is a heuristic only: exhaustive branching for ANY fixed
    orientation is complete for all retractions onto the specified target.

    FOUND includes a fully checked original-label retraction and
    inclusion. EXHAUSTED proves no ordinary retraction onto this fixed
    induced target. LIMIT_REACHED is inconclusive. The initial search
    state and each attempted b-shore assignment count against max_states.
    Existing target-first and coaffine search methods are unchanged.
    """
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if type(max_orientation_samples) is not int or max_orientation_samples < 1:
        raise ValueError("max_orientation_samples must be positive")
    pairs = _check_pairs(graph, antipodal_pairs)
    if len(pairs) < 3:
        raise ValueError("the target must have at least three antipodal pairs")
    d = len(pairs)
    if orientation is not None and (
        type(orientation) is not int or not (0 <= orientation < (1 << d))
    ):
        raise ValueError("orientation must be a pair-endpoint bitmask")

    targets = [v for pair in pairs for v in pair]
    target_set = set(targets)
    outside = [v for v in graph if v not in target_set]
    positions = {v: i for i, v in enumerate(outside)}
    outside_neighbors = [
        [positions[w] for w in graph[v] if w in positions] for v in outside
    ]
    m = len(outside)

    if orientation is None:
        # Each bitset records outside vertices nonadjacent to this endpoint.
        # Therefore a chosen b_i permits its a_i image at those vertices.
        coverage = [
            tuple(
                sum(1 << j for j, x in enumerate(outside) if endpoint not in graph[x])
                for endpoint in pair
            )
            for pair in pairs
        ]
        total = 1 << d
        candidates: Sequence[int]
        if total <= max_orientation_samples:
            candidates = range(total)
        else:
            rng = random.Random(0)
            samples = {0}
            if max_orientation_samples > 1:
                samples.add(total - 1)
            while len(samples) < max_orientation_samples and len(samples) < total:
                samples.add(rng.getrandbits(d))
            candidates = sorted(samples)
        best = (m + 1, 0)
        for trial in candidates:
            covered = 0
            for i in range(d):
                covered |= coverage[i][((trial >> i) & 1) ^ 1]
            score = (m - covered.bit_count(), trial)
            if score < best:
                best = score
                if score[0] == 0:
                    break
        orientation = best[1]

    shore_a = [pair[(orientation >> i) & 1] for i, pair in enumerate(pairs)]
    shore_b = [pair[((orientation >> i) & 1) ^ 1] for i, pair in enumerate(pairs)]
    allowed_a: list[int] = []
    allowed_b: list[int] = []
    for x in outside:
        adjacent = graph[x]
        allowed_a.append(
            sum(1 << i for i, b in enumerate(shore_b) if b not in adjacent)
        )
        allowed_b.append(
            sum(1 << i for i, a in enumerate(shore_a) if a not in adjacent)
        )
    target = graph.subgraph(targets)
    inclusion = {v: v for v in targets}
    if any(a == 0 and b == 0 for a, b in zip(allowed_a, allowed_b, strict=True)):
        return BoundedRetractionResult(
            RetractionSearchStatus.EXHAUSTED, None, None, 1, 0
        )

    selected_b = [-1] * m
    trail: list[tuple[int, int]] = []
    states_explored = 1
    assignments_tried = 0

    def rollback(size: int) -> None:
        while len(trail) > size:
            i, old_mask = trail.pop()
            allowed_a[i] = old_mask

    def forced_vertex() -> int | None:
        # The most restricted forced vertex first. This is only a branch
        # ordering heuristic: all b-shore images remain available.
        return min(
            (i for i in range(m) if selected_b[i] == -1 and allowed_a[i] == 0),
            key=lambda i: (allowed_b[i].bit_count(), i),
            default=None,
        )

    def frame_for(i: int) -> _ShoreFrame:
        mask = allowed_b[i]
        choices: list[int] = []
        while mask:
            bit = mask & -mask
            mask -= bit
            choices.append(bit.bit_length() - 1)
        return _ShoreFrame(i, tuple(choices), len(trail))

    def found() -> BoundedRetractionResult:
        projection = {v: v for v in targets}
        for i, x in enumerate(outside):
            if selected_b[i] >= 0:
                projection[x] = shore_b[selected_b[i]]
            else:
                mask = allowed_a[i]
                if not mask:
                    raise AssertionError("unassigned vertex without an a-shore image")
                projection[x] = shore_a[(mask & -mask).bit_length() - 1]
        if not verify_retraction_witness(graph, target, (projection, inclusion)):
            raise AssertionError("two-shore propagation produced an invalid witness")
        return BoundedRetractionResult(
            RetractionSearchStatus.FOUND,
            projection,
            inclusion,
            states_explored,
            assignments_tried,
        )

    initial = forced_vertex()
    if initial is None:
        return found()
    frames = [frame_for(initial)]

    while frames:
        frame = frames[-1]
        rollback(frame.trail_start)
        selected_b[frame.vertex] = -1
        if frame.next_choice == len(frame.choices):
            frames.pop()
            continue
        if states_explored >= max_states:
            return BoundedRetractionResult(
                RetractionSearchStatus.LIMIT_REACHED,
                None,
                None,
                states_explored,
                assignments_tried,
            )
        image_index = frame.choices[frame.next_choice]
        frame.next_choice += 1
        states_explored += 1
        assignments_tried += 1
        selected_b[frame.vertex] = image_index

        forbidden = 1 << image_index
        inconsistent = False
        for y in outside_neighbors[frame.vertex]:
            if selected_b[y] != -1:
                continue  # b-shore images are always pairwise compatible
            old = allowed_a[y]
            if old & forbidden:
                new = old & ~forbidden
                trail.append((y, old))
                allowed_a[y] = new
                if new == 0 and allowed_b[y] == 0:
                    inconsistent = True
                    break
        if inconsistent:
            continue
        pending = forced_vertex()
        if pending is None:
            return found()
        frames.append(frame_for(pending))

    return BoundedRetractionResult(
        RetractionSearchStatus.EXHAUSTED,
        None,
        None,
        states_explored,
        assignments_tried,
    )

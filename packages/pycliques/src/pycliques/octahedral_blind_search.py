"""Graph-independent, conflict-guided discovery of octahedral retractions.

Heuristic pair discovery is followed by logically exact completion and
independent retraction verification. All unsuccessful bounded calls return
LIMIT_REACHED, never a global nonexistence claim.
"""
from __future__ import annotations

import random
from collections.abc import Hashable
from dataclasses import dataclass
from itertools import combinations

import networkx as nx

from .octahedral_blind import (
    OctahedralBlindStatus,
    _OctahedralMasks,
    _screen_masks,
    complete_dominating_octahedron,
)
from .octahedral_joint_fibers import joint_beam_search
from .octahedral_nonlocal_search import verify_induced_octahedron
from .octahedral_retractions import find_octahedral_retraction
from .retractions import RetractionSearchStatus, verify_retraction_witness


@dataclass(frozen=True)
class BlindOctahedralRetraction:
    status: OctahedralBlindStatus
    antipodal_pairs: tuple[tuple[Hashable, Hashable], ...] | None
    projection: dict[Hashable, Hashable] | None
    partial_dimension: int
    conflicts_learned: int
    repairs_considered: int
    completion_states: int


def _learn_conflicts(
    masks: _OctahedralMasks,
    indexed: tuple[tuple[int, int], ...],
    *,
    rounds: int,
    seed: int,
) -> set[int]:
    """Sound, randomized minimization; masks encode subsets of pair indices."""
    k = len(indexed)
    all_pairs = (1 << k) - 1
    cache: dict[int, bool] = {}
    rng = random.Random(seed)

    def impossible(mask: int) -> bool:
        if mask not in cache:
            partial = tuple(p for i, p in enumerate(indexed) if mask & (1 << i))
            cache[mask] = not _screen_masks(masks, partial).possible
        return cache[mask]

    if not impossible(all_pairs):
        return set()
    found: set[int] = set()
    for _ in range(rounds):
        core = all_pairs
        ordering = list(range(k))
        rng.shuffle(ordering)
        for i in ordering:
            smaller = core & ~(1 << i)
            if impossible(smaller):
                core = smaller
        # Independently check each local deletion against the exact screen.
        assert impossible(core)
        assert all(not impossible(core & ~(1 << i))
                   for i in range(k) if core & (1 << i))
        found.add(core)
    return found


def _repair_candidates(
    masks: _OctahedralMasks,
    pairs: tuple[tuple[int, int], ...],
    cores: set[int],
    target_dimension: int,
    delete_count: int,
):
    """Keep only conflict-hitting repairs with feasible residual domination."""
    k = len(pairs)
    remaining = target_dimension - k + delete_count
    all_pairs = (1 << k) - 1
    candidates = []
    for removed in combinations(range(k), delete_count):
        deleted = sum(1 << i for i in removed)
        if any(not (deleted & core) for core in cores):
            continue
        retained = all_pairs & ~deleted
        partial = tuple(pair for i, pair in enumerate(pairs)
                        if retained & (1 << i))
        common = masks.common(partial)
        if common.bit_count() < 2 * remaining:
            continue
        if any(not (masks.nonedge[v] & common)
               for v in _bit_indices(common)):
            continue
        candidates.append((common.bit_count(), deleted, partial))
    candidates.sort(reverse=True)
    return candidates


def _bit_indices(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def find_blind_octahedral_retraction(
    graph: nx.Graph,
    *,
    target_dimension: int,
    seed: int = 24002,
    beam: int = 12,
    samples: int = 25,
    biclique_variants: int = 1,
    core_rounds: int = 1000,
    delete_count: int = 7,
    max_repairs: int = 10,
    completion_states: int = 200_000,
    retraction_states: int = 200_000,
    retraction_seed: int = 9883,
) -> BlindOctahedralRetraction:
    """Attempt a *fully blind* retraction onto an induced ``O_d``.

    No starting antipodal pairs are used. Stochastic beam exploration
    creates a partial induced octahedron together with homomorphism fibers;
    learned 2-SAT conflicts constrain deletion-and-completion repairs.
    The final existing CSP solver checks the target and constructs a full
    certificate. No negative outcome of this heuristic proves nonexistence.

    The seed fixes the random generator, **not** the graph's vertex order.
    Use explicit graph insertion order for archival reproductions.
    """
    limits = (beam, samples, biclique_variants, core_rounds, max_repairs,
              completion_states, retraction_states)
    if any(type(x) is not int or x < 1 for x in limits):
        raise ValueError("all search budgets must be positive integers")
    if type(target_dimension) is not int or target_dimension < 3:
        raise ValueError("target_dimension must be an integer >= 3")
    if type(delete_count) is not int or delete_count < 0:
        raise ValueError("delete_count must be a nonnegative integer")
    if type(seed) is not int or type(retraction_seed) is not int:
        raise ValueError("seeds must be integers")
    m = _OctahedralMasks(graph)
    fiber = joint_beam_search(
        graph, target_dimension, seed=seed, beam=beam, samples=samples,
        biclique_variants=biclique_variants, verbose=False,
    )
    if fiber is None:
        return BlindOctahedralRetraction(
            OctahedralBlindStatus.LIMIT_REACHED, None, None, 0, 0, 0, 0
        )
    k = len(fiber.pairs)
    indexed = tuple(fiber.pairs)
    # The joint search uses the same graph node iteration order as m.
    labels = tuple((m.nodes[a], m.nodes[b]) for a, b in indexed)
    if k == target_dimension and verify_induced_octahedron(
        graph, labels, require_dominating=True
    ):
        ret = find_octahedral_retraction(
            graph, labels, max_states=retraction_states, seed=retraction_seed
        )
        if ret.status is RetractionSearchStatus.FOUND:
            return _verified_result(graph, labels, ret, k, 0, 0, 0)
    if delete_count > k or k - delete_count > target_dimension:
        return BlindOctahedralRetraction(
            OctahedralBlindStatus.LIMIT_REACHED, None, None, k, 0, 0, 0
        )
    conflicts = _learn_conflicts(m, indexed, rounds=core_rounds, seed=seed)
    choices = _repair_candidates(
        m, indexed, conflicts, target_dimension, delete_count
    )
    examined = 0
    states = 0
    for _, _, remaining_pairs in choices[:max_repairs]:
        examined += 1
        initial = tuple((m.nodes[a], m.nodes[b]) for a, b in remaining_pairs)
        completion = complete_dominating_octahedron(
            graph, initial, target_dimension=target_dimension,
            max_states=completion_states,
        )
        states += completion.states_explored
        if completion.status is not OctahedralBlindStatus.FOUND:
            continue
        target = completion.antipodal_pairs
        assert target is not None
        ret = find_octahedral_retraction(
            graph, target, max_states=retraction_states, seed=retraction_seed
        )
        if ret.status is RetractionSearchStatus.FOUND:
            return _verified_result(
                graph, target, ret, k, len(conflicts), examined, states
            )
    return BlindOctahedralRetraction(
        OctahedralBlindStatus.LIMIT_REACHED, None, None, k,
        len(conflicts), examined, states
    )


def _verified_result(graph, pairs, result, k, cores, repairs, states):
    assert result.retraction is not None and result.inclusion is not None
    target = graph.subgraph([v for pair in pairs for v in pair])
    if not verify_retraction_witness(
        graph, target, (result.retraction, result.inclusion)
    ):
        raise AssertionError("octahedral retraction failed independent verification")
    return BlindOctahedralRetraction(
        OctahedralBlindStatus.FOUND, pairs, result.retraction,
        k, cores, repairs, states
    )

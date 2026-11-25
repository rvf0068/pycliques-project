"""Generic joint search for octahedral anchors and partial retraction fibers.

No graph-specific pairs or retractions are stored. A returned partial map is
verified against the supplied host and is not a complete retraction unless
all vertices are covered and all required antipodal pairs have been chosen.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ._bitsets import iter_indices as bit_indices
from ._graph_utils import is_simple_graph


def _validate_host_graph(graph: nx.Graph) -> None:
    """Require the graph model assumed by octahedral fiber search."""
    if not is_simple_graph(graph):
        raise ValueError("require a simple undirected loopless NetworkX graph")


class Masks:
    def __init__(self, g):
        _validate_host_graph(g)
        nodes = list(g)
        self.nodes = nodes
        self.index = {v: i for i, v in enumerate(nodes)}
        self.n = len(nodes)
        self.all = (1 << len(nodes)) - 1
        self.closed = [1 << i for i in range(len(nodes))]
        for a, b in g.edges:
            i = self.index[a]
            j = self.index[b]
            self.closed[i] |= 1 << j
            self.closed[j] |= 1 << i
        self.nonedge = [self.all & ~v for v in self.closed]
        self.g = g


@dataclass(frozen=True)
class State:
    pairs: tuple[tuple[int, int], ...]
    fibers: tuple[tuple[int, int], ...]  # bit masks (A_i,B_i)
    covered: int
    common: int

    def mapped(self):
        return self.covered.bit_count()


def closed_biclique(
    mask: Masks, a: int, b: int, remaining: int, rng: random.Random, variants=4
):
    """Generate a few bicliques A,B through (a,b) in complement(G).

    The Galois closure B=N_H(A), A=N_H(B) is exact, so each produced
    biclique is inclusion-maximal among the vertices not yet assigned.
    We obtain different bicliques by seeding A or B with extra vertices.
    """
    H = mask.nonedge

    def closure(A):
        B = remaining
        for v in bit_indices(A):
            B &= H[v]
        if not ((B >> b) & 1):
            return None
        new_A = remaining
        for v in bit_indices(B):
            new_A &= H[v]
        if not ((new_A >> a) & 1):
            return None
        # Fixed point follows from the two antitone Galois maps.
        return new_A, B

    results = []

    def add(A):
        v = closure(A)
        if v and v not in results:
            results.append(v)

    add(1 << a)
    # Dual star: one singleton B, all possible A.
    A = H[b] & remaining
    add(A)
    # Seek more balanced bicliques using a second A vertex.
    extra = list(bit_indices(H[b] & remaining & ~(1 << a)))
    rng.shuffle(extra)
    for z in extra[:variants]:
        add((1 << a) | (1 << z))
    # And symmetrically, a second B vertex.
    extra = list(bit_indices(H[a] & remaining & ~(1 << b)))
    rng.shuffle(extra)
    for z in extra[:variants]:
        B = (1 << b) | (1 << z)
        A = remaining
        for w in bit_indices(B):
            A &= H[w]
        add(A)
    # The largest bicliques (by covered vertices) are preferred locally.
    return sorted(results, key=lambda p: (p[0] | p[1]).bit_count(), reverse=True)


def check_partial_state(mask: Masks, state: State):
    """Costly, exact invariant check for tests and successful certificates."""
    if len(state.pairs) != len(state.fibers):
        raise ValueError("partial state must have equal numbers of pairs and fibers")
    anchored: set[int] = set()
    seen = 0
    for i, ((a, b), (A, B)) in enumerate(zip(state.pairs, state.fibers)):
        assert a != b and (mask.nonedge[a] >> b) & 1
        assert A & (1 << a) and B & (1 << b)
        assert not (A & B) and not (seen & (A | B))
        assert all((mask.nonedge[x] & B) == B for x in bit_indices(A))
        for c, d in state.pairs[:i]:
            assert all(
                mask.g.has_edge(mask.nodes[u], mask.nodes[v])
                for u in (a, b)
                for v in (c, d)
            )
        seen |= A | B
        anchored.update((a, b))
    assert seen == state.covered
    common = mask.all
    for a, b in state.pairs:
        common &= mask.closed[a] & mask.closed[b]
    assert common == state.common
    return True


def retain_fibers(mask: Masks, state: State, keep_indexes):
    """Delete whole antipodal pairs and fibers; preserve partial retraction.

    This is an exact, generally applicable operation. Retained fibers are
    still disjoint bicliques, while retained anchors still induce O_k.
    """
    indices = sorted(set(keep_indexes))
    if any(i < 0 or i >= len(state.pairs) for i in indices):
        raise ValueError("invalid kept-pair index")
    pairs = tuple(state.pairs[i] for i in indices)
    fibers = tuple(state.fibers[i] for i in indices)
    covered = 0
    for A, B in fibers:
        covered |= A | B
    common = mask.all
    for a, b in pairs:
        common &= mask.closed[a] & mask.closed[b]
    return State(pairs, fibers, covered, common)


def save_partial_state(path, mask: Masks, state: State):
    """Checkpoint a proven partial homomorphism (not a final certificate)."""
    check_partial_state(mask, state)
    data = {
        "format": "pycliques-joint-fibers-v1",
        "source_order": mask.n,
        "source_edges": mask.g.number_of_edges(),
        "pairs": [[mask.nodes[a], mask.nodes[b]] for a, b in state.pairs],
        "fibers": [
            [
                [mask.nodes[x] for x in bit_indices(A)],
                [mask.nodes[x] for x in bit_indices(B)],
            ]
            for A, B in state.fibers
        ],
        "mapped_vertices": state.mapped(),
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_partial_state(path, mask: Masks) -> State:
    """Recompute every invariant; do not trust a saved partial assignment."""
    data = json.loads(Path(path).read_text())
    if data.get("format") != "pycliques-joint-fibers-v1":
        raise ValueError("incompatible checkpoint format")
    if (data.get("source_order"), data.get("source_edges")) != (
        mask.n,
        mask.g.number_of_edges(),
    ):
        raise ValueError("checkpoint source-graph dimensions differ")
    if len(data["pairs"]) != len(data["fibers"]):
        raise ValueError("checkpoint must have equal numbers of pairs and fibers")
    pairs = tuple(tuple(mask.index[v] for v in p) for p in data["pairs"])
    fibers = tuple(
        (sum(1 << mask.index[v] for v in A), sum(1 << mask.index[v] for v in B))
        for A, B in data["fibers"]
    )
    covered = 0
    for A, B in fibers:
        covered |= A | B
    common = mask.all
    for a, b in pairs:
        common &= mask.closed[a] & mask.closed[b]
    state = State(pairs, fibers, covered, common)
    check_partial_state(mask, state)
    if data.get("mapped_vertices") != state.mapped():
        raise ValueError("checkpoint mapped-vertex count differs")
    return state


def projection(mask: Masks, state: State):
    """Return the partial projection in the original vertex labels."""
    r = {}
    for (a, b), (A, B) in zip(state.pairs, state.fibers):
        for x in bit_indices(A):
            r[mask.nodes[x]] = mask.nodes[a]
        for x in bit_indices(B):
            r[mask.nodes[x]] = mask.nodes[b]
    return r


def sample_compatible_pairs(mask, available, rng, samples):
    H = mask.nonedge
    vertices = list(bit_indices(available))
    if not vertices:
        return []
    seen = set()
    result = []
    for _ in range(samples * 6):
        if len(result) >= samples:
            break
        a = rng.choice(vertices)
        choices = available & H[a]
        if not choices:
            continue
        cc = list(bit_indices(choices))
        b = rng.choice(cc)
        pair = tuple(sorted((a, b)))
        if pair not in seen:
            result.append(pair)
            seen.add(pair)
    return result


def count_nonedge_pairs(mask, C):
    return sum((mask.nonedge[x] & C).bit_count() for x in bit_indices(C)) // 2


def ranking(state: State, target: int, n: int, mask: Masks):
    k = len(state.pairs)
    rem = target - k
    C = state.common
    e = count_nonedge_pairs(mask, C)
    # Need at least one nonedge for every remaining antipodal pair.
    if e < rem:
        return (-(10**9), 0, 0)
    c = C.bit_count()
    # A future O_rem requires an induced matching in complement(G)[C].
    # Nonedges in C are necessary raw material for it.
    if k < target - 7:
        return (e, c, state.mapped())
    return (e, state.mapped(), c)


def joint_beam_search(
    g: nx.Graph,
    dimension: int,
    *,
    seed=24002,
    beam=24,
    samples=60,
    biclique_variants=3,
    steps=None,
    verbose=True,
    callback=None,
    completion_oracle=None,
    oracle_start=14,
    oracle_per_depth=3,
    initial_states=None,
):
    """Discover anchors AND assign biclique fibers in the same beam search.

    All partial states satisfy the exact retraction condition on their
    already assigned vertices. A completed state is already a certificate,
    pending independent public API verification. No pair coordinates are
    supplied. Failure is inconclusive, even if no state survives.
    """
    assert dimension >= 3
    mask = Masks(g)
    rng = random.Random(seed)
    states = (
        list(initial_states)
        if initial_states is not None
        else [State((), (), 0, mask.all)]
    )
    if not states:
        raise ValueError("initial_states cannot be empty")
    start_depth = len(states[0].pairs)
    if any(len(x.pairs) != start_depth for x in states):
        raise ValueError("initial states must have equal dimensions")
    for state in states:
        check_partial_state(mask, state)
    seen = set()
    start = time.monotonic()
    for k in range(start_depth, dimension):
        next_states = []
        for state in states:
            free = state.common & ~state.covered
            if free.bit_count() < 2 * (dimension - k):
                continue
            if count_nonedge_pairs(mask, free) < dimension - k:
                continue
            pairs = sample_compatible_pairs(mask, free, rng, samples)
            for a, b in pairs:
                C = state.common & mask.closed[a] & mask.closed[b]
                if (C & ~state.covered).bit_count() < 2 * (dimension - k - 1):
                    continue
                if count_nonedge_pairs(mask, C) < dimension - k - 1:
                    continue
                available = mask.all & ~state.covered
                for A, B in closed_biclique(
                    mask, a, b, available, rng, biclique_variants
                ):
                    if not ((A >> a) & 1 and (B >> b) & 1):
                        continue
                    candidate = State(
                        state.pairs + ((a, b),),
                        state.fibers + ((A, B),),
                        state.covered | A | B,
                        C,
                    )
                    key = (candidate.pairs, candidate.covered)
                    if key in seen:
                        continue
                    seen.add(key)
                    next_states.append(candidate)
        if not next_states:
            if verbose:
                print(f"No feasible joint states at dimension {k + 1}", flush=True)
            break
        next_states.sort(
            key=lambda z: ranking(z, dimension, mask.n, mask), reverse=True
        )
        # Maintain some beam diversity: multiple different anchors/fibers.
        states = []
        seen_anchor_sets: list[frozenset[int]] = []
        for candidate in next_states:
            anchor_set = frozenset(u for pair in candidate.pairs for u in pair)
            if any(len(anchor_set & old) >= 2 * k + 1 for old in seen_anchor_sets):
                continue
            states.append(candidate)
            seen_anchor_sets.append(anchor_set)
            if len(states) >= beam:
                break
        if not states:
            states = next_states[:beam]
        best = states[0]
        if verbose:
            print(
                f"depth={k + 1:2d} beam={len(states):2d} "
                f"mapped={best.mapped():4d}/{mask.n} "
                f"anchor_pool={best.common.bit_count():4d} "
                f"elapsed={time.monotonic() - start:.1f}s",
                flush=True,
            )
        if callback is not None:
            for candidate in states:
                if callback(mask, candidate):
                    return candidate
        if (
            completion_oracle is not None
            and k + 1 >= oracle_start
            and k + 1 < dimension
        ):
            for j, candidate in enumerate(states[:oracle_per_depth]):
                remainder = dimension - len(candidate.pairs)
                if remainder < 2:
                    continue
                completion = completion_oracle(mask, candidate, remainder)
                if not completion:
                    continue
                # Append the *found*, graph-independent octahedral completion.
                # Each added fiber is chosen greedily, as an induced biclique
                # in complement(G), and never uses a saved snub certificate.
                extended = candidate
                for aa, bb in completion[:remainder]:
                    aa = mask.index[aa]
                    bb = mask.index[bb]
                    if not (
                        (extended.common >> aa) & 1 and (extended.common >> bb) & 1
                    ):
                        raise AssertionError("completion not cross-compatible")
                    free = mask.all & ~extended.covered
                    fibers = closed_biclique(mask, aa, bb, free, rng, biclique_variants)
                    if not fibers:
                        raise AssertionError("completion has no biclique")
                    A, B = fibers[0]
                    extended = State(
                        extended.pairs + ((aa, bb),),
                        extended.fibers + ((A, B),),
                        extended.covered | A | B,
                        extended.common & mask.closed[aa] & mask.closed[bb],
                    )
                if verbose:
                    print(
                        f"  Completion oracle: O{dimension} embedded; "
                        f"{extended.mapped()}/{mask.n} vertices mapped",
                        flush=True,
                    )
                if callback is not None and callback(mask, extended):
                    return extended
        if steps is not None and k + 1 >= steps:
            break
    return states[0] if states else None


@dataclass(frozen=True)
class PartialOctahedralFibers:
    antipodal_pairs: tuple[tuple[object, object], ...]
    projection: dict[object, object]
    covered_vertices: int
    total_vertices: int
    target_dimension: int


def discover_partial_octahedral_fibers(
    graph: nx.Graph,
    target_dimension: int,
    *,
    seed: int = 24002,
    beam: int = 12,
    samples: int = 25,
    biclique_variants: int = 1,
) -> PartialOctahedralFibers:
    """Bounded, randomized search for anchors and a partial homomorphism.

    Failure to reach the requested dimension is INCONCLUSIVE; its purpose is
    to supply compatible pairs for exact conflict-guided reconstruction.
    """
    _validate_host_graph(graph)
    if type(target_dimension) is not int or target_dimension < 3:
        raise ValueError("target_dimension must be an integer >= 3")
    if any(type(x) is not int or x < 1 for x in (beam, samples, biclique_variants)):
        raise ValueError("beam, samples, biclique_variants must be positive integers")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    result = joint_beam_search(
        graph,
        target_dimension,
        seed=seed,
        beam=beam,
        samples=samples,
        biclique_variants=biclique_variants,
        verbose=False,
    )
    if result is None:
        return PartialOctahedralFibers((), {}, 0, len(graph), target_dimension)
    masks = Masks(graph)
    check_partial_state(masks, result)
    pairs = tuple((masks.nodes[a], masks.nodes[b]) for a, b in result.pairs)
    return PartialOctahedralFibers(
        pairs,
        projection(masks, result),
        result.mapped(),
        len(graph),
        target_dimension,
    )

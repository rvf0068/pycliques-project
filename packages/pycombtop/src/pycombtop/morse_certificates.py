"""Bounded, verified facet-greedy discrete Morse certificates.

The ordinary (nonaugmented) face poset is used: each matched pair is
removed as a free face and its unique maximal coface in the *remaining*
complex. When there is no free pair, a maximal simplex is declared
critical and deleted. The final face-poset matching is independently
verified to be acyclic before any homotopy conclusion is returned.

Accepted exact wedge criteria:
* one critical vertex with all other critical cells in one dimension;
* one critical vertex plus critical cells only in dimensions m and m+1,
  m >= 2, with top rational Betti number equal to the number of critical
  (m+1)-cells. This makes the cellular boundary zero and, by Hurewicz
  for a wedge of m-spheres, all attaching maps null-homotopic.

Refs: Forman, Adv. Math. 134 (1998), 90--145; Hatcher,
Algebraic Topology, Sections 0 and 4.2 (CW complexes and Hurewicz).
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import networkx as nx
from sympy import SparseMatrix


@dataclass(frozen=True)
class MorseCertificate:
    """Verified critical-cell counts and certified sphere multiplicities."""

    critical: dict[int, int]
    spheres: dict[int, int]
    matching_pairs: int
    uses_hurewicz: bool


def _vertex_key(vertex):
    """Sort clique vertices by members, not frozenset iteration order."""
    if isinstance(vertex, (set, frozenset)):
        return ("set", tuple(sorted(_vertex_key(x) for x in vertex)))
    return ("atom", type(vertex).__name__, repr(vertex))


def _bit_list(mask: int):
    while mask:
        bit = mask & -mask
        yield bit
        mask ^= bit


def _all_nonempty_faces(facets: set[int], limit: int) -> set[int] | None:
    faces: set[int] = set()
    for facet in sorted(facets):
        if facet.bit_count() > limit.bit_length():
            return None
        face = facet
        while face:
            faces.add(face)
            if len(faces) > limit:
                return None
            face = (face - 1) & facet
    return faces


def _remove_maximal(facets: set[int], sigma: int, excluded_bit: int) -> None:
    """Delete maximal sigma (and its specified free codimension-one face)."""
    facets.remove(sigma)
    for bit in _bit_list(sigma & ~excluded_bit):
        face = sigma ^ bit
        if face and not any((face & other) == face for other in facets):
            facets.add(face)


def _free_pair(facets: set[int], order: list[int]) -> tuple[int, int] | None:
    for sigma in order:
        if sigma.bit_count() < 2:
            continue
        for bit in _bit_list(sigma):
            tau = sigma ^ bit
            if not any((tau & other) == tau for other in facets if other != sigma):
                return sigma, bit
    return None


def _run_matching(
    original: set[int],
    seed: int,
    *,
    max_facets: int = 1500,
    max_steps: int = 12000,
) -> tuple[Counter, list[tuple[int, int]]] | None:
    facets = set(original)
    rng = random.Random(seed)
    critical: Counter = Counter()
    pairs: list[tuple[int, int]] = []
    steps = 0
    while facets:
        if len(facets) > max_facets or steps >= max_steps:
            return None
        steps += 1
        # The integer set has stable membership and we sort before consuming
        # random tie-breaking values. Attempts use local, fixed RNG seeds.
        order = sorted(facets)
        order.sort(key=lambda f: (f.bit_count(), rng.random()), reverse=True)
        free = _free_pair(facets, order)
        if free is None:
            sigma = rng.choice(order)
            critical[sigma.bit_count() - 1] += 1
            bit = 0
        else:
            sigma, bit = free
            pairs.append((sigma ^ bit, sigma))
        _remove_maximal(facets, sigma, bit)
    return critical, pairs


def _valid_acyclic_matching(
    faces: set[int], pairs: list[tuple[int, int]], critical: Counter
) -> bool:
    used: set[int] = set()
    paired = set(pairs)
    if len(paired) != len(pairs):
        return False
    for low, high in pairs:
        if (
            low not in faces
            or high not in faces
            or low in used
            or high in used
            or high.bit_count() != low.bit_count() + 1
            or (low & high) != low
        ):
            return False
        used.update((low, high))
    if Counter(f.bit_count() - 1 for f in faces - used) != critical:
        return False
    gradient = nx.DiGraph()
    gradient.add_nodes_from(faces)
    for high in faces:
        for bit in _bit_list(high):
            low = high ^ bit
            if not low:
                continue
            if (low, high) in paired:
                gradient.add_edge(low, high)
            else:
                gradient.add_edge(high, low)
    return bool(nx.is_directed_acyclic_graph(gradient))


def _collapse_free_pairs(
    original: set[int],
    *,
    max_facets: int = 1500,
    max_steps: int = 12000,
) -> set[int] | None:
    """Only genuine elementary collapses; homotopy type is preserved."""
    facets = set(original)
    rng = random.Random(0)
    steps = 0
    while True:
        if len(facets) > max_facets or steps >= max_steps:
            return None
        steps += 1
        order = sorted(facets)
        order.sort(key=lambda f: (f.bit_count(), rng.random()), reverse=True)
        free = _free_pair(facets, order)
        if free is None:
            return facets
        _remove_maximal(facets, *free)


def _top_betti_matches(
    original: set[int],
    dim: int,
    count: int,
    *,
    max_top_cells: int = 120,
    max_boundary_rows: int = 500,
) -> bool:
    """Verify beta_dim(Q)=count using exact rank on a dimension-dim core.

    Never calculate ker(d_dim) alone when cells above dimension dim remain.
    """
    core = _collapse_free_pairs(original)
    if core is None or not core or max(f.bit_count() for f in core) != dim + 1:
        return False
    top = sorted(f for f in core if f.bit_count() == dim + 1)
    if len(top) > max_top_cells:
        return False
    lower = sorted({f ^ bit for f in top for bit in _bit_list(f)})
    if len(lower) > max_boundary_rows:
        return False
    row_idx = {f: i for i, f in enumerate(lower)}
    entries = {}
    for j, sigma in enumerate(top):
        for i, bit in enumerate(_bit_list(sigma)):
            entries[row_idx[sigma ^ bit], j] = (-1) ** i
    boundary = SparseMatrix(len(lower), len(top), entries)
    # Since the core is dim-dimensional, H_dim = ker(d_dim).
    return bool(len(top) - boundary.rank() == count)


def facet_morse_certificate(
    facets,
    *,
    max_faces: int = 12000,
    seeds: tuple[int, ...] = (1, 2, 3),
) -> MorseCertificate | None:
    """Try bounded matching searches; return only a checked exact certificate.

    facets: iterable of nonempty iterables of hashable vertices.
    None means undecided, not a negative statement about homotopy type.
    """
    if max_faces < 1:
        raise ValueError("max_faces must be positive")
    facets = [frozenset(f) for f in facets]
    if not facets or any(not f for f in facets):
        return None
    vertices = sorted(set().union(*facets), key=_vertex_key)
    if len(vertices) > 64:
        return None
    indices = {v: i for i, v in enumerate(vertices)}
    original = {sum(1 << indices[v] for v in f) for f in facets}
    faces = _all_nonempty_faces(original, max_faces)
    if faces is None:
        return None
    for seed in seeds:
        candidate = _run_matching(original, seed)
        if candidate is None:
            continue
        critical, pairs = candidate
        if critical.get(0) != 1:
            continue
        positive = sorted(d for d in critical if d > 0)
        admissible = (
            not positive
            or len(positive) == 1
            or (
                len(positive) == 2
                and positive[0] >= 2
                and positive[1] == positive[0] + 1
            )
        )
        if not admissible or not _valid_acyclic_matching(faces, pairs, critical):
            continue
        if not positive:
            return MorseCertificate(dict(critical), {}, len(pairs), False)
        if len(positive) == 1:
            d = positive[0]
            return MorseCertificate(dict(critical), {d: critical[d]}, len(pairs), False)
        lower, higher = positive
        if _top_betti_matches(original, higher, critical[higher]):
            return MorseCertificate(
                dict(critical),
                {lower: critical[lower], higher: critical[higher]},
                len(pairs),
                True,
            )
    return None


def graph_morse_certificate(
    graph: nx.Graph, *, max_faces: int = 12000
) -> MorseCertificate | None:
    """Apply the exact certificate search to the clique complex of a graph."""
    if graph.number_of_nodes() > 64:
        return None
    return facet_morse_certificate(nx.find_cliques(graph), max_faces=max_faces)

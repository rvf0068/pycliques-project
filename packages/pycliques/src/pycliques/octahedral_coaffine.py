"""Complete (or explicitly bounded) search for unseeded octahedral retractions.

Every retraction r:X->O_d yields a coaffine weak graph endomorphism
f = i * tau * r, where i includes O_d and tau swaps each antipodal
pair. Indeed r(f(v)) = tau(r(v)), which is a distinct nonneighbor of
r(v), so f(v) cannot equal or be adjacent to v. Moreover f^2 = i*r.

Conversely, when a coaffine endomorphism f has an idempotent square
with induced octahedral image, f^2 is a valid retraction. Searching
ALL coaffine weak endomorphisms and checking their squares is therefore
complete for detecting retractions onto any induced O_d (d>=3).
A bounded search gives no negative conclusion upon reaching a limit.

Unlike coaffinations, these endomorphisms need NOT be automorphisms.
"""
from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass

import networkx as nx

from .octahedral_nonlocal_search import verify_induced_octahedron
from .retractions import RetractionSearchStatus, verify_retraction_witness


@dataclass(frozen=True)
class CoaffineOctahedronResult:
    """Verified retraction, or exhaustive/limited search status.

    FOUND includes a verified complete projection fixing the induced
    target pointwise and its opposite pairs. EXHAUSTED proves no
    octahedral retraction of dimension >= min_dimension. LIMIT_REACHED
    proves nothing about the existence of a retraction.
    """

    status: RetractionSearchStatus
    antipodal_pairs: tuple[tuple[Hashable, Hashable], ...] | None
    retraction: dict[Hashable, Hashable] | None
    inclusion: dict[Hashable, Hashable] | None
    endomorphism: dict[Hashable, Hashable] | None
    states_explored: int
    arc_revisions: int
    assignments_tried: int

    @property
    def dimension(self) -> int | None:
        """Dimension of a found octahedral target."""
        return None if self.antipodal_pairs is None else len(self.antipodal_pairs)


def find_octahedral_retraction_coaffine(
    graph: nx.Graph,
    *,
    min_dimension: int = 3,
    max_states: int = 20_000,
    time_limit: float | None = None,
) -> CoaffineOctahedronResult:
    """Search for a retraction onto any induced octahedron with no seed.

    Find weak endomorphisms f:X->X such that f(v) lies outside the
    closed neighborhood of v; test whether f^2 is an idempotent
    projection onto an induced O_d, d>=min_dimension.

    Constraint propagation uses exact arc consistency on all source
    edges with bitmasked image domains; MRV branching searches every
    assignment unless a state/time limit is reached. Search order is
    deterministic for a fixed NetworkX vertex iteration order.

    EXHAUSTED is an exact negative certificate for ALL induced O_d
    retractions of the specified dimensions, not merely for selected
    embeddings: every such retraction r yields f=i*tau*r, f^2=i*r.
    A limit or failure to construct a host graph is never negative.

    Edge images may coincide, as in the standard clique-graph morphism
    convention. The input must be a finite simple undirected graph.

    >>> import networkx as nx
    >>> result = find_octahedral_retraction_coaffine(nx.octahedral_graph())
    >>> result.status is RetractionSearchStatus.FOUND
    True
    >>> result.dimension
    3
    """
    if (
        not isinstance(graph, nx.Graph)
        or graph.is_directed()
        or graph.is_multigraph()
        or nx.number_of_selfloops(graph)
    ):
        raise ValueError("require a simple undirected loopless NetworkX graph")
    if type(min_dimension) is not int or min_dimension < 3:
        raise ValueError("min_dimension must be an integer >= 3")
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if time_limit is not None and (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not math.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and strictly positive")

    nodes = list(graph)
    n = len(nodes)
    positions = {v: i for i, v in enumerate(nodes)}
    neighbors = [tuple(positions[w] for w in graph[v]) for v in nodes]
    closed = [
        sum(1 << j for j in neighbors[i]) | (1 << i) for i in range(n)
    ]
    full = (1 << n) - 1
    domains = [full & ~closed[i] for i in range(n)]
    start = time.monotonic()
    states = 0
    revisions = 0
    assignments = 0

    def outcome(
        status: RetractionSearchStatus,
        pairs: tuple[tuple[Hashable, Hashable], ...] | None = None,
        projection: dict[Hashable, Hashable] | None = None,
        inclusion: dict[Hashable, Hashable] | None = None,
        endomorphism: dict[Hashable, Hashable] | None = None,
    ) -> CoaffineOctahedronResult:
        return CoaffineOctahedronResult(
            status, pairs, projection, inclusion, endomorphism,
            states, revisions, assignments
        )

    if n < 2 * min_dimension or any(mask == 0 for mask in domains):
        return outcome(RetractionSearchStatus.EXHAUSTED)

    class _TimeLimit(Exception):
        pass

    def check_time() -> None:
        if time_limit is not None and time.monotonic() - start > time_limit:
            raise _TimeLimit

    def propagate(
        d: list[int], pending: deque[tuple[int, int]]
    ) -> bool:
        """Revise directed arcs (i,j): image(i)~image(j) or equality."""
        nonlocal revisions
        processed = 0
        while pending:
            processed += 1
            if processed % 128 == 0:
                check_time()
            i, j = pending.popleft()
            old = d[i]
            new = old
            bits = old
            while bits:
                bit = bits & -bits
                bits -= bit
                k = bit.bit_length() - 1
                if not (closed[k] & d[j]):
                    new ^= bit
            if new != old:
                revisions += 1
                if not new:
                    return False
                d[i] = new
                for x in neighbors[i]:
                    if x != j:
                        pending.append((x, i))
        return True

    def positive(
        d: list[int],
    ) -> tuple[
        tuple[tuple[Hashable, Hashable], ...],
        dict[Hashable, Hashable],
        dict[Hashable, Hashable],
        dict[Hashable, Hashable],
    ] | None:
        f = [mask.bit_length() - 1 for mask in d]
        if not all(i != f[i] and not graph.has_edge(nodes[i], nodes[f[i]])
                   for i in range(n)):
            raise AssertionError("Coaffine CSP returned invalid endomorphism")
        if not all(f[u] == f[v] or graph.has_edge(nodes[f[u]], nodes[f[v]])
                   for u in range(n) for v in neighbors[u]):
            raise AssertionError("Coaffine CSP returned a non-morphism")

        r = [f[f[i]] for i in range(n)]
        image = set(r)
        if len(image) < 2 * min_dimension or len(image) % 2:
            return None
        if any(r[j] != j for j in image):
            return None

        pairs: list[tuple[Hashable, Hashable]] = []
        remaining = set(image)
        while remaining:
            u = min(remaining)
            missing = [
                v for v in image
                if v != u and not graph.has_edge(nodes[u], nodes[v])
            ]
            if len(missing) != 1 or missing[0] not in remaining:
                return None
            v = missing[0]
            if sum(
                w != v and not graph.has_edge(nodes[v], nodes[w])
                for w in image
            ) != 1:
                return None
            pairs.append((nodes[u], nodes[v]))
            remaining.difference_update((u, v))

        opposite = tuple(pairs)
        projection = {nodes[i]: nodes[r[i]] for i in range(n)}
        endomorphism = {nodes[i]: nodes[f[i]] for i in range(n)}
        target = graph.subgraph(nodes[i] for i in image)
        inclusion = {v: v for v in target}
        if not verify_induced_octahedron(graph, opposite):
            raise AssertionError("Coaffine CSP produced non-octahedral target")
        if not verify_retraction_witness(
            graph, target, (projection, inclusion)
        ):
            raise AssertionError("Coaffine CSP produced invalid retraction")
        return opposite, projection, inclusion, endomorphism

    limit = object()

    def dfs(d: list[int]):
        nonlocal states, assignments
        if states >= max_states:
            return limit
        states += 1
        check_time()
        undecided = [i for i in range(n) if d[i].bit_count() > 1]
        if not undecided:
            return positive(d)
        x = min(
            undecided,
            key=lambda i: (d[i].bit_count(), -len(neighbors[i]))
        )
        choices = d[x]
        while choices:
            bit = choices & -choices
            choices -= bit
            assignments += 1
            candidate = d.copy()
            candidate[x] = bit
            if propagate(
                candidate, deque((i, x) for i in neighbors[x])
            ):
                result = dfs(candidate)
                if result is limit or result is not None:
                    return result
        return None

    try:
        if not propagate(
            domains,
            deque((i, j) for i in range(n) for j in neighbors[i]),
        ):
            return outcome(RetractionSearchStatus.EXHAUSTED)
        answer = dfs(domains)
    except (_TimeLimit, RecursionError):
        answer = limit

    if answer is limit:
        return outcome(RetractionSearchStatus.LIMIT_REACHED)
    if answer is None:
        return outcome(RetractionSearchStatus.EXHAUSTED)
    pairs, projection, inclusion, endomorphism = answer
    return outcome(
        RetractionSearchStatus.FOUND, pairs, projection, inclusion, endomorphism
    )

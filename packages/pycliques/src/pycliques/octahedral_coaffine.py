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
from collections.abc import Hashable
from dataclasses import dataclass

import networkx as nx

from ._graph_utils import is_simple_graph
from .coaffine_endomorphisms import (
    _search_coaffine_python,
    _validate_search_order,
    verify_coaffine_endomorphism,
)
from .octahedral_coaffine_native import (
    MAX_NATIVE_ORDER,
    find_coaffine_native_executable,
    run_coaffine_native,
)
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
    backend: str = "auto",
    value_order: str = "label",
    sat_max_clauses: int = 4_000_000,
    sat_memory_limit_mb: int = 2048,
) -> CoaffineOctahedronResult:
    """Search for a retraction onto any induced octahedron with no seed.

    Find weak endomorphisms f:X->X such that f(v) lies outside the
    closed neighborhood of v; test whether f^2 is an idempotent
    projection onto an induced O_d, d>=min_dimension.

    Transposed bitmasked image domains implement exact arc consistency.
    MRV branching searches every assignment unless a limit is reached.
    value_order='fiber' tries larger possible fibers first; the default
    'label' preserves the historical branch order. Ties are deterministic
    for a fixed backend, compiler and NetworkX vertex iteration order.

    EXHAUSTED is an exact negative certificate for ALL induced O_d
    retractions of the specified dimensions, not merely for selected
    embeddings: every such retraction r yields f=i*tau*r, f^2=i*r.
    A limit or failure to construct a host graph is never negative.

    Edge images may coincide, as in the standard clique-graph morphism
    convention. The input must be a finite simple undirected graph.

    The default backend='auto' uses the version-matched C++17
    bitset engine if it has been explicitly compiled; otherwise it uses
    the established portable Python engine. backend='python' forces
    the reference implementation. backend='native' requires the compiled
    executable and supports up to 2048 vertices. Build it once with
    'python -m pycliques.octahedral_coaffine_native --build'.
    This is optional; no compilation occurs during package installation or
    normal imports. Every positive native certificate is re-verified
    independently in Python. Resource limits are always inconclusive.

    backend='sat' explicitly selects the optional Linux Z3 shared-library
    backend (sudo apt install libz3-4 on Ubuntu). It streams compressed
    DIMACS with a clause/variable limit sat_max_clauses, runs Z3 in a
    subprocess with a hard address-space limit sat_memory_limit_mb (MiB),
    and interprets max_states as the SAT conflict budget. time_limit
    includes formula construction, loading and solving. The SAT result's
    states_explored counts solver conflicts, not recursive CSP states;
    arc_revisions and assignments_tried are zero. SAT has its own branching
    policy; value_order must be 'label'. No optional solver is auto-selected.

    >>> import networkx as nx
    >>> result = find_octahedral_retraction_coaffine(nx.octahedral_graph())
    >>> result.status is RetractionSearchStatus.FOUND
    True
    >>> result.dimension
    3
    """
    if not is_simple_graph(graph):
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

    _validate_search_order(value_order)
    if backend not in ("auto", "python", "native", "sat"):
        raise ValueError("backend must be 'auto', 'python', 'native', or 'sat'")
    if type(sat_max_clauses) is not int or sat_max_clauses < 1:
        raise ValueError("sat_max_clauses must be a positive integer")
    if type(sat_memory_limit_mb) is not int or not 1 <= sat_memory_limit_mb < 2**32:
        raise ValueError("sat_memory_limit_mb must be a positive 32-bit integer")
    if backend == "sat":
        if value_order != "label":
            raise ValueError("SAT backend does not support value_order='fiber'")
        if max_states >= 2**32:
            raise ValueError("SAT max_states must fit in an unsigned 32-bit integer")

    native_path = None
    if backend in ("auto", "native"):
        native_path = find_coaffine_native_executable()
        if backend == "native" and native_path is None:
            raise RuntimeError(
                "Native backend is not built; run "
                "'python -m pycliques.octahedral_coaffine_native --build'"
            )
        if backend == "native" and len(graph) > MAX_NATIVE_ORDER:
            raise ValueError(
                f"Native coaffine backend supports at most {MAX_NATIVE_ORDER} vertices"
            )
        if backend == "native" and max_states > (1 << 63) - 1:
            raise ValueError("Native max_states must fit in a signed 64-bit integer")
        if backend == "native" and min_dimension > (1 << 31) - 1:
            raise ValueError("Native min_dimension must fit in a signed 32-bit integer")
    if (
        native_path is not None
        and len(graph) <= MAX_NATIVE_ORDER
        and max_states <= (1 << 63) - 1
        and min_dimension <= (1 << 31) - 1
    ):
        # The subprocess uses 0..n-1 positions and cannot interpret arbitrary
        # NetworkX labels. Always map its result back and independently verify.
        native = run_coaffine_native(
            graph,
            min_dimension=min_dimension,
            max_states=max_states,
            time_limit=time_limit,
            executable=native_path,
            value_order=value_order,
        )
        status = RetractionSearchStatus[native.status]
        if status is not RetractionSearchStatus.FOUND:
            return CoaffineOctahedronResult(
                status,
                None,
                None,
                None,
                None,
                native.states_explored,
                native.arc_revisions,
                native.assignments_tried,
            )

        if (
            native.pairs is None
            or native.projection is None
            or native.endomorphism is None
        ):
            raise RuntimeError("Native FOUND result is missing a witness")
        labels = list(graph)
        pairs = tuple((labels[u], labels[v]) for u, v in native.pairs)
        r = {labels[i]: labels[j] for i, j in enumerate(native.projection)}
        f = {labels[i]: labels[j] for i, j in enumerate(native.endomorphism)}
        image = {v for pair in pairs for v in pair}
        target = graph.subgraph(image)
        inclusion = {v: v for v in target}
        if (
            len(image) != 2 * len(pairs)
            or set(r.values()) != image
            or any(r[u] != f[f[u]] for u in graph)
            or not verify_coaffine_endomorphism(graph, f)
            or not verify_induced_octahedron(graph, pairs)
            or not verify_retraction_witness(graph, target, (r, inclusion))
        ):
            raise RuntimeError("Native backend returned an invalid certificate")
        return CoaffineOctahedronResult(
            status,
            pairs,
            r,
            inclusion,
            f,
            native.states_explored,
            native.arc_revisions,
            native.assignments_tried,
        )

    if len(graph) < 2 * min_dimension:
        return CoaffineOctahedronResult(
            RetractionSearchStatus.EXHAUSTED, None, None, None, None, 0, 0, 0
        )

    def accept(f: dict[Hashable, Hashable]) -> bool:
        projection = {v: f[f[v]] for v in graph}
        image = set(projection.values())
        if len(image) < 2 * min_dimension or len(image) % 2:
            return False
        if any(projection[v] != v for v in image):
            return False
        # The complement on the image must be a perfect matching.
        target = graph.subgraph(image)
        return all(d == 1 for _, d in nx.complement(target).degree())

    if backend == "sat":
        from ._octahedral_sat import search_sat

        result = search_sat(
            graph,
            min_dimension=min_dimension,
            max_states=max_states,
            time_limit=time_limit,
            max_clauses=sat_max_clauses,
            memory_limit_mb=sat_memory_limit_mb,
        )
        sat_f = result.endomorphism
        if sat_f is not None and (
            not verify_coaffine_endomorphism(graph, sat_f)
            or not accept(sat_f)
            or any(sat_f[sat_f[sat_f[v]]] != sat_f[v] for v in graph)
        ):
            raise RuntimeError("SAT backend returned an invalid certificate")
    else:
        result = _search_coaffine_python(
            graph,
            max_states=max_states,
            time_limit=time_limit,
            value_order=value_order,
            accept=accept,
        )
    portable_pairs: tuple[tuple[Hashable, Hashable], ...] | None = None
    portable_projection: dict[Hashable, Hashable] | None = None
    portable_inclusion: dict[Hashable, Hashable] | None = None
    portable_f = result.endomorphism
    if portable_f is not None:
        portable_projection = {v: portable_f[portable_f[v]] for v in graph}
        image = set(portable_projection.values())
        target = graph.subgraph(image)
        portable_pairs = tuple(nx.complement(target).edges())
        portable_inclusion = {v: v for v in target}
        if not verify_induced_octahedron(graph, portable_pairs):
            raise AssertionError("Coaffine CSP produced non-octahedral target")
        if not verify_retraction_witness(
            graph, target, (portable_projection, portable_inclusion)
        ):
            raise AssertionError("Coaffine CSP produced invalid retraction")
    return CoaffineOctahedronResult(
        result.status,
        portable_pairs,
        portable_projection,
        portable_inclusion,
        portable_f,
        result.states_explored,
        result.arc_revisions,
        result.assignments_tried,
    )

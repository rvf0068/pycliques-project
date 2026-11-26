"""Octahedral retractions using the support incidence of an implicit host.

A cut pins the two disjoint anchor supports on opposite shores. A collection
of compatible anchors and cuts is a retraction certificate iff every support
is monochromatic for at least one cut. Adaptive weighted min-cut updates
and pair replacement discover these certificates without fixing dimension.
The exact alternative shares the existing bounded SAT worker.
"""

from __future__ import annotations

import math
import random
import subprocess
import tempfile
import time
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .coaffine_endomorphisms import verify_coaffine_endomorphism
from .implicit_clique_graph import ImplicitCliqueGraph
from .octahedral_coaffine import CoaffineOctahedronResult
from .octahedral_incidence_native import (
    MAX_WALK_GROUND,
    MAX_WALK_ORDER,
    find_incidence_native,
)
from .octahedral_nonlocal_search import verify_induced_octahedron
from .retractions import RetractionSearchStatus, verify_retraction_witness

Cut = tuple[int, int, int]


@dataclass(frozen=True)
class IncidenceOctahedronResult(CoaffineOctahedronResult):
    """Existing certificate fields plus an independently checked partial cover.

    Cut triples contain original active clique IDs and a ground-set bitmask.
    For walk results, states_explored is the step of the best checkpoint
    (zero without a checkpoint), not the total number of attempted moves.
    For SAT results, it counts conflicts. EXHAUSTED is scoped to the retained
    pairs, if any, and never follows from a failed walk.
    """

    partial_cuts: tuple[Cut, ...] = ()
    uncovered: int | None = None
    retained_pairs: tuple[tuple[int, int], ...] = ()


def _check_pairs(
    host: ImplicitCliqueGraph, pairs: Sequence[tuple[int, int]]
) -> None:
    endpoints = [v for pair in pairs for v in pair]
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("antipodal endpoints must be distinct")
    if any(type(v) is not int or v not in host for v in endpoints):
        raise ValueError("antipodal endpoints must be active clique IDs")
    for i, (a, b) in enumerate(pairs):
        if host.has_edge(a, b):
            raise ValueError("antipodal supports must be disjoint")
        if any(
            not host.has_edge(v, w)
            for pair in pairs[:i]
            for v in pair
            for w in (a, b)
        ):
            raise ValueError("antipodal pairs must induce an octahedron")


def _check_cuts(
    host: ImplicitCliqueGraph, cuts: tuple[Cut, ...]
) -> tuple[dict[Hashable, Hashable], int]:
    _check_pairs(host, [(a, b) for a, b, _ in cuts])
    ground = (1 << len(host.base_vertices)) - 1
    r: dict[Hashable, Hashable] = {v: v for a, b, _ in cuts for v in (a, b)}
    for a, b, cut in cuts:
        if (
            cut < 0
            or cut & ~ground
            or host.clique_masks[a] & ~cut
            or host.clique_masks[b] & cut
        ):
            raise ValueError("cut does not separate its anchor supports")
        for v in host:
            if v not in r:
                support = host.clique_masks[v]
                if not support & ~cut:
                    r[v] = a
                elif not support & cut:
                    r[v] = b
    return r, len(host) - len(r)


def _found(
    host: ImplicitCliqueGraph,
    pairs: tuple[tuple[Hashable, Hashable], ...],
    r: dict[Hashable, Hashable],
    f: dict[Hashable, Hashable],
    states: int,
    cuts: tuple[Cut, ...],
    retained: tuple[tuple[int, int], ...],
    minimum: int,
) -> IncidenceOctahedronResult:
    graph = host.to_networkx()
    assert graph is not None
    image = {v for pair in pairs for v in pair}
    target = graph.subgraph(image)
    inclusion = {v: v for v in target}
    if (
        len(pairs) < minimum
        or not verify_induced_octahedron(graph, pairs)
        or not verify_coaffine_endomorphism(graph, f)
        or not verify_retraction_witness(graph, target, (r, inclusion))
        or set(f.values()) != image
        or any(r[v] != f[f[v]] or f[r[v]] != f[v] for v in graph)
        or not all(f[a] == b and f[b] == a for a, b in retained)
    ):
        raise RuntimeError("Incidence backend returned an invalid certificate")
    return IncidenceOctahedronResult(
        RetractionSearchStatus.FOUND,
        pairs,
        r,
        inclusion,
        f,
        states,
        0,
        0,
        cuts,
        _check_cuts(host, cuts)[1] if cuts else None,
        retained,
    )


def _walk(
    host: ImplicitCliqueGraph,
    executable: Path,
    *,
    seed: int,
    max_steps: int,
    min_dimension: int,
    time_limit: float | None,
    starts: Sequence[Sequence[tuple[int, int]]],
) -> tuple[tuple[Cut, ...], int, int]:
    labels = list(host)
    position = {v: i for i, v in enumerate(labels)}
    with tempfile.TemporaryDirectory(prefix="pycliques-incidence-") as tmp:
        directory = Path(tmp)
        (directory / "supports.txt").write_text(
            "".join(f"{host.clique_masks[v]}\n" for v in labels), encoding="ascii"
        )
        (directory / "starts.txt").write_text(
            "".join(
                " ".join(
                    map(str, [len(pairs), *(position[v] for p in pairs for v in p)])
                )
                + "\n"
                for pairs in starts
            ),
            encoding="ascii",
        )
        command = [
            str(executable.resolve()),
            "weighted",
            str(seed),
            str(max_steps),
            str(min_dimension),
        ]
        try:
            run = subprocess.run(
                command,
                cwd=directory,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=time_limit,
                check=False,
                text=True,
            )
            if run.returncode:
                raise RuntimeError(f"Incidence walk failed: {run.stderr[-2000:]}")
        except subprocess.TimeoutExpired:
            # A complete best checkpoint is published atomically.
            pass
        checkpoint = directory / "best.txt"
        if not checkpoint.exists():
            return (), len(host), 0
        try:
            header, *rows = checkpoint.read_text(encoding="ascii").splitlines()
            missed, count, step = map(int, header.split())
            if len(rows) != count or not 1 <= step <= max_steps:
                raise ValueError("invalid checkpoint counters")
            cuts_list: list[Cut] = []
            for row in rows:
                a, b, mask = map(int, row.split())
                if not (0 <= a < len(labels) and 0 <= b < len(labels)):
                    raise ValueError("invalid endpoint indices")
                cuts_list.append((labels[a], labels[b], mask))
            cuts = tuple(cuts_list)
            _, checked = _check_cuts(host, cuts)
            if checked != missed:
                raise ValueError("incorrect checkpoint coverage")
        except (ValueError, IndexError) as exc:
            raise RuntimeError("Invalid incidence walk checkpoint") from exc
        return cuts, missed, step


def find_octahedral_retraction_incidence(
    host: ImplicitCliqueGraph,
    *,
    backend: str = "walk",
    min_dimension: int = 3,
    max_steps: int = 20_000,
    max_states: int = 100_000,
    time_limit: float | None = None,
    seed: int = 139,
    sat_max_clauses: int = 4_000_000,
    sat_memory_limit_mb: int = 2048,
    retained_pairs: Sequence[tuple[int, int]] = (),
    starting_matchings: Sequence[Sequence[tuple[int, int]]] = (),
    executable: Path | None = None,
) -> IncidenceOctahedronResult:
    """Search any octahedral dimension >= min_dimension in an implicit host.

    'walk' uses the optional GNU C++17 weighted cut walk (at most 64 ground
    atoms and 1280 active cliques). 'sat' uses the existing optional Linux
    Z3 shared library, exact incidence constraints and optional retained
    pairs. 'hybrid' uses half the wall budget for the walk, then tries SAT
    retaining a reproducibly sampled half of the best pairs. Failure of
    regional completion in hybrid mode is always inconclusive.

    The walk may return partial cuts on LIMIT_REACHED. starting_matchings
    are optional heuristic restart candidates, not fixed constraints.
    Retained pairs are hard constraints and are accepted only by 'sat'.
    In that mode EXHAUSTED means no solution containing those pairs.
    The time budget covers discovery/encoding/solving, not final independent
    certificate verification. Native compilation is explicit and separate.
    """
    if not isinstance(host, ImplicitCliqueGraph):
        raise TypeError("host must be an ImplicitCliqueGraph")
    if backend not in ("walk", "sat", "hybrid"):
        raise ValueError("backend must be 'walk', 'sat', or 'hybrid'")
    if type(min_dimension) is not int or min_dimension < 3:
        raise ValueError("min_dimension must be an integer >= 3")
    for value, maximum in (
        (max_steps, 2**31 - 1),
        (max_states, 2**32 - 1),
        (sat_max_clauses, 2**31 - 1),
        (sat_memory_limit_mb, 2**32 - 1),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("search budgets must be positive bounded integers")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if time_limit is not None and (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not math.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and strictly positive")
    retained = tuple(retained_pairs)
    if retained and backend != "sat":
        raise ValueError("retained_pairs requires backend='sat'")
    if starting_matchings and backend == "sat":
        raise ValueError("starting_matchings requires a walk")
    _check_pairs(host, retained)
    for pairs in starting_matchings:
        _check_pairs(host, pairs)
    started = time.monotonic()
    cuts: tuple[Cut, ...] = ()
    missed = len(host)
    states = 0

    def negative(status: RetractionSearchStatus) -> IncidenceOctahedronResult:
        return IncidenceOctahedronResult(
            status, None, None, None, None, states, 0, 0, cuts, missed, retained
        )

    if len(host) < 2 * min_dimension or host.universal_vertex() is not None:
        return negative(RetractionSearchStatus.EXHAUSTED)

    if backend != "sat":
        if len(host) > MAX_WALK_ORDER or len(host.base_vertices) > MAX_WALK_GROUND:
            raise ValueError("walk supports at most 1280 cliques and 64 ground atoms")
        remaining = (
            None if time_limit is None else time_limit - (time.monotonic() - started)
        )
        if remaining is not None and remaining <= 0:
            return negative(RetractionSearchStatus.LIMIT_REACHED)
        cuts, missed, states = _walk(
            host,
            executable or find_incidence_native(),
            seed=seed,
            max_steps=max_steps,
            min_dimension=min_dimension,
            time_limit=(
                remaining / 2
                if backend == "hybrid" and remaining is not None
                else remaining
            ),
            starts=starting_matchings,
        )
        if missed == 0 and len(cuts) >= min_dimension:
            r, _ = _check_cuts(host, cuts)
            walk_pairs = tuple((a, b) for a, b, _ in cuts)
            antipodes: dict[Hashable, Hashable] = {
                v: w for a, b in walk_pairs for v, w in ((a, b), (b, a))
            }
            walk_f = {v: antipodes[w] for v, w in r.items()}
            return _found(host, walk_pairs, r, walk_f, states, cuts, (), min_dimension)
        if backend == "walk":
            return negative(RetractionSearchStatus.LIMIT_REACHED)
        candidates = [(a, b) for a, b, _ in cuts]
        retained = tuple(random.Random(seed).sample(candidates, len(candidates) // 2))

    from ._octahedral_sat import search_sat

    remaining = (
        None if time_limit is None else time_limit - (time.monotonic() - started)
    )
    if remaining is not None and remaining <= 0:
        return negative(RetractionSearchStatus.LIMIT_REACHED)
    position = {v: i for i, v in enumerate(host)}
    result = search_sat(
        host,
        min_dimension=min_dimension,
        max_states=max_states,
        time_limit=remaining,
        max_clauses=sat_max_clauses,
        memory_limit_mb=sat_memory_limit_mb,
        retained=tuple((position[a], position[b]) for a, b in retained),
    )
    states = result.states_explored
    f = result.endomorphism
    if f is None:
        status = result.status
        if backend == "hybrid":
            status = RetractionSearchStatus.LIMIT_REACHED
        return negative(status)
    r = {v: f[f[v]] for v in host}
    image = set(r.values())
    sat_pairs = tuple((a, f[a]) for a in host if a in image and cast(int, f[a]) > a)
    return _found(host, sat_pairs, r, f, states, cuts, retained, min_dimension)

"""Bounded compressed DIMACS search for an octahedral-image endomorphism."""

from __future__ import annotations

import ctypes.util
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ._bitsets import iter_indices
from .coaffine_endomorphisms import CoaffineEndomorphismResult
from .implicit_clique_graph import ImplicitCliqueGraph
from .retractions import RetractionSearchStatus


class _Limit(Exception):
    """Formula construction exceeded its budget."""


@dataclass(frozen=True)
class _Tree:
    mask: int
    literal: int
    left: _Tree | None = None
    right: _Tree | None = None


def _write_cnf(
    complement: Sequence[Sequence[int]],
    path: Path,
    min_dimension: int,
    max_clauses: int,
    deadline: float,
    *,
    supports: Sequence[int] | None = None,
    retained: Sequence[tuple[int, int]] = (),
) -> list[dict[int, int]]:
    """Stream compressed CNF; return the source-to-target variable indices."""
    variables = clauses = 0

    def new() -> int:
        nonlocal variables
        variables += 1
        if variables > max_clauses or time.monotonic() >= deadline:
            raise _Limit
        return variables

    ends = {v for pair in retained for v in pair}
    allowed = set(range(len(complement)))
    if ends:
        allowed = ends | {
            i for i, row in enumerate(complement) if not ends.intersection(row)
        }
        allowed = {i for i in allowed if any(j in allowed for j in complement[i])}
    x = [{w: new() for w in row if w in allowed} for row in complement]
    y = [new() for _ in complement]
    pairs = {(i, j): new() for i in sorted(allowed) for j in x[i] if i < j}
    with path.open("w", encoding="ascii") as stream:
        stream.write(" " * 80 + "\n")

        def add(*literals: int) -> None:
            nonlocal clauses
            clauses += 1
            if clauses > max_clauses or time.monotonic() >= deadline:
                raise _Limit
            stream.write(" ".join(map(str, literals)) + " 0\n")

        def amo(values: list[int]) -> None:
            # Sequential at-most-one encoding.
            if len(values) < 2:
                return
            last = new()
            add(-values[0], last)
            for literal in values[1:-1]:
                state = new()
                add(-literal, state)
                add(-last, state)
                add(-literal, -last)
                last = state
            add(-values[-1], -last)

        for i, row in enumerate(x):
            values = list(row.values())
            add(*values)
            if supports is None:
                amo(values)
            for j, literal in row.items():
                add(-literal, y[j])
            # Every used endpoint has a selected reciprocal pair.
            if i in allowed:
                incident = [pairs[min(i, j), max(i, j)] for j in row]
                add(-y[i], *incident)
                if supports is not None:
                    amo(incident)
            else:
                add(-y[i])
        for i in sorted(ends):
            add(y[i])

        def tree(j: int, sources: Sequence[int]) -> _Tree:
            if len(sources) == 1:
                u = sources[0]
                return _Tree(1 << u, x[u][j])
            mid = len(sources) // 2
            left = tree(j, sources[:mid])
            right = tree(j, sources[mid:])
            literal = new()
            add(-left.literal, literal)
            add(-right.literal, literal)
            add(-literal, left.literal, right.literal)
            return _Tree(left.mask | right.mask, literal, left, right)

        def exclude(a: int, node: _Tree | None, allowed: int) -> None:
            if node is None:
                return
            if not node.mask & allowed:
                add(-a, -node.literal)
            elif node.mask & ~allowed:
                exclude(a, node.left, allowed)
                exclude(a, node.right, allowed)

        atoms: list[dict[int, int]] = [{} for _ in complement]
        if supports is None:
            trees = [tree(j, row) if row else None for j, row in enumerate(complement)]
            closed = [
                (1 << i) | sum(1 << j for j in row) for i, row in enumerate(complement)
            ]
        else:
            # Atom flags describe the unions of supports assigned to each
            # endpoint. One flag per (target, atom), shared by all sources.
            for u, row in enumerate(x):
                for j, literal in row.items():
                    for t in iter_indices(supports[u]):
                        if t not in atoms[j]:
                            atoms[j][t] = new()
                        add(-literal, atoms[j][t])

        for (i, j), literal in pairs.items():
            add(-literal, x[i][j])
            add(-literal, x[j][i])
            add(-literal, y[i])
            add(-literal, y[j])
            add(-y[i], -y[j], literal)
            if supports is None:
                # General edge-image constraints on inverse domain trees.
                for a in complement[i]:
                    exclude(x[a][i], trees[j], closed[a])
            else:
                # Opposite fibres may not share a ground atom. This is
                # equivalent to the morphism constraints for intersections.
                for t in atoms[i].keys() & atoms[j].keys():
                    add(-atoms[i][t], -atoms[j][t])

        # Backward cardinality counter: a true C(i,k) requires at least
        # k selected vertices among the first i. The reverse implication
        # is unnecessary because only the final threshold is asserted.
        threshold = 2 * min_dimension
        previous: dict[int, int] = {}
        for i, literal in enumerate(y, 1):
            current = {}
            for k in range(1, min(i, threshold) + 1):
                state = new()
                current[k] = state
                old = [previous[k]] if k in previous else []
                add(-state, literal, *old)
                if k > 1:
                    add(-state, previous[k - 1], *old)
            previous = current
        if threshold in previous:
            add(previous[threshold])
        else:
            add()
        stream.seek(0)
        stream.write(f"p cnf {variables} {clauses}".ljust(80) + "\n")
    return x


def search_sat(
    graph: nx.Graph | ImplicitCliqueGraph,
    *,
    min_dimension: int,
    max_states: int,
    time_limit: float | None,
    max_clauses: int,
    memory_limit_mb: int,
    retained: Sequence[tuple[int, int]] = (),
) -> CoaffineEndomorphismResult:
    """Search in an isolated Z3 worker; interrupted runs are inconclusive."""
    deadline = time.monotonic() + time_limit if time_limit is not None else float("inf")
    library = ctypes.util.find_library("z3")
    if not sys.platform.startswith("linux") or library is None:
        raise RuntimeError(
            "SAT backend requires Linux and the Z3 shared library "
            "(on Ubuntu: sudo apt install libz3-4)"
        )

    def limited() -> CoaffineEndomorphismResult:
        return CoaffineEndomorphismResult(
            RetractionSearchStatus.LIMIT_REACHED, None, 0, 0, 0
        )

    labels: list[Hashable] = list(graph)
    supports = (
        tuple(graph.clique_masks[i] for i in graph)
        if isinstance(graph, ImplicitCliqueGraph)
        else None
    )
    n = len(labels)
    # The public API has already checked the minimum graph order.
    # Bound the number of domain variables before allocating the rows.
    if supports is None and n * (n - 1) - 2 * graph.number_of_edges() > max_clauses:
        return limited()
    complement: list[list[int]] = []
    for i, v in enumerate(labels):
        if time.monotonic() >= deadline:
            return limited()
        if supports is not None:
            row = [j for j, mask in enumerate(supports) if not supports[i] & mask]
        else:
            assert isinstance(graph, nx.Graph)
            row = [j for j, w in enumerate(labels) if v != w and w not in graph[v]]
        if not row:
            return CoaffineEndomorphismResult(
                RetractionSearchStatus.EXHAUSTED, None, 0, 0, 0
            )
        complement.append(row)

    try:
        with tempfile.TemporaryDirectory(prefix="pycliques-sat-") as directory:
            path = Path(directory) / "problem.cnf"
            x = _write_cnf(
                complement,
                path,
                min_dimension,
                max_clauses,
                deadline,
                supports=supports,
                retained=retained,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return limited()
            worker = Path(__file__).with_name("_octahedral_sat_worker.py")
            command = [
                sys.executable,
                str(worker),
                library,
                str(path),
                str(max_states),
                str(memory_limit_mb),
                str(sum(map(len, x))),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=remaining if time_limit is not None else None,
                check=False,
            )
            # Z3 can terminate on allocation failure. A crashed worker
            # must never be interpreted as an UNSAT proof.
            if completed.returncode:
                return limited()
            data = json.loads(completed.stdout)
    except (subprocess.TimeoutExpired, _Limit, MemoryError, OSError):
        return limited()

    status = RetractionSearchStatus[data["status"]]
    conflicts = int(data["conflicts"])
    mapping: dict[Hashable, Hashable] | None = None
    if status is RetractionSearchStatus.FOUND:
        selected = set(data["selected"])
        mapping = {}
        for u, domain in enumerate(x):
            targets = [v for v, literal in domain.items() if literal in selected]
            if not targets or (supports is None and len(targets) != 1):
                raise RuntimeError("SAT backend returned an incomplete assignment")
            mapping[labels[u]] = labels[targets[0]]
    return CoaffineEndomorphismResult(status, mapping, conflicts, 0, 0)

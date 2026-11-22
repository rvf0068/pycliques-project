"""Optional C++17 backend for exact coaffine octahedral-retraction search.

The native executable is built only upon explicit request, never at import or
on an ordinary pycliques call. It accepts position-indexed graph data, not
Python object representations, and its positive output is always checked by
public graph verifiers before it is returned to the caller.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

MAX_NATIVE_ORDER = 512
_ENV_NAME = "PYCLIQUES_COAFFINE_NATIVE"


@dataclass(frozen=True)
class NativeCoaffineRawResult:
    status: str
    states_explored: int
    arc_revisions: int
    assignments_tried: int
    pairs: tuple[tuple[int, int], ...] | None = None
    projection: tuple[int, ...] | None = None
    endomorphism: tuple[int, ...] | None = None


def _source_file() -> Path:
    return Path(__file__).with_name("_coaffine_native.cpp")


def _cached_executable(*, cache_dir: Path | None = None) -> Path:
    source = _source_file()
    tag = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    root = (
        cache_dir
        if cache_dir is not None
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "pycliques"
    )
    return Path(root) / f"coaffine-native-{tag}"


def find_coaffine_native_executable() -> Path | None:
    """Return the explicitly configured or version-matched cached binary."""
    configured = os.environ.get(_ENV_NAME)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"{_ENV_NAME} does not identify an executable file")
        return path
    path = _cached_executable()
    if path.is_file() and os.access(path, os.X_OK):
        return path
    return None


def build_coaffine_native(
    *, compiler: str | None = None, cache_dir: Path | None = None
) -> Path:
    """Explicitly compile and cache the optional C++17 acceleration.

    Requires a local C++17 compiler; does not alter the Python package or
    download any dependencies. The executable is keyed by source hash so an
    updated package cannot accidentally reuse a stale cached binary.
    """
    source = _source_file()
    destination = _cached_executable(cache_dir=cache_dir)
    cxx = compiler or os.environ.get("CXX", "c++")
    if shutil.which(cxx) is None:
        raise RuntimeError(f"C++ compiler {cxx!r} not found")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        temporary_executable = Path(tmp) / "coaffine-native"
        command = [
            cxx,
            "-O3",
            "-std=c++17",
            str(source),
            "-o",
            str(temporary_executable),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"C++17 build failed (exit {completed.returncode}): "
                f"{completed.stderr[-2000:]}"
            )
        temporary_executable.replace(destination)
    return destination


def run_coaffine_native(
    graph,
    *,
    min_dimension: int,
    max_states: int,
    time_limit: float | None,
    executable: Path,
) -> NativeCoaffineRawResult:
    """Execute the bounded solver; do not trust or verify its projection here."""
    nodes = list(graph)
    n = len(nodes)
    if n > MAX_NATIVE_ORDER:
        raise ValueError(
            f"native coaffine backend supports at most {MAX_NATIVE_ORDER} vertices"
        )
    positions = {v: i for i, v in enumerate(nodes)}
    with tempfile.TemporaryDirectory() as tmp:
        graph_path = Path(tmp) / "graph.txt"
        with graph_path.open("w", encoding="ascii") as stream:
            stream.write(f"{n} {graph.number_of_edges()}\n")
            for u, v in graph.edges():
                stream.write(f"{positions[u]} {positions[v]}\n")
        # C++ is bounded by max_states and time_limit; a subprocess deadline
        # additionally protects against a stalled or malformed native binary.
        timeout = None if time_limit is None else time_limit + 5.0
        command = [
            str(executable),
            str(graph_path),
            str(max_states),
            str(time_limit if time_limit is not None else 1e12),
            str(min_dimension),
        ]
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return NativeCoaffineRawResult("LIMIT_REACHED", 0, 0, 0)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Native coaffine backend failed (exit {completed.returncode}): "
            f"{completed.stderr[-2000:]}"
        )

    lines = completed.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("Native backend returned no output")
    header = lines[0].split()
    if (
        len(header) != 5
        or header[0] != "RESULT"
        or header[1] not in ("FOUND", "EXHAUSTED", "LIMIT_REACHED")
    ):
        raise RuntimeError("Malformed native result header")
    try:
        states, revisions, assignments = (int(x) for x in header[2:])
    except ValueError as exc:
        raise RuntimeError("Non-integer native search counters") from exc
    if not (0 <= states <= max_states and revisions >= 0 and assignments >= 0):
        raise RuntimeError("Invalid native search counters")
    status = header[1]
    if status != "FOUND":
        if len(lines) != 1:
            raise RuntimeError("Unexpected witness in non-FOUND native output")
        return NativeCoaffineRawResult(status, states, revisions, assignments)
    if len(lines) != 4:
        raise RuntimeError("Missing native retraction witness fields")
    pair_line = lines[1].split()
    projection_line = lines[2].split()
    endomorphism_line = lines[3].split()
    if (
        not pair_line
        or pair_line[0] != "PAIRS"
        or not projection_line
        or projection_line[0] != "PROJECTION"
        or not endomorphism_line
        or endomorphism_line[0] != "ENDOMORPHISM"
    ):
        raise RuntimeError("Incorrect native witness field names")
    try:
        pairs = cast(
            tuple[tuple[int, int], ...],
            tuple(tuple(map(int, pair.split(","))) for pair in pair_line[1:]),
        )
        projection = tuple(map(int, projection_line[1:]))
        endomorphism = tuple(map(int, endomorphism_line[1:]))
    except ValueError as exc:
        raise RuntimeError("Malformed native witness indices") from exc
    if (
        len(pairs) < min_dimension
        or any(len(pair) != 2 for pair in pairs)
        or len(projection) != n
        or len(endomorphism) != n
        or any(i < 0 or i >= n for pair in pairs for i in pair)
        or any(i < 0 or i >= n for i in (*projection, *endomorphism))
    ):
        raise RuntimeError("Native witness has invalid dimensions or labels")
    return NativeCoaffineRawResult(
        status, states, revisions, assignments, pairs, projection, endomorphism
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build optional C++17 coaffine backend"
    )
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("pass --build to compile the optional backend")
    print(build_coaffine_native())

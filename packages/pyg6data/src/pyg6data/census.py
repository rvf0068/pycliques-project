"""Checkpointable exhaustive graph-census utilities."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import networkx as nx

from .lists import graph_generator

CensusRecord = dict[str, Any]


def _graph6(graph: nx.Graph) -> str:
    """Return a headerless ASCII graph6 encoding."""
    encoded = nx.to_graph6_bytes(graph, header=False).decode("ascii").strip()
    return cast(str, encoded)


def _evaluate_graph6(
    encoded: str,
    predicate: Callable[[nx.Graph], bool],
    evaluator: Callable[[nx.Graph], Any],
) -> tuple[str, Any]:
    """Evaluate one serialized graph in a worker process."""
    graph = nx.from_graph6_bytes(encoded.encode("ascii"))
    if not predicate(graph):
        return "filtered", None
    return "ok", evaluator(graph)


def _write_checkpoint(
    path: Path, completed: set[int], records: list[CensusRecord]
) -> None:
    """Atomically write the current census state as JSON."""
    payload = {"completed": sorted(completed), "records": records}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as checkpoint:
            json.dump(payload, checkpoint, sort_keys=True)
            checkpoint.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_checkpoint(path: Path) -> tuple[set[int], list[CensusRecord]]:
    """Read a checkpoint, rejecting malformed or incomplete JSON."""
    with path.open("r", encoding="utf-8") as checkpoint:
        payload = json.load(checkpoint)
    completed = {int(index) for index in payload["completed"]}
    records = list(payload["records"])
    return completed, records


def run_graph_census(
    order: int,
    predicate: Callable[[nx.Graph], bool],
    evaluator: Callable[[nx.Graph], Any],
    *,
    checkpoint: Path | None = None,
    checkpoint_interval: int = 100,
    workers: int = 1,
) -> list[CensusRecord]:
    """Run a resumable census over connected graphs of a fixed order.

    Graphs are streamed from :func:`pyg6data.lists.graph_generator`. Each
    record contains the source index, graph6 encoding, status, and either the
    evaluator result or exception text. A checkpoint is replaced atomically at
    the requested interval, so an interrupted run can resume without losing
    completed records.

    .. rubric:: Parameters

    order : int
        Order of the connected graphs to enumerate.
    predicate : callable
        Filter applied before evaluation.
    evaluator : callable
        Function evaluated for graphs accepted by *predicate*.
    checkpoint : pathlib.Path, optional
        JSON checkpoint file used for resume. No checkpoint is written when
        omitted.
    checkpoint_interval : int, optional
        Number of processed graph indices between checkpoint writes.
    workers : int, optional
        Number of worker processes. ``1`` runs in the calling process. When
        greater than one, *predicate* and *evaluator* must be picklable.

    .. rubric:: Returns

    list[dict]
        Result records sorted by graph index.

    .. rubric:: Examples

    >>> from pyg6data.census import run_graph_census
    >>> records = run_graph_census(6, lambda g: g.number_of_nodes() == 6,
    ...                            lambda g: g.number_of_edges())
    >>> len(records)
    112
    """
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")

    completed: set[int] = set()
    records: list[CensusRecord] = []
    if checkpoint is not None and checkpoint.exists():
        completed, records = _read_checkpoint(checkpoint)

    pending: list[tuple[int, str]] = [
        (index, _graph6(graph))
        for index, graph in enumerate(graph_generator(order, connected=True))
        if index not in completed
    ]

    def consume(index: int, encoded: str, status: str, value: Any) -> None:
        record: CensusRecord = {
            "index": index,
            "graph6": encoded,
            "status": status,
        }
        if status == "ok":
            record["result"] = repr(value)
        elif status == "filtered":
            record["result"] = None
        else:
            record["exception"] = value
        records.append(record)
        completed.add(index)

    def process_one(index: int, encoded: str) -> None:
        graph = nx.from_graph6_bytes(encoded.encode("ascii"))
        try:
            if not predicate(graph):
                consume(index, encoded, "filtered", None)
                return
            consume(index, encoded, "ok", evaluator(graph))
        except Exception as error:  # census runs must survive one bad graph
            consume(index, encoded, "error", f"{type(error).__name__}: {error}")

    if workers == 1:
        for processed, (index, encoded) in enumerate(pending, 1):
            process_one(index, encoded)
            if checkpoint is not None and processed % checkpoint_interval == 0:
                _write_checkpoint(checkpoint, completed, records)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_evaluate_graph6, encoded, predicate, evaluator): (
                    index,
                    encoded,
                )
                for index, encoded in pending
            }
            for processed, future in enumerate(futures, 1):
                index, encoded = futures[future]
                try:
                    status, value = future.result()
                    consume(index, encoded, status, value)
                except Exception as error:
                    consume(index, encoded, "error", f"{type(error).__name__}: {error}")
                if checkpoint is not None and processed % checkpoint_interval == 0:
                    _write_checkpoint(checkpoint, completed, records)

    records.sort(key=lambda record: record["index"])
    if checkpoint is not None:
        _write_checkpoint(checkpoint, completed, records)
    return records

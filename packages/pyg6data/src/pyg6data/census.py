"""Bounded, resumable censuses over the bundled connected graphs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, closing
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import networkx as nx

from .lists import _get_data_file_path, graph6_resource, iter_graph6_records

CensusRecord = dict[str, Any]
Predicate = Callable[[nx.Graph], bool]
Evaluator = Callable[[nx.Graph], Any]


def _evaluate(
    index: int, encoded: str, predicate: Predicate, evaluator: Evaluator
) -> CensusRecord:
    row: CensusRecord = {"index": index, "graph6": encoded}
    graph = nx.from_graph6_bytes(encoded.encode("ascii"))
    try:
        if predicate(graph):
            row.update(status="ok", result=repr(evaluator(graph)))
        else:
            row.update(status="filtered", result=None)
    except Exception as error:
        row.update(status="error", exception=f"{type(error).__name__}: {error}")
    return row


def _validate(row: CensusRecord, index: int, encoded: str) -> None:
    if (
        not isinstance(row, dict)
        or type(row.get("index")) is not int
        or (row["index"] != index or row.get("graph6") != encoded)
    ):
        raise ValueError(f"Checkpoint record disagrees with source at index {index}")
    status = row.get("status")
    if not (
        (status == "ok" and isinstance(row.get("result"), str))
        or (status == "filtered" and "result" in row and row["result"] is None)
        or (status == "error" and isinstance(row.get("exception"), str))
    ):
        raise ValueError(f"Invalid checkpoint result at index {index}")


def _open_checkpoint(path: Path, order: int, run_id: str | None) -> sqlite3.Connection:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(
            "Set run_id to identify the checkpointed predicate and evaluator"
        )
    resource = graph6_resource(order)
    with _get_data_file_path(resource).open("rb") as stream:
        source_hash = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            source_hash.update(block)
        digest = source_hash.hexdigest()
    identity = json.dumps([1, "connected", order, resource, digest, run_id])
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        with db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS metadata "
                "(id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT NOT NULL)"
            )
            db.execute("INSERT OR IGNORE INTO metadata VALUES (1, ?)", (identity,))
            if db.execute("SELECT identity FROM metadata WHERE id=1").fetchone() != (
                identity,
            ):
                raise ValueError(
                    "Checkpoint belongs to a different source, order, schema or run_id"
                )
            db.execute(
                "CREATE TABLE IF NOT EXISTS records "
                "(idx INTEGER PRIMARY KEY CHECK(idx>=0), record TEXT NOT NULL)"
            )
    except sqlite3.DatabaseError as error:
        db.close()
        raise ValueError(
            "Unreadable checkpoint; use import_legacy_checkpoint for historical JSON"
        ) from error
    except BaseException:
        db.close()
        raise
    return db


def import_legacy_checkpoint(
    source: Path, destination: Path, *, order: int, run_id: str
) -> None:
    """Import historical JSON to a new database after validating all rows.

    Supply the original evaluator's identifier. The old file cannot establish
    that fact. The original and any existing destination are preserved.
    """
    payload = json.loads(source.read_text(encoding="utf-8"))
    indices, rows = payload["completed"], payload["records"]
    by_index = {row["index"]: row for row in rows}
    if (
        any(type(i) is not int or i < 0 for i in indices)
        or len(indices) != len(set(indices))
        or len(by_index) != len(rows)
        or set(indices) != by_index.keys()
    ):
        raise ValueError("Legacy completed indices and records disagree")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        path = Path(directory) / "checkpoint"
        with closing(_open_checkpoint(path, order, run_id)) as db, db:
            for index, encoded in iter_graph6_records(order):
                if not by_index:
                    break
                if index in by_index:
                    row = by_index.pop(index)
                    _validate(row, index, encoded)
                    db.execute(
                        "INSERT INTO records VALUES (?, ?)", (index, json.dumps(row))
                    )
            if by_index:
                raise ValueError("Legacy checkpoint contains out-of-range indices")
        os.link(path, destination)


def iter_graph_census(
    order: int,
    predicate: Predicate,
    evaluator: Evaluator,
    *,
    checkpoint: Path | None = None,
    run_id: str | None = None,
    checkpoint_interval: int = 100,
    workers: int = 1,
) -> Iterator[CensusRecord]:
    """Yield results in original index order with at most twice workers in flight.

    A SQLite checkpoint binds the source digest and order to the caller's
    run_id. Change the identifier when the evaluator, filter or parameters
    change. Committed intervals survive interruption. Historical JSON can be
    imported with import_legacy_checkpoint. Parallel callables must be picklable.
    """
    if checkpoint_interval < 1 or workers < 1:
        raise ValueError("checkpoint_interval and workers must be positive")
    source = iter_graph6_records(order)
    with ExitStack() as stack:
        db = (
            stack.enter_context(closing(_open_checkpoint(checkpoint, order, run_id)))
            if checkpoint is not None
            else None
        )
        pool = (
            stack.enter_context(ProcessPoolExecutor(workers)) if workers > 1 else None
        )

        def results():
            for index, encoded in source:
                cached = (
                    db.execute(
                        "SELECT record FROM records WHERE idx=?", (index,)
                    ).fetchone()
                    if db is not None
                    else None
                )
                if cached is not None:
                    row = json.loads(cached[0])
                    _validate(row, index, encoded)
                    yield row, None, False
                elif pool is None:
                    yield _evaluate(index, encoded, predicate, evaluator), None, True
                else:
                    yield (
                        None,
                        pool.submit(_evaluate, index, encoded, predicate, evaluator),
                        True,
                    )

        pending = results()
        queue = deque(islice(pending, 2 * workers if pool else 1))
        dirty = 0
        count = 0
        while queue:
            row, future, fresh = queue.popleft()
            if future is not None:
                row = future.result()
            assert row is not None
            if fresh and db is not None:
                saved = db.execute(
                    "INSERT OR IGNORE INTO records VALUES (?, ?)",
                    (count, json.dumps(row)),
                )
                if saved.rowcount == 0:
                    # Another worker process using this checkpoint committed first.
                    original = db.execute(
                        "SELECT record FROM records WHERE idx=?", (count,)
                    ).fetchone()
                    cached_row = json.loads(original[0])
                    _validate(cached_row, count, row["graph6"])
                    row = cached_row
                dirty += 1
                if dirty == checkpoint_interval:
                    db.commit()
                    dirty = 0
            count += 1
            yield row
            queue.extend(islice(pending, 1))
        if db is not None:
            if db.execute(
                "SELECT 1 FROM records WHERE idx>=? LIMIT 1", (count,)
            ).fetchone():
                raise ValueError("Checkpoint contains out-of-range indices")
            db.commit()


def run_graph_census(
    order: int,
    predicate: Predicate,
    evaluator: Evaluator,
    *,
    checkpoint: Path | None = None,
    run_id: str | None = None,
    checkpoint_interval: int = 100,
    workers: int = 1,
) -> list[CensusRecord]:
    """Collect iter_graph_census in source order. Use the iterator for large runs.

    >>> rows = run_graph_census(6, lambda g: True, lambda g: len(g))
    >>> len(rows)
    112
    """
    return list(
        iter_graph_census(
            order,
            predicate,
            evaluator,
            checkpoint=checkpoint,
            run_id=run_id,
            checkpoint_interval=checkpoint_interval,
            workers=workers,
        )
    )

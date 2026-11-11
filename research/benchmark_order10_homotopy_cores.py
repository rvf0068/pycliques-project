"""Restartable, process-timeboxed benchmarks on the 60 order-10 Phase-1 cores.

Use the *derived* representatives.tsv from pycliques-homotopy-census.
This script does not modify any historical census files or claim that
homology-only outcomes determine homotopy type.

Example (from a pycliques-project checkout with its dependencies installed):

    uv run python research/benchmark_order10_homotopy_cores.py \\
      --representatives ../pycliques-homotopy-census/followups/order10_phase1/representatives.tsv \\
      --output ../pycliques-homotopy-census/followups/order10_phase2_benchmark.jsonl \\
      --limit 5

Omit --limit to finish the remaining graphs; completed records are resumed
after validating metadata and input identity. Each graph is computed in a
separate process with a hard wall-clock timeout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# Immutable Phase-1 input: exact git-blob identity from the committed
# pycliques-homotopy-census/followups/order10_phase1/representatives.tsv.
# Indices are source *order-10 census* indices, NOT consecutive 0..59 IDs.
PHASE1_REPRESENTATIVES_GIT_SHA1 = "0da9d8cf5ebd01bf2e7144faafcb6aac919f22e2"
EXPECTED_REPRESENTATIVE_INDICES = frozenset({
    1561859, 1565330, 1565364, 1565393, 1565422, 1565424,
    7923665, 7950656, 7977194, 7988753, 7991311, 7994188,
    7994391, 7994482, 7994484, 7995143, 7995295, 7995314,
    10043465, 10223998, 10362063, 10362424, 10419379, 10566451,
    10578910, 11125947, 11488990, 11519684, 11522274, 11522704,
    11523965, 11524199, 11525976, 11535601, 11537931, 11537978,
    11619731, 11620761, 11645069, 11645294, 11650233, 11655953,
    11656489, 11656499, 11656524, 11657377, 11660070, 11660425,
    11660902, 11661308, 11661343, 11661348, 11661349, 11661360,
    11661524, 11670395, 11708967, 11715924, 11716138, 11716425,
})


def git_blob_sha1(data: bytes) -> str:
    """Git's content identity, allowing comparison with the Phase-1 blob."""
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def load_representatives(path: Path) -> list[dict[str, str]]:
    if git_blob_sha1(path.read_bytes()) != PHASE1_REPRESENTATIVES_GIT_SHA1:
        raise ValueError("Representatives file differs from committed Phase-1 input")
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"representative_index", "graph6", "core_graph6", "members"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Not a Phase-1 representatives.tsv: missing columns")
        rows = list(reader)
    indices = [int(row["representative_index"]) for row in rows]
    if len(rows) != 60 or set(indices) != EXPECTED_REPRESENTATIVE_INDICES:
        raise ValueError("Expected the exact 60 Phase-1 source census indices")
    if any(not row["core_graph6"] for row in rows):
        raise ValueError("Missing core graph6")
    return rows


def tool_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def classify_child(g6: str, max_vertices: int, max_simplices: int, seed: int) -> dict:
    # Imports intentionally occur in the child process. If an algorithm hangs
    # or exhausts its own budget, the parent can safely timebox it.
    import networkx as nx
    from pycombtop import homotopy_type_with_verdict

    graph = nx.from_graph6_bytes(g6.encode("ascii"))
    verdict = homotopy_type_with_verdict(
        graph, max_vertices=max_vertices, max_simplices=max_simplices,
        dong_seed=seed,
    )
    status = (
        "EXACT" if verdict.is_exact else
        "LIMITED" if verdict.is_limited else "BETTI_ONLY"
    )
    return {
        "status": status,
        "core_order": graph.number_of_nodes(),
        "core_edges": graph.number_of_edges(),
        "is_exact": verdict.is_exact,
        "is_limited": verdict.is_limited,
        # A Betti-only vector is NOT a certified wedge decomposition.
        "certified_spheres": (
            {str(k): v for k, v in sorted(verdict.wedge.spheres.items())}
            if verdict.is_exact else None
        ),
        "reason": verdict.reason,
    }


def write_jsonl_line(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def read_completed_records(
    output: Path, rows: list[dict[str, str]]
) -> set[int]:
    """Replay the JSONL journal, rejecting mismatched inputs and malformed verdicts.

    Checking the independent input blob is not enough: a stale or edited JSONL
    record must never be silently skipped on resume.
    """
    expected_rows = {int(row["representative_index"]): row for row in rows}
    completed: set[int] = set()
    for lineno, line in enumerate(output.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("JSONL entry is not an object")
            index = record["representative_index"]
            if type(index) is not int or index not in expected_rows or index in completed:
                raise ValueError("Unexpected or repeated representative index")
            original = expected_rows[index]
            if (
                record.get("source_graph6") != original["graph6"]
                or record.get("core_graph6") != original["core_graph6"]
                or record.get("members") != original["members"].split(";")
            ):
                raise ValueError("Record does not match its Phase-1 representative")

            elapsed = record.get("elapsed_seconds")
            if (
                type(elapsed) not in (float, int) or not 0 <= elapsed < float("inf")
            ):
                raise ValueError("Invalid elapsed time")
            status = record.get("status")
            if status in {"EXACT", "BETTI_ONLY", "LIMITED"}:
                if not (
                    type(record.get("core_order")) is int
                    and type(record.get("core_edges")) is int
                    and record["core_order"] == int(original["core_order"])
                    and record["core_edges"] == int(original["core_edges"])
                    and isinstance(record.get("reason"), str)
                    and bool(record["reason"])
                    and type(record.get("is_exact")) is bool
                    and type(record.get("is_limited")) is bool
                ):
                    raise ValueError("Malformed classifier verdict")
                if status == "EXACT":
                    spheres = record.get("certified_spheres")
                    if (
                        not record["is_exact"] or record["is_limited"]
                        or not isinstance(spheres, dict)
                        or any(
                            not isinstance(k, str) or not k.isdecimal()
                            or type(v) is not int or v < 1
                            for k, v in spheres.items()
                        )
                    ):
                        raise ValueError("Malformed exact certificate summary")
                elif (
                    record["is_exact"]
                    or record["is_limited"] != (status == "LIMITED")
                    or record.get("certified_spheres") is not None
                ):
                    raise ValueError("Inexact status presented as an exact type")
            elif status == "TIMEOUT":
                if set(record) != {
                    "status", "representative_index", "source_graph6",
                    "core_graph6", "members", "elapsed_seconds",
                }:
                    raise ValueError("Malformed timeout record")
            elif status == "ERROR":
                if not isinstance(record.get("diagnostic"), str) or not record["diagnostic"]:
                    raise ValueError("Missing error diagnostic")
                if set(record) != {
                    "status", "diagnostic", "representative_index", "source_graph6",
                    "core_graph6", "members", "elapsed_seconds",
                }:
                    raise ValueError("Malformed error record")
            else:
                raise ValueError("Unknown benchmark status")
            completed.add(index)
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Invalid benchmark record at line {lineno}: {exc}") from exc
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-vertices", type=int, default=64)
    parser.add_argument("--max-simplices", type=int, default=25000)
    parser.add_argument("--dong-seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many *new* representatives")
    parser.add_argument("--_child_g6", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.max_vertices < 1 or args.max_simplices < 0 or args.timeout_seconds <= 0:
        parser.error("Invalid vertex, simplex, or time limit")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    if args._child_g6 is not None:
        print(json.dumps(
            classify_child(args._child_g6, args.max_vertices,
                           args.max_simplices, args.dong_seed),
            sort_keys=True,
        ))
        return

    if args.representatives is None or args.output is None:
        parser.error("--representatives and --output are required")
    rows = load_representatives(args.representatives)
    identity = {
        "sha256_representatives": hashlib.sha256(
            args.representatives.read_bytes()
        ).hexdigest(),
        "git_commit": tool_revision(),
        "max_vertices": args.max_vertices,
        "max_simplices": args.max_simplices,
        "dong_seed": args.dong_seed,
        "timeout_seconds": args.timeout_seconds,
        "python": sys.version.split()[0],
    }
    output = args.output
    metadata = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or metadata.exists():
        if not output.exists() or not metadata.exists():
            raise RuntimeError("Incomplete benchmark provenance; do not append")
        if json.loads(metadata.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Benchmark settings/input/revision differ; use a new output")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
        output.touch()

    completed = read_completed_records(output, rows)

    processed = 0
    for row in rows:
        index = int(row["representative_index"])
        if index in completed:
            continue
        if args.limit is not None and processed >= args.limit:
            break
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--_child_g6", row["core_graph6"],
            "--max-vertices", str(args.max_vertices),
            "--max-simplices", str(args.max_simplices),
            "--dong-seed", str(args.dong_seed),
        ]
        started = time.monotonic()
        try:
            run = subprocess.run(
                command, text=True, capture_output=True,
                timeout=args.timeout_seconds, check=False,
            )
            if run.returncode == 0:
                result = json.loads(run.stdout)
            else:
                result = {"status": "ERROR", "diagnostic": run.stderr[-1200:]}
        except subprocess.TimeoutExpired:
            result = {"status": "TIMEOUT"}
        except (OSError, json.JSONDecodeError) as error:
            result = {"status": "ERROR", "diagnostic": str(error)}
        result.update({
            "representative_index": index,
            "source_graph6": row["graph6"],
            "core_graph6": row["core_graph6"],
            "members": row["members"].split(";"),
            "elapsed_seconds": round(time.monotonic() - started, 4),
        })
        write_jsonl_line(output, result)
        processed += 1
        completed.add(index)
        print(f"{len(completed)}/60  index={index}  {result['status']}  "
              f"{result['elapsed_seconds']:.1f}s", flush=True)

    print(f"Finished this invocation: {processed}; total recorded: {len(completed)}/60")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()

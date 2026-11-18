"""Read-only audit of saved homotopy-census certificates affected by issue #16.

The historical TSV/GZ records remain untouched. Stream every record and
identify exact sphere-wedge statements that depend on the historical
vertex-decomposability + numerical Betti path. Separately flag BETTI_ONLY
and LARGE_GRAPH_BETTI records, whose claimed homology also needs review.

To independently recheck a deterministic sample, set --recheck N. The
output explicitly states how many cases have NOT been recomputed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import random

VERTEX_DECOMPOSABLE = "shedding-vertex recursion certifies nonpure vertex decomposability"
BETTI_ONLY = "BETTI_ONLY"


def _chunks(root: Path, order: int) -> list[Path]:
    stage = "through6" if order <= 6 else f"order{order}"
    folder = root / "runs" / stage / f"order{order}" / "chunks"
    files = sorted(folder.glob("*.tsv.gz"))
    if not files:
        raise FileNotFoundError(f"No saved census chunks for order {order} in {folder}")
    return files


def _category(row: dict[str, str], suffix: str) -> str | None:
    status = row[f"status_{suffix}"]
    reason = row[f"reason_{suffix}"].lower()
    if status == "EXACT" and VERTEX_DECOMPOSABLE in reason:
        if suffix == "kg" and "transferred exact type of g" in reason:
            return "EXACT_TRANSFERRED_VERTEX_DECOMPOSABLE"
        return "EXACT_VERTEX_DECOMPOSABLE"
    if status == BETTI_ONLY:
        return "BETTI_ONLY"
    if status == "EXACT":
        return None
    return None


def _saved_profile(value: str) -> dict[int, int]:
    if value == "0":
        return {}
    return {int(t.split(":")[0]): int(t.split(":")[1]) for t in value.split(";")}


def _recheck(row: dict[str, str], suffix: str, faces: int) -> dict:
    """Compare historical type with TWO independent exact GF(2) backends."""
    import networkx as nx
    from pycliques import clique_graph
    from pycombtop.bounded_betti import bounded_reduced_betti_mod2
    from pycombtop.homology_backend import HomologyResourceLimit
    from pycombtop.homotopy_type import betti_numbers_graph

    original = nx.from_graph6_bytes(row["graph6"].encode("ascii"))
    transferred = (
        suffix == "kg"
        and "transferred exact type of g" in row["reason_kg"].lower()
    )
    if suffix == "kg" and not transferred:
        graph = clique_graph(original, bound=128, max_edges=5000)
        if graph is None:
            return {"outcome": "INCOMPLETE_K_CONSTRUCTION"}
        if row["k_order"] != "UNKNOWN" and len(graph) != int(row["k_order"]):
            return {"outcome": "INVALID_SAVED_K_ORDER"}
    else:
        graph = original
    saved = _saved_profile(row[f"betti_{suffix}"])
    binary = bounded_reduced_betti_mod2(graph, max_faces=faces)
    if binary is None:
        return {"outcome": "INCOMPLETE_BINARY_BUDGET"}
    try:
        independent = betti_numbers_graph(graph)
    except HomologyResourceLimit:
        return {"outcome": "INCOMPLETE_GUDHI_BUDGET"}
    exact = {i: v for i, v in binary.items() if v != 0}
    gudhi = {i: v for i, v in enumerate(independent) if v != 0}
    if exact != gudhi:
        return {"outcome": "BACKENDS_DISAGREE", "binary": exact, "gudhi": gudhi}
    if saved != exact:
        return {"outcome": "SAVED_MISMATCH", "saved": saved, "exact_mod2": exact}
    return {"outcome": "MATCH"}


def audit(root: Path, orders: list[int], *, recheck: int = 0, faces: int = 250_000) -> dict:
    if recheck < 0:
        raise ValueError("--recheck must be nonnegative")
    totals = Counter()
    samples: dict[str, list[tuple[dict[str, str], str]]] = defaultdict(list)
    seen = Counter()
    rngs: dict[str, random.Random] = {}
    first = defaultdict(list)
    for order in orders:
        for path in _chunks(root, order):
            with gzip.open(path, "rt", encoding="utf-8", newline="") as input_file:
                reader = csv.DictReader(input_file, delimiter="\t")
                for row in reader:
                    totals[f"order{order}:records"] += 1
                    for suffix in ("g", "kg"):
                        category = _category(row, suffix)
                        if category is None:
                            continue
                        key = f"order{order}:{suffix}:{category}"
                        totals[key] += 1
                        if len(first[key]) < 5:
                            first[key].append({
                                "index": int(row["index"]), "graph6": row["graph6"],
                                "betti": row[f"betti_{suffix}"],
                            })
                        if not recheck:
                            continue
                        seen[key] += 1
                        if key not in rngs:
                            rngs[key] = random.Random(1600 + order * 17 + (suffix == "kg"))
                        bucket = samples[key]
                        entry = (row, suffix)
                        if len(bucket) < recheck:
                            bucket.append(entry)
                        else:
                            j = rngs[key].randrange(seen[key])
                            if j < recheck:
                                bucket[j] = entry
    checks = Counter()
    problems = []
    for key, cases in sorted(samples.items()):
        for row, suffix in cases:
            try:
                outcome = _recheck(row, suffix, faces)
            except Exception as exc:
                outcome = {"outcome": "RECHECK_ERROR", "error": repr(exc)}
            checks[f"{key}:{outcome['outcome']}"] += 1
            if outcome["outcome"] != "MATCH":
                problems.append({
                    "order": row["order"], "index": int(row["index"]),
                    "graph6": row["graph6"], "side": suffix, **outcome,
                })
    output = {
        "schema": 1,
        "scope": "Read-only historical saved verdict audit; no reclassification",
        "orders": orders,
        "counts": dict(sorted(totals.items())),
        "examples": dict(sorted(first.items())),
        "sample_checks": dict(sorted(checks.items())),
        "sample_size_cap_per_category": recheck,
        "unverified_count_lower_bound": sum(
            max(0, seen[k] - len(v)) for k, v in samples.items()
        ),
        "recheck_problems": problems[:200],
        "remaining_problems_truncated": max(0, len(problems) - 200),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    parser.add_argument("--recheck", type=int, default=0)
    parser.add_argument("--max-faces", type=int, default=250_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.orders, recheck=args.recheck, faces=args.max_faces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "counts": result["counts"],
        "sample_checks": result["sample_checks"],
        "recheck_problem_count": (
            len(result["recheck_problems"]) + result["remaining_problems_truncated"]
        ),
        "unverified_count_lower_bound": result["unverified_count_lower_bound"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

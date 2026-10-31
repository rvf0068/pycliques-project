"""Compute clique-complex homotopy types from indeterminate graph files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TextIO

import networkx as nx

from .homotopy_type import homotopy_type_with_verdict


def _read_graph_rows(input_path: Path):
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 3:
                raise ValueError(
                    f"{input_path}:{line_number}: expected index, order, and graph6"
                )
            try:
                index, order = int(fields[0]), int(fields[1])
                graph = nx.from_graph6_bytes(fields[2].encode("ascii"))
            except (UnicodeEncodeError, ValueError) as error:
                raise ValueError(
                    f"{input_path}:{line_number}: invalid index, order, or graph6"
                ) from error
            if graph.number_of_nodes() != order:
                raise ValueError(
                    f"{input_path}:{line_number}: declared order {order} does not "
                    f"match graph6 order {graph.number_of_nodes()}"
                )
            yield index, order, fields[2], graph


def _write_result_header(output: TextIO) -> None:
    output.write("# Clique-complex homotopy types\n")
    csv.writer(output, delimiter="\t", lineterminator="\n").writerow(
        ["original_index", "order", "graph6", "homotopy_type", "status", "reason"]
    )


def analyze_indeterminate_file(
    input_path: Path,
    output_path: Path,
    *,
    max_vertices: int | None = None,
    max_simplices: int | None = None,
) -> int:
    """Write homotopy verdicts for every graph in an indeterminate-order file."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")

    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as output:
        _write_result_header(output)
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for index, order, graph6, graph in _read_graph_rows(input_path):
            verdict = homotopy_type_with_verdict(
                graph,
                max_vertices=max_vertices,
                max_simplices=max_simplices,
            )
            status = (
                "limited"
                if verdict.is_limited
                else "exact"
                if verdict.is_exact
                else "betti-only"
            )
            writer.writerow(
                [index, order, graph6, verdict.verdict, status, verdict.reason]
            )
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute clique-complex homotopy types from an "
        "indeterminate_order_n.txt file."
    )
    parser.add_argument("input", type=Path, help="input indeterminate_order_n.txt")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output TSV path (default: <input-stem>_homotopy.txt)",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        help="skip homotopy analysis for graphs above this vertex count",
    )
    parser.add_argument(
        "--max-simplices",
        type=int,
        help="skip homotopy analysis when the clique complex exceeds this size",
    )
    args = parser.parse_args()
    output_path = args.output or args.input.with_name(f"{args.input.stem}_homotopy.txt")
    try:
        count = analyze_indeterminate_file(
            args.input,
            output_path,
            max_vertices=args.max_vertices,
            max_simplices=args.max_simplices,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote homotopy types for {count} graphs to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

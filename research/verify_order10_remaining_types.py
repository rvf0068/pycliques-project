"""Replay the twenty residual order-10 types from the original graph6 strings.

Run with ``uv run python research/verify_order10_remaining_types.py``. Each
input is a singleton representative in the Phase-1 census table. Construct
all maximal completes directly from the ten-vertex graph. The verdict is
for the actual clique graph, without relying on a saved pared core.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import networkx as nx
from pycombtop import Theorem, homotopy_type_with_verdict

RESULTS = Path(__file__).parent / "results/order10_remaining_types.jsonl"


def reconstructed_clique_graph(graph6: str) -> nx.Graph:
    graph = nx.from_graph6_bytes(graph6.encode("ascii"))
    if len(graph) != 10 or not nx.is_connected(graph):
        raise ValueError(f"Expected a connected graph of order 10 for {graph6!r}")
    cliques = [set(q) for q in sorted(tuple(sorted(q)) for q in nx.find_cliques(graph))]
    result = nx.empty_graph(len(cliques))
    result.add_edges_from(
        (i, j) for i, j in combinations(result, 2) if cliques[i] & cliques[j]
    )
    return result


def load_rows() -> list[dict]:
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    if len(rows) != 20 or len({row["index"] for row in rows}) != 20:
        raise ValueError("Expected 20 distinct order-10 census indices")
    return rows


def replay_row(row: dict) -> str:
    index = row["index"]
    graph = reconstructed_clique_graph(row["graph6"])
    if len(graph) != row["k_order"]:
        raise ValueError(f"Unexpected clique-graph order at index {index}")
    result = homotopy_type_with_verdict(graph, max_vertices=64, max_simplices=25000)
    expected = {int(d): n for d, n in row["spheres"].items()}
    if not result.is_exact or result.is_limited or result.wedge.spheres != expected:
        raise ValueError(f"Exact wedge verdict differs at index {index}: {result}")
    methods = {
        "vertex_star": Theorem.RECURSIVE_VERTEX_STAR,
        "morse": Theorem.INTEGRAL_MORSE,
        "morse_shore": Theorem.BIPARTITE_SHORE + Theorem.INTEGRAL_MORSE,
    }
    if result.reason != methods[row["method"]]:
        raise ValueError(
            f"Certificate method differs at index {index}: {result.reason}"
        )
    return result.verdict


def main() -> None:
    rows = load_rows()
    for row in rows:
        print(row["index"], row["graph6"], replay_row(row))
    print(f"{len(rows)}/{len(rows)} exact clique-graph types verified")


if __name__ == "__main__":
    main()

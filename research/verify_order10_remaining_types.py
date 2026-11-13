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
    assert len(graph) == 10 and nx.is_connected(graph)
    cliques = [set(q) for q in sorted(tuple(sorted(q)) for q in nx.find_cliques(graph))]
    result = nx.empty_graph(len(cliques))
    result.add_edges_from(
        (i, j) for i, j in combinations(result, 2) if cliques[i] & cliques[j]
    )
    return result


def main() -> None:
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    assert len(rows) == len({row["index"] for row in rows}) == 20
    methods = {
        "vertex_star": Theorem.RECURSIVE_VERTEX_STAR,
        "morse": Theorem.INTEGRAL_MORSE,
        "morse_shore": Theorem.BIPARTITE_SHORE + Theorem.INTEGRAL_MORSE,
    }
    for row in rows:
        graph = reconstructed_clique_graph(row["graph6"])
        assert len(graph) == row["k_order"]
        result = homotopy_type_with_verdict(graph, max_vertices=64, max_simplices=25000)
        expected = {int(d): n for d, n in row["spheres"].items()}
        assert result.is_exact and not result.is_limited, row["index"]
        assert (
            result.wedge.spheres == expected and result.reason == methods[row["method"]]
        ), row["index"]
        print(row["index"], row["graph6"], result.verdict)
    print(f"{len(rows)}/{len(rows)} exact clique-graph types verified")


if __name__ == "__main__":
    main()

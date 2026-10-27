"""Bounded clique-graph homotopy census over the bundled graph6 datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx
from pycliques import clique_graph, find_dominated_vertex, is_clique_helly
from pycombtop import (
    CollapseStatus,
    Simplex,
    clique_complex,
    nerve_of_cliques,
    relative_betti_numbers,
    relative_collapse,
)
from pyg6data import graph_generator


def _graph6(graph: nx.Graph) -> str:
    encoded: bytes = nx.to_graph6_bytes(
        graph, nodes=list(graph.nodes()), header=False
    )
    return encoded.decode("ascii").strip()


def matches_research_filter(graph: nx.Graph, kg: nx.Graph) -> bool:
    """Apply the reported non-Helly/no-dominated/K(G)-Helly filter."""
    return (
        not is_clique_helly(graph)
        and find_dominated_vertex(graph) is None
        and is_clique_helly(kg)
    )


def analyze_graph(
    graph: nx.Graph,
    *,
    max_states: int = 1_000,
    max_simplices: int = 2_500,
) -> dict[str, Any]:
    """Analyze one graph, using explicit budgets for costly computations."""
    if max_states < 1:
        raise ValueError("max_states must be positive")
    if max_simplices < 0:
        raise ValueError("max_simplices must be nonnegative")
    kg = clique_graph(graph)
    if kg is None:
        raise RuntimeError("unbounded clique-graph construction unexpectedly failed")

    source = clique_complex(kg)
    target = nerve_of_cliques(graph)
    clique_helly_g = is_clique_helly(graph)
    clique_helly_kg = is_clique_helly(kg)
    dominated_g = find_dominated_vertex(graph) is not None
    dominated_kg = find_dominated_vertex(kg) is not None
    source_simplices = source.all_simplices()
    target_simplices = target.all_simplices()
    if not target_simplices <= source_simplices:
        raise RuntimeError("the nerve is not a subcomplex of the clique complex")

    missing = source_simplices - target_simplices
    minimal_missing_sizes = sorted(
        {
            len(face)
            for face in missing
            if all(
                Simplex(face - {vertex}) in target_simplices
                for vertex in face
            )
        }
    )
    graph_node_ids = {
        vertex: index for index, vertex in enumerate(graph.nodes())
    }
    clique_node_ids = {
        clique: index
        for index, clique in enumerate(
            sorted(
                kg.nodes(),
                key=lambda clique: tuple(
                    sorted(graph_node_ids[vertex] for vertex in clique)
                ),
            )
        )
    }
    clique_labels = {
        str(index): sorted(graph_node_ids[vertex] for vertex in clique)
        for clique, index in clique_node_ids.items()
    }

    collapse_pairs: list[Any] = []
    if not missing:
        collapse_status = CollapseStatus.PROVED.value
        collapse_diagnostics = "Source equals target."
        explored_states = 0
    elif len(source_simplices) > max_simplices:
        collapse_status = CollapseStatus.INCONCLUSIVE.value
        collapse_diagnostics = (
            f"Not searched: {len(source_simplices)} source simplices exceed "
            f"max_simplices={max_simplices}."
        )
        explored_states = 0
    else:
        result = relative_collapse(source, target, max_states=max_states)
        collapse_status = result.status.value
        collapse_pairs = [
            [
                sorted(clique_node_ids[vertex] for vertex in free_face),
                sorted(clique_node_ids[vertex] for vertex in coface),
            ]
            for free_face, coface in result.pairs
        ]
        collapse_diagnostics = result.diagnostics
        explored_states = result.explored_states

    relative_betti = None
    if not missing:
        relative_betti = {}
    elif len(source_simplices) <= max_simplices:

        def is_in_nerve(face: Iterable) -> bool:
            return target.function(face)

        relative_betti = relative_betti_numbers(
            list(source.facet_set), is_in_nerve, method="exact"
        )

    return {
        "graph6": _graph6(graph),
        "clique_node_labels": clique_labels,
        "order": graph.number_of_nodes(),
        "clique_helly_g": clique_helly_g,
        "clique_helly_kg": clique_helly_kg,
        "dominated_vertex_g": dominated_g,
        "dominated_vertex_kg": dominated_kg,
        "defect": len(missing),
        "minimal_missing_face_sizes": minimal_missing_sizes,
        "collapse_status": collapse_status,
        "collapse_pairs": collapse_pairs,
        "collapse_diagnostics": collapse_diagnostics,
        "explored_states": explored_states,
        "relative_betti_q": relative_betti,
        "nonhelly_no_dominated_kg_helly_candidate": (
            matches_research_filter(graph, kg)
        ),
    }


def run_census(
    order: int,
    *,
    max_graphs: int | None = None,
    max_states: int = 10_000,
    max_simplices: int = 20_000,
) -> dict[str, Any]:
    if max_graphs is not None and max_graphs < 1:
        raise ValueError("max_graphs must be positive when provided")
    records: list[dict[str, Any]] = []
    for graph in graph_generator(order, connected=True):
        if max_graphs is not None and len(records) >= max_graphs:
            break
        records.append(
            analyze_graph(
                graph, max_states=max_states, max_simplices=max_simplices
            )
        )

    candidates = [
        record["graph6"]
        for record in records
        if record["nonhelly_no_dominated_kg_helly_candidate"]
    ]
    return {
        "networkx_version": nx.__version__,
        "order": order,
        "connected": True,
        "graphs_processed": len(records),
        "candidate_graph6": candidates,
        "records": records,
        "budgets": {
            "max_graphs": max_graphs,
            "max_states_per_collapse": max_states,
            "max_simplices": max_simplices,
        },
        "complete_dataset": max_graphs is None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order", type=int, choices=range(6, 11))
    parser.add_argument("--max-graphs", type=int)
    parser.add_argument("--max-states", type=int, default=1_000)
    parser.add_argument("--max-simplices", type=int, default=2_500)
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path (defaults to research/homotopy-census-orderN.json)",
    )
    args = parser.parse_args()
    output = args.output or Path(__file__).with_name(
        f"homotopy-census-order{args.order}.json"
    )
    report = run_census(
        args.order,
        max_graphs=args.max_graphs,
        max_states=args.max_states,
        max_simplices=args.max_simplices,
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "graphs_processed": report["graphs_processed"],
                "candidate_graph6": report["candidate_graph6"],
                "complete_dataset": report["complete_dataset"],
                "networkx_version": report["networkx_version"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

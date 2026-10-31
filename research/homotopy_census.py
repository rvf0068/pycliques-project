"""Bounded clique-graph homotopy census over the bundled graph6 datasets."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Iterable
from importlib.metadata import version
from itertools import chain, combinations
from pathlib import Path
from typing import Any

import networkx as nx
from pycliques import clique_graph, find_dominated_vertex, is_clique_helly
from pycombtop import (
    CollapseStatus,
    Simplex,
    SimplicialComplex,
    relative_betti_numbers,
    relative_collapse,
    verify_relative_collapse,
)
from pyg6data import graph_generator


def _graph6(graph: nx.Graph) -> str:
    encoded: bytes = nx.to_graph6_bytes(graph, nodes=list(graph.nodes()), header=False)
    return encoded.decode("ascii").strip()


def matches_research_filter(graph: nx.Graph, kg: nx.Graph) -> bool:
    """Apply the reported non-Helly/no-dominated/K(G)-Helly filter."""
    return (
        not is_clique_helly(graph)
        and find_dominated_vertex(graph) is None
        and is_clique_helly(kg)
    )


def _bounded_complex(
    facets: Iterable[Iterable],
    base_set: Iterable,
    max_simplices: int,
) -> tuple[SimplicialComplex, set[Simplex]] | None:
    """Build a complex only while its distinct simplex count fits the budget.

    The empty simplex counts toward ``max_simplices``, matching
    :meth:`SimplicialComplex.all_simplices`.
    """
    maximal_faces: set[Simplex] = set()
    simplices: set[Simplex] = set()
    for facet in facets:
        simplex_facet = Simplex(facet)
        maximal_faces.add(simplex_facet)
        subsets = chain.from_iterable(
            combinations(simplex_facet, size) for size in range(len(simplex_facet) + 1)
        )
        for subset in subsets:
            simplex = Simplex(subset)
            if simplex not in simplices:
                simplices.add(simplex)
                if len(simplices) > max_simplices:
                    return None
    return SimplicialComplex(base_set, facet_set=maximal_faces), simplices


def _nerve_facets(graph: nx.Graph, kg: nx.Graph) -> set[Simplex]:
    """Return a facet description of the nerve without testing all clique sets."""
    cliques_by_vertex: dict[Any, set] = {vertex: set() for vertex in graph}
    for clique in kg:
        for vertex in clique:
            cliques_by_vertex[vertex].add(clique)

    facets: set[Simplex] = set()
    for containing_cliques in cliques_by_vertex.values():
        if containing_cliques:
            facets.add(Simplex(containing_cliques))

    return {facet for facet in facets if not any(facet < other for other in facets)}


def analyze_graph(
    graph: nx.Graph,
    *,
    max_states: int = 1_000,
    max_simplices: int = 2_500,
    max_clique_graph_edges: int = 10_000,
) -> dict[str, Any]:
    """Analyze one graph, using explicit budgets for costly computations."""
    if max_states < 1:
        raise ValueError("max_states must be positive")
    if max_simplices < 0:
        raise ValueError("max_simplices must be nonnegative")
    if max_clique_graph_edges < 0:
        raise ValueError("max_clique_graph_edges must be nonnegative")
    kg = clique_graph(graph, bound=max_simplices, max_edges=max_clique_graph_edges)
    if kg is None:
        return {
            "graph6": _graph6(graph),
            "clique_node_labels": {},
            "order": graph.number_of_nodes(),
            "clique_helly_g": None,
            "clique_helly_kg": None,
            "dominated_vertex_g": None,
            "dominated_vertex_kg": None,
            "defect": None,
            "minimal_missing_face_sizes": None,
            "collapse_status": CollapseStatus.INCONCLUSIVE.value,
            "collapse_pairs": [],
            "collapse_diagnostics": (
                "Not analyzed: clique-graph construction exceeded "
                f"max_simplices={max_simplices} or "
                f"max_clique_graph_edges={max_clique_graph_edges}."
            ),
            "explored_states": 0,
            "relative_betti_q": None,
            "nonhelly_no_dominated_kg_helly_candidate": None,
        }

    graph_node_ids = {vertex: index for index, vertex in enumerate(graph.nodes())}
    ordered_cliques = sorted(
        kg.nodes(),
        key=lambda clique: tuple(sorted(graph_node_ids[vertex] for vertex in clique)),
    )
    clique_node_ids = {clique: index for index, clique in enumerate(ordered_cliques)}
    clique_labels = {
        str(index): sorted(graph_node_ids[vertex] for vertex in clique)
        for clique, index in clique_node_ids.items()
    }

    bounded_source = _bounded_complex(nx.find_cliques(kg), kg.nodes(), max_simplices)
    if bounded_source is None:
        return {
            "graph6": _graph6(graph),
            "clique_node_labels": clique_labels,
            "order": graph.number_of_nodes(),
            "clique_helly_g": None,
            "clique_helly_kg": None,
            "dominated_vertex_g": None,
            "dominated_vertex_kg": None,
            "defect": None,
            "minimal_missing_face_sizes": None,
            "collapse_status": CollapseStatus.INCONCLUSIVE.value,
            "collapse_pairs": [],
            "collapse_diagnostics": (
                "Not analyzed: source simplex enumeration exceeded "
                f"max_simplices={max_simplices} (including the empty simplex)."
            ),
            "explored_states": 0,
            "relative_betti_q": None,
            "nonhelly_no_dominated_kg_helly_candidate": None,
        }
    source, source_simplices = bounded_source

    target_facets = _nerve_facets(graph, kg)
    bounded_target = _bounded_complex(target_facets, kg.nodes(), max_simplices)
    if bounded_target is None:
        return {
            "graph6": _graph6(graph),
            "clique_node_labels": clique_labels,
            "order": graph.number_of_nodes(),
            "clique_helly_g": None,
            "clique_helly_kg": None,
            "dominated_vertex_g": None,
            "dominated_vertex_kg": None,
            "defect": None,
            "minimal_missing_face_sizes": None,
            "collapse_status": CollapseStatus.INCONCLUSIVE.value,
            "collapse_pairs": [],
            "collapse_diagnostics": (
                "Not analyzed: nerve simplex enumeration exceeded "
                f"max_simplices={max_simplices} (including the empty simplex)."
            ),
            "explored_states": 0,
            "relative_betti_q": None,
            "nonhelly_no_dominated_kg_helly_candidate": None,
        }
    target, target_simplices = bounded_target

    clique_helly_g = is_clique_helly(graph)
    clique_helly_kg = is_clique_helly(kg)
    dominated_g = find_dominated_vertex(graph) is not None
    dominated_kg = find_dominated_vertex(kg) is not None
    if not target_simplices <= source_simplices:
        raise RuntimeError("the nerve is not a subcomplex of the clique complex")

    missing = source_simplices - target_simplices
    minimal_missing_sizes = sorted(
        {
            len(face)
            for face in missing
            if all(Simplex(face - {vertex}) in target_simplices for vertex in face)
        }
    )
    collapse_pairs: list[Any] = []
    if not missing:
        collapse_status = CollapseStatus.PROVED.value
        collapse_diagnostics = "Source equals target."
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
    else:

        def is_in_nerve(face: Iterable) -> bool:
            return target.function(face)

        relative_betti = relative_betti_numbers(
            list(source.facet_set), is_in_nerve, method="exact"
        )

    if collapse_status == CollapseStatus.PROVED.value:
        serialized_pairs = tuple(
            (
                Simplex(ordered_cliques[index] for index in free_face),
                Simplex(ordered_cliques[index] for index in coface),
            )
            for free_face, coface in collapse_pairs
        )
        if not verify_relative_collapse(source, target, serialized_pairs):
            raise RuntimeError("serialized relative-collapse certificate failed replay")

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
    max_clique_graph_edges: int = 100_000,
) -> dict[str, Any]:
    if max_graphs is not None and max_graphs < 1:
        raise ValueError("max_graphs must be positive when provided")
    if max_clique_graph_edges < 0:
        raise ValueError("max_clique_graph_edges must be nonnegative")
    records: list[dict[str, Any]] = []
    for graph in graph_generator(order, connected=True):
        if max_graphs is not None and len(records) >= max_graphs:
            break
        records.append(
            analyze_graph(
                graph,
                max_states=max_states,
                max_simplices=max_simplices,
                max_clique_graph_edges=max_clique_graph_edges,
            )
        )

    candidates = [
        record["graph6"]
        for record in records
        if record["nonhelly_no_dominated_kg_helly_candidate"]
    ]
    candidate_filter_inconclusive = sum(
        record["nonhelly_no_dominated_kg_helly_candidate"] is None for record in records
    )
    return {
        "networkx_version": nx.__version__,
        "order": order,
        "connected": True,
        "graphs_processed": len(records),
        "candidate_graph6": candidates,
        "candidate_filter_inconclusive_graphs": candidate_filter_inconclusive,
        "candidate_filter_complete": candidate_filter_inconclusive == 0,
        "records": records,
        "budgets": {
            "max_graphs": max_graphs,
            "max_states_per_collapse": max_states,
            "max_simplices": max_simplices,
            "max_clique_graph_edges": max_clique_graph_edges,
        },
        "complete_dataset": max_graphs is None,
    }


def run_census_jsonl(
    order: int,
    output: Path,
    *,
    max_graphs: int | None = None,
    max_states: int = 10_000,
    max_simplices: int = 20_000,
    max_clique_graph_edges: int = 100_000,
    checkpoint_interval: int = 100,
    resume: bool = False,
) -> dict[str, Any]:
    """Stream census records to JSONL and safely resume from that file.

    The first JSONL object stores input and budget metadata. Each following
    object stores one graph index, graph6 encoding, and analysis result.
    Existing output is accepted only with ``resume=True`` and matching
    metadata; completed indices are checked against the current graph6 input.
    """
    if max_graphs is not None and max_graphs < 1:
        raise ValueError("max_graphs must be positive when provided")
    if max_states < 1:
        raise ValueError("max_states must be positive")
    if max_simplices < 0:
        raise ValueError("max_simplices must be nonnegative")
    if max_clique_graph_edges < 0:
        raise ValueError("max_clique_graph_edges must be nonnegative")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if output.exists() and not resume:
        raise FileExistsError(f"{output} exists; pass resume=True to continue it")
    if resume and not output.exists():
        raise FileNotFoundError(f"cannot resume missing JSONL output: {output}")

    metadata = {
        "type": "metadata",
        "schema_version": 1,
        "python_version": platform.python_version(),
        "networkx_version": nx.__version__,
        "package_versions": {
            package: version(package)
            for package in ("pycliques", "pycombtop", "pyg6data")
        },
        "order": order,
        "connected": True,
        "budgets": {
            "max_graphs": max_graphs,
            "max_states_per_collapse": max_states,
            "max_simplices": max_simplices,
            "max_clique_graph_edges": max_clique_graph_edges,
        },
    }

    records_by_index: dict[int, str] = {}
    candidate_by_index: dict[int, str] = {}
    candidate_unknown_indices: set[int] = set()
    if resume:
        with output.open("r", encoding="utf-8") as source:
            first_line = source.readline()
            try:
                existing_metadata = json.loads(first_line)
            except json.JSONDecodeError as error:
                raise ValueError("JSONL output has invalid metadata") from error
            if existing_metadata != metadata:
                raise ValueError(
                    "JSONL metadata does not match the requested order, "
                    "versions, and budgets"
                )
            for line_number, line in enumerate(source, start=2):
                try:
                    row = json.loads(line)
                    index = row["index"]
                    graph6 = row["graph6"]
                    result = row["result"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(
                        f"invalid JSONL record at line {line_number}"
                    ) from error
                if (
                    not isinstance(row, dict)
                    or not isinstance(result, dict)
                    or not isinstance(index, int)
                    or index < 0
                    or not isinstance(graph6, str)
                    or row.get("type") != "record"
                    or index in records_by_index
                    or (max_graphs is not None and index >= max_graphs)
                    or result.get("nonhelly_no_dominated_kg_helly_candidate")
                    not in (True, False, None)
                ):
                    raise ValueError(
                        f"invalid or duplicate census index at line {line_number}"
                    )
                records_by_index[index] = graph6
                if result["nonhelly_no_dominated_kg_helly_candidate"]:
                    candidate_by_index[index] = graph6
                elif result["nonhelly_no_dominated_kg_helly_candidate"] is None:
                    candidate_unknown_indices.add(index)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if resume else "w", encoding="utf-8") as destination:
        if not resume:
            destination.write(json.dumps(metadata, sort_keys=True) + "\n")
        pending_since_flush = 0
        graphs_seen = 0
        input_count = 0
        for index, graph in enumerate(graph_generator(order, connected=True)):
            if max_graphs is not None and index >= max_graphs:
                break
            input_count += 1
            encoded = _graph6(graph)
            previous = records_by_index.get(index)
            if previous is not None:
                if previous != encoded:
                    raise ValueError(
                        f"graph6 input changed at census index {index}; "
                        "resume is unsafe"
                    )
                continue
            result = analyze_graph(
                graph,
                max_states=max_states,
                max_simplices=max_simplices,
                max_clique_graph_edges=max_clique_graph_edges,
            )
            destination.write(
                json.dumps(
                    {
                        "index": index,
                        "type": "record",
                        "graph6": encoded,
                        "result": result,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            records_by_index[index] = encoded
            if result["nonhelly_no_dominated_kg_helly_candidate"]:
                candidate_by_index[index] = encoded
            elif result["nonhelly_no_dominated_kg_helly_candidate"] is None:
                candidate_unknown_indices.add(index)
            graphs_seen += 1
            pending_since_flush += 1
            if pending_since_flush >= checkpoint_interval:
                destination.flush()
                pending_since_flush = 0
        destination.flush()
    if any(index >= input_count for index in records_by_index):
        raise ValueError("JSONL output contains indices absent from the input dataset")

    summary = {
        "order": order,
        "connected": True,
        "graphs_processed": len(records_by_index),
        "graphs_processed_this_run": graphs_seen,
        "candidate_graph6": [
            candidate_by_index[index] for index in sorted(candidate_by_index)
        ],
        "candidate_filter_inconclusive_graphs": len(candidate_unknown_indices),
        "candidate_filter_complete": not candidate_unknown_indices,
        "budgets": metadata["budgets"],
        "complete_dataset": max_graphs is None,
        "jsonl_output": str(output),
        "networkx_version": nx.__version__,
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order", type=int, choices=range(6, 11))
    parser.add_argument("--max-graphs", type=int)
    parser.add_argument("--max-states", type=int, default=1_000)
    parser.add_argument("--max-simplices", type=int, default=2_500)
    parser.add_argument("--max-clique-graph-edges", type=int, default=100_000)
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="stream one result per line and write a separate summary",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a compatible existing JSONL output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path (defaults to research/homotopy-census-orderN.json)",
    )
    args = parser.parse_args()
    if args.resume and not args.jsonl:
        parser.error("--resume requires --jsonl")
    output = args.output or Path(__file__).with_name(
        f"homotopy-census-order{args.order}{'.jsonl' if args.jsonl else '.json'}"
    )
    if args.jsonl:
        report = run_census_jsonl(
            args.order,
            output,
            max_graphs=args.max_graphs,
            max_states=args.max_states,
            max_simplices=args.max_simplices,
            max_clique_graph_edges=args.max_clique_graph_edges,
            resume=args.resume,
        )
    else:
        report = run_census(
            args.order,
            max_graphs=args.max_graphs,
            max_states=args.max_states,
            max_simplices=args.max_simplices,
            max_clique_graph_edges=args.max_clique_graph_edges,
        )
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "graphs_processed": report["graphs_processed"],
                "candidate_graph6": report["candidate_graph6"],
                "candidate_filter_complete": report["candidate_filter_complete"],
                "complete_dataset": report["complete_dataset"],
                "networkx_version": report["networkx_version"],
                "summary_output": (
                    str(output.with_suffix(output.suffix + ".summary.json"))
                    if args.jsonl
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

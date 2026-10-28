"""Bounded experiments for the H-operator homotopy conjecture.

The direct construction is the reference path: M = K(H(G)), followed by the
existing ``analyze_graph`` certificate search for Delta(K(M)) relative to the
nerve of M's maximal cliques. ``double_bound_clique_graph`` is an optional,
bounded realization of the same M using Proposition 7.5.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from collections.abc import Iterable, Iterator
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
from pycliques import clique_graph, homotopy_clique_graph
from pycombtop import relative_betti_numbers

from research.homotopy_census import _bounded_complex, analyze_graph

PROVED_BY_COLLAPSE = "PROVED_BY_COLLAPSE"
HOMOLOGY_MISMATCH_COUNTEREXAMPLE = "HOMOLOGY_MISMATCH_COUNTEREXAMPLE"
INCONCLUSIVE_BUDGET = "INCONCLUSIVE_BUDGET"
INCONCLUSIVE_NO_COLLAPSE_CERTIFICATE = "INCONCLUSIVE_NO_COLLAPSE_CERTIFICATE"
INCONCLUSIVE_HOMOLOGY_MISMATCH_UNCONFIRMED = (
    "INCONCLUSIVE_HOMOLOGY_MISMATCH_UNCONFIRMED"
)


def _graph6(graph: nx.Graph) -> str:
    encoded: bytes = nx.to_graph6_bytes(graph, nodes=list(graph.nodes()), header=False)
    return encoded.decode("ascii").strip()


def _is_complete_face(graph: nx.Graph, face: frozenset) -> bool:
    return all(graph.has_edge(u, v) for u, v in combinations(face, 2))


def _homotopy_graph_counts(graph: nx.Graph) -> tuple[int, int]:
    """Count H(G)'s vertices and edges without constructing H(G)."""
    cliques_by_vertex: dict[Any, set[frozenset]] = {vertex: set() for vertex in graph}
    for clique in nx.find_cliques(graph):
        frozen_clique = frozenset(clique)
        for vertex in frozen_clique:
            cliques_by_vertex[vertex].add(frozen_clique)
    vertex_count = sum(len(cliques) for cliques in cliques_by_vertex.values())
    edge_count = sum(
        math.comb(len(cliques), 2) for cliques in cliques_by_vertex.values()
    )
    edge_count += sum(
        len(cliques_by_vertex[first] & cliques_by_vertex[second]) ** 2
        for first, second in graph.edges()
    )
    return vertex_count, edge_count


def double_bound_clique_graph(
    graph: nx.Graph,
    *,
    max_faces: int | None = None,
    max_candidates: int | None = None,
    max_edges: int | None = None,
) -> nx.Graph | None:
    """Build the Proposition 7.5 realization ``DB(P(G)'')``.

    Vertices are distinct coatomic-closed nonempty faces ``c(A)`` of *graph*.
    Distinct closed faces are adjacent exactly when they intersect and their
    union is complete in *graph*. ``max_candidates`` bounds the nonempty
    complete faces examined; all configured budgets return ``None`` when
    exceeded, not a negative result.
    """
    if max_faces is not None and max_faces < 0:
        raise ValueError("max_faces must be nonnegative or None")
    if max_candidates is not None and max_candidates < 0:
        raise ValueError("max_candidates must be nonnegative or None")
    if max_edges is not None and max_edges < 0:
        raise ValueError("max_edges must be nonnegative or None")

    maximal_cliques = [frozenset(clique) for clique in nx.find_cliques(graph)]
    vertex_order = {vertex: index for index, vertex in enumerate(graph.nodes())}
    closed_faces: set[frozenset] = set()
    candidate_count = 0
    for clique in maximal_cliques:
        for size in range(1, len(clique) + 1):
            for face_tuple in combinations(clique, size):
                candidate_count += 1
                if max_candidates is not None and candidate_count > max_candidates:
                    return None
                face = frozenset(face_tuple)
                containing = [maximal for maximal in maximal_cliques if face <= maximal]
                closed_faces.add(frozenset.intersection(*containing))
                if max_faces is not None and len(closed_faces) > max_faces:
                    return None

    result = nx.Graph()
    ordered_faces = sorted(
        closed_faces,
        key=lambda face: tuple(sorted(vertex_order[vertex] for vertex in face)),
    )
    result.add_nodes_from(ordered_faces)
    edge_count = 0
    for first, second in combinations(ordered_faces, 2):
        union = first | second
        if first.isdisjoint(second) or not _is_complete_face(graph, union):
            continue
        if max_edges is not None and edge_count >= max_edges:
            return None
        result.add_edge(first, second)
        edge_count += 1
    return result


def atlas_graphs(max_order: int = 6) -> Iterator[nx.Graph]:
    """Yield every nonempty graph in NetworkX's atlas through *max_order*."""
    if not 1 <= max_order <= 7:
        raise ValueError("max_order must be between 1 and 7")
    yield from (
        graph.copy()
        for graph in nx.graph_atlas_g()
        if 0 < graph.number_of_nodes() <= max_order
    )


def graph6_graphs(path: Path) -> Iterator[nx.Graph]:
    """Read one graph6 graph per nonblank line, with or without a header."""
    with path.open(encoding="ascii") as source:
        for line_number, line in enumerate(source, start=1):
            encoded = line.strip()
            if not encoded or encoded.startswith(">>graph6<<"):
                continue
            try:
                yield nx.from_graph6_bytes(encoded.encode("ascii"))
            except (UnicodeEncodeError, nx.NetworkXError) as error:
                raise ValueError(
                    f"invalid graph6 data on line {line_number}"
                ) from error


def _betti_numbers(graph: nx.Graph, max_simplices: int) -> list[int] | None:
    bounded = _bounded_complex(nx.find_cliques(graph), graph.nodes(), max_simplices)
    if bounded is None:
        return None
    complex_, _ = bounded
    ordinary = relative_betti_numbers(
        complex_.facet_set, lambda _face: False, method="exact"
    )
    if not ordinary:
        return []
    reduced = dict(ordinary)
    if 0 in reduced:
        reduced[0] -= 1
    highest = max(reduced)
    values = [reduced.get(dimension, 0) for dimension in range(highest + 1)]
    while values and values[-1] == 0:
        values.pop()
    return values


def _gf2_rank(columns: list[int]) -> int:
    """Return the rank of a binary matrix given as integer-encoded columns."""
    pivots: dict[int, int] = {}
    for column in columns:
        value = column
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def _betti_numbers_mod2(graph: nx.Graph, max_simplices: int) -> list[int] | None:
    """Compute reduced Betti numbers independently using binary boundaries."""
    simplices: set[frozenset] = set()
    for clique in nx.find_cliques(graph):
        frozen_clique = frozenset(clique)
        for size in range(len(frozen_clique) + 1):
            for face in combinations(frozen_clique, size):
                simplices.add(frozenset(face))
                if len(simplices) > max_simplices:
                    return None

    vertex_order = {vertex: i for i, vertex in enumerate(graph.nodes())}
    by_dimension: dict[int, list[frozenset]] = {}
    for simplex in simplices:
        if simplex:
            by_dimension.setdefault(len(simplex) - 1, []).append(simplex)
    if not by_dimension:
        return []

    for faces in by_dimension.values():
        faces.sort(key=lambda face: tuple(sorted(vertex_order[v] for v in face)))

    boundary_ranks = {0: 0}
    for dimension in range(1, max(by_dimension) + 1):
        rows = {
            face: index
            for index, face in enumerate(by_dimension.get(dimension - 1, []))
        }
        columns = []
        for simplex in by_dimension.get(dimension, []):
            boundary = 0
            for vertex in simplex:
                boundary ^= 1 << rows[simplex - {vertex}]
            columns.append(boundary)
        boundary_ranks[dimension] = _gf2_rank(columns)

    betti_numbers = []
    for dimension in range(max(by_dimension) + 1):
        rank = (
            len(by_dimension.get(dimension, []))
            - boundary_ranks.get(dimension, 0)
            - boundary_ranks.get(dimension + 1, 0)
        )
        if dimension == 0:
            rank -= 1
        betti_numbers.append(rank)
    while betti_numbers and betti_numbers[-1] == 0:
        betti_numbers.pop()
    return betti_numbers


def analyze_h_operator_graph(
    graph: nx.Graph,
    *,
    max_cliques: int = 10_000,
    max_edges: int = 100_000,
    max_states: int = 1_000,
    max_simplices: int = 2_500,
    max_candidate_faces: int = 100_000,
    use_compact_realization: bool = False,
) -> dict[str, Any]:
    """Analyze the conjecture for one graph without treating search limits as failure.

    The compact Proposition 7.5 realization is opt-in. Before enabling it in
    a larger census, cross-check it against ``K(H(G))`` on the desired input
    class using ``double_bound_clique_graph`` and ``nx.is_isomorphic``.
    """
    if max_cliques < 0:
        raise ValueError("max_cliques must be nonnegative")
    if max_edges < 0:
        raise ValueError("max_edges must be nonnegative")
    if max_states < 1:
        raise ValueError("max_states must be positive")
    if max_simplices < 0:
        raise ValueError("max_simplices must be nonnegative")
    if max_candidate_faces < 0:
        raise ValueError("max_candidate_faces must be nonnegative")

    if use_compact_realization:
        h_order, h_size = _homotopy_graph_counts(graph)
        middle = double_bound_clique_graph(
            graph,
            max_faces=max_cliques,
            max_candidates=max_candidate_faces,
            max_edges=max_edges,
        )
    else:
        homotopy_graph = homotopy_clique_graph(graph)
        h_order = homotopy_graph.number_of_nodes()
        h_size = homotopy_graph.number_of_edges()
        middle = clique_graph(homotopy_graph, bound=max_cliques, max_edges=max_edges)

    base: dict[str, Any] = {
        "graph6": _graph6(graph),
        "order": graph.number_of_nodes(),
        "size": graph.number_of_edges(),
        "h_order": h_order,
        "h_size": h_size,
        "middle_order": None if middle is None else middle.number_of_nodes(),
        "middle_size": None if middle is None else middle.number_of_edges(),
        "k2_order": None,
        "k2_size": None,
        "betti_field": "rational",
        "betti_method": "exact integer boundary matrices via pycombtop",
        "betti_g": None,
        "betti_k2": None,
        "betti_mod2_g": None,
        "betti_mod2_k2": None,
        "homology_mismatch_independently_confirmed": False,
        "relative_betti_q": None,
        "collapse_pairs": [],
        "collapse_diagnostics": None,
        "collapse_certificate_replayed": False,
        "resource_diagnostics": None,
        "status": INCONCLUSIVE_BUDGET,
        "versions": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "pycliques": version("pycliques"),
            "pycombtop": version("pycombtop"),
        },
        "budgets": {
            "max_cliques": max_cliques,
            "max_edges": max_edges,
            "max_states": max_states,
            "max_simplices": max_simplices,
            "max_candidate_faces": max_candidate_faces,
            "use_compact_realization": use_compact_realization,
        },
    }
    if middle is None:
        base["resource_diagnostics"] = (
            "Construction of K(H(G)) exceeded max_cliques/max_edges."
            if not use_compact_realization
            else (
                "Compact DB(P(G)'') construction exceeded "
                "max_cliques/max_candidate_faces/max_edges."
            )
        )
        return base

    k2 = clique_graph(middle, bound=max_cliques, max_edges=max_edges)
    if k2 is None:
        base["resource_diagnostics"] = (
            "Construction of K^2(H(G)) exceeded max_cliques/max_edges."
        )
        return base
    base["k2_order"] = k2.number_of_nodes()
    base["k2_size"] = k2.number_of_edges()

    analysis = analyze_graph(
        middle,
        max_states=max_states,
        max_simplices=max_simplices,
        max_clique_graph_edges=max_edges,
    )
    base["relative_betti_q"] = analysis["relative_betti_q"]
    base["collapse_pairs"] = analysis["collapse_pairs"]
    base["collapse_diagnostics"] = analysis["collapse_diagnostics"]
    base["collapse_certificate_replayed"] = analysis["collapse_status"] == "PROVED"

    betti_g = _betti_numbers(graph, max_simplices)
    betti_k2 = _betti_numbers(k2, max_simplices)
    base["betti_g"] = betti_g
    base["betti_k2"] = betti_k2

    if base["collapse_certificate_replayed"]:
        base["status"] = PROVED_BY_COLLAPSE
    elif betti_g is not None and betti_k2 is not None and betti_g != betti_k2:
        betti_g_mod2 = _betti_numbers_mod2(graph, max_simplices)
        betti_k2_mod2 = _betti_numbers_mod2(k2, max_simplices)
        base["betti_mod2_g"] = betti_g_mod2
        base["betti_mod2_k2"] = betti_k2_mod2
        if (
            betti_g_mod2 is not None
            and betti_k2_mod2 is not None
            and betti_g_mod2 != betti_k2_mod2
        ):
            base["status"] = HOMOLOGY_MISMATCH_COUNTEREXAMPLE
            base["homology_mismatch_independently_confirmed"] = True
        else:
            base["status"] = INCONCLUSIVE_HOMOLOGY_MISMATCH_UNCONFIRMED
    elif analysis["collapse_diagnostics"].startswith("Not analyzed:"):
        base["status"] = INCONCLUSIVE_BUDGET
        base["resource_diagnostics"] = analysis["collapse_diagnostics"]
    else:
        base["status"] = INCONCLUSIVE_NO_COLLAPSE_CERTIFICATE

    base["analysis"] = analysis
    return base


def run_h_operator_census(
    graphs: Iterable[nx.Graph],
    *,
    output_jsonl: Path | None = None,
    resume: bool = False,
    max_graphs: int | None = None,
    max_cliques: int = 10_000,
    max_edges: int = 100_000,
    max_states: int = 1_000,
    max_simplices: int = 2_500,
    max_candidate_faces: int = 100_000,
    use_compact_realization: bool = False,
) -> dict[str, Any]:
    """Run a bounded census and optionally save resumable JSONL records."""
    if max_graphs is not None and max_graphs < 1:
        raise ValueError("max_graphs must be positive when provided")
    if max_cliques < 0:
        raise ValueError("max_cliques must be nonnegative")
    if max_edges < 0:
        raise ValueError("max_edges must be nonnegative")
    if max_states < 1:
        raise ValueError("max_states must be positive")
    if max_simplices < 0:
        raise ValueError("max_simplices must be nonnegative")
    if max_candidate_faces < 0:
        raise ValueError("max_candidate_faces must be nonnegative")
    if resume and output_jsonl is None:
        raise ValueError("resume requires output_jsonl")

    graph_list = list(graphs)
    if max_graphs is not None:
        graph_list = graph_list[:max_graphs]
    limits = {
        "max_graphs": max_graphs,
        "max_cliques": max_cliques,
        "max_edges": max_edges,
        "max_states": max_states,
        "max_simplices": max_simplices,
        "max_candidate_faces": max_candidate_faces,
        "use_compact_realization": use_compact_realization,
    }
    metadata = {
        "type": "metadata",
        "schema_version": 1,
        "limits": limits,
        "python_version": platform.python_version(),
        "networkx_version": nx.__version__,
        "package_versions": {
            package: version(package) for package in ("pycliques", "pycombtop")
        },
    }
    previous_records: dict[int, dict[str, Any]] = {}
    if output_jsonl is not None:
        if output_jsonl.exists() and not resume:
            raise FileExistsError(
                f"{output_jsonl} exists; pass resume=True to continue it"
            )
        if resume and not output_jsonl.exists():
            raise FileNotFoundError(f"cannot resume missing JSONL: {output_jsonl}")
        if resume:
            with output_jsonl.open(encoding="utf-8") as source:
                try:
                    existing_metadata = json.loads(source.readline())
                except json.JSONDecodeError as error:
                    raise ValueError("JSONL output has invalid metadata") from error
                if existing_metadata != metadata:
                    raise ValueError("JSONL metadata does not match requested census")
                for line_number, line in enumerate(source, start=2):
                    try:
                        record = json.loads(line)
                        index = record["index"]
                        if (
                            record["type"] != "record"
                            or not isinstance(index, int)
                            or index < 0
                            or index >= len(graph_list)
                            or not isinstance(record["graph6"], str)
                            or not isinstance(record["result"], dict)
                            or index in previous_records
                        ):
                            raise ValueError
                        previous_records[index] = record
                    except (
                        json.JSONDecodeError,
                        KeyError,
                        TypeError,
                        ValueError,
                    ) as error:
                        raise ValueError(
                            f"invalid JSONL record at line {line_number}"
                        ) from error
            for index, record in previous_records.items():
                if record["graph6"] != _graph6(graph_list[index]):
                    raise ValueError(f"graph6 input changed at census index {index}")

        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "w"
        with output_jsonl.open(mode, encoding="utf-8") as destination:
            if not resume:
                destination.write(json.dumps(metadata, sort_keys=True) + "\n")
            for index, graph in enumerate(graph_list):
                encoded = _graph6(graph)
                if index in previous_records:
                    if previous_records[index]["graph6"] != encoded:
                        raise ValueError(
                            f"graph6 input changed at census index {index}"
                        )
                    continue
                result = analyze_h_operator_graph(
                    graph,
                    max_cliques=max_cliques,
                    max_edges=max_edges,
                    max_states=max_states,
                    max_simplices=max_simplices,
                    max_candidate_faces=max_candidate_faces,
                    use_compact_realization=use_compact_realization,
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
                previous_records[index] = {
                    "graph6": encoded,
                    "result": result,
                }
    else:
        for index, graph in enumerate(graph_list):
            previous_records[index] = {
                "graph6": _graph6(graph),
                "result": analyze_h_operator_graph(
                    graph,
                    max_cliques=max_cliques,
                    max_edges=max_edges,
                    max_states=max_states,
                    max_simplices=max_simplices,
                    max_candidate_faces=max_candidate_faces,
                    use_compact_realization=use_compact_realization,
                ),
            }

    results = [previous_records[index]["result"] for index in sorted(previous_records)]
    counts = {
        status: sum(record["status"] == status for record in results)
        for status in (
            PROVED_BY_COLLAPSE,
            HOMOLOGY_MISMATCH_COUNTEREXAMPLE,
            INCONCLUSIVE_BUDGET,
            INCONCLUSIVE_NO_COLLAPSE_CERTIFICATE,
            INCONCLUSIVE_HOMOLOGY_MISMATCH_UNCONFIRMED,
        )
    }
    return {
        "graphs_processed": len(results),
        "status_counts": counts,
        "limits": limits,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--atlas-order", type=int, choices=range(1, 8))
    source.add_argument("--graph6", type=Path)
    parser.add_argument("--max-graphs", type=int)
    parser.add_argument("--max-cliques", type=int, default=10_000)
    parser.add_argument("--max-edges", type=int, default=100_000)
    parser.add_argument("--max-states", type=int, default=1_000)
    parser.add_argument("--max-simplices", type=int, default=2_500)
    parser.add_argument("--max-candidate-faces", type=int, default=100_000)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.atlas_order is not None:
        graphs = atlas_graphs(args.atlas_order)
    elif args.graph6 is not None:
        graphs = graph6_graphs(args.graph6)
    else:  # pragma: no cover - enforced by the required argument group
        parser.error("select an atlas order or graph6 input")
    report = run_h_operator_census(
        graphs,
        output_jsonl=args.output,
        resume=args.resume,
        max_graphs=args.max_graphs,
        max_cliques=args.max_cliques,
        max_edges=args.max_edges,
        max_states=args.max_states,
        max_simplices=args.max_simplices,
        max_candidate_faces=args.max_candidate_faces,
        use_compact_realization=args.compact,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))


if __name__ == "__main__":
    main()

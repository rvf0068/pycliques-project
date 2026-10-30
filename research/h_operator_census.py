"""Bounded experiments for the H-operator homotopy conjecture.

The direct construction is the reference path: M = K(H(G)), followed by the
existing ``analyze_graph`` certificate search for Delta(K(M)) relative to the
nerve of M's maximal cliques. ``double_bound_clique_graph`` is an optional,
bounded realization of the same M using Proposition 7.5. An exact 3-Helly
recognition test can certify affirmative cases before either route constructs
an iterated clique graph; pass ``use_3_helly_certificate=False`` to disable it.
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
from pycliques import (
    ThreeHellyStatus,
    clique_graph,
    homotopy_clique_graph,
    maximal_clique_hypergraph_is_3_helly,
)
from pycombtop import (
    CollapseStatus,
    Simplex,
    SimplicialComplex,
    dismantling_certificate,
    relative_betti_numbers,
    relative_collapse,
    verify_relative_collapse,
    verify_s_collapse_move,
)

from research.homotopy_census import _bounded_complex, analyze_graph

PROVED_BY_COLLAPSE = "PROVED_BY_COLLAPSE"
PROVED_BY_3_HELLY = "PROVED_BY_3_HELLY"
HOMOLOGY_MISMATCH_COUNTEREXAMPLE = "HOMOLOGY_MISMATCH_COUNTEREXAMPLE"
INCONCLUSIVE_BUDGET = "INCONCLUSIVE_BUDGET"
INCONCLUSIVE_NO_COLLAPSE_CERTIFICATE = "INCONCLUSIVE_NO_COLLAPSE_CERTIFICATE"
INCONCLUSIVE_HOMOLOGY_MISMATCH_UNCONFIRMED = (
    "INCONCLUSIVE_HOMOLOGY_MISMATCH_UNCONFIRMED"
)
PROVED_BY_SPHERE_CERTIFICATE = "PROVED_BY_SPHERE_CERTIFICATE"


def _graph6(graph: nx.Graph) -> str:
    encoded: bytes = nx.to_graph6_bytes(graph, nodes=list(graph.nodes()), header=False)
    return encoded.decode("ascii").strip()


def _encode_certificate_label(label: Any) -> Any:
    if isinstance(label, frozenset):
        return {
            "$frozenset": [
                _encode_certificate_label(item)
                for item in sorted(label, key=repr)
            ]
        }
    if isinstance(label, tuple):
        return {"$tuple": [_encode_certificate_label(item) for item in label]}
    if label is None or isinstance(label, (str, int, float, bool)):
        return label
    raise TypeError(f"unsupported certificate label type: {type(label).__name__}")


def _decode_certificate_label(label: Any) -> Any:
    if isinstance(label, dict) and set(label) == {"$frozenset"}:
        return frozenset(
            _decode_certificate_label(item) for item in label["$frozenset"]
        )
    if isinstance(label, dict) and set(label) == {"$tuple"}:
        return tuple(_decode_certificate_label(item) for item in label["$tuple"])
    return label


def _encode_s_reduction_certificate(
    moves: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    encoded = []
    for move in moves:
        item = {
            "kind": move["kind"],
            "dismantling": [
                [
                    _encode_certificate_label(removed),
                    _encode_certificate_label(dominator),
                ]
                for removed, dominator in move["dismantling"]
            ],
        }
        if move["kind"] == "vertex":
            item["vertex"] = _encode_certificate_label(move["vertex"])
        else:
            item["edge"] = [
                _encode_certificate_label(endpoint) for endpoint in move["edge"]
            ]
        encoded.append(item)
    return encoded


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


coatomic_double_bound_graph = double_bound_clique_graph


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


def _s_reduce_graph(
    graph: nx.Graph,
    max_moves: int,
) -> tuple[nx.Graph, list[dict[str, Any]], bool]:
    """Run deterministic vertex/edge s-moves to a fixed point."""
    current = graph.copy()
    moves: list[dict[str, Any]] = []
    while True:
        found_move = False
        for vertex in current:
            local = current.subgraph(set(current[vertex])).copy()
            certificate = dismantling_certificate(local)
            local_pairs = None if certificate is None else list(certificate)
            if local_pairs is None:
                continue
            if len(moves) >= max_moves:
                return current, moves, False
            moves.append(
                {"kind": "vertex", "vertex": vertex, "dismantling": local_pairs}
            )
            current.remove_node(vertex)
            found_move = True
            break
        if found_move:
            continue

        for first, second in current.edges():
            common = set(current[first]) & set(current[second])
            local = current.subgraph(common).copy()
            certificate = dismantling_certificate(local)
            local_pairs = None if certificate is None else list(certificate)
            if local_pairs is None:
                continue
            if len(moves) >= max_moves:
                return current, moves, False
            moves.append(
                {
                    "kind": "edge",
                    "edge": (first, second),
                    "dismantling": local_pairs,
                }
            )
            current.remove_edge(first, second)
            found_move = True
            break
        if not found_move:
            return current, moves, True


def verify_s_reduction_certificate(
    graph: nx.Graph,
    moves: list[dict[str, Any]],
    expected_core: nx.Graph | None = None,
) -> bool:
    """Replay graph s-moves on the evolving graph and verify the claimed core."""
    current = graph.copy()
    decoded_moves = []
    try:
        for move in moves:
            decoded = {
                "kind": move["kind"],
                "dismantling": [
                    (
                        _decode_certificate_label(removed),
                        _decode_certificate_label(dominator),
                    )
                    for removed, dominator in move["dismantling"]
                ],
            }
            if move["kind"] == "vertex":
                decoded["vertex"] = _decode_certificate_label(move["vertex"])
            elif move["kind"] == "edge":
                decoded["edge"] = tuple(
                    _decode_certificate_label(endpoint) for endpoint in move["edge"]
                )
            else:
                return False
            decoded_moves.append(decoded)
    except (KeyError, TypeError, ValueError):
        return False
    for move in decoded_moves:
        if not verify_s_collapse_move(current, move):
            return False
    if expected_core is None:
        return True
    return set(current.nodes()) == set(expected_core.nodes()) and {
        frozenset(edge) for edge in current.edges()
    } == {frozenset(edge) for edge in expected_core.edges()}


def _complex_from_clique_graph(
    graph: nx.Graph, max_simplices: int
) -> tuple[SimplicialComplex, set[Simplex]] | None:
    return _bounded_complex(nx.find_cliques(graph), graph.nodes(), max_simplices)


def _sphere_collapse_certificate(
    complex_: SimplicialComplex,
    *,
    max_states: int,
) -> dict[str, Any] | None:
    """Find a top-facet deletion whose remaining complex collapses to a point."""
    facets = list(complex_.facet_set)
    if not facets or len({len(facet) for facet in facets}) != 1:
        return None
    simplices = complex_.all_simplices()
    if not complex_.base_set:
        return None
    ordered_facets = sorted(
        facets, key=lambda face: tuple(sorted(map(repr, face)))
    )
    for deleted_facet in ordered_facets:
        remaining = simplices - {deleted_facet}
        maximal = {
            face
            for face in remaining
            if not any(face < other for other in remaining)
        }
        source = SimplicialComplex(complex_.base_set, facet_set=maximal)
        vertices = sorted(
            (vertex for vertex in complex_.base_set if Simplex({vertex}) in remaining),
            key=repr,
        )
        if not vertices:
            continue
        target = SimplicialComplex({vertices[0]}, facet_set=[{vertices[0]}])
        result = relative_collapse(source, target, max_states=max_states)
        if (
            result.status is CollapseStatus.PROVED
            and verify_relative_collapse(source, target, result.pairs)
        ):
            certificate = {
                "dimension": len(deleted_facet) - 1,
                "deleted_facet": [
                    _encode_certificate_label(vertex)
                    for vertex in sorted(deleted_facet, key=repr)
                ],
                "collapse_pairs": [
                    [
                        [
                            _encode_certificate_label(vertex)
                            for vertex in sorted(free_face, key=repr)
                        ],
                        [
                            _encode_certificate_label(vertex)
                            for vertex in sorted(coface, key=repr)
                        ],
                    ]
                    for free_face, coface in result.pairs
                ],
                "collapse_certificate_replayed": True,
            }
            if verify_sphere_collapse_certificate(complex_, certificate):
                return certificate
    return None


def verify_sphere_collapse_certificate(
    complex_: SimplicialComplex,
    certificate: dict[str, Any],
) -> bool:
    """Replay a serialized top-facet deletion and collapse-to-a-point proof."""
    try:
        facets = list(complex_.facet_set)
        if not facets or len({len(facet) for facet in facets}) != 1:
            return False
        deleted_facet = Simplex(
            _decode_certificate_label(vertex)
            for vertex in certificate["deleted_facet"]
        )
        if deleted_facet not in complex_.facet_set:
            return False
        if certificate["dimension"] != len(deleted_facet) - 1:
            return False
        simplices = complex_.all_simplices() - {deleted_facet}
        maximal = {
            face
            for face in simplices
            if not any(face < other for other in simplices)
        }
        source = SimplicialComplex(complex_.base_set, facet_set=maximal)
        pairs = [
            (
                Simplex(_decode_certificate_label(vertex) for vertex in free_face),
                Simplex(_decode_certificate_label(vertex) for vertex in coface),
            )
            for free_face, coface in certificate["collapse_pairs"]
        ]
        current_simplices = source.all_simplices()
        for vertex in complex_.base_set:
            target = SimplicialComplex({vertex}, facet_set=[{vertex}])
            if target.all_simplices() <= current_simplices and verify_relative_collapse(
                source, target, pairs
            ):
                return True
        return False
    except (KeyError, TypeError, ValueError):
        return False


def analyze_h_operator_graph(
    graph: nx.Graph,
    *,
    max_cliques: int = 10_000,
    max_edges: int = 100_000,
    max_states: int = 1_000,
    max_simplices: int = 2_500,
    max_candidate_faces: int = 100_000,
    use_3_helly_certificate: bool = True,
    max_3helly_4sets: int = 100_000,
    use_compact_realization: bool = False,
    enable_sphere_certificate: bool = False,
    max_s_reduction_moves: int = 10_000,
) -> dict[str, Any]:
    """Analyze one graph without treating search limits as failure.

    The compact Proposition 7.5 realization is opt-in. Before enabling it in
    a larger census, cross-check it against ``K(H(G))`` on the desired input
    class using ``double_bound_clique_graph`` and ``nx.is_isomorphic``. By
    default, a complete maximal-clique 3-Helly test returns an affirmative
    theorem certificate before constructing ``H(G)`` or any clique graph.
    Disable this shortcut with ``use_3_helly_certificate=False``.
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
    if max_3helly_4sets < 0:
        raise ValueError("max_3helly_4sets must be nonnegative")
    if max_s_reduction_moves < 0:
        raise ValueError("max_s_reduction_moves must be nonnegative")

    base: dict[str, Any] = {
        "graph6": _graph6(graph),
        "order": graph.number_of_nodes(),
        "size": graph.number_of_edges(),
        "h_order": None,
        "h_size": None,
        "middle_order": None,
        "middle_size": None,
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
        "s_reduction_certificate": [],
        "s_reduction_certificate_replayed": False,
        "reduced_core_order": None,
        "reduced_core_size": None,
        "reduced_core_f_vector": None,
        "sphere_certificate": None,
        "sphere_search_diagnostics": None,
        "input_sphere_certificate": None,
        "input_sphere_search_diagnostics": None,
        "sphere_certificate_replayed": False,
        "three_helly_audit": None,
        "analysis": None,
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
            "use_3_helly_certificate": use_3_helly_certificate,
            "max_3helly_4sets": max_3helly_4sets,
            "use_compact_realization": use_compact_realization,
            "enable_sphere_certificate": enable_sphere_certificate,
            "max_s_reduction_moves": max_s_reduction_moves,
        },
    }
    if use_3_helly_certificate:
        certificate = maximal_clique_hypergraph_is_3_helly(
            graph,
            max_maximal_cliques=max_cliques,
            max_4sets=max_3helly_4sets,
        )
        vertex_indices = {
            vertex: index for index, vertex in enumerate(graph.nodes())
        }
        audit: dict[str, Any] = {
            "status": certificate.status.value,
            "exact_test": (
                certificate.status is not ThreeHellyStatus.INCONCLUSIVE_BUDGET
            ),
            "theorem": (
                "Dourado, Protti, and Szwarcfiter, EJC Dynamic Surveys DS17 "
                "(2009), Theorem 4.2, DOI 10.37236/38"
            ),
            "recognition_test": (
                "For every four-set W, maximal cliques meeting W in at least "
                "three vertices have nonempty total intersection, unless "
                "there are no such cliques."
            ),
            "maximal_cliques_count": certificate.maximal_cliques_count,
            "maximal_cliques_complete": certificate.maximal_cliques_complete,
            "four_sets_checked": certificate.four_sets_checked,
            "budgets": {
                "max_maximal_cliques": certificate.max_maximal_cliques,
                "max_4sets": certificate.max_4sets,
            },
            "witness": None,
        }
        if certificate.witness is not None:
            audit["witness"] = {
                "vertices": sorted(vertex_indices[v] for v in certificate.witness),
                "relevant_maximal_cliques": [
                    sorted(vertex_indices[v] for v in clique)
                    for clique in certificate.relevant_cliques
                ],
            }
        base["three_helly_audit"] = audit
        if certificate.status is ThreeHellyStatus.PROVED:
            base["status"] = PROVED_BY_3_HELLY
            return base

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
    base["h_order"] = h_order
    base["h_size"] = h_size
    base["middle_order"] = None if middle is None else middle.number_of_nodes()
    base["middle_size"] = None if middle is None else middle.number_of_edges()

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

    sphere_proof = False
    if enable_sphere_certificate:
        reduced, s_moves, reduction_complete = _s_reduce_graph(
            k2, max_s_reduction_moves
        )
        replayed = verify_s_reduction_certificate(k2, s_moves, reduced)
        base["s_reduction_certificate"] = _encode_s_reduction_certificate(s_moves)
        base["s_reduction_certificate_replayed"] = replayed and reduction_complete
        if not reduction_complete:
            base["resource_diagnostics"] = (
                f"S-reduction stopped at max_s_reduction_moves="
                f"{max_s_reduction_moves}; partial certificate retained."
            )
        elif not replayed:
            base["resource_diagnostics"] = (
                "Generated s-reduction certificate did not replay."
            )
        else:
            base["reduced_core_order"] = reduced.number_of_nodes()
            base["reduced_core_size"] = reduced.number_of_edges()
            bounded_core = _complex_from_clique_graph(reduced, max_simplices)
            if bounded_core is None:
                base["resource_diagnostics"] = (
                    "Reduced core simplex enumeration exceeded max_simplices."
                )
            else:
                core_complex, core_simplices = bounded_core
                base["reduced_core_f_vector"] = [
                    sum(len(simplex) == dimension + 1 for simplex in core_simplices)
                    for dimension in range(max(0, core_complex.dimension() + 1))
                ]
                sphere_certificate = _sphere_collapse_certificate(
                    core_complex, max_states=max_states
                )
                base["sphere_certificate"] = sphere_certificate
                if sphere_certificate is None:
                    base["sphere_search_diagnostics"] = (
                        "No replayable top-facet deletion collapse was found; "
                        "this is inconclusive."
                    )
                input_complex_result = _complex_from_clique_graph(
                    graph, max_simplices
                )
                input_sphere_certificate = (
                    None
                    if input_complex_result is None
                    else _sphere_collapse_certificate(
                        input_complex_result[0], max_states=max_states
                    )
                )
                base["input_sphere_certificate"] = input_sphere_certificate
                if input_sphere_certificate is None:
                    base["input_sphere_search_diagnostics"] = (
                        "No replayable top-facet deletion collapse was found "
                        "for the input complex; this is inconclusive."
                        if input_complex_result is not None
                        else "Input complex enumeration exceeded max_simplices."
                    )
                if (
                    sphere_certificate is not None
                    and input_sphere_certificate is not None
                    and sphere_certificate["dimension"]
                    == input_sphere_certificate["dimension"]
                    and replayed
                ):
                    base["sphere_certificate_replayed"] = True
                    sphere_proof = True

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
        if base["resource_diagnostics"] is None:
            base["resource_diagnostics"] = analysis["collapse_diagnostics"]
        else:
            base["resource_diagnostics"] += f" {analysis['collapse_diagnostics']}"
    else:
        base["status"] = INCONCLUSIVE_NO_COLLAPSE_CERTIFICATE

    if sphere_proof:
        base["status"] = PROVED_BY_SPHERE_CERTIFICATE
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
    use_3_helly_certificate: bool = True,
    max_3helly_4sets: int = 100_000,
    use_compact_realization: bool = False,
    enable_sphere_certificate: bool = False,
    max_s_reduction_moves: int = 10_000,
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
    if max_3helly_4sets < 0:
        raise ValueError("max_3helly_4sets must be nonnegative")
    if max_s_reduction_moves < 0:
        raise ValueError("max_s_reduction_moves must be nonnegative")
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
        "use_3_helly_certificate": use_3_helly_certificate,
        "max_3helly_4sets": max_3helly_4sets,
        "use_compact_realization": use_compact_realization,
        "enable_sphere_certificate": enable_sphere_certificate,
        "max_s_reduction_moves": max_s_reduction_moves,
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
                    use_3_helly_certificate=use_3_helly_certificate,
                    max_3helly_4sets=max_3helly_4sets,
                    use_compact_realization=use_compact_realization,
                    enable_sphere_certificate=enable_sphere_certificate,
                    max_s_reduction_moves=max_s_reduction_moves,
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
                    use_3_helly_certificate=use_3_helly_certificate,
                    max_3helly_4sets=max_3helly_4sets,
                    use_compact_realization=use_compact_realization,
                    enable_sphere_certificate=enable_sphere_certificate,
                    max_s_reduction_moves=max_s_reduction_moves,
                ),
            }

    results = [previous_records[index]["result"] for index in sorted(previous_records)]
    counts = {
        status: sum(record["status"] == status for record in results)
        for status in (
            PROVED_BY_3_HELLY,
            PROVED_BY_COLLAPSE,
            HOMOLOGY_MISMATCH_COUNTEREXAMPLE,
            INCONCLUSIVE_BUDGET,
            INCONCLUSIVE_NO_COLLAPSE_CERTIFICATE,
            INCONCLUSIVE_HOMOLOGY_MISMATCH_UNCONFIRMED,
            PROVED_BY_SPHERE_CERTIFICATE,
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
    parser.add_argument(
        "--no-3-helly-certificate",
        dest="use_3_helly_certificate",
        action="store_false",
        help="disable the early affirmative 3-Helly recognition shortcut",
    )
    parser.add_argument("--max-3helly-4sets", type=int, default=100_000)
    parser.set_defaults(use_3_helly_certificate=True)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--sphere-certificate", action="store_true")
    parser.add_argument("--max-s-reduction-moves", type=int, default=10_000)
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
        use_3_helly_certificate=args.use_3_helly_certificate,
        max_3helly_4sets=args.max_3helly_4sets,
        use_compact_realization=args.compact,
        enable_sphere_certificate=args.sphere_certificate,
        max_s_reduction_moves=args.max_s_reduction_moves,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))


if __name__ == "__main__":
    main()

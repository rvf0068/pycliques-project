"""Replayable conical/contractible edge reductions and vertex-star wedges."""

import base64
import gzip
from collections import Counter
from dataclasses import replace
from itertools import combinations

import networkx as nx
import pytest
from pycombtop import (
    FlagMove,
    FlagReductionCertificate,
    reduce_flag_graph,
    reduced_star_wedge_certificate,
    verify_flag_reduction,
    verify_reduced_star_wedge,
    verify_vertex_star_wedge,
    vertex_star_wedge_certificate,
)

# A 54-vertex, 491-edge reduced flag complex from K^4(Susp(C_9)).
# The fixture verifies the structural wedge independently of the huge
# edge-reduction search.
CORE_G6_GZIP_B64 = (
    "H4sICE8X/moC/3N1c3BfYzlfY29udHJhY3RpYmxlX2xpbmtfY29yZS5nNgAVTm1PwkAM/"
    "u6fei4c0J1yN8iGixp2smxuxIRj6Dm3YX+7pUmTts9bY7ceyDEPr3668dp66M0n8xfgo"
    "Qj3+gCohXIeeJa1kp6ExgOwKIHkTItYkKCuDFDL888vF9DC6ufg3lIkgJ7dC5uIxAq/t"
    "99lNQF1CqVv8Sin0l6yVSMq45/aANqSHUS4FZeGH/PifV5BmS5I4qE5XXCQqc5GQI1Fv"
    "TPyIM/53jArYJmNnv8Eyq0kJydSO7rGmJLumyw8/AMJGXRA8QAAAA=="
)


def c9_core():
    data = gzip.decompress(base64.b64decode(CORE_G6_GZIP_B64))
    return nx.from_graph6_bytes(data.strip())


def test_cone_edge_and_fold_replay():
    triangle = nx.complete_graph(3)
    reduced = nx.path_graph(3)
    proof = FlagReductionCertificate(
        moves=(FlagMove("cone_edge", 0, 2, 1),),
        remaining_vertices=tuple(triangle),
        remaining_edges=tuple(reduced.edges),
    )
    assert verify_flag_reduction(triangle, proof)
    assert not verify_flag_reduction(
        triangle, replace(proof, moves=(FlagMove("cone_edge", 0, 2, 0),))
    )
    path = nx.path_graph(3)
    folded = nx.path_graph([1, 2])
    fold = FlagReductionCertificate(
        moves=(FlagMove("fold_vertex", 0, 1),),
        remaining_vertices=tuple(folded),
        remaining_edges=tuple(folded.edges),
    )
    assert verify_flag_reduction(path, fold)


def test_contractible_link_not_necessarily_cone():
    # Common neighborhood is the path P4: contractible, but no cone apex.
    graph = nx.path_graph(4)
    graph.add_edge(4, 5)
    graph.add_edges_from((u, v) for u in (4, 5) for v in range(4))
    after = graph.copy()
    after.remove_edge(4, 5)
    proof = FlagReductionCertificate(
        moves=(FlagMove("contractible_edge", 4, 5),),
        remaining_vertices=tuple(after),
        remaining_edges=tuple(after.edges),
    )
    assert verify_flag_reduction(graph, proof)
    noncontractible = nx.cycle_graph(4)
    noncontractible.add_edge(4, 5)
    noncontractible.add_edges_from((u, v) for u in (4, 5) for v in range(4))
    assert not verify_flag_reduction(noncontractible, proof)


def _explicit_critical_faces(graph):
    """Independent augmented matching on all explicit clique faces."""
    labels = sorted(graph)
    indexed = nx.relabel_nodes(graph, {v: i for i, v in enumerate(labels)})
    faces = {0}
    faces.update(
        sum(1 << i for i in face) for face in nx.enumerate_all_cliques(indexed)
    )
    for pivot in range(len(indexed)):
        bit = 1 << pivot
        matching = set()
        for face in faces:
            if not face & bit and face | bit in faces:
                matching.add(face)
                matching.add(face | bit)
        faces.difference_update(matching)
    return dict(Counter(face.bit_count() - 1 for face in faces))


def test_core_star_wedge_and_tamper_detection():
    graph = c9_core()
    assert (len(graph), graph.number_of_edges()) == (54, 491)
    certificate = vertex_star_wedge_certificate(graph, candidates=[22])
    assert certificate is not None
    assert certificate.deletion_critical == {0: 1, 1: 1, 6: 1}
    assert certificate.link_critical == {1: 1, 5: 1}
    deleted = graph.copy()
    deleted.remove_node(22)
    link = graph.subgraph(graph[22])
    assert _explicit_critical_faces(deleted) == certificate.deletion_critical
    assert _explicit_critical_faces(link) == certificate.link_critical
    assert certificate.deletion_spheres == {6: 1}
    assert certificate.link_spheres == {1: 1, 5: 1}
    assert certificate.spheres == {2: 1, 6: 2}
    assert verify_vertex_star_wedge(graph, certificate)
    assert not verify_vertex_star_wedge(
        graph, replace(certificate, spheres={2: 1, 6: 3})
    )
    altered = graph.copy()
    altered.remove_edge(*next(iter(altered.edges)))
    assert not verify_vertex_star_wedge(altered, certificate)


def test_small_graph_fail_closed_limits_and_loops():
    graph = nx.complete_graph(4)
    assert reduce_flag_graph(graph, max_moves=1) is None
    result = reduce_flag_graph(graph, cone_passes=2)
    assert result is not None
    core, certificate = result
    assert verify_flag_reduction(graph, certificate)
    assert len(core) <= len(graph)
    looped = nx.Graph([(0, 0), (0, 1)])
    assert reduce_flag_graph(looped) is None
    assert vertex_star_wedge_certificate(looped) is None
    mixed = nx.Graph()
    mixed.add_edges_from([(1, "two"), ("two", (3,)), ((3,), 1)])
    reduced = reduce_flag_graph(mixed, cone_passes=2)
    assert reduced is not None
    assert verify_flag_reduction(mixed, reduced[1])


def _clique_graph(graph):
    cliques = [frozenset(q) for q in nx.find_cliques(graph)]
    result = nx.Graph()
    result.add_nodes_from(range(len(cliques)))
    incidence = {v: [] for v in graph}
    for j, q in enumerate(cliques):
        for v in q:
            incidence[v].append(j)
    for ids in incidence.values():
        result.add_edges_from(combinations(ids, 2))
    return result


@pytest.mark.slow
def test_full_c9_fourth_iterate_homotopy_certificate():
    graph = nx.cycle_graph(9)
    graph.add_nodes_from((9, 10))
    graph.add_edges_from((u, v) for u in range(9) for v in (9, 10))
    for _ in range(4):
        graph = _clique_graph(graph)
    assert (len(graph), graph.number_of_edges()) == (803, 183_222)
    complement = nx.complement(graph)
    apices = [
        u
        for u in complement
        if complement.degree[u] == 1
        and complement.degree[next(iter(complement[u]))] == 1
    ]
    assert len(apices) == 2
    graph.remove_nodes_from(apices)
    factor = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    assert (len(factor), factor.number_of_edges()) == (801, 181_620)
    certificate = reduced_star_wedge_certificate(factor, candidates=[22])
    assert certificate is not None
    assert certificate.spheres == {2: 1, 6: 2}
    assert verify_reduced_star_wedge(factor, certificate)
    core = certificate.reduction
    assert len(core.remaining_vertices) == 54
    assert len(core.remaining_edges) == 491

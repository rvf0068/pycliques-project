"""Regression tests for antipodal domains and partial-checkpoint integrity."""

import json

import networkx as nx
import pytest
from pycliques.octahedral_blind import screen_partial_octahedral_retraction
from pycliques.octahedral_joint_fibers import (
    Masks,
    State,
    check_partial_state,
    load_partial_state,
    save_partial_state,
)


def test_possible_image_requires_nonadjacency_to_its_antipode():
    for missing, image in ((1, 0), (0, 1)):
        graph = nx.complete_multipartite_graph(2, 2, 2)
        graph.add_node(6)
        graph.add_edges_from((6, v) for v in range(6) if v != missing)
        pairs = ((0, 1), (2, 3), (4, 5))
        report = screen_partial_octahedral_retraction(graph, pairs)
        assert report.possible
        assert report.singleton_domains == 7
        projection = {v: v for v in range(6)}
        projection[6] = image
        target = graph.subgraph(range(6))
        for a, b in graph.edges:
            assert projection[a] == projection[b] or target.has_edge(
                projection[a], projection[b]
            )


def test_opposite_forced_images_conflict_on_an_edge():
    graph = nx.complete_multipartite_graph(2, 2, 2)
    graph.add_nodes_from((6, 7))
    graph.add_edges_from((6, v) for v in range(6) if v != 1)
    graph.add_edges_from((7, v) for v in range(6) if v != 0)
    graph.add_edge(6, 7)
    report = screen_partial_octahedral_retraction(graph, ((0, 1), (2, 3), (4, 5)))
    assert not report.possible
    assert report.reason == "FORCED_EDGE"


def test_unmatched_anchor_pairs_and_fibers_rejected(tmp_path):
    graph = nx.complete_multipartite_graph(2, 2, 2, 2)
    masks = Masks(graph)
    fibers = (((1 << 0), (1 << 1)),)
    common = masks.closed[0] & masks.closed[1] & masks.closed[2] & masks.closed[3]
    bad = State(((0, 1), (2, 3)), fibers, 3, common)
    with pytest.raises(ValueError, match="equal numbers of pairs and fibers"):
        check_partial_state(masks, bad)
    valid_common = masks.closed[0] & masks.closed[1]
    valid = State(((0, 1),), fibers, 3, valid_common)
    path = tmp_path / "partial.json"
    save_partial_state(path, masks, valid)
    data = json.loads(path.read_text())
    data["pairs"].append([2, 3])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="equal numbers of pairs and fibers"):
        load_partial_state(path, masks)
    data = json.loads(path.read_text())
    data["pairs"].pop()
    data["fibers"].append([[2], [3]])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="equal numbers of pairs and fibers"):
        load_partial_state(path, masks)

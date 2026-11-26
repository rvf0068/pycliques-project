"""Regression tests for certified topological products."""

from collections import Counter
from dataclasses import replace
from itertools import combinations

import networkx as nx
import pytest
from pycombtop import (
    MooreSpaceType,
    ProductType,
    SphereType,
    WedgeType,
    product_type_certificate,
    verify_product_type,
)


@pytest.mark.parametrize("lengths", [(5, 7), (4, 4, 4), (5, 5, 5)])
def test_tori_without_coordinate_labels(lengths):
    graph = nx.cycle_graph(lengths[0])
    for n in lengths[1:]:
        graph = nx.strong_product(graph, nx.cycle_graph(n))
    graph = nx.convert_node_labels_to_integers(graph)
    cert = product_type_certificate(graph)
    assert cert is not None
    assert cert.homotopy_type == ProductType((SphereType(1),) * len(lengths))
    assert verify_product_type(graph, cert)
    assert cert.verdict == r"\(" + r"\times ".join(["S^{1}"] * len(lengths)) + r"\)"


def test_spheres_wedges_and_contractible_factors():
    circle = nx.cycle_graph(5)
    octahedron = nx.complete_multipartite_graph(2, 2, 2)
    graph = nx.strong_product(circle, octahedron)
    cert = product_type_certificate(graph)
    assert cert is not None
    assert Counter(cert.homotopy_type.factors) == {
        SphereType(1): 1,
        SphereType(2): 1,
    }
    assert verify_product_type(graph, cert)

    graph = nx.strong_product(circle, nx.complete_bipartite_graph(2, 3))
    cert = product_type_certificate(graph)
    assert cert is not None
    assert Counter(cert.homotopy_type.factors) == {
        SphereType(1): 1,
        WedgeType(((SphereType(1), 2),)): 1,
    }
    assert verify_product_type(graph, cert)

    graph = nx.strong_product(circle, nx.path_graph(3))
    cert = product_type_certificate(graph)
    assert cert is not None
    assert cert.homotopy_type == ProductType((SphereType(1),))
    assert verify_product_type(graph, cert)


def test_projective_plane_times_circle():
    triangles = (
        (1, 2, 3),
        (1, 2, 4),
        (1, 3, 5),
        (1, 4, 6),
        (1, 5, 6),
        (2, 3, 6),
        (2, 4, 5),
        (2, 5, 6),
        (3, 4, 5),
        (3, 4, 6),
    )
    faces = {
        frozenset(face)
        for triangle in triangles
        for k in range(1, 4)
        for face in combinations(triangle, k)
    }
    # The order complex of the face poset is a flag triangulation of RP2.
    projective = nx.Graph()
    projective.add_nodes_from(faces)
    projective.add_edges_from((a, b) for a in faces for b in faces if a < b)
    graph = nx.strong_product(projective, nx.cycle_graph(4))
    cert = product_type_certificate(graph)
    assert cert is not None
    assert Counter(cert.homotopy_type.factors) == {
        MooreSpaceType(2): 1,
        SphereType(1): 1,
    }
    assert verify_product_type(graph, cert)
    assert not verify_product_type(graph, cert, max_simplices=1)


def test_product_limits_and_replay():
    graph = nx.strong_product(nx.cycle_graph(5), nx.cycle_graph(5))
    assert product_type_certificate(graph, max_vertices=24) is None
    assert product_type_certificate(graph, max_simplices=1) is None
    assert product_type_certificate(nx.cycle_graph(5)) is None
    for kwargs in (
        {"max_vertices": 0},
        {"max_simplices": 0},
        {"max_partitions": 0},
    ):
        with pytest.raises(ValueError):
            product_type_certificate(graph, **kwargs)
    # Only factors need to fit the simplex budget.
    cert = product_type_certificate(graph, max_simplices=11)
    assert cert is not None
    assert verify_product_type(graph, cert, max_simplices=11)
    assert not verify_product_type(graph, replace(cert, left=WedgeType()))
    assert not verify_product_type(graph, replace(cert, right=None))
    assert not verify_product_type(graph, None)
    with pytest.raises(ValueError):
        verify_product_type(graph, cert, max_simplices=0)
    graph.remove_edge((0, 0), (1, 1))
    assert not verify_product_type(graph, cert)

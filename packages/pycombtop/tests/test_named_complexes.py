"""Named complexes: known face counts, surface links, and topology."""

from collections import defaultdict

import networkx as nx
import pytest
from pycliques import matching_graph, octahedron
from pycombtop import (
    SimplicialComplex,
    betti_numbers_sc,
    chessboard_complex,
    clique_complex,
    cross_polytope_boundary,
    independence_complex,
    klein_bottle_complex,
    matching_complex,
    projective_plane_complex,
    simplex_boundary,
    sphere_complex,
    torus_complex,
    vietoris_rips_complex,
)


def test_independence_matching_and_chessboard_complexes():
    g = nx.path_graph(4)
    assert independence_complex(g) == clique_complex(nx.complement(g))
    assert matching_graph(g).number_of_nodes() == 3
    assert matching_graph(g).number_of_edges() == 1
    assert matching_complex(g).dimension() == 1
    assert matching_complex(5) == clique_complex(matching_graph(5))
    assert matching_graph(5).number_of_edges() == 15
    assert chessboard_complex(2, 3).dimension() == 1
    assert betti_numbers_sc(chessboard_complex(2, 3)) == [0, 1]
    assert len(chessboard_complex(3, 4).facet_set) == 24
    for complex_ in (
        independence_complex(nx.Graph()),
        matching_complex(0),
        chessboard_complex(0, 3),
    ):
        assert complex_ == SimplicialComplex(set(), [set()])


def test_named_spheres():
    for d in range(4):
        x = sphere_complex(d)
        assert x.dimension() == d
        assert len(x.vertex_set) == d + 2
        assert betti_numbers_sc(x) == [0] * d + [1]
        y = cross_polytope_boundary(d + 1)
        assert y == clique_complex(octahedron(d + 1))
        assert len(y.vertex_set) == 2 * (d + 1)
        assert betti_numbers_sc(y) == [0] * d + [1]
    assert sphere_complex(-1).facet_set == cross_polytope_boundary(0).facet_set
    assert simplex_boundary(0).facet_set == set()


def _orientable_closed_surface(x):
    incidence = defaultdict(list)
    for face in x.facet_set:
        triangle = sorted(face)
        for i in range(3):
            edge = tuple(v for j, v in enumerate(triangle) if j != i)
            incidence[edge].append((face, (-1) ** i))
    assert all(len(items) == 2 for items in incidence.values())
    adjacency = defaultdict(list)
    for (a, sa), (b, sb) in incidence.values():
        adjacency[a].append((b, -sa * sb))
        adjacency[b].append((a, -sa * sb))
    start = next(iter(x.facet_set))
    signs = {start: 1}
    todo = [start]
    while todo:
        a = todo.pop()
        for b, sign in adjacency[a]:
            target = signs[a] * sign
            if b in signs:
                if signs[b] != target:
                    return False
            else:
                signs[b] = target
                todo.append(b)
    assert set(signs) == x.facet_set
    return True


@pytest.mark.parametrize(
    "constructor,counts,betti,orientable",
    [
        (projective_plane_complex, (6, 15, 10), [0, 1, 1], False),
        (torus_complex, (16, 48, 32), [0, 2, 1], True),
        (klein_bottle_complex, (16, 48, 32), [0, 2, 1], False),
    ],
)
def test_closed_surface_triangulations(constructor, counts, betti, orientable):
    x = constructor()
    assert tuple(len(list(x.faces_of_dimension(d))) for d in range(3)) == counts
    assert nx.is_connected(x.one_skeleton_graph())
    for vertex in x.vertex_set:
        link = x.link(vertex).one_skeleton_graph()
        assert len(link) >= 3
        assert nx.is_connected(link)
        assert all(d == 2 for _, d in link.degree())
    assert _orientable_closed_surface(x) is orientable
    assert betti_numbers_sc(x) == betti


def test_rips_threshold_and_duplicate_points():
    points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert betti_numbers_sc(vietoris_rips_complex(points, 1)) == [0, 1]
    assert vietoris_rips_complex(points, 2).dimension() == 3
    x = vietoris_rips_complex([(0,), (0,), (1,)], 0)
    assert x.facet_set == {frozenset({0, 1}), frozenset({2})}
    assert vietoris_rips_complex([], 1).facet_set == {frozenset()}


def test_invalid_inputs():
    for constructor in (simplex_boundary, cross_polytope_boundary, matching_complex):
        with pytest.raises((ValueError, nx.NetworkXError)):
            constructor(-1)
    with pytest.raises(ValueError):
        sphere_complex(-2)
    with pytest.raises(ValueError):
        chessboard_complex(2, -1)
    for graph in (nx.DiGraph(), nx.MultiGraph(), nx.Graph([(0, 0)])):
        with pytest.raises(ValueError, match="simple"):
            matching_graph(graph)
        with pytest.raises(ValueError, match="simple"):
            independence_complex(graph)
    for epsilon in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="epsilon"):
            vietoris_rips_complex([], epsilon)
    with pytest.raises(ValueError, match="dimension"):
        vietoris_rips_complex([(1,), (1, 2)], 1)
    with pytest.raises(ValueError, match="finite"):
        vietoris_rips_complex([(float("nan"),)], 1)

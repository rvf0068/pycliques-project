"""Regression tests for established clique-graph facts and fixtures."""

import networkx as nx
from pycliques.cliques import clique_graph
from pycliques.dominated import find_dominated_vertex
from pycliques.helly import is_clique_helly
from pycliques.homotopy_invariance import theorem15_hypothesis_holds
from pycliques.named import (
    collapse_obstruction_fixture,
    dominated_vertex_free_non_helly,
    octahedron,
    suspension_of_cycle,
)
from pycliques.retractions import has_induced
from pycliques.small import CliqueSequence, _make_clique_retraction_test
from pycliques.surfaces import open_neighborhood
from pycombtop.homotopy_type import WedgeOfSpheres, collapse, homotopy_type_with_verdict
from pycombtop.simplex import clique_complex, relative_betti_numbers


def _assert_exact_type(graph, expected):
    """Assert an exact homotopy verdict has the expected wedge type."""
    verdict = homotopy_type_with_verdict(graph)
    assert verdict.is_exact is True
    assert verdict.wedge == expected


def test_octahedron_and_clique_graph_have_expected_spheres():
    """The standard octahedron is S^2 while its clique graph is S^3."""
    _assert_exact_type(octahedron(3), WedgeOfSpheres.sphere(2))
    _assert_exact_type(clique_graph(octahedron(3)), WedgeOfSpheres.sphere(3))


def test_suspension_of_c5_iterates_have_expected_types():
    """Suspension of C5 is good but not very good."""
    graph = suspension_of_cycle(5)
    _assert_exact_type(graph, WedgeOfSpheres.sphere(2))
    first = clique_graph(graph)
    second = clique_graph(first)
    _assert_exact_type(first, WedgeOfSpheres.sphere(2))
    _assert_exact_type(second, WedgeOfSpheres.sphere(3))


def test_suspension_of_c5_is_good_and_clique_divergent():
    """Goodness and clique divergence are independent properties."""
    classifier = _make_clique_retraction_test(
        nx.complement(nx.cycle_graph(10)), "retracts to Comp(C_10)"
    )
    result = classifier(CliqueSequence(suspension_of_cycle(5)))
    assert result is not None
    assert result[0].name == "DIVERGENT"


def test_smallest_non_clique_helly_graph_has_k4_clique_graph():
    """The documented six-vertex example has K(G) isomorphic to K4."""
    graph = nx.Graph(
        [
            (0, 1),
            (1, 2),
            (0, 2),
            (2, 3),
            (3, 4),
            (2, 4),
            (4, 5),
            (5, 0),
            (4, 0),
        ]
    )
    kg = clique_graph(graph)
    assert not is_clique_helly(graph)
    assert nx.is_isomorphic(kg, nx.complete_graph(4))
    _assert_exact_type(kg, WedgeOfSpheres.contractible())


def test_fixture_1_has_documented_invariants():
    """Fixture H?qdvbU defeats dominated-vertex and contractible-link strategies."""
    graph = dominated_vertex_free_non_helly()
    kg = clique_graph(graph)
    assert graph.order() == 9
    assert not is_clique_helly(graph)
    assert is_clique_helly(kg)
    assert find_dominated_vertex(graph) is None
    assert all(not nx.is_connected(open_neighborhood(graph, v)) for v in graph)


def test_fixture_2_has_documented_invariants():
    """Fixture HUZv~zz carries the O4 obstruction and satisfies Theorem 15."""
    graph = collapse_obstruction_fixture()
    kg = clique_graph(graph)
    assert graph.order() == 9
    assert kg.order() == 16
    assert has_induced(kg, octahedron(4))
    assert theorem15_hypothesis_holds(graph)[0] is True

    relative = relative_betti_numbers(
        nx.find_cliques(kg),
        lambda simplex: (
            not simplex or bool(set.intersection(*(set(clique) for clique in simplex)))
        ),
    )
    assert all(value == 0 for value in relative.values())

    # The ordinary full clique complex remains collapsible; the obstruction
    # concerns the restricted violation-set collapse strategy.
    assert len(collapse(clique_complex(kg)).vertex_set) == 1

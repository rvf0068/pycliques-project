from itertools import combinations
from math import comb

import networkx as nx
import pycliques.poset_homotopy as poset_homotopy
import pytest
from pycliques import (
    Theorem57Status,
    ThreeHellyStatus,
    closed_clique_intersections,
    double_bound_graph,
    h_closure_from_neighbors,
    maximal_clique_hypergraph_is_3_helly,
    strong_collapse_certificate,
    theorem57_hypothesis_holds,
    verify_strong_collapse_certificate,
)
from pycliques.cliques import clique_graph
from pycliques.named import octahedron


def test_three_helly_is_trivial_through_three_vertices():
    graphs = [graph for graph in nx.graph_atlas_g() if graph.number_of_nodes() <= 3]
    graphs.append(nx.empty_graph(0))

    for graph in graphs:
        result = maximal_clique_hypergraph_is_3_helly(
            graph, max_maximal_cliques=0, max_4sets=0
        )
        assert result.status is ThreeHellyStatus.PROVED
        assert result.four_sets_checked == 0
        assert result.maximal_cliques_count == 0
        assert result.maximal_cliques_complete is False


@pytest.mark.parametrize(
    "graph",
    [
        nx.complete_graph(6),
        *(nx.cycle_graph(order) for order in range(4, 9)),
        *(nx.path_graph(order) for order in range(4, 9)),
    ],
)
def test_three_helly_complete_cycles_and_triangle_free_paths(graph):
    result = maximal_clique_hypergraph_is_3_helly(graph)

    assert result.status is ThreeHellyStatus.PROVED
    assert result.four_sets_checked == comb(graph.number_of_nodes(), 4)
    assert result.maximal_cliques_complete is True
    assert result.maximal_cliques_count == len(list(nx.find_cliques(graph)))


@pytest.mark.parametrize("dimension", [4, 5])
def test_octahedra_fail_three_helly_without_claiming_conjecture_failure(dimension):
    result = maximal_clique_hypergraph_is_3_helly(octahedron(dimension))

    assert result.status is ThreeHellyStatus.OBSTRUCTION
    assert result.witness is not None
    assert len(result.witness) == 4
    assert result.relevant_cliques
    assert not set.intersection(*(set(clique) for clique in result.relevant_cliques))


def test_three_helly_is_weaker_than_edge_intersection_helly():
    a = {"a", "b"}
    b = {"c", "d"}
    c = {"e", "f"}
    cliques = (
        a | b | {"p1"},
        a | c | {"p2"},
        b | c | {"p3"},
        a | b | c,
    )
    graph = nx.Graph()
    graph.add_nodes_from(set().union(*cliques))
    for clique in cliques:
        graph.add_edges_from(combinations(clique, 2))
    maximal_cliques = {frozenset(clique) for clique in nx.find_cliques(graph)}
    assert maximal_cliques == {frozenset(clique) for clique in cliques}

    first_three = cliques[:3]
    pair_intersections = [
        first_three[first] & first_three[second]
        for first, second in combinations(range(3), 2)
    ]
    assert all(pair_intersections)
    assert not set.intersection(*(set(clique) for clique in first_three))
    result = maximal_clique_hypergraph_is_3_helly(graph)
    assert result.status is ThreeHellyStatus.PROVED


def test_three_helly_recognition_matches_brute_force_on_atlas_through_six():
    graphs = [
        graph
        for graph in nx.graph_atlas_g()
        if graph.number_of_nodes() > 0 and graph.number_of_nodes() <= 6
    ]
    assert len(graphs) == 208

    for graph in graphs:
        maximal_cliques = [frozenset(clique) for clique in nx.find_cliques(graph)]
        direct_is_3_helly = True
        for family_size in range(1, len(maximal_cliques) + 1):
            for family in combinations(maximal_cliques, family_size):
                total_intersection = set.intersection(
                    *(set(clique) for clique in family)
                )
                all_small_intersections_nonempty = all(
                    set.intersection(*(set(clique) for clique in subfamily))
                    for subfamily_size in range(1, min(3, family_size) + 1)
                    for subfamily in combinations(family, subfamily_size)
                )
                if all_small_intersections_nonempty and not total_intersection:
                    direct_is_3_helly = False
                    break
            if not direct_is_3_helly:
                break

        result = maximal_clique_hypergraph_is_3_helly(graph)
        assert (result.status is ThreeHellyStatus.PROVED) is direct_is_3_helly


@pytest.mark.parametrize(
    ("graph", "limits", "expected_status", "expected_checked"),
    [
        (
            nx.complete_graph(4),
            {"max_maximal_cliques": 0},
            ThreeHellyStatus.INCONCLUSIVE_BUDGET,
            0,
        ),
        (
            nx.complete_graph(4),
            {"max_maximal_cliques": 1},
            ThreeHellyStatus.PROVED,
            1,
        ),
        (
            nx.path_graph(4),
            {"max_4sets": 0},
            ThreeHellyStatus.INCONCLUSIVE_BUDGET,
            0,
        ),
        (
            nx.path_graph(4),
            {"max_4sets": 1},
            ThreeHellyStatus.PROVED,
            1,
        ),
    ],
)
def test_three_helly_budget_boundaries(
    graph, limits, expected_status, expected_checked
):
    result = maximal_clique_hypergraph_is_3_helly(graph, **limits)

    assert result.status is expected_status
    assert result.four_sets_checked == expected_checked


def test_three_helly_requires_complete_maximal_clique_enumeration():
    result = maximal_clique_hypergraph_is_3_helly(
        nx.path_graph(4), max_maximal_cliques=1
    )

    assert result.status is ThreeHellyStatus.INCONCLUSIVE_BUDGET
    assert result.maximal_cliques_count == 1
    assert result.four_sets_checked == 0


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_three_helly_rejects_invalid_budgets(budget):
    with pytest.raises(ValueError):
        maximal_clique_hypergraph_is_3_helly(nx.complete_graph(4), max_4sets=budget)


def test_three_helly_supports_noninteger_labels():
    graph = nx.relabel_nodes(
        octahedron(4), {vertex: ("vertex", vertex) for vertex in octahedron(4)}
    )

    result = maximal_clique_hypergraph_is_3_helly(graph)

    assert result.status is ThreeHellyStatus.OBSTRUCTION
    assert result.witness is not None
    assert all(isinstance(vertex, tuple) for vertex in result.witness)
    assert all(
        isinstance(vertex, tuple)
        for clique in result.relevant_cliques
        for vertex in clique
    )


def _direct_h_closure(graph, complete):
    complete = frozenset(complete)
    containing = [
        frozenset(maximal)
        for maximal in nx.find_cliques(graph)
        if complete <= frozenset(maximal)
    ]
    return frozenset.intersection(*containing)


def test_double_bound_graph_builds_full_nonempty_face_poset():
    graph = nx.complete_graph(3)

    double_bound = double_bound_graph(graph)

    assert double_bound is not None
    assert double_bound.number_of_nodes() == 7
    assert double_bound.number_of_edges() == 15


def test_double_bound_graph_budget_exits_and_validates():
    graph = nx.complete_graph(3)

    assert double_bound_graph(graph, max_faces=6) is None
    assert double_bound_graph(graph, max_candidates=0) is None
    assert double_bound_graph(graph, max_edges=11) is None
    assert double_bound_graph(nx.empty_graph(0)).number_of_nodes() == 0
    for budget in ("max_faces", "max_candidates", "max_edges"):
        with pytest.raises(ValueError):
            double_bound_graph(graph, **{budget: -1})


def test_double_bound_graph_supports_noninteger_labels_and_disconnected_graphs():
    graph = nx.Graph()
    graph.add_nodes_from([("a", 1), "b", "isolated"])
    graph.add_edge(("a", 1), "b")

    double_bound = double_bound_graph(graph)

    assert double_bound is not None
    assert set(double_bound) == {
        frozenset({("a", 1)}),
        frozenset({"b"}),
        frozenset({"isolated"}),
        frozenset({("a", 1), "b"}),
    }


def test_h_closure_from_neighbors_matches_direct_definition_on_atlas():
    graph_count = 0
    complete_count = 0
    for graph in nx.graph_atlas_g():
        if not graph or graph.number_of_nodes() > 6:
            continue
        graph_count += 1
        for complete in nx.enumerate_all_cliques(graph):
            assert h_closure_from_neighbors(graph, complete) == _direct_h_closure(
                graph, complete
            )
            complete_count += 1

    assert graph_count == 208
    assert complete_count == 3196


@pytest.mark.parametrize(
    ("graph", "complete"),
    [
        (nx.path_graph(3), ()),
        (nx.path_graph(3), (0, 3)),
        (nx.path_graph(3), (0, 2)),
    ],
)
def test_h_closure_from_neighbors_rejects_invalid_completes(graph, complete):
    with pytest.raises(ValueError):
        h_closure_from_neighbors(graph, complete)


def test_closed_clique_intersections_are_exact_and_nonempty():
    graph = nx.path_graph(3)

    result = closed_clique_intersections(graph)

    assert result.is_exact is True
    assert result.is_limited is False
    assert set(result.closed_sets) == {
        frozenset({0, 1}),
        frozenset({1}),
        frozenset({1, 2}),
    }


@pytest.mark.parametrize(
    "limits",
    [
        {"max_maximal_cliques": 0},
        {"max_closed_sets": 0},
    ],
)
def test_closed_clique_intersections_reports_inconclusive_limits(limits):
    result = closed_clique_intersections(nx.path_graph(3), **limits)

    assert result.is_exact is False
    assert result.is_limited is True


def test_closed_clique_intersections_empty_graph_is_exact():
    result = closed_clique_intersections(nx.empty_graph(0))

    assert result.is_exact is True
    assert result.closed_sets == ()


@pytest.mark.parametrize(
    "budget",
    [
        {"max_maximal_cliques": -1},
        {"max_closed_sets": -1},
    ],
)
def test_closed_clique_intersections_rejects_negative_limits(budget):
    with pytest.raises(ValueError):
        closed_clique_intersections(nx.path_graph(3), **budget)


def test_strong_collapse_certificate_is_verified_independently():
    facets = ({0, 1, 2}, {0, 1, 3})

    certificate = strong_collapse_certificate(facets)

    assert certificate is not None
    assert verify_strong_collapse_certificate(facets, certificate)
    assert not verify_strong_collapse_certificate(facets, ((0, 2),))
    assert strong_collapse_certificate(({0, 1}, {1, 2}, {0, 2})) is None


def test_strong_collapse_certificate_supports_noninteger_labels():
    facets = ({("a", 1), "b"}, {("a", 1), "c"})

    certificate = strong_collapse_certificate(facets)

    assert certificate is not None
    assert verify_strong_collapse_certificate(facets, certificate)


def test_theorem57_driver_reports_proof_and_replays_certificate():
    result = theorem57_hypothesis_holds(nx.empty_graph(1))

    assert result.status is Theorem57Status.PROVED
    assert result.checked_h_images == 1
    assert len(result.certificates) == 1
    for h_image, moves in result.certificates.items():
        facets = tuple(frozenset(clique) for clique in h_image)
        assert verify_strong_collapse_certificate(facets, moves)


def test_theorem57_driver_reports_budget_exits_as_inconclusive():
    result = theorem57_hypothesis_holds(
        nx.complete_graph(3),
        max_faces=0,
    )

    assert result.status is Theorem57Status.INCONCLUSIVE
    assert result.checked_h_images == 0


def test_theorem57_nonzero_exact_homology_is_a_counterexample(monkeypatch):
    facets = (
        frozenset({0, 1}),
        frozenset({1, 2}),
        frozenset({0, 2}),
    )
    intersection_graph = nx.complete_graph(facets)
    monkeypatch.setattr(
        poset_homotopy,
        "double_bound_graph",
        lambda *_args, **_kwargs: nx.Graph(),
    )
    monkeypatch.setattr(
        poset_homotopy,
        "clique_graph",
        lambda _graph: intersection_graph,
    )

    result = theorem57_hypothesis_holds(nx.empty_graph(0))

    assert result.status is Theorem57Status.COUNTEREXAMPLE
    assert result.witness == frozenset(facets)
    assert "exact rational homology" in result.reason


def test_theorem57_counterexample_count_includes_earlier_unknown_images(
    monkeypatch,
):
    earlier = frozenset({frozenset({0, 1}), frozenset({1, 2})})
    obstructed = frozenset({frozenset({3, 4}), frozenset({4, 5}), frozenset({3, 5})})
    intersections = poset_homotopy.ClosedCliqueIntersections(
        (earlier, obstructed),
        is_exact=True,
    )
    monkeypatch.setattr(
        poset_homotopy,
        "double_bound_graph",
        lambda *_args, **_kwargs: nx.Graph(),
    )
    monkeypatch.setattr(
        poset_homotopy,
        "clique_graph",
        lambda _graph: nx.Graph(),
    )
    monkeypatch.setattr(
        poset_homotopy,
        "closed_clique_intersections",
        lambda *_args, **_kwargs: intersections,
    )
    monkeypatch.setattr(
        poset_homotopy,
        "strong_collapse_certificate",
        lambda _facets: None,
    )
    monkeypatch.setattr(
        "pycombtop.simplex.relative_betti_numbers",
        lambda facets, _membership, *, method: (
            {0: 1}
            if frozenset(frozenset(face) for face in facets) == earlier
            else {0: 1, 1: 1}
        ),
    )

    result = theorem57_hypothesis_holds(nx.empty_graph(0))

    assert result.status is Theorem57Status.COUNTEREXAMPLE
    assert result.checked_h_images == 2
    assert result.witness == obstructed


def test_theorem57_o4_certifies_every_closed_h_image():
    graph = octahedron(4)
    double_bound = double_bound_graph(graph)
    assert double_bound is not None
    assert (double_bound.number_of_nodes(), double_bound.number_of_edges()) == (
        80,
        848,
    )
    second_clique_graph = clique_graph(double_bound)
    assert second_clique_graph is not None
    assert (
        second_clique_graph.number_of_nodes(),
        second_clique_graph.number_of_edges(),
    ) == (192, 6688)
    assert len(list(nx.find_cliques(second_clique_graph))) == 976

    result = theorem57_hypothesis_holds(graph)

    assert result.status is Theorem57Status.PROVED
    assert result.checked_h_images == 19040
    assert len(result.certificates) == 19040
    for h_image, moves in result.certificates.items():
        facets = tuple(frozenset(clique) for clique in h_image)
        assert verify_strong_collapse_certificate(facets, moves)


def test_full_double_bound_construction_is_not_coatomic_compact_graph():
    graph = nx.from_graph6_bytes(b"Gslyf{")
    from research.h_operator_census import double_bound_clique_graph

    full = double_bound_graph(graph)
    coatomic = double_bound_clique_graph(graph)

    assert full is not None
    assert coatomic is not None
    assert full.number_of_nodes() == 47
    assert coatomic.number_of_nodes() == 19

from dataclasses import replace
from itertools import combinations
from math import comb

import networkx as nx
import pycliques.cliques as cliques_module
import pycliques.poset_homotopy as poset_homotopy
import pytest
from pycliques import (
    Theorem57Status,
    ThreeHellyStatus,
    clique_graph_of_double_bound,
    closed_clique_intersections,
    double_bound_graph,
    h_closure_from_neighbors,
    maximal_clique_hypergraph_is_3_helly,
    strong_collapse_certificate,
    support_nerve_cone_certificate,
    support_nerve_facets,
    theorem57_hypothesis_holds,
    verify_strong_collapse_certificate,
    verify_support_nerve_cone_certificate,
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


def test_direct_clique_graph_of_double_bound_matches_explicit_construction():
    graph = nx.from_graph6_bytes(b"G~]uf?")
    double_bound = double_bound_graph(graph)
    assert double_bound is not None
    expected = clique_graph(double_bound)
    assert expected is not None

    direct = clique_graph_of_double_bound(graph)

    assert direct is not None
    assert graph.number_of_edges() == 18
    assert direct.number_of_nodes() == 60
    assert set(direct) == set(expected)
    assert {frozenset(edge) for edge in direct.edges} == {
        frozenset(edge) for edge in expected.edges
    }


def _assert_direct_matches_explicit(graph):
    double_bound = double_bound_graph(graph)
    assert double_bound is not None
    expected = clique_graph(double_bound)
    assert expected is not None
    direct = clique_graph_of_double_bound(graph)
    assert direct is not None
    assert set(direct) == set(expected)
    assert {frozenset(edge) for edge in direct.edges} == {
        frozenset(edge) for edge in expected.edges
    }


def test_direct_clique_graph_matches_explicit_on_feasible_atlas_graphs():
    checked = 0
    for graph in nx.graph_atlas_g():
        if not graph or graph.number_of_nodes() > 6:
            continue
        direct = clique_graph_of_double_bound(graph)
        if direct is None:
            continue
        double_bound = double_bound_graph(graph)
        assert double_bound is not None
        expected = clique_graph(double_bound)
        assert expected is not None
        assert set(direct) == set(expected)
        assert {frozenset(edge) for edge in direct.edges} == {
            frozenset(edge) for edge in expected.edges
        }
        checked += 1
    assert checked == 207


@pytest.mark.parametrize(
    "graph",
    [octahedron(4), nx.icosahedral_graph(), nx.from_graph6_bytes(b"G~]uf?")],
)
def test_direct_clique_graph_matches_explicit_on_named_graphs(graph):
    _assert_direct_matches_explicit(graph)


def test_direct_clique_graph_matches_explicit_on_disconnected_and_labeled_graphs():
    disconnected = nx.disjoint_union(nx.path_graph(3), nx.empty_graph(2))
    labeled = nx.relabel_nodes(
        nx.cycle_graph(4),
        {vertex: ("vertex", vertex) for vertex in range(4)},
    )
    _assert_direct_matches_explicit(disconnected)
    _assert_direct_matches_explicit(labeled)


def test_direct_clique_graph_uses_exact_support_adjacency_criterion():
    graph = nx.Graph()
    graph.add_edges_from(
        combinations(
            (0, 1, 2),
            2,
        )
    )
    graph.add_edges_from(combinations((1, 2, 3), 2))
    direct = clique_graph_of_double_bound(graph)
    assert direct is not None

    vertices = tuple(direct)
    supports = {family: frozenset().union(*family) for family in vertices}
    for first_index, first in enumerate(vertices):
        for second in vertices[first_index + 1 :]:
            if supports[first] == supports[second]:
                expected_adjacent = True
            else:
                common_support = supports[first] & supports[second]
                expected_adjacent = bool(common_support) and (
                    common_support in first and common_support in second
                )
            assert direct.has_edge(first, second) is expected_adjacent


def test_direct_clique_graph_of_double_bound_budgets_are_exact():
    graph = nx.complete_graph(3)

    assert clique_graph_of_double_bound(graph, max_vertices=2) is None
    assert clique_graph_of_double_bound(graph, max_edges=0) is None
    empty_result = clique_graph_of_double_bound(nx.empty_graph(0))
    assert empty_result is not None
    assert empty_result.number_of_nodes() == 0
    assert empty_result.number_of_edges() == 0
    with pytest.raises(ValueError):
        clique_graph_of_double_bound(graph, max_vertices=True)
    with pytest.raises(ValueError):
        clique_graph_of_double_bound(graph, max_edges=-1)


def test_direct_rank_six_and_vertex_budgets_refuse_before_family_decoding(
    monkeypatch,
):
    def unexpected(_rank):
        raise AssertionError("family contents should not be decoded")

    monkeypatch.setattr(
        cliques_module,
        "_maximal_intersecting_family_library",
        unexpected,
    )

    assert clique_graph_of_double_bound(nx.complete_graph(6)) is None


def test_direct_family_library_rank_counts():
    expected_counts = (1, 2, 4, 12, 81)
    for rank, expected_count in enumerate(expected_counts, start=1):
        families = cliques_module._maximal_intersecting_family_library(rank)
        assert families is not None
        assert len(families) == expected_count
    assert cliques_module._maximal_intersecting_family_count(6) == 2646
    assert (
        clique_graph_of_double_bound(
            nx.complete_graph(5),
            max_vertices=80,
            max_edges=None,
        )
        is None
    )


def test_support_nerve_facets_and_cone_certificate():
    first_support = frozenset({"a", "b"})
    second_support = frozenset({"b", "c"})
    family = (
        (frozenset({"a", "b"}), frozenset({"b"})),
        (frozenset({"b", "c"}), frozenset({"b"})),
    )

    facets = support_nerve_facets(family)
    certificate = support_nerve_cone_certificate(family)

    assert facets == (frozenset({first_support, second_support}),)
    assert certificate is not None
    assert certificate.apex_support in {first_support, second_support}
    assert all(mask & (1 << certificate.apex_index) for mask in certificate.facet_masks)
    assert certificate.support_order[certificate.apex_index] == certificate.apex_support
    assert verify_support_nerve_cone_certificate(family, certificate)


def test_support_nerve_without_common_apex_is_not_a_certificate():
    family = (
        (frozenset({"a", "b"}), frozenset({"a"})),
        (frozenset({"b", "c"}), frozenset({"c"})),
    )

    assert set(support_nerve_facets(family)) == {
        frozenset({frozenset({"a", "b"})}),
        frozenset({frozenset({"b", "c"})}),
    }
    assert support_nerve_cone_certificate(family) is None


def test_support_nerve_verifier_rejects_corrupted_certificate_components():
    family = (
        (frozenset({"a", "b"}), frozenset({"b"})),
        (frozenset({"b", "c"}), frozenset({"b"})),
    )
    certificate = support_nerve_cone_certificate(family)
    assert certificate is not None
    first, second = certificate.support_order

    corrupted = (
        replace(certificate, apex_support=frozenset({"wrong"})),
        replace(certificate, apex_index=len(certificate.support_order)),
        replace(certificate, apex_index=1),
        replace(certificate, support_order=(second, first)),
        replace(certificate, support_order=(first, first)),
        replace(certificate, support_order=(first,)),
        replace(certificate, facet_masks=()),
        replace(certificate, facet_masks=certificate.facet_masks[:-1]),
        replace(certificate, facet_masks=(*certificate.facet_masks, 0b01)),
        replace(certificate, facet_masks=(0b01,)),
        replace(certificate, facet_masks=(0b10,)),
        replace(certificate, facet_masks=(0b11, 0b11)),
        replace(certificate, facet_masks=(0b100,)),
    )
    for forged in corrupted:
        assert not verify_support_nerve_cone_certificate(family, forged)


def test_support_nerve_verifier_accepts_repeated_members_and_hashable_labels():
    first = (("label", 1), "other")
    second = (("label", 1), "third")
    family_member = (
        frozenset(first),
        frozenset({("label", 1)}),
    )
    family = (
        family_member,
        family_member,
        (frozenset(second), frozenset({("label", 1)})),
    )

    certificate = support_nerve_cone_certificate(family)

    assert certificate is not None
    assert verify_support_nerve_cone_certificate(family, certificate)


def test_support_nerve_verifier_rejects_forged_nonconical_certificate():
    family = (
        (frozenset({"a", "b"}), frozenset({"a"})),
        (frozenset({"b", "c"}), frozenset({"c"})),
    )
    assert support_nerve_cone_certificate(family) is None
    forged = poset_homotopy.SupportNerveConeCertificate(
        apex_support=frozenset({"a", "b"}),
        apex_index=0,
        support_order=(frozenset({"a", "b"}), frozenset({"b", "c"})),
        facet_masks=(0b01, 0b10),
    )

    assert not verify_support_nerve_cone_certificate(family, forged)


def test_support_nerve_rejects_invalid_families():
    with pytest.raises(ValueError):
        support_nerve_facets(())
    with pytest.raises(ValueError):
        support_nerve_facets(((frozenset({"a"}), frozenset({"b"})),))


def test_support_nerve_verifier_rejects_malformed_families():
    certificate = support_nerve_cone_certificate(
        ((frozenset({"a", "b"}), frozenset({"b"})),)
    )
    assert certificate is not None
    assert not verify_support_nerve_cone_certificate((), certificate)
    assert not verify_support_nerve_cone_certificate(
        ((frozenset(),),),
        certificate,
    )
    assert not verify_support_nerve_cone_certificate(
        ((frozenset({"a"}),),),
        certificate,
    )


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


def test_theorem57_driver_reports_proof_and_replays_certificate(monkeypatch):
    monkeypatch.setattr(
        poset_homotopy,
        "support_nerve_cone_certificate",
        lambda _family: None,
    )
    result = theorem57_hypothesis_holds(nx.empty_graph(1), construction="explicit")

    assert result.status is Theorem57Status.PROVED
    assert result.checked_h_images == 1
    assert len(result.certificates) == 1
    assert not result.support_certificates
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
    assert result.construction_used == "explicit"
    assert "explicit-only budgets" in result.construction_reason


@pytest.mark.parametrize(
    ("graph", "budget", "reason"),
    [
        (nx.complete_graph(7), {}, "rank greater than six"),
        (nx.path_graph(3), {"max_direct_vertices": 0}, "max_direct_vertices"),
        (nx.path_graph(3), {"max_direct_edges": 0}, "max_direct_edges"),
    ],
)
def test_theorem57_direct_refusals_are_inconclusive(graph, budget, reason):
    result = theorem57_hypothesis_holds(
        graph,
        construction="direct",
        **budget,
    )

    assert result.status is Theorem57Status.INCONCLUSIVE
    assert result.construction_used == "direct"
    assert reason in result.reason


def test_theorem57_construction_modes_reject_or_route_mode_specific_budgets():
    with pytest.raises(ValueError, match="apply only to explicit"):
        theorem57_hypothesis_holds(
            nx.path_graph(3),
            construction="direct",
            max_faces=100,
        )

    automatic = theorem57_hypothesis_holds(nx.path_graph(3), max_faces=0)
    assert automatic.construction_used == "explicit"
    assert automatic.status is Theorem57Status.INCONCLUSIVE


def test_theorem57_direct_mode_preserves_closure_budget_semantics():
    result = theorem57_hypothesis_holds(
        nx.path_graph(3),
        construction="direct",
        max_closed_sets=0,
    )

    assert result.status is Theorem57Status.INCONCLUSIVE
    assert result.construction_used == "direct"
    assert "closed h-image" in result.reason


@pytest.mark.parametrize(
    "graph",
    [
        nx.path_graph(4),
        nx.cycle_graph(5),
        octahedron(3),
        nx.from_graph6_bytes(b"G~]uf?"),
        nx.disjoint_union(nx.path_graph(3), nx.empty_graph(2)),
        nx.relabel_nodes(nx.cycle_graph(4), {i: ("v", i) for i in range(4)}),
    ],
)
def test_theorem57_direct_and_explicit_modes_agree(graph):
    direct = theorem57_hypothesis_holds(graph, construction="direct")
    explicit = theorem57_hypothesis_holds(graph, construction="explicit")

    assert direct.construction_used == "direct"
    assert explicit.construction_used == "explicit"
    assert direct.status is explicit.status
    assert direct.checked_h_images == explicit.checked_h_images


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
    monkeypatch.setattr(
        poset_homotopy,
        "support_nerve_cone_certificate",
        lambda _family: None,
    )

    result = theorem57_hypothesis_holds(nx.empty_graph(0), construction="explicit")

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
        "support_nerve_cone_certificate",
        lambda _family: None,
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

    result = theorem57_hypothesis_holds(nx.empty_graph(0), construction="explicit")

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
    assert len(result.support_certificates) == 19040
    assert not result.certificates
    assert (
        sum(
            len(certificate.facet_masks) > 1
            for certificate in result.support_certificates.values()
        )
        == 80
    )
    for certificate in result.support_certificates.values():
        assert all(
            mask & (1 << certificate.apex_index) for mask in certificate.facet_masks
        )
    for h_image, certificate in result.support_certificates.items():
        assert verify_support_nerve_cone_certificate(h_image, certificate)

    explicit = theorem57_hypothesis_holds(graph, construction="explicit")
    assert explicit.status is result.status
    assert explicit.checked_h_images == result.checked_h_images
    assert explicit.construction_used == "explicit"


def test_full_double_bound_construction_is_not_coatomic_compact_graph():
    from research.h_operator_census import double_bound_clique_graph

    graph = nx.from_graph6_bytes(b"G~]uf?")
    full = double_bound_graph(graph)
    coatomic = double_bound_clique_graph(graph)

    assert full is not None
    second_clique_graph = clique_graph(full)
    assert second_clique_graph is not None
    assert graph.number_of_edges() == 18
    assert second_clique_graph.number_of_nodes() == 60
    assert coatomic is not None
    assert full.number_of_nodes() == 47
    assert coatomic.number_of_nodes() == 19

import networkx as nx
import pytest
from pycliques.cutpoints import (
    InverseCutpointExtension,
    contract_local_bridge,
    edge_in_triangle,
    edges_in_no_triangle,
    inverse_cutpoint_extensions,
    inverse_cutpoint_extensions_at,
    is_admissible_inverse_extension,
    is_local_bridge,
    local_bridge_contractions,
    local_bridges,
    local_cutpoints,
    neighborhood_components,
    non_triangle_edge_removals,
    remove_edge_not_in_triangle,
)


def test_local_bridge_on_path_and_bridge():
    graph = nx.path_graph(4)

    assert is_local_bridge(graph, 1, 2)
    assert is_local_bridge(graph, 0, 1)


def test_short_alternate_path_is_not_local_bridge():
    assert not is_local_bridge(nx.cycle_graph(4), 0, 1)


def test_triangle_edge_is_not_local_bridge():
    assert not is_local_bridge(nx.complete_graph(3), 0, 1)


def test_non_edge_is_not_local_bridge():
    assert not is_local_bridge(nx.path_graph(3), 0, 2)


def test_local_bridge_labels_and_contraction():
    graph = nx.Graph([("left", "right"), ("right", "tail")])

    contracted = contract_local_bridge(graph, "left", "right")

    assert set(contracted) == {"left", "tail"}
    assert contracted.has_edge("left", "tail")
    assert set(graph) == {"left", "right", "tail"}


def test_local_bridges_and_contractions():
    graph = nx.path_graph(4)

    assert set(local_bridges(graph)) == {(0, 1), (1, 2), (2, 3)}
    assert len(list(local_bridge_contractions(graph))) == 3


def test_contract_local_bridge_rejects_invalid_edges():
    with pytest.raises(ValueError, match="not an edge"):
        contract_local_bridge(nx.path_graph(3), 0, 2)
    with pytest.raises(ValueError, match="not a local bridge"):
        contract_local_bridge(nx.cycle_graph(4), 0, 1)


def test_triangle_edges_are_not_removable():
    graph = nx.complete_graph(3)

    assert edge_in_triangle(graph, 0, 1)
    assert list(edges_in_no_triangle(graph)) == []
    with pytest.raises(ValueError, match="contained in a triangle"):
        remove_edge_not_in_triangle(graph, 0, 1)


def test_path_edges_are_removable_and_labels_are_preserved():
    graph = nx.Graph([(("a", 1), ("b", 2)), (("b", 2), ("c", 3))])

    assert not edge_in_triangle(graph, ("a", 1), ("b", 2))
    assert set(edges_in_no_triangle(graph)) == {
        (("a", 1), ("b", 2)),
        (("b", 2), ("c", 3)),
    }
    removed = remove_edge_not_in_triangle(graph, ("a", 1), ("b", 2))
    assert set(removed) == set(graph)
    assert not removed.has_edge(("a", 1), ("b", 2))
    assert graph.has_edge(("a", 1), ("b", 2))


def test_non_triangle_edge_removals():
    assert len(list(non_triangle_edge_removals(nx.path_graph(3)))) == 2


def test_icosahedron_plus_antipodal_edge():
    graph = nx.icosahedral_graph()
    antipodes = (0, 3)
    graph.add_edge(*antipodes)

    assert not edge_in_triangle(graph, *antipodes)
    removed = remove_edge_not_in_triangle(graph, *antipodes)
    assert nx.is_isomorphic(removed, nx.icosahedral_graph())


# ---------------------------------------------------------------------------
# Local cutpoints and inverse cutpoint extensions
# ---------------------------------------------------------------------------


def test_local_cutpoints_path_and_complete_graph():
    assert sorted(local_cutpoints(nx.path_graph(4))) == [1, 2]
    assert list(local_cutpoints(nx.complete_graph(4))) == []


def test_neighborhood_components_path():
    assert neighborhood_components(nx.path_graph(5), 2) == [
        frozenset({1}),
        frozenset({3}),
    ]


def _bowtie_graph() -> nx.Graph:
    """Two triangles sharing a single vertex; vertex 2 is a local cutpoint."""
    return nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)])


def test_bowtie_has_single_local_cutpoint_with_two_branches():
    graph = _bowtie_graph()

    assert list(local_cutpoints(graph)) == [2]
    assert neighborhood_components(graph, 2) == [frozenset({0, 1}), frozenset({3, 4})]


def test_inverse_cutpoint_extensions_at_bowtie():
    graph = _bowtie_graph()

    extensions = list(inverse_cutpoint_extensions_at(graph, 2))
    assert len(extensions) == 1

    extension = extensions[0]
    assert isinstance(extension, InverseCutpointExtension)
    assert extension.cutpoint == 2
    assert {extension.u_branch, extension.v_branch} == {
        frozenset({0, 1}),
        frozenset({3, 4}),
    }
    assert extension.graph.has_edge(extension.u, extension.v)
    assert extension.graph.order() == graph.order() + 1
    assert is_admissible_inverse_extension(extension)


def test_inverse_cutpoint_extensions_finds_no_candidates_without_local_cutpoint():
    assert list(inverse_cutpoint_extensions(nx.complete_graph(4))) == []


def test_inverse_cutpoint_extensions_at_respects_max_candidates():
    # A cutpoint whose neighborhood has 4 singleton components admits
    # 2**3 - 1 = 7 distinct bipartitions into two nonempty sides.
    graph = nx.Graph()
    for i in range(4):
        graph.add_edge("w", i)
    assert neighborhood_components(graph, "w") == [frozenset({i}) for i in range(4)]

    assert len(list(inverse_cutpoint_extensions_at(graph, "w"))) == 7
    assert len(list(inverse_cutpoint_extensions_at(graph, "w", max_candidates=3))) == 3


def test_inverse_cutpoint_extension_rejects_split_within_a_component():
    """Splitting inside a single neighborhood component is not admissible.

    Only whole connected components of N(w) may be assigned to one side; an
    arbitrary split that separates two adjacent neighbors of ``w`` violates
    Theorem 6.1's local-bridge hypothesis and must be rejected.
    """
    graph = nx.Graph()
    graph.add_edges_from([("u", 0), ("u", 3), ("v", 1), ("v", 4), (0, 1), (3, 4)])
    graph.add_edge("u", "v")

    extension = InverseCutpointExtension(
        cutpoint=2,
        u="u",
        v="v",
        u_branch=frozenset({0, 3}),
        v_branch=frozenset({1, 4}),
        graph=graph,
    )

    assert not is_admissible_inverse_extension(extension)


def test_motivating_example_inverse_cutpoint_extension_recovers_g():
    """The contracted vertex of the motivating example is a local cutpoint
    whose admissible split, followed by removing the added edge, recovers a
    graph isomorphic to the original clockwork graph.
    """
    from pycliques.clockwork import clockwork_graph

    g = clockwork_graph(6 * [1], [[0] for _ in range(6)], 2, [0, 1])
    h = g.copy()
    h.add_edge(12, 15)
    h = nx.contracted_edge(h, (12, 15), self_loops=False)

    assert list(local_cutpoints(h)) == [12]

    found = False
    for extension in inverse_cutpoint_extensions(h):
        if not is_admissible_inverse_extension(extension):
            continue
        u, v = extension.u, extension.v
        if edge_in_triangle(extension.graph, u, v):
            continue
        target = remove_edge_not_in_triangle(extension.graph, u, v)
        if nx.is_isomorphic(target, g):
            found = True
    assert found


def test_icosahedron_plus_degree_two_vertex():
    graph = nx.icosahedral_graph()
    graph.add_edges_from([(12, 0), (12, 3)])

    assert not edge_in_triangle(graph, 12, 0)
    removed = remove_edge_not_in_triangle(graph, 12, 0)
    assert removed.has_edge(12, 3)
    assert removed.degree(12) == 1

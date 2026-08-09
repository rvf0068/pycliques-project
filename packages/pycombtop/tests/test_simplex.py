import networkx as nx
from pycombtop import (
    Simplex,
    SimplicialComplex,
    all_subsets,
    bounded_degree,
    bounded_degree_complex,
    clique_complex,
    complex_of_forests,
    directed_neighborhood_complex,
    dong_matching,
    is_oriented_simplex,
    neighborhood_complex,
    nerve_of_cliques,
    nerve_of_sets,
    oriented_complex,
)

# ---------- Simplex tests ----------


def test_simplex_creation():
    """Simplex can be created from a list or set."""
    s = Simplex([1, 2, 3])
    assert 1 in s
    assert len(s) == 3
    assert isinstance(s, frozenset)


def test_simplex_repr():
    """Non-empty simplex displays as a set literal."""
    s = Simplex([1, 2])
    r = repr(s)
    assert r == "{1, 2}" or r == "{2, 1}"


def test_empty_simplex_repr():
    """Empty simplex displays as {}."""
    assert repr(Simplex([])) == "{}"


def test_simplex_dimension():
    """Dimension is len - 1."""
    assert Simplex({0, 1, 2}).dimension() == 2
    assert Simplex({5}).dimension() == 0
    assert Simplex([]).dimension() == -1


def test_simplex_equality():
    """Simplices compare equal to frozensets with the same elements."""
    assert Simplex([1, 2]) == frozenset([1, 2])


# ---------- SimplicialComplex construction ----------


def test_complex_from_facets():
    """Build complex from explicit facets."""
    sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
    assert sc.dimension() == 2
    assert sc.vertex_set == {0, 1, 2}


def test_complex_from_function():
    """Build complex from a membership function."""
    sc = SimplicialComplex({0, 1, 2}, function=lambda s: len(s) <= 2)
    assert sc.dimension() == 1


def test_empty_complex_dimension():
    """Empty complex has dimension -1."""
    sc = SimplicialComplex(set(), facet_set=set())
    assert sc.dimension() == -1


def test_complex_equality():
    """Two complexes with the same vertex set and facets are equal."""
    sc1 = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    sc2 = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    assert sc1 == sc2


def test_complex_inequality():
    """Complexes with different facets are not equal."""
    sc1 = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
    sc2 = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1}, {1, 2}])
    assert sc1 != sc2


def test_complex_not_equal_to_non_complex():
    """Comparing a SimplicialComplex with a non-complex returns False."""
    sc = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    assert sc != 42
    assert sc != "not a complex"


def test_complex_repr():
    """Repr mentions vertex_set and facets."""
    sc = SimplicialComplex({0}, facet_set=[{0}])
    r = repr(sc)
    assert "vertex_set" in r
    assert "facets" in r


# ---------- SimplicialComplex methods ----------


def test_deletion():
    """Deleting a vertex removes it and updates facets."""
    sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
    d = sc.deletion(2)
    assert 2 not in d.vertex_set
    assert d.dimension() == 1


def test_deletion_preserves_non_containing_facets():
    """Facets not containing the deleted vertex persist unchanged."""
    sc = SimplicialComplex({0, 1, 2, 3}, facet_set=[{0, 1, 2}, {2, 3}])
    d = sc.deletion(0)
    assert Simplex({2, 3}) in d.facet_set


def test_link():
    """Link of a vertex in a triangle has the opposite edge."""
    sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
    lk = sc.link(0)
    assert 0 not in lk.vertex_set
    assert lk.dimension() == 1


def test_link_of_isolated_vertex():
    """Link of a vertex not in any facet is the empty complex."""
    sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1}])
    lk = sc.link(2)
    assert lk.dimension() == -1


def test_skeleton():
    """n-skeleton keeps only simplices up to dimension n."""
    sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
    sk = sc.skeleton(1)
    assert sk.dimension() == 1


def test_skeleton_zero():
    """0-skeleton is just the vertex set as 0-simplices."""
    sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
    sk = sc.skeleton(0)
    assert sk.dimension() == 0


def test_one_skeleton_graph():
    """1-skeleton graph has correct nodes and edges."""
    sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}])
    g = sc.one_skeleton_graph()
    assert sorted(g.nodes()) == [0, 1, 2]
    assert g.number_of_edges() == 3


def test_is_clique_complex_true():
    """Clique complex of a graph is a clique complex."""
    cc = clique_complex(nx.cycle_graph(5))
    assert cc.is_clique_complex()


def test_is_clique_complex_false():
    """The boundary of a triangle is not a clique complex."""
    sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1}, {1, 2}, {0, 2}])
    assert not sc.is_clique_complex()


def test_all_simplices():
    """all_simplices returns every face including the empty set."""
    sc = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    simplices = sc.all_simplices()
    # Should contain: {}, {0}, {1}, {0,1}
    assert len(simplices) == 4
    assert Simplex([]) in simplices
    assert Simplex([0]) in simplices


def test_dong_matching():
    """Dong matching returns a subset of all simplices."""
    sc = SimplicialComplex({0, 1}, facet_set=[{0, 1}])
    critical = dong_matching(sc, order_function=sorted)
    assert critical.issubset(sc.all_simplices())


# ---------- Module functions ----------


def test_all_subsets():
    """all_subsets yields every non-empty subset."""
    subs = list(all_subsets({0, 1}))
    assert len(subs) == 3  # {0,1}, {0}, {1}
    assert all(isinstance(s, Simplex) for s in subs)


def test_all_subsets_single():
    """Single-element set has one subset."""
    subs = list(all_subsets({42}))
    assert len(subs) == 1


def test_nerve_of_sets():
    """Nerve has expected dimension for overlapping sets."""
    n = nerve_of_sets([{1, 2}, {2, 3}, {3, 4}])
    assert n.dimension() == 1


def test_nerve_of_sets_full_intersection():
    """Three sets with a common element form a 2-simplex in the nerve."""
    n = nerve_of_sets([{1, 2}, {2, 3}, {1, 2, 3}])
    assert n.dimension() == 2


def test_clique_complex_cycle():
    """Clique complex of a cycle is 1-dimensional."""
    cc = clique_complex(nx.cycle_graph(5))
    assert cc.dimension() == 1


def test_clique_complex_complete():
    """Clique complex of K4 is a tetrahedron."""
    cc = clique_complex(nx.complete_graph(4))
    assert cc.dimension() == 3


def test_nerve_of_cliques():
    """Nerve of cliques of C4 is 1-dimensional."""
    n = nerve_of_cliques(nx.cycle_graph(4))
    assert n.dimension() == 1


def test_bounded_degree_within_bounds():
    """Edges within the degree bound return True."""
    g = nx.path_graph(3)
    assert bounded_degree(g, {0: 1, 1: 1, 2: 1}, [(0, 1)])


def test_bounded_degree_exceeds():
    """Edges that exceed the bound return False."""
    g = nx.path_graph(3)
    assert not bounded_degree(g, {0: 1, 1: 1, 2: 1}, [(0, 1), (1, 2)])


def test_bounded_degree_complex():
    """Bounded degree complex has correct dimension."""
    g = nx.cycle_graph(3)
    lv = {0: 1, 1: 1, 2: 1}
    bdc = bounded_degree_complex(g, lv)
    assert bdc.dimension() == 0


def test_is_oriented_simplex_true():
    """A transitive tournament is an oriented simplex."""
    d = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    assert is_oriented_simplex(d)


def test_is_oriented_simplex_false_cycle():
    """A directed cycle is not an oriented simplex."""
    d = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    assert not is_oriented_simplex(d)


def test_is_oriented_simplex_false_symmetric():
    """A graph with symmetric edges is not an oriented simplex."""
    d = nx.DiGraph([(0, 1), (1, 0)])
    assert not is_oriented_simplex(d)


def test_oriented_complex():
    """Oriented complex of a transitive tournament on 3 vertices has dim 2."""
    d = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    oc = oriented_complex(d)
    assert oc.dimension() == 2


def test_oriented_complex_empty():
    """Oriented complex of a single directed edge is 1-dimensional."""
    d = nx.DiGraph([(0, 1)])
    oc = oriented_complex(d)
    assert oc.dimension() == 1


# ---------- complex_of_forests ----------


def test_complex_of_forests():
    """Forest complex of C4 is 2-dimensional (3 vertices can form a forest)."""
    g = nx.cycle_graph(4)
    cf = complex_of_forests(g)
    assert cf.dimension() == 2


def test_complex_of_forests_with_max_deg():
    """Restricting max degree reduces the dimension."""
    g = nx.cycle_graph(4)
    cf = complex_of_forests(g, max_deg=1)
    assert cf.dimension() == 1


def test_complex_of_forests_complete():
    """Forest complex of K3 is 1-dimensional (any 2 vertices form a forest)."""
    g = nx.complete_graph(3)
    cf = complex_of_forests(g)
    assert cf.dimension() == 1


# ---------- neighborhood_complex ----------


def test_neighborhood_complex_cycle():
    """Neighborhood complex of C5 is 1-dimensional (each vertex has 2 neighbors)."""
    nc = neighborhood_complex(nx.cycle_graph(5))
    assert nc.dimension() == 1


def test_neighborhood_complex_complete():
    """Neighborhood complex of K4: each N(v) has 3 vertices, giving dim 2."""
    nc = neighborhood_complex(nx.complete_graph(4))
    assert nc.dimension() == 2


def test_neighborhood_complex_vertex_set():
    """Neighborhood complex preserves the graph's vertex set."""
    g = nx.cycle_graph(4)
    nc = neighborhood_complex(g)
    assert nc.vertex_set == set(g.nodes())


def test_neighborhood_complex_facets_are_neighborhoods():
    """Each facet of the neighborhood complex is a neighborhood of some vertex."""
    g = nx.path_graph(4)
    nc = neighborhood_complex(g)
    expected = {frozenset(g.neighbors(v)) for v in g.nodes() if list(g.neighbors(v))}
    assert {frozenset(f) for f in nc.facet_set} == expected


def test_neighborhood_complex_no_edges():
    """Graph with no edges yields an empty neighborhood complex."""
    g = nx.empty_graph(3)
    nc = neighborhood_complex(g)
    assert nc.dimension() == -1


# ---------- directed_neighborhood_complex ----------


def test_directed_neighborhood_complex_in():
    """In-neighborhood complex: facets are in-neighborhoods of each vertex."""
    d = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    nc = directed_neighborhood_complex(d)
    assert nc.dimension() == 1


def test_directed_neighborhood_complex_out():
    """Out-neighborhood complex: facets are out-neighborhoods of each vertex."""
    d = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    nc = directed_neighborhood_complex(d, use_out_neighborhood=True)
    assert nc.dimension() == 1


def test_directed_neighborhood_complex_in_facets():
    """In-neighborhoods of a path digraph are singletons."""
    d = nx.DiGraph([(0, 1), (1, 2)])
    nc = directed_neighborhood_complex(d)
    # N⁻(1)={0}, N⁻(2)={1}; N⁻(0)={} is dropped
    assert nc.dimension() == 0


def test_directed_neighborhood_complex_out_facets():
    """Out-neighborhoods of a path digraph are singletons."""
    d = nx.DiGraph([(0, 1), (1, 2)])
    nc = directed_neighborhood_complex(d, use_out_neighborhood=True)
    # N⁺(0)={1}, N⁺(1)={2}; N⁺(2)={} is dropped
    assert nc.dimension() == 0


def test_directed_neighborhood_complex_vertex_set():
    """Directed neighborhood complex preserves the digraph's node set."""
    d = nx.DiGraph([(0, 1), (1, 2)])
    nc = directed_neighborhood_complex(d)
    assert nc.vertex_set == set(d.nodes())


def test_directed_neighborhood_complex_no_edges():
    """Digraph with no edges yields an empty directed neighborhood complex."""
    d = nx.DiGraph()
    d.add_nodes_from([0, 1, 2])
    nc = directed_neighborhood_complex(d)
    assert nc.dimension() == -1

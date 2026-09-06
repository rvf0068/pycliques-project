"""Tests for the homotopy_type module."""

import networkx as nx
from pycombtop import (
    Simplex,
    SimplicialComplex,
)
from pycombtop.homotopy_type import (
    HomotopyVerdict,
    Theorem,
    WedgeOfSpheres,
    betti_numbers_graph,
    betti_numbers_sc,
    collapse,
    homotopy_type_large_graph,
    homotopy_type_sc_with_verdict,
    homotopy_type_with_verdict,
    intersection_complex,
    is_vertex_decomposable,
    star,
    star_cluster,
)

# ---------- HomotopyVerdict ----------


def test_homotopy_verdict_fields():
    """HomotopyVerdict stores wedge, reason, is_exact."""
    w = WedgeOfSpheres.contractible()
    v = HomotopyVerdict(wedge=w, reason="test", is_exact=True)
    assert v.verdict == "Contractible"
    assert v.reason == "test"
    assert v.is_exact is True


def test_homotopy_verdict_default_exact():
    """is_exact defaults to True."""
    w = WedgeOfSpheres.contractible()
    v = HomotopyVerdict(wedge=w, reason="test")
    assert v.is_exact is True


# ---------- collapse ----------


def test_collapse_single_vertex():
    """Collapsing a single vertex complex returns it unchanged."""
    sc = SimplicialComplex({0}, facet_set={Simplex({0})})
    result = collapse(sc)
    assert len(result.vertex_set) == 1


def test_collapse_edge():
    """An edge complex collapses to a single vertex."""
    sc = SimplicialComplex({0, 1}, facet_set={Simplex({0, 1})})
    result = collapse(sc)
    assert len(result.vertex_set) == 1


def test_collapse_triangle():
    """A filled triangle collapses to a single vertex."""
    sc = SimplicialComplex({0, 1, 2}, facet_set={Simplex({0, 1, 2})})
    result = collapse(sc)
    assert len(result.vertex_set) == 1


def test_collapse_boundary_triangle():
    """The boundary of a triangle (hollow) is not collapsible to a point."""
    sc = SimplicialComplex(
        {0, 1, 2},
        facet_set={Simplex({0, 1}), Simplex({1, 2}), Simplex({0, 2})},
    )
    result = collapse(sc)
    # The boundary of a triangle has the homotopy type of S^1;
    # elementary collapses cannot reduce it to a point.
    assert len(result.vertex_set) >= 2


# ---------- is_vertex_decomposable ----------


def test_simplex_is_vertex_decomposable():
    """A simplex is vertex decomposable."""
    sc = SimplicialComplex({0, 1, 2}, facet_set={Simplex({0, 1, 2})})
    assert is_vertex_decomposable(sc) is True


def test_boundary_triangle_vertex_decomposable():
    """The boundary of a triangle is vertex decomposable."""
    sc = SimplicialComplex(
        {0, 1, 2},
        facet_set={Simplex({0, 1}), Simplex({1, 2}), Simplex({0, 2})},
    )
    assert is_vertex_decomposable(sc) is True


# ---------- star / star_cluster / intersection_complex ----------


def test_star_of_vertex():
    """Star of a vertex contains exactly the facets containing it."""
    sc = SimplicialComplex(
        {0, 1, 2, 3},
        facet_set={Simplex({0, 1, 2}), Simplex({2, 3})},
    )
    st = star(sc, 0)
    assert Simplex({0, 1, 2}) in st.facet_set
    assert Simplex({2, 3}) not in st.facet_set


def test_star_cluster_union():
    """Star cluster of a set is the union of stars of its elements."""
    sc = SimplicialComplex(
        {0, 1, 2, 3},
        facet_set={Simplex({0, 1}), Simplex({2, 3})},
    )
    sc_result = star_cluster(sc, {0, 2})
    assert Simplex({0, 1}) in sc_result.facet_set
    assert Simplex({2, 3}) in sc_result.facet_set


def test_intersection_complex_basic():
    """Intersection of two complexes shares simplices in both."""
    sc1 = SimplicialComplex({0, 1, 2}, facet_set={Simplex({0, 1, 2})})
    sc2 = SimplicialComplex({0, 1, 3}, facet_set={Simplex({0, 1, 3})})
    inter = intersection_complex(sc1, sc2)
    assert 0 in inter.vertex_set
    assert 1 in inter.vertex_set


# ---------- betti_numbers_sc ----------


def test_betti_numbers_point():
    """A single point has trivial reduced Betti numbers."""
    sc = SimplicialComplex({0}, facet_set={Simplex({0})})
    bettis = betti_numbers_sc(sc)
    assert bettis == [] or all(b == 0 for b in bettis)


def test_betti_numbers_circle():
    """Boundary of a triangle (S^1) has beta_1 = 1."""
    sc = SimplicialComplex(
        {0, 1, 2},
        facet_set={Simplex({0, 1}), Simplex({1, 2}), Simplex({0, 2})},
    )
    bettis = betti_numbers_sc(sc)
    assert len(bettis) >= 2
    assert bettis[1] == 1


# ---------- betti_numbers_graph ----------


def test_betti_numbers_complete_graph():
    """K_n is contractible (all reduced Betti numbers zero)."""
    g = nx.complete_graph(4)
    bettis = betti_numbers_graph(g)
    assert bettis == [] or all(b == 0 for b in bettis)


def test_betti_numbers_cycle_5():
    """C_5 has the homotopy type of S^1."""
    g = nx.cycle_graph(5)
    bettis = betti_numbers_graph(g)
    assert len(bettis) >= 2
    assert bettis[1] == 1


# ---------- homotopy_type_with_verdict ----------


def test_single_vertex_contractible():
    """A single vertex graph is contractible."""
    g = nx.Graph()
    g.add_node(0)
    v = homotopy_type_with_verdict(g)
    assert v.verdict == "Contractible"
    assert v.is_exact is True


def test_complete_graph_contractible():
    """K_n is contractible (the clique complex is a simplex)."""
    g = nx.complete_graph(5)
    v = homotopy_type_with_verdict(g)
    assert v.verdict == "Contractible"
    assert v.is_exact is True


def test_cycle_graph_c5():
    """C_5 has homotopy type S^1."""
    g = nx.cycle_graph(5)
    v = homotopy_type_with_verdict(g)
    assert "S^{1}" in v.verdict
    assert v.is_exact is True


def test_edge_graph_contractible():
    """A single edge is contractible (dismantlable to a point)."""
    g = nx.Graph()
    g.add_edge(0, 1)
    v = homotopy_type_with_verdict(g)
    assert v.verdict == "Contractible"


def test_path_graph_contractible():
    """A path graph is contractible (dismantlable)."""
    g = nx.path_graph(5)
    v = homotopy_type_with_verdict(g)
    assert v.verdict == "Contractible"


def test_two_isolated_vertices_s0():
    """Two isolated vertices (graph atlas #2) have homotopy type S^0."""
    g = nx.graph_atlas_g()[2]
    v = homotopy_type_with_verdict(g)
    assert "S^{0}" in v.verdict
    assert v.is_exact is True


# ---------- homotopy_type_sc_with_verdict ----------


def test_sc_simplex_contractible():
    """A simplex complex is contractible."""
    sc = SimplicialComplex({0, 1, 2}, facet_set={Simplex({0, 1, 2})})
    v = homotopy_type_sc_with_verdict(sc)
    assert v.verdict == "Contractible"
    assert v.is_exact is True


def test_sc_boundary_triangle():
    """Boundary of triangle has homotopy type S^1."""
    sc = SimplicialComplex(
        {0, 1, 2},
        facet_set={Simplex({0, 1}), Simplex({1, 2}), Simplex({0, 2})},
    )
    v = homotopy_type_sc_with_verdict(sc)
    assert "S^{1}" in v.verdict


# ---------- homotopy_type_large_graph ----------


def test_large_graph_delegates_for_small():
    """For small graphs, homotopy_type_large_graph delegates to full analysis."""
    g = nx.complete_graph(4)
    v = homotopy_type_large_graph(g, bound=100)
    assert v.verdict == "Contractible"


# ---------- Theorem labels ----------


def test_theorem_labels_are_strings():
    """All Theorem constants are non-empty strings."""
    for name in dir(Theorem):
        if name.startswith("_"):
            continue
        val = getattr(Theorem, name)
        assert isinstance(val, str)
        assert len(val) > 0


# ---------- WedgeOfSpheres ----------


def test_wedge_contractible():
    """Contractible wedge has no spheres."""
    w = WedgeOfSpheres.contractible()
    assert w.is_contractible()
    assert w.to_latex() == "Contractible"
    assert w.spheres == {}


def test_wedge_sphere():
    """A single sphere of given dimension."""
    w = WedgeOfSpheres.sphere(2)
    assert not w.is_contractible()
    assert w.to_latex() == "\\(S^{2}\\)"
    assert w.spheres == {2: 1}


def test_wedge_add_spheres():
    """Adding spheres accumulates counts."""
    w = WedgeOfSpheres.contractible()
    w2 = w.add_spheres(1, 3)
    assert w2.spheres == {1: 3}
    w3 = w2.add_spheres(1, 2)
    assert w3.spheres == {1: 5}
    w4 = w3.add_spheres(2, 1)
    assert w4.spheres == {1: 5, 2: 1}


def test_wedge_join():
    """Wedge of two WedgeOfSpheres combines sphere counts."""
    a = WedgeOfSpheres.sphere(1)
    b = WedgeOfSpheres(spheres={1: 2, 3: 1})
    c = a.wedge(b)
    assert c.spheres == {1: 3, 3: 1}


def test_wedge_suspend():
    """Suspension raises each sphere dimension by 1."""
    w = WedgeOfSpheres(spheres={1: 2, 3: 1})
    s = w.suspend()
    assert s.spheres == {2: 2, 4: 1}


def test_wedge_from_betti():
    """Construct from Betti numbers."""
    # beta_0=0, beta_1=3, beta_2=1  →  3 copies of S^1 and 1 copy of S^2
    w = WedgeOfSpheres.from_betti([0, 3, 1])
    assert w.spheres == {1: 3, 2: 1}


def test_wedge_from_betti_contractible():
    """All-zero Betti numbers → contractible."""
    w = WedgeOfSpheres.from_betti([0, 0, 0])
    assert w.is_contractible()


def test_wedge_to_betti():
    """Round-trip: from_betti → to_betti."""
    bettis = [0, 3, 0, 1]
    w = WedgeOfSpheres.from_betti(bettis)
    assert w.to_betti() == bettis


def test_wedge_to_latex_multiple():
    """LaTeX string for a wedge of multiple sphere types."""
    w = WedgeOfSpheres(spheres={1: 3, 2: 1})
    latex = w.to_latex()
    assert "\\vee_{3}S^{1}" in latex
    assert "S^{2}" in latex


def test_verdict_uses_wedge():
    """HomotopyVerdict.verdict delegates to WedgeOfSpheres.to_latex()."""
    w = WedgeOfSpheres.sphere(1)
    v = HomotopyVerdict(wedge=w, reason="test")
    assert v.verdict == "\\(S^{1}\\)"


def test_verdict_wedge_field():
    """HomotopyVerdict exposes wedge field for programmatic access."""
    g = nx.cycle_graph(5)
    v = homotopy_type_with_verdict(g)
    assert isinstance(v.wedge, WedgeOfSpheres)
    assert 1 in v.wedge.spheres


# ---------- is_contractible_via_flag_apex ----------


def test_is_contractible_via_flag_apex_star_graph_true():
    """A star graph has a universal vertex (the center): trivially a cone."""
    from pycombtop.homotopy_type import is_contractible_via_flag_apex

    assert is_contractible_via_flag_apex(nx.star_graph(4)) is True


def test_is_contractible_via_flag_apex_cycle_false():
    """C5 is contractible? No -- it's S^1, and has no universal vertex either."""
    from pycombtop.homotopy_type import is_contractible_via_flag_apex

    assert is_contractible_via_flag_apex(nx.cycle_graph(5)) is False


def test_is_contractible_via_flag_apex_single_vertex():
    """A single vertex is trivially a cone (on the empty complex)."""
    from pycombtop.homotopy_type import is_contractible_via_flag_apex

    assert is_contractible_via_flag_apex(nx.empty_graph(1)) is True


def test_is_contractible_via_flag_apex_disconnected_but_contractible_case():
    """A vertex dominating everything is a cone even if the link looks
    disconnected at first glance (distinguishes 'trivially a cone' from
    'contractible via more general means')."""
    from pycombtop.homotopy_type import is_contractible_via_flag_apex

    # Two disjoint edges plus a vertex adjacent to all four endpoints: the
    # complex is a cone from the apex, even though removing the apex leaves
    # a disconnected graph.
    g = nx.Graph([(0, 1), (2, 3), (4, 0), (4, 1), (4, 2), (4, 3)])
    assert is_contractible_via_flag_apex(g) is True


def test_is_contractible_via_flag_apex_on_simplicial_complex():
    """Works on a SimplicialComplex directly, not just a networkx graph."""
    from pycombtop import SimplicialComplex
    from pycombtop.homotopy_type import is_contractible_via_flag_apex

    sc = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1}, {0, 2}])
    assert is_contractible_via_flag_apex(sc) is True
    sc_no_apex = SimplicialComplex({0, 1, 2}, facet_set=[{0, 1}, {1, 2}, {0, 2}])
    assert is_contractible_via_flag_apex(sc_no_apex) is False


# ---------- link/antistar homotopy pushout (Priority 1c) ----------


def test_antistar_matches_whole_complex_when_link_contractible():
    """When lk(sigma) is contractible, the whole complex is homotopy
    equivalent to its antistar -- checked here on a complex where the
    antistar is genuinely not flag (see test_simplex.py for the structural
    check), using the SC-based homotopy-type path throughout."""
    from pycombtop import SimplicialComplex, antistar, link

    sc = SimplicialComplex(
        {0, 1, 2, 3, 4},
        facet_set=[{0, 1, 2}, {2, 3}, {3, 4}, {2, 4}],
    )
    lk = link(sc, {0})
    assert homotopy_type_sc_with_verdict(lk).wedge.is_contractible()

    whole = homotopy_type_sc_with_verdict(sc)
    ast = homotopy_type_sc_with_verdict(antistar(sc, {0}))
    assert whole.wedge == ast.wedge


def test_antistar_matches_graph_view_when_flag():
    """When the antistar happens to be flag, the SC-based path and the plain
    graph-based path (via its 1-skeleton) must agree."""
    from pycombtop import SimplicialComplex, antistar

    sc = SimplicialComplex(
        {0, 1, 2, 3},
        facet_set=[{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}],
    )
    ast = antistar(sc, {0, 1})
    assert ast.is_clique_complex() is True
    sc_verdict = homotopy_type_sc_with_verdict(ast)
    graph_verdict = homotopy_type_with_verdict(ast.one_skeleton_graph())
    assert sc_verdict.wedge == graph_verdict.wedge

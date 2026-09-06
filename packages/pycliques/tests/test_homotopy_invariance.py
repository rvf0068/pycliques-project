import networkx as nx
from pycliques.cliques import clique_graph
from pycliques.helly import is_clique_helly
from pycliques.homotopy_invariance import (
    completes_of_size,
    completes_with_empty_intersection,
    delta_of_clique_set,
    h_closure,
    has_center_at_all,
    is_center,
    minimal_violation,
    neckties,
    theorem11_hypothesis_holds,
    theorem15_hypothesis_holds,
    witnesses_and_extension,
)
from pycliques.named import (
    collapse_obstruction_fixture,
    dominated_vertex_free_non_helly,
    octahedron,
    suspension_of_cycle,
)

# ---------- Shared fixtures (Priority 6, added early per project instructions) ---

FIXTURE_1 = dominated_vertex_free_non_helly  # graph6 "H?qdvbU"
FIXTURE_2 = collapse_obstruction_fixture  # graph6 "HUZv~zz"

# ---------- 1a: completes, centers, neckties ----------


def test_completes_with_empty_intersection_k4_is_empty():
    """K4's clique graph is a single node: no bad completes exist."""
    kg = clique_graph(nx.complete_graph(4))
    assert list(completes_with_empty_intersection(kg)) == []


def test_completes_with_empty_intersection_octahedron_nonempty():
    """The octahedron is the textbook example of a non-clique-Helly graph."""
    kg = clique_graph(octahedron(3))
    bad = list(completes_with_empty_intersection(kg))
    assert len(bad) > 0
    assert all(not set.intersection(*(set(q) for q in X)) for X in bad)


def test_completes_with_empty_intersection_max_size():
    """max_size stops enumeration once completes exceed the given size."""
    kg = clique_graph(octahedron(3))
    bad = list(completes_with_empty_intersection(kg, max_size=3))
    assert all(len(X) <= 3 for X in bad)


def test_completes_of_size_single_clique_graph():
    """K4's clique graph has exactly one complete of size 1: itself."""
    kg = clique_graph(nx.complete_graph(4))
    result = list(completes_of_size(kg, 1))
    assert len(result) == 1
    assert result[0] == frozenset(kg.nodes())


def test_completes_of_size_matches_brute_force():
    """completes_of_size(kg, k) agrees with a direct combinatorial check."""
    import itertools

    kg = clique_graph(octahedron(3))
    for k in (1, 2, 3):
        expected = {
            frozenset(c)
            for c in itertools.combinations(kg.nodes(), k)
            if all(kg.has_edge(a, b) for a, b in itertools.combinations(c, 2))
        }
        assert set(completes_of_size(kg, k)) == expected


def test_completes_of_size_zero_is_empty():
    """Size 0 (or negative) yields nothing."""
    kg = clique_graph(octahedron(3))
    assert list(completes_of_size(kg, 0)) == []


def test_neckties_octahedron_are_maximal_bad_completes():
    """Neckties of the octahedron's clique graph are maximal and bad."""
    kg = clique_graph(octahedron(3))
    nt = neckties(kg)
    assert len(nt) > 0
    for necktie in nt:
        assert not set.intersection(*(set(q) for q in necktie))
        # maximal: no other node of kg is adjacent to every member
        candidates = set(kg.nodes()) - necktie
        for q in candidates:
            assert not all(kg.has_edge(q, member) for member in necktie)


def test_is_center_true_and_false_on_octahedron():
    """A concrete bad complete of the octahedron has both a center and a non-center."""
    kg = clique_graph(octahedron(3))
    X = next(completes_with_empty_intersection(kg))
    assert sorted(X, key=sorted) == [
        frozenset({0, 2, 4}),
        frozenset({0, 3, 5}),
        frozenset({1, 2, 5}),
    ]
    assert is_center(frozenset({0, 2, 5}), X) is True
    assert is_center(frozenset({0, 3, 4}), X) is False


def test_has_center_at_all_reports_bool():
    """has_center_at_all mirrors is_center over every node of kg."""
    kg = clique_graph(octahedron(3))
    X = next(completes_with_empty_intersection(kg))
    expected = any(is_center(q0, X) for q0 in kg.nodes())
    assert has_center_at_all(X, kg) == expected


def test_theorem11_hypothesis_holds_for_complete_graph():
    """K4 is clique-Helly, so Theorem 11's hypothesis holds vacuously."""
    holds, bad = theorem11_hypothesis_holds(nx.complete_graph(4))
    assert holds is True
    assert bad is None


def test_theorem11_hypothesis_holds_for_suspension_of_c5():
    """The suspension of C5 is a known-good case from the research session."""
    holds, _ = theorem11_hypothesis_holds(suspension_of_cycle(5))
    assert holds is True


def test_theorem11_hypothesis_fixture_1_holds():
    """Fixture 1 (H?qdvbU): K(G) is clique-Helly, so no bad completes exist."""
    holds, bad = theorem11_hypothesis_holds(FIXTURE_1())
    assert holds is True
    assert bad is None


def test_theorem11_hypothesis_fixture_2_fails():
    """Fixture 2 (HUZv~zz) is the sharpest known stress test: Theorem 11's
    direct hypothesis fails here even though G is (separately) known to be
    good -- this is exactly why Theorem 15 (1d) is the strongest tool."""
    holds, bad = theorem11_hypothesis_holds(FIXTURE_2())
    assert holds is False
    assert bad is not None


# ---------- 1b: minimal violation and witnesses ----------


def test_minimal_violation_none_for_clique_helly_graph():
    """K4 is clique-Helly: no bad complete exists at all."""
    kg = clique_graph(nx.complete_graph(4))
    assert minimal_violation(kg) is None


def test_minimal_violation_size_three_for_octahedron():
    """The octahedron's minimal violation is a triangle of cliques (size 3)."""
    kg = clique_graph(octahedron(3))
    mv = minimal_violation(kg)
    assert mv is not None
    assert len(mv) == 3
    assert not set.intersection(*(set(q) for q in mv))


def test_minimal_violation_is_actually_minimal():
    """Every proper subset of the returned violation has nonempty intersection."""
    import itertools

    kg = clique_graph(octahedron(3))
    mv = minimal_violation(kg)
    for r in range(1, len(mv)):
        for sub in itertools.combinations(mv, r):
            assert set.intersection(*(set(q) for q in sub))


def test_minimal_violation_matches_brute_force_on_fixtures():
    """The optimized incremental search finds a smallest bad complete of the
    same size (and equally minimal) as a direct brute-force search, on both
    canonical fixtures. The specific complete found need not be unique --
    only its size and minimality are guaranteed."""
    import itertools

    def brute_force_minimal_violation(kg):
        for size in range(3, kg.order() + 1):
            for c in itertools.combinations(kg.nodes(), size):
                if set.intersection(*(set(q) for q in c)):
                    continue
                if all(
                    set.intersection(*(set(q) for q in sub))
                    for r in range(1, size)
                    for sub in itertools.combinations(c, r)
                ):
                    return frozenset(c)
        return None

    for fixture in (FIXTURE_1, FIXTURE_2):
        kg = clique_graph(fixture())
        got = minimal_violation(kg)
        expected = brute_force_minimal_violation(kg)
        assert got is not None
        assert len(got) == len(expected)
        assert not set.intersection(*(set(q) for q in got))
        for r in range(1, len(got)):
            for sub in itertools.combinations(got, r):
                assert set.intersection(*(set(q) for q in sub))


def test_witnesses_and_extension_octahedron():
    """Witnesses are pairwise distinct and extend to a clique of G."""
    g = octahedron(3)
    kg = clique_graph(g)
    mv = minimal_violation(kg)
    witnesses, q_star = witnesses_and_extension(g, kg, mv)
    assert len(witnesses) == len(mv)
    assert len(set(witnesses)) == len(witnesses)
    assert set(witnesses) <= set(q_star)
    # q_star is provably distinct from every element of the violation
    assert all(frozenset(q_star) != frozenset(q) for q in mv)


def test_witnesses_and_extension_rejects_foreign_violation():
    """A violation that isn't made of kg's own nodes raises ValueError."""
    import pytest

    g = octahedron(3)
    kg = clique_graph(g)
    with pytest.raises(ValueError):
        witnesses_and_extension(g, kg, [frozenset({99, 100})])


def test_witnesses_and_extension_fixtures():
    """Witness construction succeeds on both canonical fixtures."""
    for fixture in (FIXTURE_1, FIXTURE_2):
        g = fixture()
        kg = clique_graph(g)
        mv = minimal_violation(kg)
        witnesses, q_star = witnesses_and_extension(g, kg, mv)
        assert len(set(witnesses)) == len(mv)
        assert set(witnesses) <= set(q_star)


# ---------- 1d: h-closure and Theorem 15 ----------


def test_h_closure_is_idempotent():
    """h(h(X)) == h(X) for every complete X of a clique graph."""
    kg = clique_graph(octahedron(3))
    all_maximal = [frozenset(c) for c in nx.find_cliques(kg)]
    for X in nx.enumerate_all_cliques(kg):
        hx = h_closure(frozenset(X), kg, all_maximal)
        assert h_closure(hx, kg, all_maximal) == hx


def test_h_closure_of_maximal_clique_is_itself():
    """A maximal clique of kg is already a fixed point of h."""
    kg = clique_graph(octahedron(3))
    all_maximal = [frozenset(c) for c in nx.find_cliques(kg)]
    for q in all_maximal:
        assert h_closure(q, kg, all_maximal) == q


def test_delta_of_clique_set_single_clique_is_complete_graph():
    """Delta of a single clique is the complete graph on its vertices."""
    d = delta_of_clique_set([frozenset({0, 1, 2})])
    assert sorted(d.nodes()) == [0, 1, 2]
    assert d.number_of_edges() == 3


def test_delta_of_clique_set_empty_is_empty_graph():
    """Delta of an empty clique set has no vertices."""
    d = delta_of_clique_set([])
    assert d.number_of_nodes() == 0


def test_theorem15_hypothesis_holds_for_complete_graph():
    """K4 trivially satisfies Theorem 15's hypothesis."""
    holds, bad = theorem15_hypothesis_holds(nx.complete_graph(4))
    assert holds is True
    assert bad is None


def test_theorem15_hypothesis_fails_for_octahedron():
    """The octahedron is a known NOT-good example: K(octahedron(3)) is
    S^3 while octahedron(3) itself is S^2, so Theorem 15's hypothesis must
    correctly fail here (see also Priority 6's octahedron regression fact)."""
    holds, bad = theorem15_hypothesis_holds(octahedron(3))
    assert holds is False
    assert bad is not None


def test_theorem15_hypothesis_holds_fixture_1():
    """Fixture 1 (H?qdvbU): Theorem 15's hypothesis holds."""
    holds, _ = theorem15_hypothesis_holds(FIXTURE_1())
    assert holds is True


def test_theorem15_hypothesis_holds_fixture_2():
    """Fixture 2 (HUZv~zz): the sharpest stress test in the whole toolkit.
    Elementary collapse provably cannot succeed here, yet Theorem 15's
    hypothesis holds cleanly -- exactly the documented result."""
    holds, _ = theorem15_hypothesis_holds(FIXTURE_2())
    assert holds is True


def test_theorem15_hypothesis_size_cap_inconclusive():
    """A tiny size_cap makes the check bail out as inconclusive (None), not
    raise or silently hang."""
    holds, hx = theorem15_hypothesis_holds(FIXTURE_2(), size_cap=1)
    assert holds is None
    assert hx is not None


def test_smallest_non_clique_helly_graph_has_k4_clique_graph():
    """The smallest non-clique-Helly graph (6 vertices, three cliques with a
    common-triple-empty-but-pairwise-nonempty intersection pattern) has
    K(G) = K4."""
    g = nx.Graph()
    # Three triangles sharing vertices pairwise but not all together.
    g.add_edges_from(
        [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4), (4, 5), (5, 0), (4, 0)]
    )
    assert not is_clique_helly(g)
    kg = clique_graph(g)
    assert nx.is_isomorphic(kg, nx.complete_graph(4))

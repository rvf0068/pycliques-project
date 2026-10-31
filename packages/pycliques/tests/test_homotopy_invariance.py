from dataclasses import dataclass, replace

import networkx as nx
import pytest
from pycliques.cliques import clique_graph
from pycliques.helly import is_clique_helly
from pycliques.homotopy_invariance import (
    completes_of_size,
    completes_with_empty_intersection,
    delta_of_clique_set,
    discrete_morse_matching,
    h_closure,
    has_center_at_all,
    is_center,
    minimal_violation,
    neckties,
    theorem11_hypothesis_holds,
    theorem15_hypothesis_holds,
    verify_discrete_morse_matching,
    witnesses_and_extension,
)
from pycliques.homotopy_research import (
    EnumerationStatus,
    homotopy_structure_certificates,
    verify_homotopy_structure_certificates,
)
from pycliques.named import (
    collapse_obstruction_fixture,
    dominated_vertex_free_non_helly,
    octahedron,
    suspension_of_cycle,
)
from pycombtop import (
    HomotopyVerdict,
    Simplex,
    SimplicialComplex,
    WedgeOfSpheres,
    relative_betti_numbers,
)
from pycombtop.homotopy_type import betti_numbers_sc

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


@dataclass(frozen=True)
class _HashableLabel:
    value: str


def test_homotopy_structure_certificates_support_arbitrary_hashable_labels():
    graph = nx.octahedral_graph()
    graph = nx.relabel_nodes(
        graph, {vertex: _HashableLabel(str(vertex)) for vertex in graph}
    )

    result = homotopy_structure_certificates(graph)

    assert result.status is EnumerationStatus.COMPLETE
    assert result.graph6
    assert result.missing_faces
    assert result.to_json()
    assert verify_homotopy_structure_certificates(graph, result)
    certificate = result.missing_faces[0]
    invalid_certificate = replace(
        certificate,
        witness_vertices=(len(graph) + 1,) * len(certificate.clique_ids),
    )
    invalid_result = replace(
        result,
        missing_faces=(invalid_certificate, *result.missing_faces[1:]),
    )
    assert not verify_homotopy_structure_certificates(graph, invalid_result)


def test_homotopy_structure_budget_is_explicitly_inconclusive():
    graph = nx.octahedral_graph()

    result = homotopy_structure_certificates(graph, max_cliques=1)

    assert result.status is EnumerationStatus.INCONCLUSIVE
    assert result.explored_cliques == 0
    assert result.explored_graph_cliques == 2
    assert verify_homotopy_structure_certificates(graph, result)


def test_homotopy_structure_edge_budget_is_inconclusive():
    graph = nx.octahedral_graph()

    result = homotopy_structure_certificates(
        graph, max_cliques=100, max_clique_graph_edges=0
    )

    assert result.status is EnumerationStatus.INCONCLUSIVE
    assert result.explored_graph_cliques == 8
    assert result.explored_graph_edges == 1
    assert "max_clique_graph_edges=0" in result.diagnostics
    assert verify_homotopy_structure_certificates(graph, result)


def test_homotopy_structure_without_violations_verifies_none_minimum():
    graph = nx.complete_graph(4)
    result = homotopy_structure_certificates(graph)

    assert result.smallest_violation_ids is None
    assert verify_homotopy_structure_certificates(graph, result)


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
    """A clique of kg is already a fixed point of h."""
    kg = clique_graph(octahedron(3))
    all_maximal = [frozenset(c) for c in nx.find_cliques(kg)]
    for q in all_maximal:
        assert h_closure(q, kg, all_maximal) == q


def test_delta_of_clique_set_single_clique_is_a_full_simplex():
    """Delta of a single clique has that clique as its only facet."""
    d = delta_of_clique_set([frozenset({0, 1, 2})])
    assert d.vertex_set == {0, 1, 2}
    assert d.facet_set == {Simplex({0, 1, 2})}
    assert (
        d.all_simplices()
        == SimplicialComplex({0, 1, 2}, facet_set=[{0, 1, 2}]).all_simplices()
    )


def test_delta_of_clique_set_empty_is_empty_complex():
    """Delta of an empty clique set has no vertices."""
    d = delta_of_clique_set([])
    assert not d.vertex_set
    assert not d.facet_set


def test_delta_of_clique_set_preserves_nonflag_triangle_boundary():
    """Three edge facets remain an unfilled triangle."""
    d = delta_of_clique_set(
        [
            frozenset({0, 1}),
            frozenset({1, 2}),
            frozenset({0, 2}),
        ]
    )
    assert d.facet_set == {
        Simplex({0, 1}),
        Simplex({1, 2}),
        Simplex({0, 2}),
    }
    assert not d.is_clique_complex()
    assert betti_numbers_sc(d) == [0, 1]


def test_delta_of_h_closed_family_preserves_missing_triangle_face():
    """The graph6 octahedron example remains the genuine non-flag complex."""
    graph = nx.from_graph6_bytes(b"EznW")
    family = frozenset(
        {
            frozenset({0, 4, 5}),
            frozenset({1, 3, 5}),
            frozenset({2, 3, 4}),
        }
    )
    kg = clique_graph(graph)
    assert kg is not None
    assert family <= set(kg)
    maximal_kg = [frozenset(clique) for clique in nx.find_cliques(kg)]
    assert h_closure(family, kg, maximal_kg) == family

    delta = delta_of_clique_set(family)
    expected = SimplicialComplex(
        set().union(*family), facet_set={Simplex(clique) for clique in family}
    )
    assert delta.all_simplices() == expected.all_simplices()
    assert not delta.is_clique_complex()
    assert betti_numbers_sc(delta) == [0, 1]


def test_delta_of_clique_set_ignores_nonmaximal_generators():
    delta = delta_of_clique_set([frozenset({0, 1}), frozenset({0})])
    assert delta.facet_set == {Simplex({0, 1})}


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


def test_theorem15_hypothesis_fails_fixture_2_with_nonflag_homology():
    """The genuine h-closed family has nonzero rational first homology."""
    holds, witness = theorem15_hypothesis_holds(FIXTURE_2())
    assert holds is False
    assert witness is not None
    delta = delta_of_clique_set(witness)
    betti_numbers = relative_betti_numbers(
        delta.facet_set, lambda _face: False, method="exact"
    )
    assert betti_numbers[1] == 1


def test_theorem15_hypothesis_size_cap_inconclusive():
    """A tiny size_cap makes the check bail out as inconclusive (None), not
    raise or silently hang."""
    holds, hx = theorem15_hypothesis_holds(FIXTURE_2(), size_cap=1)
    assert holds is None
    assert hx is not None


def test_theorem15_complete_set_budget_stops_enumeration_early(monkeypatch):
    """A complete-set budget prevents exhaustive enumeration of K(G)."""
    enumerate_all_cliques = nx.enumerate_all_cliques
    enumerated = 0

    def tracked_enumeration(graph):
        nonlocal enumerated
        for complete in enumerate_all_cliques(graph):
            enumerated += 1
            yield complete

    monkeypatch.setattr(nx, "enumerate_all_cliques", tracked_enumeration)

    result = theorem15_hypothesis_holds(
        nx.star_graph(8),
        max_complete_sets=3,
    )

    assert result == (None, None)
    assert enumerated == 4


def test_theorem15_maximal_clique_budget_stops_indexing_early(monkeypatch):
    """A maximal-clique budget prevents materializing the full index."""
    clique_graph = nx.complete_bipartite_graph(4, 4)
    find_cliques = nx.find_cliques
    indexed = 0

    def tracked_find_cliques(graph):
        nonlocal indexed
        for clique in find_cliques(graph):
            indexed += 1
            yield clique

    monkeypatch.setattr(
        "pycliques.homotopy_invariance.clique_graph",
        lambda *_args, **_kwargs: clique_graph,
    )
    monkeypatch.setattr(nx, "find_cliques", tracked_find_cliques)

    def unexpected_enumeration(_graph):
        raise AssertionError("complete sets must not be enumerated after a budget exit")

    monkeypatch.setattr(nx, "enumerate_all_cliques", unexpected_enumeration)

    assert theorem15_hypothesis_holds(
        nx.path_graph(3),
        max_maximal_cliques=2,
    ) == (None, None)
    assert indexed == 3


@pytest.mark.parametrize(
    "graph_budget",
    [
        "max_clique_graph_vertices",
        "max_clique_graph_edges",
    ],
)
def test_theorem15_clique_graph_budget_exits_before_enumeration(
    monkeypatch, graph_budget
):
    def unexpected_enumeration(_graph):
        raise AssertionError("complete sets must not be enumerated after a budget exit")

    monkeypatch.setattr(nx, "enumerate_all_cliques", unexpected_enumeration)

    assert theorem15_hypothesis_holds(
        nx.path_graph(3),
        **{graph_budget: 0},
    ) == (None, None)


@pytest.mark.parametrize(
    "budget_name",
    [
        "max_complete_sets",
        "max_closed_h_images",
        "max_maximal_cliques",
        "max_clique_graph_vertices",
        "max_simplices_per_h_image",
    ],
)
def test_theorem15_rejects_negative_resource_budgets(budget_name):
    with pytest.raises(ValueError):
        theorem15_hypothesis_holds(nx.path_graph(3), **{budget_name: -1})


def test_theorem15_forwards_per_image_simplex_budget(monkeypatch):
    observed = {}

    def fake_homotopy_type(s_complex, *, max_simplices=None):
        observed["complex"] = s_complex
        observed["max_simplices"] = max_simplices
        return HomotopyVerdict(
            wedge=WedgeOfSpheres.contractible(),
            reason="test contractible result",
        )

    monkeypatch.setattr(
        "pycombtop.homotopy_type.homotopy_type_sc_with_verdict",
        fake_homotopy_type,
    )

    assert theorem15_hypothesis_holds(
        nx.path_graph(3),
        max_simplices_per_h_image=13,
    ) == (True, None)
    assert isinstance(observed["complex"], SimplicialComplex)
    assert observed["max_simplices"] == 13


def test_theorem15_rejects_negative_simplex_budget_before_computation(
    monkeypatch,
):
    def unexpected_computation(*_args, **_kwargs):
        raise AssertionError("invalid budgets must be rejected before computation")

    monkeypatch.setattr(
        "pycliques.homotopy_invariance.clique_graph",
        unexpected_computation,
    )

    with pytest.raises(ValueError, match="max_simplices_per_h_image"):
        theorem15_hypothesis_holds(
            nx.path_graph(3),
            max_simplices_per_h_image=-1,
        )


def test_theorem15_default_enumerates_closed_images_not_all_completes(monkeypatch):
    def unexpected_enumeration(_graph):
        raise AssertionError("closed-image enumeration must not enumerate completes")

    monkeypatch.setattr(nx, "enumerate_all_cliques", unexpected_enumeration)

    assert theorem15_hypothesis_holds(nx.path_graph(3)) == (True, None)


def test_theorem15_closed_image_budget_exits_inconclusively():
    assert theorem15_hypothesis_holds(
        nx.path_graph(3),
        max_closed_h_images=0,
    ) == (None, None)


def test_theorem15_per_image_limited_verdict_is_inconclusive(monkeypatch):
    limited = HomotopyVerdict(
        wedge=WedgeOfSpheres.contractible(),
        reason="test resource limit",
        is_exact=False,
        is_limited=True,
    )
    monkeypatch.setattr(
        "pycombtop.homotopy_type.homotopy_type_sc_with_verdict",
        lambda _complex, *, max_simplices=None: limited,
    )

    holds, witness = theorem15_hypothesis_holds(
        nx.path_graph(3),
        max_simplices_per_h_image=0,
    )

    assert holds is None
    assert witness is not None


def test_theorem15_large_simplex_budget_preserves_unlimited_result():
    graph = nx.path_graph(3)

    assert theorem15_hypothesis_holds(
        graph,
        max_simplices_per_h_image=10_000,
    ) == theorem15_hypothesis_holds(graph)


def test_theorem15_inexact_contractible_betti_result_is_inconclusive(monkeypatch):
    """An inexact zero-Betti result does not prove contractibility."""
    monkeypatch.setattr(
        "pycombtop.homotopy_type.homotopy_type_sc_with_verdict",
        lambda _complex, *, max_simplices=None: HomotopyVerdict(
            wedge=WedgeOfSpheres.contractible(),
            reason="test-only Betti numbers",
            is_exact=False,
        ),
    )
    holds, witness = theorem15_hypothesis_holds(nx.path_graph(3))
    assert holds is None
    assert witness is not None


def test_theorem15_limited_flag_takes_precedence_over_exactness(monkeypatch):
    """The limited flag takes precedence over an exactness default."""
    monkeypatch.setattr(
        "pycombtop.homotopy_type.homotopy_type_sc_with_verdict",
        lambda _complex, *, max_simplices=None: HomotopyVerdict(
            wedge=WedgeOfSpheres.contractible(),
            reason="test resource limit",
            is_limited=True,
        ),
    )
    holds, witness = theorem15_hypothesis_holds(nx.path_graph(3))
    assert holds is None
    assert witness is not None


def test_theorem15_recursive_limited_verdict_stays_inconclusive(monkeypatch):
    """Theorem 15 must preserve an upstream limited placeholder as unknown."""
    from pycombtop.homotopy_type import _try_special_neighbourhood

    limited = HomotopyVerdict(
        wedge=WedgeOfSpheres.contractible(),
        reason="test resource limit",
        is_exact=False,
        is_limited=True,
    )
    monkeypatch.setattr(
        "pycombtop.homotopy_type.homotopy_type_with_verdict",
        lambda _graph: limited,
    )
    recursively_derived = _try_special_neighbourhood(nx.cycle_graph(4))
    assert recursively_derived is not None
    assert recursively_derived.is_limited is True

    monkeypatch.setattr(
        "pycombtop.homotopy_type.homotopy_type_sc_with_verdict",
        lambda _complex, *, max_simplices=None: recursively_derived,
    )
    holds, witness = theorem15_hypothesis_holds(nx.path_graph(3))

    assert holds is None
    assert witness is not None


def test_theorem15_inexact_nonzero_betti_result_proves_failure(monkeypatch):
    """A nonzero reduced Betti number proves noncontractibility."""
    monkeypatch.setattr(
        "pycombtop.homotopy_type.homotopy_type_sc_with_verdict",
        lambda _complex, *, max_simplices=None: HomotopyVerdict(
            wedge=WedgeOfSpheres.sphere(1),
            reason="test-only nonzero reduced Betti number",
            is_exact=False,
        ),
    )
    holds, witness = theorem15_hypothesis_holds(nx.path_graph(3))
    assert holds is False
    assert witness is not None


def test_theorem15_exact_noncontractible_verdict_proves_failure(monkeypatch):
    monkeypatch.setattr(
        "pycombtop.homotopy_type.homotopy_type_sc_with_verdict",
        lambda _complex, *, max_simplices=None: HomotopyVerdict(
            wedge=WedgeOfSpheres.sphere(1),
            reason="test-only exact homotopy type",
            is_exact=True,
        ),
    )
    holds, witness = theorem15_hypothesis_holds(nx.path_graph(3))
    assert holds is False
    assert witness is not None


# ---------- 1e: discrete Morse matching ----------


def _bad_complete_faces(graph):
    """Return all bad completes of the graph's clique graph."""
    kg = clique_graph(graph)
    return [
        frozenset(complete)
        for complete in nx.enumerate_all_cliques(kg)
        if not set.intersection(*(set(clique) for clique in complete))
    ], kg


def test_discrete_morse_matching_is_reciprocal_and_acyclic():
    """The staged matcher returns a matching accepted by its verifier."""
    faces = [frozenset(face) for face in ({1}, {2}, {1, 2})]
    matching = discrete_morse_matching(faces, lambda face: len(face) >= 1)
    assert matching == {(frozenset({1}), frozenset({1, 2}))}
    assert verify_discrete_morse_matching(faces, matching, lambda face: len(face) >= 1)


def test_verify_discrete_morse_matching_rejects_shared_faces():
    """The verifier rejects a claimed matching that reuses a face."""
    faces = [frozenset(face) for face in (set(), {1}, {2}, {1, 2})]
    matching = {
        (frozenset(), frozenset({1})),
        (frozenset({1}), frozenset({1, 2})),
    }
    assert not verify_discrete_morse_matching(faces, matching, lambda _: True)


def test_discrete_morse_matching_handles_collapse_obstruction_fixture():
    """Fixture 2 receives a valid matching despite elementary collapse failure."""
    faces, kg = _bad_complete_faces(FIXTURE_2())
    matching = discrete_morse_matching(
        faces,
        lambda complete: not set.intersection(*(set(clique) for clique in complete)),
        ground_set=kg.nodes(),
    )
    assert matching
    assert verify_discrete_morse_matching(
        faces,
        matching,
        lambda complete: not set.intersection(*(set(clique) for clique in complete)),
    )


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

import networkx as nx
import pycliques.small as small
import pytest
from pycliques import CliqueBehavior, classify_clique_behavior
from pycliques.named import (
    complement_of_cycle,
    dominated_vertex_free_non_helly,
    snub_disphenoid,
    suspension_of_cycle,
)
from pycliques.retractions import retracts
from pycliques.small import (
    CliqueSequence,
    ReferenceStatus,
    Verdict,
    _classify_reference_dependency,
    _classify_suspension_2_coaffination,
    _make_clique_retraction_test,
    clear_reference_graphs,
    eventually_retracts_specially,
    register_reference_graph,
    suspension_bases,
)
from pycliques.small import (
    test_eventually_helly as public_test_eventually_helly,
)
from pyg6data.lists import list_graphs


def test_eventually_helly_immediately():
    result = public_test_eventually_helly(nx.cycle_graph(4))

    assert result == (Verdict.CONVERGENT, "is eventually Helly (index 0)", None)


def test_classify_clique_behavior_convergent():
    """A clique-Helly graph is classified as convergent."""
    result = classify_clique_behavior(nx.cycle_graph(4))

    assert isinstance(result, CliqueBehavior)
    assert result.verdict is Verdict.CONVERGENT
    assert result.iterations_checked >= 1
    assert result.bound_exceeded is False


def test_clique_sequence_exposes_read_only_state():
    """Sequence clients can inspect cache state without private attributes."""
    seq = CliqueSequence(nx.cycle_graph(4))

    assert seq.graph_count == 1
    assert seq.exhausted is False
    assert seq[1] is not None
    assert seq.graph_count == 2


def _labelled_suspension(base: nx.Graph) -> nx.Graph:
    """Build a suspension while preserving the base graph's labels."""
    graph = base.copy()
    graph.add_nodes_from(["u", "v"])
    for vertex in base:
        graph.add_edge("u", vertex)
        graph.add_edge("v", vertex)
    return graph


def test_suspension_bases_detects_connected_base_with_arbitrary_labels():
    """Structural suspension detection does not require integer labels."""
    labels = [("x", 1), ("x", 2), 7, "a", "b", ("z", 0)]
    base = nx.cycle_graph(labels)
    graph = _labelled_suspension(base)

    found = list(suspension_bases(graph))

    assert len(found) == 1
    u, v, detected_base = found[0]
    assert {u, v} == {"u", "v"}
    assert set(detected_base) == set(base)
    assert set(detected_base.edges()) == set(base.edges())


def test_theorem_4_6_accepts_noninvolutive_exact_2_coaffination(monkeypatch):
    """Theorem 4.6 accepts a radius-2 automorphism without requiring involution."""
    base = nx.cycle_graph(6)
    graph = _labelled_suspension(base)
    tau = {vertex: (vertex + 2) % 6 for vertex in base}

    monkeypatch.setattr(small, "coaffinations", lambda _graph, radius: iter([tau]))
    result = _classify_suspension_2_coaffination(graph)

    assert result is not None
    verdict, reason, certificate = result
    assert verdict is Verdict.DIVERGENT
    assert "Theorem 4.6" in reason
    assert certificate is not None
    assert certificate.rule == "theorem_4_6_suspension_2_coaffination"
    assert certificate.target_status == "proven_divergent"
    assert certificate.radius == 2
    assert certificate.suspension_vertices is not None
    assert set(certificate.suspension_vertices) == {"u", "v"}
    assert certificate.base_graph is not None
    assert nx.is_isomorphic(certificate.base_graph, base)
    assert certificate.base_coaffination == tau
    assert all(
        nx.shortest_path_length(base, vertex, tau[vertex]) >= 2 for vertex in base
    )
    assert any(tau[tau[vertex]] != vertex for vertex in base)


def test_classify_clique_behavior_uses_theorem_4_6_certificate():
    """The fast classifier applies Theorem 4.6 before pared-graph rules."""
    base = nx.cycle_graph(6)
    result = classify_clique_behavior(_labelled_suspension(base))

    assert result.verdict is Verdict.DIVERGENT
    assert result.certificate is not None
    assert result.certificate.rule == "theorem_4_6_suspension_2_coaffination"


def test_suspension_bases_require_connected_base():
    """A disconnected base is not a Theorem 4.6 suspension candidate."""
    base = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4))
    graph = _labelled_suspension(base)

    assert list(suspension_bases(graph)) == []
    assert _classify_suspension_2_coaffination(graph) is None


def test_theorem_4_6_rejects_connected_base_without_2_coaffination():
    """A genuine suspension without a 2-coaffination does not trigger."""
    graph = _labelled_suspension(nx.path_graph(4))

    assert list(suspension_bases(graph))
    assert _classify_suspension_2_coaffination(graph) is None


@pytest.mark.parametrize(
    "graph",
    [
        nx.star_graph(3),
        nx.complete_graph(4),
        nx.cycle_graph(5),
        nx.cycle_graph(4),
    ],
)
def test_suspension_bases_reject_false_candidates(graph):
    """Universal, adjacent, nonuniversal, and disconnected pairs are rejected."""
    assert list(suspension_bases(graph)) == []


def test_theorem_4_6_checks_multiple_suspension_pairs(monkeypatch):
    """The classifier continues after a valid pair whose base has no coaffination."""
    graph = nx.empty_graph(1)
    first_base = nx.path_graph(4)
    second_base = nx.cycle_graph(6)
    tau = {vertex: (vertex + 2) % 6 for vertex in second_base}

    monkeypatch.setattr(
        small,
        "suspension_bases",
        lambda _graph: iter([("a", "b", first_base), ("c", "d", second_base)]),
    )

    def fake_coaffinations(base, radius):
        return iter([tau]) if base is second_base else iter(())

    monkeypatch.setattr(small, "coaffinations", fake_coaffinations)

    result = _classify_suspension_2_coaffination(graph)

    assert result is not None
    assert result[2] is not None
    assert result[2].suspension_vertices == ("c", "d")


def test_eventual_helly_public_api_agrees_with_classifier():
    """The public Helly search and classifier share the same iteration path."""
    graph = nx.triangular_lattice_graph(3, 3)

    assert public_test_eventually_helly(graph) == (
        Verdict.CONVERGENT,
        "is eventually Helly (index 1)",
        None,
    )
    assert classify_clique_behavior(graph).verdict is Verdict.CONVERGENT


def test_classify_clique_behavior_bound_exceeded():
    """The public result reports when the clique bound aborts iteration."""
    result = classify_clique_behavior(dominated_vertex_free_non_helly(), bound=3)

    assert result.verdict is Verdict.INDETERMINATE
    assert result.bound_exceeded is True
    assert result.reason == "clique count exceeded bound"


def test_clique_sequence_reports_bound_exhaustion():
    """A failed bounded iteration is exposed as sequence exhaustion."""
    seq = CliqueSequence(nx.octahedral_graph(), bound=3)

    assert seq[1] is None
    assert seq.exhausted is True
    assert seq.graph_count == 1


def test_classify_clique_behavior_respects_iteration_limit():
    """An exhausted iteration limit is distinct from a clique-bound abort."""
    result = classify_clique_behavior(dominated_vertex_free_non_helly(), tries=1)

    assert result.verdict is Verdict.INDETERMINATE
    assert result.iterations_checked == 1
    assert result.bound_exceeded is False


@pytest.mark.parametrize("keyword", ["tries", "bound"])
def test_classify_clique_behavior_rejects_invalid_limits(keyword):
    """The public classifier rejects limits that cannot inspect an input graph."""
    with pytest.raises(ValueError, match=keyword):
        classify_clique_behavior(nx.cycle_graph(4), **{keyword: 0})


def test_eventually_retracts_specially_cycle_is_false():
    # C4 is Helly, so its iterated clique graphs converge
    assert eventually_retracts_specially(nx.cycle_graph(4)) is None


def test_eventually_retracts_specially_complete_is_false():
    # K3 is Helly and collapses immediately
    assert eventually_retracts_specially(nx.complete_graph(3)) is None


def test_eventually_retracts_specially_single_vertex():
    # A single vertex has order <= 1, should return False immediately
    assert eventually_retracts_specially(nx.trivial_graph()) is None


def test_eventually_retracts_specially_respects_max_steps():
    # With 0 steps, no iteration occurs so nothing can be found
    assert eventually_retracts_specially(nx.octahedral_graph(), tries=0) is None


def test_eventually_retracts_specially_path_is_false():
    # P4 is dismantlable and converges
    assert eventually_retracts_specially(nx.path_graph(4)) is None


def test_eventually_retracts():
    g = list_graphs(8)[11045]
    assert retracts(g, nx.octahedral_graph()) is False
    assert eventually_retracts_specially(g)


def test_special_octahedron_public_api_agrees_with_classifier():
    """The special-octahedron search agrees with classifier certification."""
    g = list_graphs(8)[11045]

    assert eventually_retracts_specially(g)
    result = classify_clique_behavior(g)
    assert result.verdict is Verdict.DIVERGENT


# ---------- Coverage for test_eventually_helly edge cases ----------


def test_test_eventually_helly_bound_exceeded():
    """A clique bound abort produces an indeterminate result."""
    # The octahedral graph is NOT clique-Helly and has 8 cliques.
    # With bound=3, clique_graph will return None.
    result = public_test_eventually_helly(nx.octahedral_graph(), bound=3)

    assert result[0] is Verdict.INDETERMINATE
    assert "bound" in result[1]


def test_test_eventually_helly_tries_exhausted():
    """A finite failure to find a Helly iterate is indeterminate."""
    # With tries=0, the loop never runs.  The octahedral graph is not Helly.
    result = public_test_eventually_helly(nx.octahedral_graph(), tries=0)

    assert result[0] is Verdict.INDETERMINATE
    assert "finite tested range" in result[1]


# ---------- Coverage for eventually_retracts_specially edge cases ----------


def test_eventually_retracts_specially_bound_exceeded():
    """When the clique bound is exceeded, return None."""
    # C6 has no special octahedra, and clique_graph(C6, bound=1) returns None.
    assert eventually_retracts_specially(nx.cycle_graph(6), bound=1) is None


# ---------- CLI tests for small-behavior ----------


def test_small_parse_args():
    """_parse_args parses the graph order and default bound correctly."""
    from pycliques.small import _parse_args

    args = _parse_args(["6"])
    assert args.n == 6
    assert args.bound == 30


def test_small_parse_args_bound():
    """_parse_args accepts an explicit clique bound."""
    from pycliques.small import _parse_args

    args = _parse_args(["--bound", "12", "6"])
    assert args.bound == 12


def test_small_parse_args_from_indeterminate_file():
    """_parse_args accepts the indeterminate-file second-pass flag."""
    from pycliques.small import _parse_args

    args = _parse_args(["--from-indeterminate-file", "9"])
    assert args.from_indeterminate_file is True


def test_small_parse_args_exclude_conjectured_divergent():
    """_parse_args accepts the conjectured-divergent filter flag."""
    from pycliques.small import _parse_args

    args = _parse_args(["--exclude-conjectured-divergent", "9"])
    assert args.exclude_conjectured_divergent is True


def test_small_parse_args_verbose():
    """_parse_args sets DEBUG loglevel with -v."""
    import logging

    from pycliques.small import _parse_args

    args = _parse_args(["-v", "6"])
    assert args.loglevel == logging.DEBUG


def test_small_main_invalid_order(capsys):
    """_main exits with error for unavailable order."""
    from pycliques.small import _main

    with pytest.raises(SystemExit):
        _main(["5"])


def test_small_main_runs_successfully():
    """_main completes for order 6 (112 connected graphs)."""
    from pycliques.small import _main

    # This processes all 112 connected graphs on 6 vertices.
    _main(["6"])


# ---------- Coverage for _make_clique_retraction_test ----------


def test_clique_retraction_to_comp_c10_positive():
    """K(Susp(C_5)) retracts to Comp(C_10), so the classifier returns DIVERGENT."""
    classifier = _make_clique_retraction_test(
        complement_of_cycle(10), "clique graph retracts to Comp(C_10)"
    )
    seq = CliqueSequence(suspension_of_cycle(5))
    result = classifier(seq)
    assert result is not None
    assert result[0] is Verdict.DIVERGENT


def test_clique_retraction_to_comp_c10_negative():
    """K4 is Helly; its clique graph does not retract to Comp(C_10)."""
    classifier = _make_clique_retraction_test(
        complement_of_cycle(10), "clique graph retracts to Comp(C_10)"
    )
    seq = CliqueSequence(nx.complete_graph(4))
    result = classifier(seq)
    assert result is None


# ---------- Coverage for --skip-dominated ----------


def test_small_parse_args_skip_dominated_default():
    """--skip-dominated defaults to True."""
    from pycliques.small import _parse_args

    args = _parse_args(["6"])
    assert args.skip_dominated is True


def test_small_parse_args_no_skip_dominated():
    """--no-skip-dominated sets skip_dominated to False."""
    from pycliques.small import _parse_args

    args = _parse_args(["--no-skip-dominated", "6"])
    assert args.skip_dominated is False


def test_small_main_skip_dominated(tmp_path):
    """With --skip-dominated, graphs with dominated vertices are skipped."""
    from pycliques.small import _main

    output = tmp_path / "verdicts.tsv"
    _main(["6", "--output-file", str(output)])
    lines = output.read_text().splitlines()
    reducible_lines = [line for line in lines if "REDUCIBLE" in line]
    assert len(reducible_lines) > 0


def test_ref_dep_retraction_proven_certificate():
    """A retraction to a proven divergent reference graph yields
    a divergent certificate.
    """
    clear_reference_graphs()
    try:
        h = nx.octahedral_graph()
        register_reference_graph(
            h, status=ReferenceStatus.PROVEN, label="octahedral_graph"
        )

        result = _classify_reference_dependency(h)

        assert result is not None
        verdict, reason, certificate = result
        assert verdict is Verdict.DIVERGENT
        assert certificate is not None
        assert certificate.rule == "retracts"
        assert certificate.target_status == "proven_divergent"
        assert certificate.target_label == "octahedral_graph"
        assert "proven" in reason.lower()
    finally:
        clear_reference_graphs()
        register_reference_graph(
            snub_disphenoid(),
            status=ReferenceStatus.CONJECTURED,
            label="snub_disphenoid",
        )
        register_reference_graph(
            nx.octahedral_graph(),
            status=ReferenceStatus.PROVEN,
            label="octahedral_graph",
        )


def test_ref_dep_retraction_conjectural_indeterminate():
    """A conjectural divergent target may only yield a conditional
    indeterminate verdict.
    """
    clear_reference_graphs()
    try:
        h = snub_disphenoid()
        register_reference_graph(
            h, status=ReferenceStatus.CONJECTURED, label="snub_disphenoid"
        )

        result = _classify_reference_dependency(h)

        assert result is not None
        verdict, reason, certificate = result
        assert verdict is Verdict.INDETERMINATE
        assert certificate is not None
        assert certificate.rule == "retracts"
        assert certificate.target_status == "conjectured_divergent"
        assert certificate.target_label == "snub_disphenoid"
        assert "conditional" in reason.lower()
    finally:
        clear_reference_graphs()
        register_reference_graph(
            snub_disphenoid(),
            status=ReferenceStatus.CONJECTURED,
            label="snub_disphenoid",
        )
        register_reference_graph(
            nx.octahedral_graph(),
            status=ReferenceStatus.PROVEN,
            label="octahedral_graph",
        )


def test_small_main_no_skip_dominated(tmp_path):
    """With --no-skip-dominated, no graphs are marked REDUCIBLE."""
    from pycliques.small import _main

    output = tmp_path / "verdicts.tsv"
    _main(["6", "--no-skip-dominated", "--output-file", str(output)])
    lines = output.read_text().splitlines()
    reducible_lines = [line for line in lines if "REDUCIBLE" in line]
    assert len(reducible_lines) == 0
    assert all("UNKNOWN" not in line for line in lines)


def test_classify_local_bridge_no_local_bridge():
    """A graph without local bridges returns None from classify_local_bridge."""
    from pycliques.small import classify_local_bridge

    assert classify_local_bridge(nx.complete_graph(3)) is None


def test_classify_local_bridge_divergent():
    """Contracting a local bridge attached to octahedral graph yields DIVERGENT."""
    from pycliques import classify_local_bridge

    # Octahedral graph + a leaf node attached by a bridge
    g = nx.octahedral_graph()
    g.add_edge(0, "leaf")

    local_res = classify_local_bridge(g)
    assert local_res is not None
    verdict, reason, certificate = local_res
    assert verdict is Verdict.DIVERGENT
    assert "Theorem 6.1" in reason
    assert certificate is not None
    assert certificate.rule == "local_bridge"
    assert certificate.target_status == "proven_divergent"
    assert certificate.target_label == "octahedral_graph"
    assert certificate.edge == (0, "leaf")


def test_classify_local_bridge_conjectured_divergent():
    """Contracting a local bridge attached to snub disphenoid yields
    INDETERMINATE with conjectured status.
    """
    from pycliques import classify_local_bridge

    # Snub disphenoid + a leaf node attached by a bridge
    g = snub_disphenoid()
    g.add_edge(0, "leaf")

    local_res = classify_local_bridge(g)
    assert local_res is not None
    verdict, reason, certificate = local_res
    assert verdict is Verdict.INDETERMINATE
    assert "conjectured" in reason
    assert "Theorem 6.1" in reason
    assert certificate is not None
    assert certificate.rule == "local_bridge"
    assert certificate.target_status == "conjectured_divergent"
    assert certificate.target_label == "snub_disphenoid"
    assert certificate.edge == (0, "leaf")


def test_classify_non_triangle_edge_none():
    """A complete graph has no edges in no triangle, returning None."""
    from pycliques import classify_non_triangle_edge

    assert classify_non_triangle_edge(nx.complete_graph(3)) is None


def test_classify_non_triangle_edge_divergent():
    """Deleting a non-triangle edge that yields a divergent graph certifies
    divergence.
    """
    from pycliques import classify_non_triangle_edge

    o3 = nx.octahedral_graph()
    g = o3.copy()
    g.add_edge("x", 0)
    g.add_edge("y", 3)
    g.add_edge("x", "y")

    res = classify_non_triangle_edge(g)
    assert res is not None
    verdict, reason, certificate = res
    assert verdict is Verdict.DIVERGENT
    assert "Theorem 6.2" in reason
    assert certificate is not None
    assert certificate.rule == "non_triangle_edge"
    assert certificate.target_status == "proven_divergent"
    assert certificate.edge is not None
    assert list(nx.common_neighbors(g, *certificate.edge)) == []


def test_classify_non_triangle_edge_conjectured_divergent():
    """Deleting a non-triangle edge that yields a conjectured divergent graph
    returns INDETERMINATE with conjectured status.
    """
    from pycliques import classify_non_triangle_edge

    snub = snub_disphenoid()
    c5 = nx.cycle_graph(5)
    c5 = nx.relabel_nodes(c5, {i: 10 + i for i in range(5)})

    g = nx.disjoint_union(snub, c5)
    g.add_edge(0, "x")
    g.add_edge(8, "y")
    g.add_edge("x", "y")

    res = classify_non_triangle_edge(g)
    assert res is not None
    verdict, reason, certificate = res
    assert verdict is Verdict.INDETERMINATE
    assert "conjectured" in reason
    assert "Theorem 6.2" in reason
    assert certificate is not None
    assert certificate.rule == "non_triangle_edge"
    assert certificate.target_status == "conjectured_divergent"
    assert certificate.target_label == "snub_disphenoid"
    assert certificate.edge is not None
    assert list(nx.common_neighbors(g, *certificate.edge)) == []


# ---------------------------------------------------------------------------
# Inverse cutpoint extension
# ---------------------------------------------------------------------------


def _motivating_example() -> tuple[nx.Graph, nx.Graph]:
    """Return (g, h) from the inverse-cutpoint-extension motivating example.

    ``g`` is a clockwork graph, known clique divergent.  ``h`` is obtained
    from ``g`` by identifying two vertices at distance four (add the edge,
    then contract it); the identified vertex is a local cutpoint of ``h``,
    but not an ordinary articulation point.
    """
    from pycliques.clockwork import clockwork_graph

    g = clockwork_graph(6 * [1], [[0] for _ in range(6)], 2, [0, 1])
    h = g.copy()
    h.add_edge(12, 15)
    h = nx.contracted_edge(h, (12, 15), self_loops=False)
    return g, h


def test_classify_clique_behavior_motivating_example_is_divergent():
    """The motivating example is classified DIVERGENT via the inverse
    cutpoint extension, chained with Theorem 6.2 and clockwork recognition.
    """
    _, h = _motivating_example()

    result = classify_clique_behavior(h)

    assert result.verdict is Verdict.DIVERGENT
    assert result.certificate is not None
    assert result.certificate.rule == "inverse_cutpoint_extension"
    assert result.certificate.target_status == "proven_divergent"


def test_classify_inverse_cutpoint_extension_divergent_directly():
    """Calling the rule directly finds the same admissible extension."""
    from pycliques import classify_inverse_cutpoint_extension

    _, h = _motivating_example()

    res = classify_inverse_cutpoint_extension(h)
    assert res is not None
    verdict, reason, certificate = res
    assert verdict is Verdict.DIVERGENT
    assert "Theorem 6.1" in reason
    assert "Theorem 6.2" in reason
    assert certificate is not None
    assert certificate.rule == "inverse_cutpoint_extension"
    assert certificate.target_status == "proven_divergent"
    assert certificate.map is not None
    assert certificate.map["cutpoint"] == 12


def test_classify_clique_behavior_detects_cutpoint_regardless_of_labels():
    """Relabeling with arbitrary hashables does not affect the classification."""
    _, h = _motivating_example()

    relabeled = nx.relabel_nodes(
        h,
        {n: (str(n), n, "tag") if n != 12 else "special" for n in h.nodes()},
    )

    result = classify_clique_behavior(relabeled)
    assert result.verdict is Verdict.DIVERGENT
    assert result.certificate is not None
    assert result.certificate.rule == "inverse_cutpoint_extension"


def test_classify_inverse_cutpoint_extension_none_without_divergent_target():
    """A local cutpoint whose only split yields a convergent target returns None."""
    from pycliques import classify_inverse_cutpoint_extension

    # Two triangles sharing vertex 2; splitting it gives two disjoint
    # triangles, which is convergent, not divergent.
    bowtie = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)])

    assert classify_inverse_cutpoint_extension(bowtie) is None
    assert classify_clique_behavior(bowtie).verdict is Verdict.CONVERGENT


def test_inverse_cutpoint_extension_rejects_invalid_split():
    """A split that separates two adjacent neighbors of the cutpoint is
    rejected and never produces a divergence certificate."""
    from pycliques.cutpoints import (
        InverseCutpointExtension,
        is_admissible_inverse_extension,
    )

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


def test_classify_inverse_cutpoint_extension_conjectured_divergent():
    """A split whose target is only conjectured divergent yields a
    conditional INDETERMINATE result, never a proven DIVERGENT verdict.
    """
    from pycliques import classify_inverse_cutpoint_extension
    from pycliques.cutpoints import contract_local_bridge

    s = snub_disphenoid()
    branch = s.copy()
    branch.add_edge(0, "p")
    branch.add_edge(1, "q")
    extended = branch.copy()
    extended.add_edge("p", "q")
    h = contract_local_bridge(extended, "p", "q")

    res = classify_inverse_cutpoint_extension(h)
    assert res is not None
    verdict, reason, certificate = res
    assert verdict is Verdict.INDETERMINATE
    assert "conjectured" in reason
    assert certificate is not None
    assert certificate.rule == "inverse_cutpoint_extension"
    assert certificate.target_status == "conjectured_divergent"
    assert certificate.target_label == "snub_disphenoid"

    # The global classifier must not silently promote this to DIVERGENT.
    overall = classify_clique_behavior(h)
    assert overall.verdict is not Verdict.DIVERGENT


def test_classify_inverse_cutpoint_extension_respects_budget():
    """A zero extension budget disables the rule immediately."""
    from pycliques import classify_inverse_cutpoint_extension

    _, h = _motivating_example()

    assert classify_inverse_cutpoint_extension(h, extension_budget=0) is None


def test_save_indeterminate_includes_certificate_metadata(tmp_path):
    """Saved indeterminate rows include certificate metadata when available."""
    from pycliques.small import _save_indeterminate

    g = nx.disjoint_union(snub_disphenoid(), nx.cycle_graph(5))
    g.add_edge(0, "x")
    g.add_edge(8, "y")
    g.add_edge("x", "y")

    result = classify_clique_behavior(g)
    assert result.certificate is not None

    _save_indeterminate(9, [(17, result.pared_graph, result.certificate)], tmp_path)

    saved = (tmp_path / "indeterminate_order_9.txt").read_text().splitlines()
    row = next(line for line in saved if line and not line.startswith("#"))
    assert "retracts" in row
    assert "conjectured_divergent" in row
    assert "snub_disphenoid" in row


def test_load_indeterminate_graphs_ignores_certificate_metadata(tmp_path):
    """Loading indeterminate graphs accepts the extended file format."""
    from pycliques.small import _load_indeterminate_graphs

    g6 = nx.to_graph6_bytes(nx.path_graph(4), header=False).decode("ascii").strip()
    path = tmp_path / "indeterminate_order_9.txt"
    path.write_text(
        "# Indeterminate clique behavior - connected graphs of order 9\n"
        "# Format: original_index pared_order graph6 certificate_rule "
        "certificate_target_status certificate_target_label\n"
        f"7 4 {g6} retracts conjectured_divergent snub_disphenoid\n"
    )

    known = _load_indeterminate_graphs(10, tmp_path)
    assert len(known[4]) == 1


def test_small_main_rechecks_and_removes_resolved_indeterminate_graphs(
    monkeypatch, tmp_path
):
    """The file second pass rewrites the file with unresolved graphs only."""
    import pycliques.small as small
    from pycliques.small import _main, _save_indeterminate

    convergent = nx.path_graph(4)
    indeterminate = nx.cycle_graph(5)
    _save_indeterminate(
        9,
        [(3, convergent, None), (7, indeterminate, None)],
        tmp_path,
    )

    def fake_classify(graph, *, bound):
        verdict = (
            Verdict.CONVERGENT
            if graph.number_of_edges() == 3
            else Verdict.INDETERMINATE
        )
        return CliqueBehavior(verdict, "fake", 1, False, graph)

    monkeypatch.setattr(small, "classify_clique_behavior", fake_classify)
    _main(["9", "--from-indeterminate-file", "--data-dir", str(tmp_path)])

    rows = [
        line
        for line in (tmp_path / "indeterminate_order_9.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert len(rows) == 1
    assert rows[0].split()[0] == "7"


def test_small_main_excludes_conjectured_divergent_from_recheck(monkeypatch, tmp_path):
    """The conjectured-divergent filter skips testing but preserves rows."""
    import pycliques.small as small
    from pycliques.small import Certificate, _main, _save_indeterminate

    ordinary = nx.path_graph(4)
    conjectured = nx.cycle_graph(5)
    _save_indeterminate(
        9,
        [
            (3, ordinary, None),
            (
                7,
                conjectured,
                Certificate(
                    rule="retracts",
                    target_status="conjectured_divergent",
                    target_label="example",
                ),
            ),
        ],
        tmp_path,
    )

    tested = []

    def fake_classify(graph, *, bound):
        tested.append(graph)
        return CliqueBehavior(Verdict.INDETERMINATE, "fake", 1, False, graph)

    monkeypatch.setattr(small, "classify_clique_behavior", fake_classify)
    _main(
        [
            "9",
            "--from-indeterminate-file",
            "--exclude-conjectured-divergent",
            "--data-dir",
            str(tmp_path),
        ]
    )

    assert len(tested) == 1
    rows = [
        line
        for line in (tmp_path / "indeterminate_order_9.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert {row.split()[0] for row in rows} == {"3", "7"}
    assert "conjectured_divergent" in next(row for row in rows if row.startswith("7 "))

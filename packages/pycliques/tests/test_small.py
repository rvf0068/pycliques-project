import networkx as nx
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
    _make_clique_retraction_test,
    clear_reference_graphs,
    eventually_retracts_specially,
    is_eventually_helly,
    register_reference_graph,
)
from pyg6data.lists import list_graphs


def test_eventually_helly():
    assert is_eventually_helly(nx.triangular_lattice_graph(4, 4))


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


def test_eventual_helly_public_api_agrees_with_classifier():
    """The public Helly search and classifier share the same iteration path."""
    graph = nx.triangular_lattice_graph(3, 3)

    assert is_eventually_helly(graph)
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


# ---------- Coverage for is_eventually_helly edge cases ----------


def test_is_eventually_helly_bound_exceeded():
    """When the clique bound is exceeded, is_eventually_helly returns False."""
    # The octahedral graph is NOT clique-Helly and has 8 cliques.
    # With bound=3, clique_graph will return None.
    assert is_eventually_helly(nx.octahedral_graph(), bound=3) is False


def test_is_eventually_helly_tries_exhausted():
    """When tries are exhausted without finding a Helly iterate, return False."""
    # With tries=0, the loop never runs.  The octahedral graph is not Helly.
    assert is_eventually_helly(nx.octahedral_graph(), tries=0) is False


# ---------- Coverage for eventually_retracts_specially edge cases ----------


def test_eventually_retracts_specially_bound_exceeded():
    """When the clique bound is exceeded, return None."""
    # C6 has no special octahedra, and clique_graph(C6, bound=1) returns None.
    assert eventually_retracts_specially(nx.cycle_graph(6), bound=1) is None


# ---------- CLI tests for small-behavior ----------


def test_small_parse_args():
    """_parse_args parses the graph order correctly."""
    from pycliques.small import _parse_args

    args = _parse_args(["6"])
    assert args.n == 6


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
        register_reference_graph(h, status=ReferenceStatus.PROVEN)

        result = _classify_reference_dependency(h)

        assert result is not None
        verdict, reason, certificate = result
        assert verdict is Verdict.DIVERGENT
        assert certificate is not None
        assert certificate.rule == "retracts"
        assert certificate.target_status == "proven_divergent"
        assert "proven" in reason.lower()
    finally:
        clear_reference_graphs()
        register_reference_graph(snub_disphenoid(), status=ReferenceStatus.CONJECTURED)
        register_reference_graph(nx.octahedral_graph(), status=ReferenceStatus.PROVEN)


def test_ref_dep_retraction_conjectural_indeterminate():
    """A conjectural divergent target may only yield a conditional
    indeterminate verdict.
    """
    clear_reference_graphs()
    try:
        h = snub_disphenoid()
        register_reference_graph(h, status=ReferenceStatus.CONJECTURED)

        result = _classify_reference_dependency(h)

        assert result is not None
        verdict, reason, certificate = result
        assert verdict is Verdict.INDETERMINATE
        assert certificate is not None
        assert certificate.rule == "retracts"
        assert certificate.target_status == "conjectured_divergent"
        assert "conditional" in reason.lower()
    finally:
        clear_reference_graphs()
        register_reference_graph(snub_disphenoid(), status=ReferenceStatus.CONJECTURED)
        register_reference_graph(nx.octahedral_graph(), status=ReferenceStatus.PROVEN)


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

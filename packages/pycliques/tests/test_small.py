import networkx as nx
import pytest
from pycliques.named import complement_of_cycle, suspension_of_cycle
from pycliques.retractions import retracts
from pycliques.small import (
    CliqueSequence,
    Verdict,
    _make_clique_retraction_test,
    eventually_retracts_specially,
    is_eventually_helly,
)
from pyg6data.lists import list_graphs


def test_eventually_helly():
    assert is_eventually_helly(nx.triangular_lattice_graph(4, 4))


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


# ---------- Coverage for is_eventually_helly edge cases ----------


def test_is_eventually_helly_bound_exceeded():
    """When the clique bound is exceeded, is_eventually_helly returns False."""
    # The octahedral graph is NOT clique-Helly and has 8 maximal cliques.
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

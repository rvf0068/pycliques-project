import networkx as nx
from pycliques.retractions import retracts
from pycliques.small import eventually_retracts_specially, is_eventually_helly
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

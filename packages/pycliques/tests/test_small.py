import networkx as nx
from pycliques.small import eventually_retracts_specially


def test_eventually_retracts_specially_cycle_is_false():
    # C4 is Helly, so its iterated clique graphs converge
    assert eventually_retracts_specially(nx.cycle_graph(4)) is False


def test_eventually_retracts_specially_complete_is_false():
    # K3 is Helly and collapses immediately
    assert eventually_retracts_specially(nx.complete_graph(3)) is False


def test_eventually_retracts_specially_single_vertex():
    # A single vertex has order <= 1, should return False immediately
    assert eventually_retracts_specially(nx.trivial_graph()) is False


def test_eventually_retracts_specially_respects_max_steps():
    # With 0 steps, no iteration occurs so nothing can be found
    assert eventually_retracts_specially(nx.octahedral_graph(), max_steps=0) is False


def test_eventually_retracts_specially_path_is_false():
    # P4 is dismantlable and converges
    assert eventually_retracts_specially(nx.path_graph(4)) is False

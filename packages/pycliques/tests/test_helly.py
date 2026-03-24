import networkx as nx
from pycliques.helly import (
    is_clique_helly,
    is_hereditary_clique_helly,
    n_closed,
    n_open,
    u_closed,
    u_open,
)


def test_n_open_complete_graph():
    """Common open neighbors in K4 are the other two vertices."""
    result = n_open(nx.complete_graph(4), 0, 1)
    assert result == {2, 3}


def test_n_open_path_graph():
    """Vertices 0 and 2 in P4 share only neighbor 1."""
    assert n_open(nx.path_graph(4), 0, 2) == {1}


def test_n_open_non_adjacent_no_common():
    """Vertices 0 and 3 in P4 have no common open neighbors."""
    assert n_open(nx.path_graph(4), 0, 3) == set()


def test_n_closed_complete_graph():
    """Closed common neighborhood in K4 includes all four vertices."""
    result = n_closed(nx.complete_graph(4), 0, 1)
    assert result == {0, 1, 2, 3}


def test_n_closed_path_graph():
    """Vertices 0 and 2 in P4: open common is {1}, plus {0, 2}."""
    assert n_closed(nx.path_graph(4), 0, 2) == {0, 1, 2}


def test_n_closed_non_adjacent():
    """Non-adjacent vertices with no common neighbors still appear."""
    assert n_closed(nx.path_graph(4), 0, 3) == {0, 3}


def test_u_open_complete_graph():
    """In K4 every common neighbor is universal."""
    result = u_open(nx.complete_graph(4), 0, 1)
    assert result == {2, 3}


def test_u_open_cycle_graph():
    g = nx.cycle_graph(5)
    assert u_open(g, 0, 1) == set()


def test_u_closed_complete_graph():
    """In K4 all vertices are universal in the closed neighborhood."""
    result = u_closed(nx.complete_graph(4), 0, 1)
    assert result == {0, 1, 2, 3}


def test_u_closed_path_graph():
    """In P4, u_closed(0, 2) should contain vertex 1 (dominates {0,1,2})."""
    g = nx.path_graph(4)
    result = u_closed(g, 0, 2)
    # n_closed = {0, 1, 2}, vertex 1 has closed neighborhood {0, 1, 2}
    assert 1 in result


def test_is_clique_helly_complete_graph():
    """Complete graphs are clique-Helly."""
    assert is_clique_helly(nx.complete_graph(4))


def test_is_clique_helly_octahedral():
    """The octahedral graph is not clique-Helly."""
    assert not is_clique_helly(nx.octahedral_graph())


def test_is_clique_helly_cycle():
    """Cycle graphs are clique-Helly (their cliques are edges)."""
    assert is_clique_helly(nx.cycle_graph(5))


def test_is_clique_helly_path():
    """Path graphs are clique-Helly."""
    assert is_clique_helly(nx.path_graph(4))


def test_is_clique_helly_empty_graph():
    """A graph with no edges is trivially clique-Helly."""
    assert is_clique_helly(nx.empty_graph(3))


def test_is_hereditary_clique_helly_complete_graph():
    """Complete graphs are hereditary clique-Helly."""
    assert is_hereditary_clique_helly(nx.complete_graph(4))


def test_is_hereditary_clique_helly_path():
    """Path graphs are hereditary clique-Helly."""
    assert is_hereditary_clique_helly(nx.path_graph(4))


def test_is_hereditary_clique_helly_cycle():
    """Cycle graphs are hereditary clique-Helly."""
    assert is_hereditary_clique_helly(nx.cycle_graph(5))


def test_is_hereditary_clique_helly_empty_graph():
    """A graph with no edges is trivially hereditary clique-Helly."""
    assert is_hereditary_clique_helly(nx.empty_graph(3))


def test_clique_helly_but_not_hereditary():
    """There exist graphs that are clique-Helly but not hereditary."""
    # The complement of C7 is known to be clique-Helly but not hereditary
    g = nx.complement(nx.cycle_graph(7))
    assert is_clique_helly(g)
    assert not is_hereditary_clique_helly(g)

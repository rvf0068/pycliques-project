import networkx as nx
import pytest
from pycliques import ClosedNecktieIndex, check_closed_centers
from pycombtop import check_closed_trace_contractibility


def test_center_failure_does_not_prevent_proof():
    graph = nx.from_graph6_bytes(b"Iy~s~V]ng")
    index = ClosedNecktieIndex(nx.find_cliques(graph))
    assert check_closed_centers(index).holds is False
    result = check_closed_trace_contractibility(index)
    assert result.proved
    assert result.visited > 0
    assert result.pruned_intervals > 0
    assert result.covered_maximal_completes > 0
    assert result.unresolved is None


def test_unresolved_and_limits():
    index = ClosedNecktieIndex([{0, 1}, {1, 2}, {2, 0}])
    result = check_closed_trace_contractibility(index)
    assert not result.proved
    assert result.unresolved == 7
    limited = check_closed_trace_contractibility(index, max_nodes=0)
    assert not limited.proved
    assert limited.visited == 0
    assert limited.unresolved is None
    with pytest.raises(ValueError):
        check_closed_trace_contractibility(index, max_nodes=-1)
    assert check_closed_trace_contractibility(ClosedNecktieIndex([])).proved


def test_orbit_accounting():
    index = ClosedNecktieIndex([{0, 1}, {1, 2}, {2, 3}, {3, 0}])
    result = check_closed_trace_contractibility(index, generators=[(1, 2, 3, 0)])
    assert result.proved
    assert result.completed_orbits == 1
    assert result.covered_maximal_completes == 4


def test_vertex_strategy_without_maximal_enumeration(monkeypatch):
    graph = nx.from_graph6_bytes(b"Iy~s~V]ng")
    index = ClosedNecktieIndex(nx.find_cliques(graph), lazy=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("maximal completes must not be enumerated")

    monkeypatch.setattr(index, "maximal_complete_orbits", forbidden)
    result = check_closed_trace_contractibility(index, strategy="vertex")
    assert result.proved
    assert result.completed_orbits == len(index.family)
    assert result.covered_maximal_completes == 0
    assert result.strategy == "vertex"
    limited = check_closed_trace_contractibility(index, strategy="vertex", max_nodes=0)
    assert not limited.proved
    assert limited.visited == 0
    with pytest.raises(ValueError):
        check_closed_trace_contractibility(index, strategy="invalid")

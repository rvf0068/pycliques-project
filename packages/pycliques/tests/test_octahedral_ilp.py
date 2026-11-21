"""MILP induced-octahedron regressions, including independent brute-force checks."""

import itertools
import random
from types import SimpleNamespace

import networkx as nx
import pytest
from pycliques.octahedral_ilp import (
    OctahedronILPStatus,
    find_induced_octahedron_ilp,
)

pytest.importorskip("scipy")


def _check_pairs(graph, pairs, dominating):
    assert pairs is not None
    endpoints = [v for pair in pairs for v in pair]
    assert len(endpoints) == len(set(endpoints))
    for i, (u, v) in enumerate(pairs):
        assert not graph.has_edge(u, v)
        for a, b in pairs[:i]:
            assert all(graph.has_edge(x, y) for x in (u, v) for y in (a, b))
    if dominating:
        assert not any(
            all(w == u or graph.has_edge(w, u) for u in endpoints) for w in graph
        )


def _brute_max_dimension(graph, dominating):
    edges = list(nx.non_edges(graph))
    best = 0
    for d in range(2, len(graph) // 2 + 1):
        for pairs in itertools.combinations(edges, d):
            endpoints = [v for pair in pairs for v in pair]
            if len(set(endpoints)) < 2 * d:
                continue
            if any(
                not graph.has_edge(x, y)
                for i, pair in enumerate(pairs)
                for other in pairs[i + 1 :]
                for x in pair
                for y in other
            ):
                continue
            if dominating and any(
                all(v == u or graph.has_edge(v, u) for u in endpoints) for v in graph
            ):
                continue
            best = max(best, d)
    return best


@pytest.mark.parametrize("d", [3, 4, 5])
def test_exact_maximum_on_octahedra(d):
    graph = nx.complete_multipartite_graph(*([2] * d))
    found = find_induced_octahedron_ilp(graph, time_limit=10)
    assert found.status is OctahedronILPStatus.OPTIMAL
    assert found.dimension == d
    _check_pairs(graph, found.antipodal_pairs, True)


def test_arbitrary_labels_and_feasibility_objective():
    original = nx.complete_multipartite_graph(2, 2, 2, 2)
    graph = nx.relabel_nodes(original, {v: ("vertex", v) for v in original})
    found = find_induced_octahedron_ilp(
        graph, min_dimension=3, maximize_dimension=False, time_limit=10
    )
    assert found.status is OctahedronILPStatus.FOUND
    assert found.dimension >= 3
    _check_pairs(graph, found.antipodal_pairs, True)


def test_domination_is_separate_from_inducedness():
    graph = nx.octahedral_graph()
    graph.add_edges_from((6, v) for v in range(6))
    impossible = find_induced_octahedron_ilp(graph)
    assert impossible.status is OctahedronILPStatus.INFEASIBLE
    allowed = find_induced_octahedron_ilp(graph, require_dominating=False)
    assert allowed.status is OctahedronILPStatus.OPTIMAL
    assert allowed.dimension == 3
    _check_pairs(graph, allowed.antipodal_pairs, False)


@pytest.mark.parametrize(
    "graph", [nx.complete_graph(6), nx.empty_graph(6), nx.cycle_graph(5)]
)
def test_unsatisfiable_instances(graph):
    result = find_induced_octahedron_ilp(graph, min_dimension=3)
    assert result.status is OctahedronILPStatus.INFEASIBLE
    assert result.antipodal_pairs is None


def test_matches_brute_force_on_small_random_graphs():
    rng = random.Random(20261121)
    for order in (6, 7, 8):
        for _ in range(4):
            graph = nx.gnp_random_graph(
                order, rng.uniform(0.45, 0.9), seed=rng.randrange(1_000_000)
            )
            for dominating in (False, True):
                answer = _brute_max_dimension(graph, dominating)
                result = find_induced_octahedron_ilp(
                    graph,
                    min_dimension=2,
                    maximize_dimension=True,
                    require_dominating=dominating,
                    time_limit=10,
                )
                if answer < 2:
                    assert result.status is OctahedronILPStatus.INFEASIBLE
                else:
                    assert result.status is OctahedronILPStatus.OPTIMAL
                    assert result.dimension == answer
                    _check_pairs(graph, result.antipodal_pairs, dominating)


def test_limit_with_incumbent_is_not_falsely_optimal(monkeypatch):
    import scipy.optimize as optimize

    real_milp = optimize.milp

    def simulated_limit(*args, **kwargs):
        success = real_milp(*args, **kwargs)
        return SimpleNamespace(status=1, x=success.x, message="simulated limit")

    monkeypatch.setattr(optimize, "milp", simulated_limit)
    result = find_induced_octahedron_ilp(nx.octahedral_graph())
    assert result.status is OctahedronILPStatus.FEASIBLE_LIMIT
    _check_pairs(nx.octahedral_graph(), result.antipodal_pairs, True)


@pytest.mark.parametrize(
    "status,expected",
    [
        (1, OctahedronILPStatus.LIMIT_REACHED),
        (4, OctahedronILPStatus.SOLVER_ERROR),
    ],
)
def test_no_incumbent_is_inconclusive(monkeypatch, status, expected):
    import scipy.optimize as optimize

    monkeypatch.setattr(
        optimize,
        "milp",
        lambda *a, **kw: SimpleNamespace(status=status, x=None, message="halt"),
    )
    result = find_induced_octahedron_ilp(nx.octahedral_graph())
    assert result.status is expected
    assert result.antipodal_pairs is None


@pytest.mark.parametrize("graph", [nx.DiGraph(), nx.MultiGraph(), nx.Graph([(0, 0)])])
def test_invalid_graphs(graph):
    with pytest.raises(ValueError, match="simple undirected"):
        find_induced_octahedron_ilp(graph)


@pytest.mark.parametrize(
    "params",
    [
        {"min_dimension": 1},
        {"min_dimension": 3.0},
        {"min_dimension": True},
        {"maximize_dimension": 1},
        {"require_dominating": 1},
        {"time_limit": 0},
        {"time_limit": -1},
        {"time_limit": float("inf")},
        {"time_limit": True},
    ],
)
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        find_induced_octahedron_ilp(nx.octahedral_graph(), **params)


P4_GRAPH6 = (
    "~?@?~~~~~~~~~~~~~~~v~~~~~~}~~~~~~~~{~~~N~~z~~~f~~~~~~~^~~~~~~~~~~~~~~~"
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~r|~~f}N~~}|v^F~~~u^N~~~wN~~~~~~|~wV~~~"
    "Fv^~~~~~|@~~~~~wR~~~~~}k^~~~~}G~~~~~vI^~~f~~~m^~~t~{^~~~~nyy~~~"
    "~zvw^~~~~}~^v~x~~~~f~~{^~~~w~~~f~~~nZe~~~~~~{F~~~~~~~M~h~~~~~~"
    "n^c~~~~~~~fxN~~~~~~~N~nx~~~~~s~{~~~~~~v~U^~~~~~~r~xN~~~~~~~~jZ^"
    "~~~~~~~~~gZ~~~~~"
)


@pytest.mark.slow
def test_census_fourth_iterate_optimal_dominating_o12():
    # McKay 10510367, archived P4 in census PR #60. This is NOT a retract test.
    graph = nx.from_graph6_bytes(P4_GRAPH6.encode())
    assert (len(graph), graph.number_of_edges()) == (64, 1844)
    result = find_induced_octahedron_ilp(
        graph,
        min_dimension=12,
        maximize_dimension=True,
        require_dominating=True,
        time_limit=30,
    )
    assert result.status is OctahedronILPStatus.OPTIMAL
    assert result.dimension == 12
    _check_pairs(graph, result.antipodal_pairs, True)

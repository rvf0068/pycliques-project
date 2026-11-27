"""Named-poset examples and exact small-group enumeration regressions."""

from itertools import combinations

import networkx as nx
import pytest
from sympy.combinatorics.named_groups import (
    AbelianGroup,
    CyclicGroup,
    DihedralGroup,
    SymmetricGroup,
)

from pycombtop import (
    Poset,
    SubgroupEnumerationLimit,
    beat_core,
    betti_numbers_sc,
    boolean_lattice,
    brown_poset,
    enumerate_subgroups,
    induced_permutation,
    nontrivial_partitions,
    order_complex,
    partition_graph,
    partition_lattice,
    quillen_poset,
    refines,
    subgroup_conjugation,
    subgroup_lattice,
)


def test_boolean_lattice_and_truncation():
    p = boolean_lattice(4)
    assert len(p) == 16
    assert p.minimal_elements() == {frozenset()}
    assert p.maximal_elements() == {frozenset(range(4))}
    assert p.join([frozenset({0}), frozenset({1})]) == frozenset({0, 1})
    proper = boolean_lattice(4, proper=True)
    assert len(proper) == 14
    assert order_complex(proper).dimension() == 2
    assert betti_numbers_sc(order_complex(proper)) == [0, 0, 1]
    truncated = boolean_lattice(4, min_rank=1, max_rank=2)
    assert len(truncated) == 10
    assert order_complex(truncated).dimension() == 1
    assert betti_numbers_sc(order_complex(truncated)) == [0, 3]
    mixed = boolean_lattice(["a", 7], proper=True)
    assert set(mixed) == {frozenset({"a"}), frozenset({7})}
    assert len(boolean_lattice(0)) == 1
    assert len(boolean_lattice(0, proper=True)) == 0
    assert len(boolean_lattice(1, proper=True)) == 0
    for ground in (-1, True, [1, 1]):
        with pytest.raises(ValueError):
            boolean_lattice(ground)
    with pytest.raises(ValueError):
        boolean_lattice(4, min_rank=3, max_rank=2)
    with pytest.raises(ValueError):
        boolean_lattice(4, max_rank=5)
    with pytest.raises(ValueError):
        boolean_lattice(4, min_rank=False)


@pytest.mark.parametrize("n,bell", [(0, 1), (1, 1), (2, 2), (3, 5), (4, 15), (5, 52)])
def test_partition_lattice(n, bell):
    p = partition_lattice(n)
    assert len(p) == bell
    assert len(p.minimal_elements()) == len(p.maximal_elements()) == 1
    assert all(p.leq(a, b) == refines(a, b) for a in p for b in p)
    proper = partition_lattice(n, proper=True)
    assert len(proper) == max(0, bell - 2)
    if n >= 3:
        graph, labels = partition_graph(n)
        assert labels == nontrivial_partitions(n) == list(proper)
        indexed = nx.relabel_nodes(
            proper.comparability_graph(), {part: i for i, part in enumerate(labels)}
        )
        assert set(graph.edges) == set(indexed.edges)
        g = SymmetricGroup(n).generators[0]
        action = induced_permutation(
            g, n, labels, {part: i for i, part in enumerate(labels)}
        )
        relabeled = nx.relabel_nodes(graph, dict(enumerate(action.array_form)))
        assert {frozenset(edge) for edge in graph.edges} == {
            frozenset(edge) for edge in relabeled.edges
        }


def test_partition_lattice_topology_and_rank_selection():
    p = partition_lattice(4, proper=True)
    assert betti_numbers_sc(order_complex(p)) == [0, 6]
    rank_one = partition_lattice(4, min_rank=1, max_rank=1)
    assert len(rank_one) == 6
    assert not rank_one.comparability_graph().edges
    assert len(partition_lattice(5, min_rank=1, max_rank=2)) == 35
    with pytest.raises(ValueError):
        partition_lattice(-1)
    with pytest.raises(ValueError):
        partition_lattice(4, max_rank=4)
    with pytest.raises(ValueError):
        nontrivial_partitions(2)


def _brute_subgroups(group):
    elements = tuple(group.generate_schreier_sims())
    identity = group.identity
    return {
        h
        for r in range(1, len(elements) + 1)
        for subset in combinations(elements, r)
        if identity in (h := frozenset(subset))
        and all(x * y in h for x in h for y in h)
    }


@pytest.mark.parametrize(
    "group,count",
    [
        (CyclicGroup(1), 1),
        (CyclicGroup(6), 4),
        (SymmetricGroup(3), 6),
        (AbelianGroup(2, 2), 5),
        (DihedralGroup(4), 10),
    ],
)
def test_subgroups_exact(group, count):
    subgroups = enumerate_subgroups(group)
    assert len(subgroups) == count
    assert set(subgroups) == _brute_subgroups(group)
    p = subgroup_lattice(group)
    assert set(p) == set(subgroups)
    assert all(p.leq(a, b) == (a <= b) for a in p for b in p)
    assert len(beat_core(p).core) == 1
    proper = subgroup_lattice(group, proper=True)
    assert len(proper) == max(0, count - 2)
    assert proper == p.induced(proper)
    for a in p:
        for b in p:
            assert p.meet([a, b]) == a & b
            assert a | b <= p.join([a, b])


def test_brown_and_quillen_differ_on_cyclic_p_group():
    group = CyclicGroup(4)
    b = brown_poset(group, 2)
    q = quillen_poset(group, 2)
    assert sorted(map(len, b)) == [2, 4]
    assert sorted(map(len, q)) == [2]
    assert q == b.induced(q)
    assert not len(brown_poset(group, 3))
    assert not len(quillen_poset(group, 3))
    assert not len(brown_poset(CyclicGroup(1), 2))
    assert not len(quillen_poset(CyclicGroup(1), 2))


def test_elementary_abelian_and_conjugation():
    v = AbelianGroup(2, 2)
    assert brown_poset(v, 2) == quillen_poset(v, 2)
    assert len(quillen_poset(v, 2)) == 4
    assert betti_numbers_sc(order_complex(subgroup_lattice(v, proper=True))) == [2]
    g = SymmetricGroup(3)
    q = quillen_poset(g, 2)
    assert q == brown_poset(g, 2)
    assert len(q) == 3
    assert betti_numbers_sc(order_complex(q)) == [2]
    assert len(quillen_poset(g, 3)) == 1
    for generator in g.generators:
        action = subgroup_conjugation(q, generator)
        assert set(action.values()) == set(q)
        assert q.is_order_preserving(q, action)
    with pytest.raises(ValueError):
        subgroup_conjugation(Poset([0]), g.identity)


def test_symmetric_group_four_and_filtered_enumeration():
    g = SymmetricGroup(4)
    p = subgroup_lattice(g)
    assert len(p) == 30
    for prime in (2, 3):
        b = brown_poset(g, prime)
        q = quillen_poset(g, prime)
        expected_b = []
        expected_q = []
        for h in p:
            k = len(h)
            while k % prime == 0:
                k //= prime
            if len(h) > 1 and k == 1:
                expected_b.append(h)
                if all(x.order() in (1, prime) for x in h) and all(
                    x * y == y * x for x in h for y in h
                ):
                    expected_q.append(h)
        assert b == p.induced(expected_b)
        assert q == p.induced(expected_q)
        assert betti_numbers_sc(order_complex(b)) == betti_numbers_sc(order_complex(q))


def test_subgroup_validation_and_budgets():
    group = SymmetricGroup(3)
    for prime in (1, 4, -2, True):
        with pytest.raises(ValueError):
            brown_poset(group, prime)
        with pytest.raises(ValueError):
            quillen_poset(group, prime)
    with pytest.raises(ValueError):
        enumerate_subgroups(group, elementary=True)
    with pytest.raises(TypeError):
        subgroup_lattice("S3")
    with pytest.raises(ValueError):
        subgroup_lattice(group, max_group_order=0)
    with pytest.raises(ValueError):
        subgroup_lattice(group, max_subgroups=0)
    with pytest.raises(SubgroupEnumerationLimit, match="max_group_order"):
        subgroup_lattice(group, max_group_order=5)
    with pytest.raises(SubgroupEnumerationLimit, match="max_subgroups"):
        subgroup_lattice(group, max_subgroups=5)
    assert len(subgroup_lattice(group, max_subgroups=6)) == 6
    assert len(subgroup_lattice(group, max_group_order=None, max_subgroups=None)) == 6

import networkx as nx
import pytest
from sympy.combinatorics import Permutation

from research.partition_lattice import (
    decompose,
    induced_permutation,
    nontrivial_partitions,
    partition_graph,
    refines,
)


def test_nontrivial_partitions_exclude_finest_and_coarsest():
    partitions = nontrivial_partitions(4)

    assert len(partitions) == 13
    assert len(set(partitions)) == 13
    assert all(1 < len(partition) < 4 for partition in partitions)
    assert all(
        set().union(*partition) == set(range(4)) and all(block for block in partition)
        for partition in partitions
    )


@pytest.mark.parametrize("function", [nontrivial_partitions, partition_graph])
def test_partition_lattice_helpers_require_at_least_three_elements(function):
    with pytest.raises(ValueError, match="at least 3"):
        function(2)


def test_refinement_relation_and_comparability_graph():
    finer = frozenset({frozenset({0}), frozenset({1}), frozenset({2, 3})})
    coarser = frozenset({frozenset({0, 1}), frozenset({2, 3})})
    incomparable = frozenset({frozenset({0, 2}), frozenset({1, 3})})

    assert refines(finer, coarser)
    assert not refines(coarser, finer)
    assert not refines(finer, incomparable)

    graph, partitions = partition_graph(4)
    index = {partition: i for i, partition in enumerate(partitions)}
    assert graph.has_edge(index[finer], index[coarser])
    assert not graph.has_edge(index[finer], index[incomparable])


def test_symmetric_group_action_preserves_partition_comparability():
    graph, partitions = partition_graph(4)
    index = {partition: i for i, partition in enumerate(partitions)}
    action = induced_permutation(Permutation(0, 1), 4, partitions, index)
    relabeling = dict(enumerate(action.array_form))
    relabeled_graph = nx.relabel_nodes(graph, relabeling)

    assert set(graph.edges) == set(relabeled_graph.edges)
    assert sorted(action(i) for i in range(len(partitions))) == list(
        range(len(partitions))
    )


def test_small_partition_lattice_homology_decompositions():
    n_three = decompose(3, verbose=False)
    assert n_three == {0: {(3,): 1, (2, 1): 1}}

    n_four = decompose(4, verbose=False)
    assert n_four == {
        0: {(4,): 1},
        1: {(2, 1, 1): 1, (3, 1): 1},
    }


def test_decompose_rejects_invalid_size_and_degree():
    with pytest.raises(ValueError, match="at least 3"):
        decompose(2, verbose=False)
    with pytest.raises(ValueError, match="max_dim"):
        decompose(3, max_dim=-1, verbose=False)

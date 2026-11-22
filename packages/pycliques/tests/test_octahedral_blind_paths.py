"""Behavioral coverage of generic octahedral search and failure paths.

These are small, deterministic graph tests; the census-scale snub replay
remains an optional slow regression test.
"""

import json
import random

import networkx as nx
import pytest

from pycliques import (
    octahedral_blind as ob,
    octahedral_blind_search as bs,
    octahedral_joint_fibers as jf,
)


def octahedron(d):
    return nx.complete_multipartite_graph(*([2] * d))


def base_pairs(d):
    return tuple((2 * i, 2 * i + 1) for i in range(d))


def test_fiber_operations_checkpoint_and_relabeling(tmp_path):
    graph = nx.relabel_nodes(octahedron(4), {i: f"v{i}" for i in range(8)})
    masks = jf.Masks(graph)
    all_nodes = masks.all
    left = masks.index["v0"]
    right = masks.index["v1"]
    pair = ((left, right),)
    common = masks.closed[left] & masks.closed[right]
    state = jf.State(pair, (((1 << left), (1 << right)),),
                     (1 << left) | (1 << right), common)
    assert jf.check_partial_state(masks, state)
    assert state.mapped() == 2
    assert jf.projection(masks, state) == {"v0": "v0", "v1": "v1"}
    assert jf.count_nonedge_pairs(masks, common) == 3
    assert len(jf.sample_compatible_pairs(masks, common, random.Random(2), 12)) == 3
    bicliques = jf.closed_biclique(masks, left, right, all_nodes,
                                  random.Random(4), variants=4)
    assert bicliques
    for a, b in bicliques:
        assert (a >> left) & 1 and (b >> right) & 1
        assert all((masks.nonedge[x] & b) == b for x in jf.bit_indices(a))

    dest = tmp_path / "partial.json"
    jf.save_partial_state(dest, masks, state)
    assert jf.load_partial_state(dest, masks) == state
    original = json.loads(dest.read_text())
    modified = dict(original, mapped_vertices=999)
    dest.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="mapped-vertex count"):
        jf.load_partial_state(dest, masks)
    dest.write_text(json.dumps(dict(original, source_order=999)))
    with pytest.raises(ValueError, match="dimensions differ"):
        jf.load_partial_state(dest, masks)
    dest.write_text(json.dumps(dict(original, format="old-format")))
    with pytest.raises(ValueError, match="checkpoint format"):
        jf.load_partial_state(dest, masks)

    empty = jf.retain_fibers(masks, state, ())
    assert empty.pairs == () and empty.covered == 0 and empty.common == all_nodes
    assert jf.check_partial_state(masks, empty)
    assert jf.retain_fibers(masks, state, (0, 0)) == state
    with pytest.raises(ValueError, match="kept-pair"):
        jf.retain_fibers(masks, state, (3,))
    with pytest.raises(ValueError, match="kept-pair"):
        jf.retain_fibers(masks, state, (-1,))


def test_joint_search_reports_dead_ends_and_resumes_states():
    complete = nx.complete_graph(8)
    mask = jf.Masks(complete)
    assert jf.sample_compatible_pairs(mask, mask.all, random.Random(1), 20) == []
    result = jf.joint_beam_search(complete, 3, beam=2, samples=3, verbose=False)
    assert result is not None
    with pytest.raises(ValueError, match="cannot be empty"):
        jf.joint_beam_search(octahedron(3), 3, initial_states=[], verbose=False)
    graph = octahedron(4)
    masks = jf.Masks(graph)
    empty = jf.State((), (), 0, masks.all)
    partial = jf.State(((0, 1),), (((1 << 0), (1 << 1)),),
                       (1 << 0) | (1 << 1), masks.closed[0] & masks.closed[1])
    with pytest.raises(ValueError, match="equal dimensions"):
        jf.joint_beam_search(graph, 4, initial_states=[empty, partial], verbose=False)
    resumed = jf.joint_beam_search(graph, 4, initial_states=[partial],
                                   seed=9, beam=5, samples=20, steps=2,
                                   verbose=False)
    assert resumed is not None
    assert 1 <= len(resumed.pairs) <= 2
    assert jf.check_partial_state(masks, resumed)


def test_joint_search_completion_oracle_and_callback():
    graph = octahedron(4)
    seen = []

    def oracle(masks, state, remainder):
        used = {a // 2 for p in state.pairs for a in p}
        completion = [(masks.nodes[2 * i], masks.nodes[2 * i + 1])
                      for i in range(4) if i not in used]
        assert len(completion) == remainder
        return completion

    def callback(masks, state):
        assert jf.check_partial_state(masks, state)
        seen.append(len(state.pairs))
        return len(state.pairs) == 4

    result = jf.joint_beam_search(graph, 4, seed=17, beam=5, samples=15,
                                   completion_oracle=oracle, oracle_start=1,
                                   oracle_per_depth=2, callback=callback,
                                   verbose=False)
    assert result is not None and len(result.pairs) == 4
    assert 4 in seen
    assert jf.check_partial_state(jf.Masks(graph), result)


def test_joint_search_early_callback_and_empty_oracle():
    graph = octahedron(4)
    first = jf.joint_beam_search(graph, 4, seed=22, beam=4, samples=12,
                                  callback=lambda m, s: len(s.pairs) == 1,
                                  verbose=False)
    assert first is not None and len(first.pairs) == 1
    second = jf.joint_beam_search(graph, 4, seed=22, beam=4, samples=12,
                                   completion_oracle=lambda m, s, r: (),
                                   oracle_start=1, steps=2, verbose=False)
    assert second is not None and len(second.pairs) >= 1


def test_fiber_public_validation_and_inconclusive_graph():
    with pytest.raises(ValueError, match="target_dimension"):
        jf.discover_partial_octahedral_fibers(octahedron(3), 2)
    for keyword, value in (("seed", True), ("beam", 0), ("samples", 0),
                           ("biclique_variants", 0)):
        with pytest.raises(ValueError):
            jf.discover_partial_octahedral_fibers(octahedron(3), 3,
                                                 **{keyword: value})
    result = jf.discover_partial_octahedral_fibers(nx.complete_graph(8), 3,
                                                  seed=19, beam=2, samples=4)
    assert result.antipodal_pairs == () and result.projection == {}
    assert result.covered_vertices == 0


def test_learn_conflicts_and_conflict_guided_candidate_filter():
    graph = octahedron(3)
    graph.add_nodes_from(("left", "right"))
    graph.add_edges_from(("left", v) for v in (0, 2, 3, 4, 5))
    graph.add_edges_from(("right", v) for v in (1, 2, 3, 4, 5))
    graph.add_edge("left", "right")
    masks = ob._OctahedralMasks(graph)
    full = masks.pairs(graph, base_pairs(3))
    cores = bs._learn_conflicts(masks, full, rounds=12, seed=19)
    assert cores
    for core in cores:
        indexed = tuple(p for i, p in enumerate(full) if core & (1 << i))
        assert not ob._screen_masks(masks, indexed).possible
        for i in range(3):
            if core & (1 << i):
                smaller = tuple(p for j, p in enumerate(full)
                                if (core & (1 << j)) and j != i)
                assert ob._screen_masks(masks, smaller).possible
    good_graph = octahedron(4)
    good = ob._OctahedralMasks(good_graph)
    assert bs._learn_conflicts(good, good.pairs(good_graph, base_pairs(3)),
                               rounds=3, seed=1) == set()
    candidates = bs._repair_candidates(good, good.pairs(good_graph,
                                          base_pairs(3)), {0b001}, 4, 1)
    assert candidates and all(deleted & 0b001 for _, deleted, _ in candidates)
    assert candidates == sorted(candidates, reverse=True)


def test_blind_coordinator_early_return_and_budgets(monkeypatch):
    graph = octahedron(3)
    invalid_cases = [
        {"beam": 0},
        {"delete_count": -1},
        {"target_dimension": 2},
        {"seed": True},
        {"retraction_seed": True},
        {"completion_states": 0},
    ]
    for overrides in invalid_cases:
        options = {"target_dimension": 3, "delete_count": 0, **overrides}
        with pytest.raises(ValueError):
            bs.find_blind_octahedral_retraction(graph, **options)
    monkeypatch.setattr(bs, "joint_beam_search", lambda *a, **kw: None)
    outcome = bs.find_blind_octahedral_retraction(graph, target_dimension=3)
    assert outcome.status is ob.OctahedralBlindStatus.LIMIT_REACHED
    assert outcome.partial_dimension == 0


def test_blind_coordinator_accepts_complete_fiber_retraction(monkeypatch):
    graph = octahedron(3)
    masks = jf.Masks(graph)
    pairs = tuple((2 * i, 2 * i + 1) for i in range(3))
    state = jf.State(pairs, tuple(((1 << a), (1 << b)) for a, b in pairs),
                     masks.all, 0)
    monkeypatch.setattr(bs, "joint_beam_search", lambda *a, **kw: state)
    outcome = bs.find_blind_octahedral_retraction(graph, target_dimension=3,
                                                  delete_count=0,
                                                  retraction_states=500)
    assert outcome.status is ob.OctahedralBlindStatus.FOUND
    assert len(outcome.antipodal_pairs) == 3
    assert set(outcome.projection) == set(graph)


def test_blind_coordinator_repairs_partial_target(monkeypatch):
    graph = octahedron(4)
    masks = jf.Masks(graph)
    pairs = base_pairs(3)
    state = jf.State(pairs, tuple(((1 << a), (1 << b)) for a, b in pairs),
                     sum((1 << a) | (1 << b) for a, b in pairs),
                     masks.closed[0] & masks.closed[1] &
                     masks.closed[2] & masks.closed[3] &
                     masks.closed[4] & masks.closed[5])
    monkeypatch.setattr(bs, "joint_beam_search", lambda *a, **kw: state)
    outcome = bs.find_blind_octahedral_retraction(graph, target_dimension=4,
                                                  delete_count=0,
                                                  core_rounds=4, max_repairs=4,
                                                  completion_states=1000)
    assert outcome.status is ob.OctahedralBlindStatus.FOUND
    assert outcome.partial_dimension == 3
    assert outcome.repairs_considered >= 1
    assert len(outcome.antipodal_pairs) == 4


def test_blind_coordinator_unfinished_and_oversized_deletion(monkeypatch):
    graph = octahedron(4)
    masks = jf.Masks(graph)
    pair = ((0, 1),)
    state = jf.State(pair, (((1 << 0), (1 << 1)),), 3,
                     masks.closed[0] & masks.closed[1])
    monkeypatch.setattr(bs, "joint_beam_search", lambda *a, **kw: state)
    limit = bs.find_blind_octahedral_retraction(graph, target_dimension=4,
                                                delete_count=4)
    assert limit.status is ob.OctahedralBlindStatus.LIMIT_REACHED
    assert limit.partial_dimension == 1
    monkeypatch.setattr(bs, "complete_dominating_octahedron", lambda *a, **kw:
                        ob.OctahedralCompletion(ob.OctahedralBlindStatus.EXHAUSTED,
                                                 None, 3, 2))
    result = bs.find_blind_octahedral_retraction(graph, target_dimension=4,
                                                 delete_count=0, core_rounds=2)
    assert result.status is ob.OctahedralBlindStatus.LIMIT_REACHED
    assert result.repairs_considered >= 1 and result.completion_states >= 3


def test_conflict_minimization_limits_and_completion_edge_cases():
    graph = octahedron(3)
    assert ob.screen_partial_octahedral_retraction(graph, []).reason == "NO_PAIRS"
    assert ob.minimize_octahedral_conflict(graph, [], max_checks=1) is None
    with pytest.raises(ValueError):
        ob.minimize_octahedral_conflict(graph, base_pairs(3), max_checks=0)
    with pytest.raises(ValueError, match="more initial pairs"):
        ob.complete_dominating_octahedron(graph, base_pairs(3),
                                         target_dimension=2)
    for bad in (nx.DiGraph(), nx.MultiGraph(), nx.Graph([(0, 0)])):
        with pytest.raises(ValueError, match="loopless"):
            ob.screen_partial_octahedral_retraction(bad, [])
    with pytest.raises(ValueError, match="induce an octahedron"):
        ob.screen_partial_octahedral_retraction(graph, ((0, 2),))


def test_weighted_repair_empty_invalid_and_budgeted_conflict():
    empty = ob.minimum_octahedral_repair([], 4)
    assert empty.cost == 0 and empty.deleted_indices == ()
    with pytest.raises(ValueError, match="positive integers"):
        ob.minimum_octahedral_repair([0b01], 2, weights=[1, 0])
    with pytest.raises(ValueError, match="number_of_pairs"):
        ob.minimum_octahedral_repair([], -1)
    with pytest.raises(ValueError, match="nonempty subsets"):
        ob.minimum_octahedral_repair([0], 2)
    costs = ob.minimum_octahedral_repair([0b011, 0b110, 0b111], 3,
                                         weights=[9, 2, 1])
    assert costs.cost == 2 and costs.deleted_indices == (1,)

    graph = octahedron(3)
    graph.add_node(6)
    graph.add_edges_from((6, v) for v in graph if v != 6)
    report = ob.screen_partial_octahedral_retraction(graph, base_pairs(3))
    assert report.possible is False
    partial = ob.minimize_octahedral_conflict(graph, base_pairs(3), max_checks=1)
    assert partial is not None and not partial.deletion_minimal
    assert partial.checks == 1

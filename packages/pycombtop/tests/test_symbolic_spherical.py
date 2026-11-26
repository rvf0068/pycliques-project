"""Integral generation by octahedral sphere cycles."""

from dataclasses import replace

import networkx as nx
import pytest
from pycliques import CliqueSequence, enumerate_induced_octahedra
from pycombtop import (
    WedgeType,
    symbolic_integral_morse_certificate,
    verify_symbolic_spherical,
    verify_symbolic_sq2,
)
from pycombtop._symbolic_morse_gradient import signed_projector
from pycombtop._symbolic_spherical import spherical_generation


def _wedge_octahedra(*dimensions):
    graph = nx.Graph()
    offset = 1
    for d in dimensions:
        sphere = nx.complete_multipartite_graph(*([2] * (d + 1)))
        graph.add_edges_from(
            (0 if u == 0 else u + offset - 1, 0 if v == 0 else v + offset - 1)
            for u, v in sphere.edges
        )
        offset += len(sphere) - 1
    return graph


def test_multiple_required_degrees_and_replay():
    graph = _wedge_octahedra(2, 4, 5)
    old = symbolic_integral_morse_certificate(graph)
    assert old is not None
    assert old.betti == {2: 1, 4: 1, 5: 1}
    assert old.wedge_spheres is None
    assert old.spherical is None
    proof = symbolic_integral_morse_certificate(graph, max_spherical_states=10000)
    assert proof is not None
    assert proof.spherical is not None and proof.spherical.complete
    assert proof.wedge_spheres == old.betti
    assert [d.degree for d in proof.spherical.degrees] == [4, 5]
    assert all(d.smith_factors == (1,) for d in proof.spherical.degrees)
    assert verify_symbolic_spherical(graph, proof)
    d = proof.spherical.degrees[0]
    c = d.cycles[0]
    forged = replace(c, projection=tuple(2 * x for x in c.projection))
    bad = replace(
        proof.spherical,
        degrees=(replace(d, cycles=(forged,)), *proof.spherical.degrees[1:]),
    )
    assert not verify_symbolic_spherical(graph, replace(proof, spherical=bad))


def test_partial_seed_replay_and_budget():
    graph = _wedge_octahedra(2, 4, 5)
    proof = symbolic_integral_morse_certificate(graph, max_spherical_states=10000)
    assert proof is not None and proof.spherical is not None
    first = tuple(c.pairs for c in proof.spherical.degrees[0].cycles)
    partial = symbolic_integral_morse_certificate(
        graph,
        max_spherical_states=10000,
        spherical_seed=first,
        spherical_search=False,
    )
    assert partial is not None and partial.spherical is not None
    assert not partial.spherical.complete and partial.wedge_spheres is None
    assert partial.spherical.degrees[0].complete
    assert not partial.spherical.degrees[1].complete
    assert not verify_symbolic_spherical(graph, partial)
    resumed = symbolic_integral_morse_certificate(
        graph, max_spherical_states=10000, spherical_seed=first
    )
    assert resumed is not None and verify_symbolic_spherical(graph, resumed)
    limited = symbolic_integral_morse_certificate(graph, max_spherical_states=1)
    assert limited is not None and limited.spherical is not None
    assert limited.spherical.limited and not limited.spherical.complete
    assert limited.betti == proof.betti and limited.wedge_spheres is None


def test_odd_index_is_not_integral_generation(monkeypatch):
    # Synthetic projection of an index-three subgroup of Z. Full mod-2
    # rank must never be interpreted as integral generation.
    graph = nx.complete_multipartite_graph(*([2] * 5))
    monkeypatch.setattr(
        "pycombtop._symbolic_spherical._project_sphere", lambda *args: (3,)
    )
    proof = spherical_generation(
        graph,
        tuple(graph),
        {4: [31]},
        {},
        {2: 1, 4: 1},
        lambda _: None,
        10000,
    )
    assert not proof.complete and not proof.limited
    assert proof.degrees[0].lattice_rank == 1
    assert proof.degrees[0].smith_factors == (3,)


def test_validation_and_steenrod_compatibility():
    graph = _wedge_octahedra(2, 4)
    with pytest.raises(ValueError):
        symbolic_integral_morse_certificate(graph, max_spherical_states=-1)
    with pytest.raises(ValueError):
        symbolic_integral_morse_certificate(
            graph, max_spherical_states=1000, spherical_seed=(((0, 0),),)
        )
    graph = _wedge_octahedra(3, 5)
    proof = symbolic_integral_morse_certificate(
        graph, max_sq2_states=10000, max_spherical_states=10000
    )
    assert proof is not None and proof.sq2 is not None
    assert proof.sq2.rank == 0 and proof.spherical is None
    assert proof.wedge_spheres == {3: 1, 5: 1}


def test_cubic13_first_iterate():
    cubic = nx.from_graph6_bytes(b"M}KGGGA?gA?H?I?D_")
    graph = CliqueSequence(
        nx.complement(cubic), bound=1000, relabel=True, pare_initial=True
    )[1]
    assert graph is not None and len(graph) == 38
    proof = symbolic_integral_morse_certificate(
        graph, max_critical=128, max_spherical_states=500_000
    )
    assert proof is not None and proof.spherical is not None
    assert proof.betti == {4: 5, 7: 1}
    assert proof.wedge_spheres == proof.betti
    assert verify_symbolic_spherical(graph, proof)


def test_combined_square_and_spherical_verifiers():
    graph = _wedge_octahedra(2, 4)
    proof = symbolic_integral_morse_certificate(
        graph, steenrod_degree=2, max_sq2_states=10000, max_spherical_states=10000
    )
    assert proof is not None
    assert proof.sq2 is not None and proof.sq2.rank == 0
    assert proof.spherical is not None and proof.spherical.complete
    assert verify_symbolic_sq2(graph, proof)
    assert verify_symbolic_spherical(graph, proof)
    forged = replace(proof, wedge_spheres={4: 1})
    assert not verify_symbolic_sq2(graph, forged)
    assert not verify_symbolic_spherical(graph, forged)
    forged = replace(proof, sq2_homotopy_type=WedgeType())
    assert not verify_symbolic_spherical(graph, forged)


def test_integral_lattice_refinement_after_full_rational_rank(monkeypatch):
    # Include an incoming boundary, then generate an index-three subgroup.
    # A second batch supplies an odd-prime correction without increasing rank.
    graph = nx.complete_multipartite_graph(*([2] * 8))
    seeds = list(enumerate_induced_octahedra(graph, target_dimension=5))
    projections = iter([(3, 0)] * 16 + [(2, 0)] * 16)
    monkeypatch.setattr(
        "pycombtop._symbolic_spherical._project_sphere",
        lambda *args: next(projections),
    )
    proof = spherical_generation(
        graph,
        tuple(graph),
        {4: [31, 62], 5: [63]},
        {5: ((0,), (1,))},
        {2: 1, 4: 1},
        lambda _: None,
        10000,
        seed=[seeds[0], *seeds],
    )
    assert proof.complete and not proof.limited
    assert proof.candidates == 32
    assert proof.degrees[0].lattice_rank == 2
    assert proof.degrees[0].smith_factors == (1, 1)


def test_spherical_invalid_chain_and_seed_budget(monkeypatch):
    graph = nx.complete_multipartite_graph(*([2] * 5))
    pairs = next(enumerate_induced_octahedra(graph, target_dimension=5))
    with pytest.raises(ValueError, match="too many spherical seeds"):
        spherical_generation(
            graph,
            tuple(graph),
            {4: [31]},
            {},
            {2: 1, 4: 1},
            lambda _: None,
            1,
            seed=(pairs, pairs),
        )
    monkeypatch.setattr(
        "pycombtop._symbolic_spherical._project_sphere", lambda *args: (1, 0)
    )
    with pytest.raises(ArithmeticError, match="not a Morse cycle"):
        spherical_generation(
            graph,
            tuple(graph),
            {3: [15], 4: [31, 62]},
            {4: ((1, 0),)},
            {2: 1, 4: 1},
            lambda _: None,
            10000,
        )


def test_partial_square_and_malformed_spherical_replay():
    graph = _wedge_octahedra(2, 4)
    with pytest.raises(ValueError, match="spherical_search"):
        symbolic_integral_morse_certificate(graph, spherical_search=1)
    partial = symbolic_integral_morse_certificate(
        graph,
        steenrod_degree=2,
        max_sq2_states=10000,
        max_spherical_states=1,
    )
    assert partial is not None and partial.spherical is not None
    assert not partial.spherical.complete and partial.wedge_spheres is None
    assert verify_symbolic_sq2(graph, partial)
    assert not verify_symbolic_sq2(graph, replace(partial, wedge_spheres=partial.betti))
    assert not verify_symbolic_sq2(graph, replace(partial, connectivity_order=()))

    proof = symbolic_integral_morse_certificate(graph, max_spherical_states=10000)
    assert proof is not None and proof.spherical is not None
    assert not verify_symbolic_spherical(graph, proof, max_vertices=1)
    assert not verify_symbolic_spherical(graph, replace(proof, vertex_order=()))


def test_signed_projector_rejects_inconsistent_matching():
    # These are invalid matching descriptions, not valid Dong matchings.
    # A projection must reject them rather than returning a false chain.
    project = signed_projector([4], lambda _: None, lambda: None)
    with pytest.raises(ArithmeticError, match="unexpected critical"):
        project(1)
    project = signed_projector([4], {1: 1, 2: 0}.__getitem__, lambda: None)
    with pytest.raises(ArithmeticError, match="cyclic Morse gradient"):
        project(1)

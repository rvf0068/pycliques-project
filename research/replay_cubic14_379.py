"""Blind, package-only reproduction of cubic complement 379."""

import networkx as nx
import pycliques as pc
import pycombtop as pt


def main():
    graph = nx.complement(nx.from_graph6_bytes(b"M{O___IA_H?_?T?R?"))
    sequence = pc.CliqueSequence(
        graph, bound=1000, relabel=True, pare_initial=True
    )
    p0, p1, p2 = sequence[0], sequence[1], sequence[2]
    assert p0 is not None and p1 is not None and p2 is not None
    assert [len(p) for p in (p0, p1, p2)] == [14, 34, 541]
    assert p2.number_of_edges() == 132098
    h0 = pt.homotopy_type_with_verdict(p0)
    h1 = pt.homotopy_type_with_verdict(p1)
    assert h0.is_exact and h0.wedge.spheres == {}
    assert h1.is_exact and h1.wedge.spheres == {5: 1}
    result = pc.find_octahedral_retraction_coaffine(
        p2,
        backend="native",
        value_order="fiber",
        max_states=10000,
        time_limit=30,
    )
    assert result.status is pc.RetractionSearchStatus.FOUND
    assert result.dimension == 20
    assert result.inclusion is not None and result.retraction is not None
    assert result.endomorphism is not None
    target = p2.subgraph(result.inclusion.values())
    assert pc.verify_coaffine_endomorphism(p2, result.endomorphism)
    assert pc.verify_retraction_witness(
        p2, target, (result.retraction, result.inclusion)
    )
    assert all(
        result.retraction[v] == result.endomorphism[result.endomorphism[v]]
        for v in p2
    )
    print("Exact homotopy:", h0.wedge.spheres, h1.wedge.spheres)
    print("Blind O20 retraction:", result.states_explored, "states")
    print("Integral H_19 contains a split Z summand by the retraction.")


if __name__ == "__main__":
    main()

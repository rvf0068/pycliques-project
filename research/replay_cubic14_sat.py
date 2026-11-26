"""Blind SAT reproduction of two difficult order-14 cubic complements."""

from time import perf_counter

import networkx as nx
import pycliques as pc
from pyg6data import list_cubic_graphs
from pycliques.retractions import RetractionSearchStatus

cubic = list_cubic_graphs(14)
assert len(cubic) == 509
for index, expected_order in [(47, 210), (419, 401)]:
    start = perf_counter()
    seq = pc.CliqueSequence(
        nx.complement(cubic[index - 1]),
        bound=10_000,
        relabel=True,
        pare_initial=True,
    )
    host = seq[2]
    assert host is not None and len(host) == expected_order
    result = pc.find_octahedral_retraction_coaffine(
        host, backend="sat", max_states=1_000_000, time_limit=120,
        sat_max_clauses=4_000_000, sat_memory_limit_mb=2048,
    )
    print(
        index, len(host), result.status.name, result.dimension,
        result.states_explored, round(perf_counter() - start, 2),
        flush=True,
    )
    assert result.status is RetractionSearchStatus.FOUND
    assert result.antipodal_pairs is not None
    assert result.retraction is not None and result.inclusion is not None
    assert result.endomorphism is not None
    f = result.endomorphism
    target = host.subgraph(result.retraction.values())
    assert pc.verify_coaffine_endomorphism(host, f)
    assert all(f[f[f[v]]] == f[v] for v in host)
    assert all(result.retraction[v] == f[f[v]] for v in host)
    assert pc.verify_induced_octahedron(host, result.antipodal_pairs)
    assert pc.verify_retraction_witness(
        host, target, (result.retraction, result.inclusion)
    )

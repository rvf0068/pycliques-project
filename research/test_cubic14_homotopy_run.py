"""Research-only serial SAT tranche after PR #66. No saved retractions."""

import json
import time

import networkx as nx
import pycliques as pc
from pyg6data import list_cubic_graphs
from pycliques.retractions import RetractionSearchStatus

IDS = [441, 206, 94, 442, 185, 451, 283, 308]


def test_research_cubic14_sat(capsys):
    cubic = list_cubic_graphs(14)
    assert len(cubic) == 509
    with capsys.disabled():
        for index in IDS:
            start = time.monotonic()
            seq = pc.CliqueSequence(
                nx.complement(cubic[index - 1]), bound=10_000,
                relabel=True, pare_initial=True,
            )
            host = seq[2]
            assert host is not None
            proof = pc.find_octahedral_retraction_coaffine(
                host, backend="sat", max_states=2_000_000,
                time_limit=65, sat_max_clauses=8_000_000,
                sat_memory_limit_mb=3072,
            )
            row = dict(id=index, n=len(host), m=host.number_of_edges(),
                       status=proof.status.name, dimension=proof.dimension,
                       conflicts=proof.states_explored,
                       seconds=round(time.monotonic()-start, 3))
            if proof.status is RetractionSearchStatus.FOUND:
                f = proof.endomorphism
                r = proof.retraction
                assert f is not None and r is not None
                assert proof.inclusion is not None and proof.antipodal_pairs
                assert pc.verify_coaffine_endomorphism(host, f)
                assert all(f[f[f[u]]] == f[u] and r[u] == f[f[u]] for u in host)
                assert pc.verify_induced_octahedron(host, proof.antipodal_pairs)
                assert pc.verify_retraction_witness(
                    host, host.subgraph(r.values()), (r, proof.inclusion)
                )
                row.update(mapping=[f[u] for u in range(len(host))],
                           pairs=proof.antipodal_pairs)
            print("POST66_ROW " + json.dumps(row), flush=True)

"""Research-only incidence walk tranche after PR #67."""
import json
import time

import networkx as nx
import pycliques as pc
from pyg6data import list_cubic_graphs

IDS = [451, 441, 442, 94, 185, 206, 283, 308, 320, 344, 363, 473, 127, 287, 368, 462, 497]

def test_research_cubic14_incidence(capsys):
    cubic = list_cubic_graphs(14)
    assert len(cubic) == 509
    exe = pc.build_incidence_native()
    with capsys.disabled():
        for index in IDS:
            seq = pc.CliqueSequence(nx.complement(cubic[index-1]), bound=3000,
                                    relabel=True, pare_initial=True)
            base = seq[1]
            assert base is not None
            raw = pc.ImplicitCliqueGraph.from_graph(base, max_cliques=3000)
            assert raw is not None
            pared = raw.pare()
            assert pc.verify_implicit_paring(raw, pared)
            host = pared.core
            start = time.monotonic()
            result = pc.find_octahedral_retraction_incidence(
                host, backend="walk", executable=exe, seed=137,
                max_steps=250_000 if index in (451,441,442) else 50_000,
                time_limit=30,
            )
            row = dict(index=index, order=len(host), ground=len(host.base_vertices),
                       status=result.status.name, dimension=result.dimension,
                       partial_dimension=len(result.partial_cuts), missed=result.uncovered,
                       states=result.states_explored, seconds=round(time.monotonic()-start,3))
            row.update(cuts=result.partial_cuts)
            if result.retraction is not None:
                graph = host.to_networkx()
                assert graph is not None
                assert result.inclusion is not None and result.antipodal_pairs
                assert pc.verify_induced_octahedron(graph, result.antipodal_pairs)
                assert pc.verify_retraction_witness(
                    graph, graph.subgraph(result.retraction.values()),
                    (result.retraction,result.inclusion))
                row.update(pairs=result.antipodal_pairs, projection=result.retraction)
            print("POST67_ROW " + json.dumps(row), flush=True)

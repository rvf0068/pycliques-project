"""Bounded incidence search on cubic complements, without archived maps."""
import json
from time import perf_counter

import networkx as nx
import pycliques as pc
from pyg6data import list_cubic_graphs

cubic = list_cubic_graphs(14)
assert len(cubic) == 509
exe = pc.build_incidence_native()
for index in (441, 442, 451):
    seq = pc.CliqueSequence(nx.complement(cubic[index - 1]), bound=3000,
                            relabel=True, pare_initial=True)
    base = seq[1]
    assert base is not None
    raw = pc.ImplicitCliqueGraph.from_graph(base, max_cliques=3000)
    assert raw is not None
    pared = raw.pare()
    assert pc.verify_implicit_paring(raw, pared)
    host = pared.core
    for backend in ("walk", "hybrid", "sat"):
        start = perf_counter()
        result = pc.find_octahedral_retraction_incidence(
            host, backend=backend, executable=exe, seed=139,
            max_steps=100_000, max_states=500_000, time_limit=25,
        )
        row = dict(index=index, backend=backend, ground=len(host.base_vertices),
                   order=len(host), status=result.status.name,
                   dimension=result.dimension, missed=result.uncovered,
                   partial_dimension=len(result.partial_cuts),
                   states=result.states_explored, elapsed=round(perf_counter()-start, 3))
        if result.retraction is not None:
            row.update(pairs=result.antipodal_pairs, projection=result.retraction,
                       cuts=result.partial_cuts)
        print("INCIDENCE_ROW " + json.dumps(row), flush=True)
        if result.retraction is not None:
            break

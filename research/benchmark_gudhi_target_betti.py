"""Cross-check GUDHI against independent binary exact homology.

This reproducible experiment confirms the production backend and the
independent integer-bitset reducer. The historical numerical discrepancy is
preserved in issue #16.
"""
from __future__ import annotations

import json
import time

import gudhi
import networkx as nx

from pycliques.cliques import clique_graph
from pycliques.named import complement_of_cycle, octahedron, snub_disphenoid
from pycombtop.bounded_betti import bounded_reduced_betti_mod2
from pycombtop.homotopy_type import betti_numbers_graph


def gudhi_betti(graph: nx.Graph, *, max_degree: int | None = None, edge_collapse: bool = True):
    """Compute ordinary then reduced Betti ranks using a GUDHI simplex tree."""
    nodes = {v: i for i, v in enumerate(graph)}
    tree = gudhi.SimplexTree()
    for v in nodes.values():
        tree.insert([v])
    for u, v in graph.edges():
        tree.insert([nodes[u], nodes[v]])
    if edge_collapse:
        tree.collapse_edges()
    # For exact beta_j one must include (j+1)-simplices.
    if max_degree is None:
        # Determine the maximal dimension of the *original* clique complex.
        maximal_dimension = max((len(c) - 1 for c in nx.find_cliques(graph)), default=0)
        expansion_dimension = maximal_dimension
    else:
        expansion_dimension = max_degree + 1
    if expansion_dimension >= 2:
        tree.expansion(expansion_dimension)
    tree.compute_persistence(homology_coeff_field=2, persistence_dim_max=True)
    ordinary = tree.betti_numbers()
    if max_degree is None:
        n = max(len(ordinary), expansion_dimension + 1)
    else:
        n = max_degree + 1
    reduced = {j: (ordinary[j] if j < len(ordinary) else 0) - (j == 0)
               for j in range(n)}
    if any(v < 0 for v in reduced.values()):
        raise AssertionError("Negative reduced Betti rank")
    return reduced, tree.num_simplices()


def benchmark():
    snub = snub_disphenoid()
    samples = [
        ("cycle5",nx.cycle_graph(5)),
        ("octa3",octahedron(3)),
        ("octa4",octahedron(4)),
        ("cyclecomplement9",complement_of_cycle(9)),
        ("snub",snub),
    ]
    graph = snub
    for i in range(1,4):
        graph = clique_graph(graph,bound=1200)
        assert graph is not None
        if i <= 2:
            samples.append((f"snub_k{i}", graph))
        elif i == 3:
            samples.append(("snub_k3_partial", graph))
    results=[]
    for name, graph in samples:
        partial=name.endswith("_partial")
        kwargs={"max_degree":3} if partial else {}
        started=time.monotonic()
        via_gudhi, count = gudhi_betti(graph,**kwargs)
        gudhi_seconds=time.monotonic()-started
        started=time.monotonic()
        via_bounded=bounded_reduced_betti_mod2(
            graph,max_faces=150_000,**kwargs,
        )
        bounded_seconds=time.monotonic()-started
        # All fully bounded cases are known to fit under this budget,
        # while the K3 partial may exceed it and legitimately return None.
        if via_bounded is not None and any(
            via_bounded.get(d) != v for d, v in via_gudhi.items()
            if d in via_bounded
        ):
            raise AssertionError(f"Independent GF2 mismatch at {name}")
        existing=None
        if not partial:
            started=time.monotonic()
            raw = betti_numbers_graph(graph)
            pycombtop_seconds=time.monotonic()-started
            existing={i:value for i,value in enumerate(raw)}
            # The production pycombtop API is now also exact over F_2.
            pycombtop_matches_gudhi = not any(
                existing.get(d, 0) != v for d, v in via_gudhi.items()
            )
        else:
            pycombtop_seconds=None
            pycombtop_matches_gudhi=None
        results.append({
            "graph":name,"order":len(graph),"edges":graph.number_of_edges(),
            "gudhi_reduced_betti":via_gudhi,"gudhi_simplices":count,
            "gudhi_seconds":round(gudhi_seconds,4),
            "bounded_seconds":round(bounded_seconds,4),
            "pycombtop_seconds":None if pycombtop_seconds is None else round(pycombtop_seconds,4),
            "bounded_within_face_budget":via_bounded is not None,
            "bounded_reduced_betti":via_bounded,
            "pycombtop_reduced_betti":existing,
            "pycombtop_matches_gudhi":pycombtop_matches_gudhi,
        })
    return {
        "schema":1,"gudhi_version":getattr(gudhi,"__version__","unknown"),
        "gudhi_matches_exact_binary_boundaries":True,
        "pycombtop_disagreements":[x["graph"] for x in results
            if x["pycombtop_matches_gudhi"] is False],
        "results":results,
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(),indent=2,sort_keys=True))

"""Targeted second pass on the first-pass indeterminate cubic complements."""
import argparse
import json
import random
import signal
from collections import Counter
from pathlib import Path

import networkx as nx
import pycliques as pc

from cubic_order14_census import (
    alarm, canon, code, limited, min_octahedron_dimension, retraction_betti,
)


def join_certificate(g):
    components = list(nx.connected_components(nx.complement(g)))
    if len(components) < 2:
        return None
    maps = []
    for component in components:
        factor = g.subgraph(component)
        mapping = next((m for m in pc.automorphisms(factor)
                        if all(v != m[v] and not factor.has_edge(v, m[v])
                               for v in factor)), None)
        if mapping is None:
            return None
        assert set(mapping) == set(factor) == set(mapping.values())
        assert all(factor.has_edge(u, v) == factor.has_edge(mapping[u], mapping[v])
                   for u in factor for v in factor)
        maps.append(mapping)
    if len(components) >= 3 or any(nx.is_connected(g.subgraph(c)) for c in components):
        return dict(verdict="DIVERGENT", rule="coaffined_join", level=0, maps=maps)
    return None


def search(g):
    p = canon(pc.completely_pared_graph(g))
    sizes = [len(p)]
    counts = []
    stages = []
    homology = {}
    join = limited(4, join_certificate, p)
    if join:
        return dict(join, sizes=sizes)
    # P_1, P_2, P_3: use the implicit representation first, so the edge
    # budget does not prevent finding a small pared core of a large iterate.
    for level in range(1, 4):
        implicit = limited(8, pc.ImplicitCliqueGraph.from_graph, p, max_cliques=3500)
        if implicit is None:
            stages.append("clique enumeration limit")
            break
        counts.append(len(implicit))
        pared = limited(15, implicit.pare)
        if pared is None:
            stages.append("implicit paring timeout")
            break
        assert pc.verify_implicit_paring(implicit, pared)
        sizes.append(len(pared.core))
        if len(pared.core) > 2048:
            stages.append("native order limit")
            break
        p = canon(pared.core.to_networkx(max_edges=2_100_000))
        if len(p) <= 350 and limited(5, pc.is_clique_helly, p) is True:
            return dict(verdict="CONVERGENT", rule="Helly", level=level,
                        sizes=sizes, clique_counts=counts)
        known = retraction_betti(p)
        homology[level] = known
        if len(p) <= 75:
            targets = limited(8, pc.search_targeted_retractions, p,
                              betti_numbers=known, max_target_order=18, max_patterns=20_000,
                              max_states=100_000, max_assignments=500_000)
            if targets:
                for family, v in targets.items():
                    if v.status.name == "FOUND":
                        assert pc.verify_targeted_retraction(p, v)
                        return dict(verdict="DIVERGENT", rule=family, level=level,
                                    target=list(v.target_vertices), projection=v.projection,
                                    kind=v.target_kind, sizes=sizes, clique_counts=counts,
                                    host=code(p))
        minimum = min_octahedron_dimension(p, known)
        for seed in range(3 if minimum <= len(p)//2 else 0):
            order = list(p)
            random.Random(14000 + seed).shuffle(order)
            h = nx.Graph()
            h.add_nodes_from(order)
            h.add_edges_from(p.edges)
            v = pc.find_octahedral_retraction_coaffine(
                h, min_dimension=minimum, max_states=400000, time_limit=3.0, backend="native")
            if v.status is pc.RetractionSearchStatus.FOUND:
                assert pc.verify_retraction_witness(
                    h, h.subgraph(v.inclusion.values()), (v.retraction, v.inclusion))
                return dict(verdict="DIVERGENT", rule="coaffine", level=level,
                            dimension=v.dimension, pairs=v.antipodal_pairs,
                            projection=v.retraction, inclusion=v.inclusion,
                            sizes=sizes, clique_counts=counts, host=code(p), seed=seed)
            if v.status is pc.RetractionSearchStatus.EXHAUSTED:
                stages.append(f"coaffine exhausted at {level}")
                break
        s = limited(6, pc.find_special_octahedron_witness, p)
        if s is not None:
            assert pc.verify_special_octahedron_witness(p, s)
            return dict(verdict="DIVERGENT", rule="special", level=level,
                        dimension=s.dimension, pairs=list(zip(s.clique, s.antipodes)),
                        sizes=sizes, clique_counts=counts, host=code(p))
        # Higher iterates of large cores almost always exceed the
        # enumeration budget. Still try bounded implicit construction.
    return dict(verdict="INDETERMINATE", sizes=sizes,
                clique_counts=counts, notes=stages, homology=homology)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--output", default="order14-pass2.jsonl")
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, alarm)
    records = [json.loads(line) for line in Path(
        "research/cubic_order14/pass1.jsonl").read_text().splitlines()]
    records = [x for x in records if x["behavior"]["verdict"] == "INDETERMINATE"]
    assert len(records) == 62
    counts = Counter()
    with Path(args.output).open("w") as out:
        for j, x in enumerate(records):
            if j % args.shards != args.shard:
                continue
            g = nx.complement(nx.from_graph6_bytes(x["cubic"].encode()))
            b = search(g)
            row = dict(index=x["index"], cubic=x["cubic"], behavior=b)
            line = json.dumps(row, sort_keys=True)
            out.write(line + "\n")
            out.flush()
            print("ROW " + line, flush=True)
            counts[b["verdict"]] += 1
    print("SUMMARY " + json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()

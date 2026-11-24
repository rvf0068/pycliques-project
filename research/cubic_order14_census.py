"""Order-14 cubic complement census; run from the uv workspace.

Indices are one-based in the connected pyg6data list. Disconnected graphs
are appended, grouped by component orders. All graphs use integer labels.
"""
import argparse
import json
import random
import signal
from collections import Counter
from itertools import product
from pathlib import Path

import networkx as nx
import pycliques as pc
import pycombtop as pt
from pyg6data import cubic_graph_generator


def code(g):
    return nx.to_graph6_bytes(g, header=False).decode().strip()


def canon(g):
    return nx.convert_node_labels_to_integers(g, ordering="sorted")


def cubic_graphs():
    small = {4: [nx.complete_graph(4)],
             6: [nx.circular_ladder_graph(3), nx.complete_bipartite_graph(3, 3)]}
    for n in (8, 10, 14):
        small[n] = [g for g in cubic_graph_generator(n) if nx.is_connected(g)]
    cases = [(g, [14]) for g in small[14]]
    for orders in [(4, 10), (6, 8), (4, 4, 6)]:
        for components in product(*(small[n] for n in orders)):
            cases.append((nx.disjoint_union_all(components), list(orders)))
    assert len(small[14]) == 509, {n: len(v) for n, v in small.items()}
    assert len(cases) == 540, len(cases)
    assert all(len(g) == 14 and all(d == 3 for _, d in g.degree()) for g, _ in cases)
    return cases


class Timeout(Exception):
    pass


def alarm(_sig, _frame):
    raise Timeout()


def limited(seconds, f, *args, **kwargs):
    signal.alarm(seconds)
    try:
        return f(*args, **kwargs)
    except Timeout:
        return None
    finally:
        signal.alarm(0)


def hom(g):
    h = limited(12, pt.homotopy_type_with_verdict, g, max_simplices=100_000)
    if h is None:
        return {"status": "TIMEOUT"}
    return {"status": "EXACT" if h.is_exact else "LIMITED" if h.is_limited else "BETTI_ONLY",
            "spheres": h.wedge.spheres, "reason": h.reason}


def behavior(g, deep):
    p = canon(pc.completely_pared_graph(g))
    sizes = []
    log = []
    seen = []
    for k in range(3 if deep else 2):
        sizes.append(len(p))
        for j, old in enumerate(seen):
            if len(old) == len(p) and nx.is_isomorphic(old, p):
                # A repeated pared iterate is sufficient: paring is a retraction,
                # but repeating pared cores alone does not prove ordinary convergence.
                log.append(f"repeated pared iterate {j}")
        seen.append(p)
        if limited(5, pc.is_clique_helly, p) is True:
            return dict(verdict="CONVERGENT", rule="Helly", level=k, sizes=sizes)
        if len(p) <= 350:
            s = limited(5, pc.find_special_octahedron_witness, p)
            if s is not None:
                assert pc.verify_special_octahedron_witness(p, s)
                return dict(verdict="DIVERGENT", rule="special", level=k,
                            dimension=s.dimension, pairs=list(zip(s.clique, s.antipodes)), sizes=sizes)
        if k == 0 or len(p) <= 70:
            t = limited(8, pc.search_targeted_retractions, p,
                        families=("octahedra", "cycle_complements", "suspensions"),
                        max_target_order=16, max_patterns=3000,
                        max_states=10000, max_assignments=50000)
            if t:
                for f, v in t.items():
                    if v.status.name == "FOUND":
                        assert pc.verify_targeted_retraction(p, v)
                        return dict(verdict="DIVERGENT", rule=f, level=k,
                                    target=list(v.target_vertices), projection=v.projection,
                                    kind=v.target_kind, sizes=sizes)
        if len(p) <= 500:
            # Native search is independently checked by the package.
            for seed in range(2 if deep else 1):
                order = list(p)
                random.Random(seed).shuffle(order)
                h = nx.Graph()
                h.add_nodes_from(order)
                h.add_edges_from(p.edges)
                v = pc.find_octahedral_retraction_coaffine(h, max_states=100000,
                    time_limit=1.0 if deep else 0.3, backend="auto")
                if v.status is pc.RetractionSearchStatus.FOUND:
                    assert pc.verify_retraction_witness(h, h.subgraph(v.inclusion.values()),
                                                       (v.retraction, v.inclusion))
                    return dict(verdict="DIVERGENT", rule="coaffine", level=k,
                                dimension=v.dimension, pairs=v.antipodal_pairs,
                                projection=v.retraction, inclusion=v.inclusion, sizes=sizes)
        if k == (2 if deep else 1):
            break
        nxt = limited(5, pc.clique_graph, p, bound=500 if deep else 150,
                      max_edges=125000 if deep else 11200)
        if nxt is None:
            log.append("clique bound/time")
            break
        # Stable order by increasing integer masks, independent of hash ordering.
        labels = sorted(nxt, key=lambda q: sum(1 << v for v in q))
        nxt = nx.relabel_nodes(nxt, {q: i for i, q in enumerate(labels)})
        p = canon(pc.completely_pared_graph(nxt))
    return dict(verdict="INDETERMINATE", sizes=sizes, notes=log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--output", default="order14.jsonl")
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, alarm)
    graphs = cubic_graphs()
    counts = Counter()
    with Path(args.output).open("w") as out:
        for i, (cubic, components) in enumerate(graphs, 1):
            if (i - 1) % args.shards != args.shard:
                continue
            g = nx.complement(cubic)
            h = hom(g)
            b = behavior(g, args.deep)
            row = dict(index=i, cubic=code(cubic), components=components, homotopy=h, behavior=b)
            line = json.dumps(row, sort_keys=True)
            out.write(line + "\n")
            out.flush()
            print("ROW " + line, flush=True)
            counts[b["verdict"]] += 1
            counts["H_" + h["status"]] += 1
    print("SUMMARY " + json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()

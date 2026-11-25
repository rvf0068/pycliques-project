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
from pycombtop.bounded_betti import bounded_reduced_betti_mod2


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


def retraction_betti(p, previous_hom=None):
    """Use exact ranks or proved upper bounds, never guessed zeros.

    The original graph and its completely pared core have the same
    homotopy type. Later iterates need their own information.
    """
    if previous_hom is not None and previous_hom["status"] == "EXACT":
        ranks = {int(d): b for d, b in previous_hom["spheres"].items()}
        return {d: ranks.get(d, 0) for d in range(len(p))}
    known = {}
    if len(p) <= 75:
        for degree in (None, 3):
            betti = limited(3, bounded_reduced_betti_mod2, p,
                            max_faces=100000, max_degree=degree)
            if betti is not None:
                known.update(betti)
                if degree is None:
                    known.update({d: 0 for d in range(max(betti, default=-1)+1, len(p))})
                break
    # Optional until the new library PR is merged. Zero entries are
    # exact even in the original API; positive upper bounds are not.
    recursive = getattr(pt, "recursive_betti_upper_bounds", None)
    if recursive is not None and len(p) <= 2048:
        bounds = limited(3, recursive, p, max_degree=6, max_states=10000)
        if bounds is not None:
            for d, upper in bounds.upper_bounds.items():
                if d in known:
                    assert known[d] <= upper
                elif upper == 0:
                    known[d] = 0
    return known


def min_octahedron_dimension(p, known):
    d = 3
    while d <= len(p)//2 and known.get(d-1) == 0:
        d += 1
    return d


def behavior(g, deep, original_hom=None):
    p = canon(pc.completely_pared_graph(g))
    sizes = []
    log = []
    seen = []
    homology = {}
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
        known = retraction_betti(p, original_hom if k == 0 else None)
        homology[k] = known
        if k == 0 or len(p) <= 70:
            t = limited(8, pc.search_targeted_retractions, p,
                        families=("octahedra", "cycle_complements", "suspensions"),
                        betti_numbers=known,
                        max_target_order=16, max_patterns=3000,
                        max_states=10000, max_assignments=50000)
            if t:
                for f, v in t.items():
                    if v.status.name == "FOUND":
                        assert pc.verify_targeted_retraction(p, v)
                        return dict(verdict="DIVERGENT", rule=f, level=k,
                                    target=list(v.target_vertices), projection=v.projection,
                                    kind=v.target_kind, sizes=sizes)
        min_dimension = min_octahedron_dimension(p, known)
        if len(p) <= 500 and min_dimension <= len(p)//2:
            # Native search is independently checked by the package.
            for seed in range(2 if deep else 1):
                order = list(p)
                random.Random(seed).shuffle(order)
                h = nx.Graph()
                h.add_nodes_from(order)
                h.add_edges_from(p.edges)
                v = pc.find_octahedral_retraction_coaffine(h, min_dimension=min_dimension, max_states=100000,
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
    return dict(verdict="INDETERMINATE", sizes=sizes, notes=log, homology=homology)


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
            b = behavior(g, args.deep, h)
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

"""Focused bounded search on contractible order-14 complements."""
import argparse
import json
import random
import signal
from pathlib import Path

import networkx as nx
import pycliques as pc
import pycombtop as pt

from cubic_order14_census import (
    alarm, canon, code, limited, min_octahedron_dimension, retraction_betti,
)

CUBICS = {182: "M{S__SE@?A?C?F?F?", 379: "M{O___IA_H?_?T?R?"}


# Previously verified in order14-homological-filters.zip (README.md,
# verify_inputs.py, verify_reduction.py, matchtree.cpp, check_vanish.cpp).
# These facts refer to the relative pared iterates of the exact cubic codes
# above, not to arbitrary graphs of the same order.
EXPECTED_ORDERS = {182: (13, 27, 199), 379: (14, 34, 541)}


def certified_profile(index, level):
    order = EXPECTED_ORDERS[index][level]
    if index == 182 or level == 0:
        return dict.fromkeys(range(order), 0)
    if level == 1:
        return {d: int(d == 5) for d in range(order)}
    return dict.fromkeys(range(11), 0)


# Verified lower bounds from reference_verified.json in the same archive.
# Graph 300's lower bound comes from the pass2.jsonl O_21 retraction.
REFERENCE_PROFILES = {
    i: {1: {5: b}}
    for i, b in {113: 2, 193: 1, 204: 2, 210: 1, 216: 1,
                 356: 1, 495: 5, 517: 12}.items()
}
REFERENCE_PROFILES[300] = {2: {20: 1}}


def reference_obstructions(index, level):
    screen = getattr(pc, "find_homological_retraction_obstruction", None)
    if screen is None:
        return None  # Requires library PR #54.
    host = {j-level: certified_profile(index, j) for j in range(level, 3)}
    obstructions = {}
    for target, profile in REFERENCE_PROFILES.items():
        witness = screen(host, profile)
        if witness is not None:
            obstructions[target] = dict(
                iterate=witness.iterate, degree=witness.degree,
                host_upper_bound=witness.host_upper_bound,
                target_lower_bound=witness.target_lower_bound,
            )
    return obstructions


def report(**kw):
    print("ROW " + json.dumps(kw, sort_keys=True, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int, choices=CUBICS)
    parser.add_argument("task", choices=("topology", "retractions", "structural"))
    a = parser.parse_args()
    signal.signal(signal.SIGALRM, alarm)
    g = nx.complement(nx.from_graph6_bytes(CUBICS[a.index].encode()))
    p = canon(pc.completely_pared_graph(g))
    for k in range(3):
        report(index=a.index, level=k, order=len(p), edges=p.number_of_edges(),
               host=code(p) if k < 2 else None)
        if a.task == "topology":
            h = limited(30, pt.homotopy_type_with_verdict, p, max_simplices=100000)
            report(index=a.index, level=k, task="homotopy", status="TIMEOUT" if h is None else "EXACT" if h.is_exact else "INEXACT", spheres=None if h is None else h.wedge.spheres, reason=None if h is None else h.reason)
        elif a.task == "retractions":
            assert len(p) == EXPECTED_ORDERS[a.index][k]
            known = retraction_betti(p)
            for d, b in certified_profile(a.index, k).items():
                assert d not in known or known[d] == b
                known[d] = b
            minimum = min_octahedron_dimension(p, known)
            report(index=a.index, level=k, task="homological-screen",
                   exact_betti=known, min_dimension=minimum,
                   all_octahedra_excluded=minimum > len(p)//2,
                   reference_obstructions=reference_obstructions(a.index, k))
            for seed in range((1 if k < 2 else 6) if minimum <= len(p)//2 else 0):
                order = list(p)
                random.Random(seed+182379).shuffle(order)
                h = nx.Graph()
                h.add_nodes_from(order)
                h.add_edges_from(p.edges)
                v = pc.find_octahedral_retraction_coaffine(h, min_dimension=minimum, max_states=3000000, time_limit=20, backend="native")
                report(index=a.index, level=k, task="coaffine-oct", seed=seed, status=v.status.name, states=v.states_explored, pairs=v.antipodal_pairs, projection=v.retraction)
                if v.status is pc.RetractionSearchStatus.FOUND:
                    assert pc.verify_retraction_witness(h, h.subgraph(v.inclusion.values()), (v.retraction, v.inclusion))
                    return
                if v.status is pc.RetractionSearchStatus.EXHAUSTED:
                    break
            v = limited(20, pc.find_coaffine_endomorphism, p, max_states=1000000, time_limit=15)
            report(index=a.index, level=k, task="coaffine-any", result=str(v))
        else:
            if k < 2:
                v = limited(30, pc.classify_clique_behavior_with_clockwork_pair_search, p, tries=2, bound=60, clockwork_fallback=True, induced_clockwork_depth=1, clockwork_retract_depth=1, clockwork_max_candidates=5000)
                report(index=a.index, level=k, task="classifier", verdict=None if v is None else v.verdict.name, reason=None if v is None else v.reason)
            v = limited(20, pc.search_proper_stiff_retraction, p, max_target_order=12) if len(p) <= 34 else None
            report(index=a.index, level=k, task="proper-stiff", result=str(v))
        if k < 2:
            nxt = pc.clique_graph(p, bound=1000)
            labels = sorted(nxt, key=lambda q: sum(1 << v for v in q))
            p = canon(pc.completely_pared_graph(nx.relabel_nodes(nxt, {q:i for i,q in enumerate(labels)})))


if __name__ == "__main__":
    main()

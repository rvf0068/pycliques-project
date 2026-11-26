"""Research-only census of all order-14 cubic complements.

Run on the installed workspace. No production code is modified.
Rows are emitted directly to the Actions log for reproducibility.
"""
import json
import multiprocessing as mp
import os
import random
import signal
import time
from collections import Counter

import networkx as nx
import pycliques as pc
import pycombtop as pt
from pyg6data import list_cubic_graphs


class CaseTimeout(BaseException):
    pass


def _alarm(_signum, _frame):
    raise CaseTimeout()


def _inputs():
    c8, c10, c14 = (list_cubic_graphs(n) for n in (8, 10, 14))
    assert (len(c8), len(c10), len(c14)) == (5, 19, 509)
    k4 = nx.complete_graph(4)
    c6 = [nx.complete_bipartite_graph(3, 3), nx.circular_ladder_graph(3)]
    rows = [(str(i), g) for i, g in enumerate(c14, 1)]
    rows += [
        (f"4+10:{i}", nx.disjoint_union(k4, g))
        for i, g in enumerate(c10, 1)
    ]
    rows += [
        (f"6+8:{j}:{i}", nx.disjoint_union(h, g))
        for j, h in enumerate(c6, 1)
        for i, g in enumerate(c8, 1)
    ]
    rows += [
        (f"4+4+6:{j}", nx.disjoint_union_all([k4, k4, h]))
        for j, h in enumerate(c6, 1)
    ]
    assert len(rows) == 540
    return rows



IDS = ["4+10:4", "67", "20", "22", "34", "75", "294", "4+10:10", "4+10:12", "4+10:2", "4+10:9", "4+4+6:2"]


def _classify(item):
    index, cubic = item
    g = pc.CliqueSequence(
        nx.complement(cubic), bound=1000, relabel=True, pare_initial=True
    )[1]
    assert g is not None
    row = {"id": index, "n": len(g), "m": g.number_of_edges(), "attempts": []}
    order = pt.connected_link_order(g)
    assert order is not None and pt.verify_connected_link_order(g, order)
    signal.signal(signal.SIGALRM, _alarm)
    start = time.monotonic()
    row["status"] = "LIMITED"
    natural = sorted(g)
    orders = [natural, sorted(natural, key=lambda v: (-len(g[v]), v)),
              sorted(natural, key=lambda v: (len(g[v]), v))]
    # Deterministic randomized alternatives.
    for seed in (17, 43):
        order = natural.copy()
        random.Random(seed).shuffle(order)
        orders.append(order)
    seen = set()
    sphere_seed = ()
    for order in orders:
        if tuple(order) in seen:
            continue
        seen.add(tuple(order))
        attempt = {"order": order}
        row["attempts"].append(attempt)
        signal.setitimer(signal.ITIMER_REAL, 40)
        t = time.monotonic()
        try:
            p = pt.symbolic_integral_morse_certificate(
                g, vertex_order=order, max_vertices=100,
                max_nodes=1_500_000, max_critical=1024,
                max_gradient_states=2_000_000, max_sq2_states=5_000_000,
                max_spherical_states=6_000_000, spherical_seed=sphere_seed,
            )
            if p is None:
                attempt["status"] = "LIMITED"
            else:
                attempt.update(status="HOMOLOGY", betti=p.betti, torsion=p.torsion,
                               critical=p.critical, nodes=p.diagram_nodes,
                               gradient_states=p.gradient_states, faces=p.total_faces)
                row.update(status="HOMOLOGY", betti=p.betti, torsion=p.torsion)
                if p.sq2 is not None:
                    attempt["sq2"] = dict(degree=p.sq2.degree, rank=p.sq2.rank,
                                          states=p.sq2.states)
                if p.spherical is not None:
                    attempt["spherical"] = dict(
                        complete=p.spherical.complete, limited=p.spherical.limited,
                        candidates=p.spherical.candidates, states=p.spherical.states,
                        degrees=[dict(degree=d.degree, betti=d.betti, rank=d.lattice_rank,
                                      smith=d.smith_factors, complete=d.complete,
                                      pairs=[c.pairs for c in d.cycles])
                                 for d in p.spherical.degrees])
                    sphere_seed = tuple(c.pairs for d in p.spherical.degrees for c in d.cycles)
                if p.homotopy_type is not None:
                    row.update(status="EXACT_WEDGE" if p.wedge_spheres is not None else "EXACT_OTHER",
                               spheres=p.wedge_spheres, type=p.homotopy_type.to_latex(),
                               method="SPHERICAL" if p.spherical and p.spherical.complete else
                               "SQ2" if p.sq2_homotopy_type is not None else "INTEGRAL_MORSE")
                    break
        except CaseTimeout:
            attempt["status"] = "TIMEOUT"
        except Exception as exc:
            attempt.update(status="ERROR", error=repr(exc))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            attempt["seconds"] = round(time.monotonic() - t, 4)
    row["seconds"] = round(time.monotonic() - start, 4)
    return row


def test_research_cubic14_census(capsys):
    inputs = dict(_inputs())
    rows = []
    workers = min(2, os.cpu_count() or 1)
    with capsys.disabled():
        print("POST64_START", len(IDS), workers, flush=True)
        with mp.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
            for row in pool.imap_unordered(_classify, [(i, inputs[i]) for i in IDS], chunksize=1):
                rows.append(row)
                print("POST64_ROW " + json.dumps(row, sort_keys=True), flush=True)
        print("POST64_SUMMARY", json.dumps(Counter(r["status"] for r in rows)), flush=True)
    assert len(rows) == len(IDS)
    assert not any(a["status"] == "ERROR" for r in rows for a in r["attempts"])

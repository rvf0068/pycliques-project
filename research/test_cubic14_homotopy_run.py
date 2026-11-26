"""Research-only census of all order-14 cubic complements.

Run on the installed workspace. No production code is modified.
Rows are emitted directly to the Actions log for reproducibility.
"""
import json
import multiprocessing as mp
import os
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


def _classify(item):
    index, cubic = item
    seq = pc.CliqueSequence(
        nx.complement(cubic), bound=1000, relabel=True, pare_initial=True
    )
    rows = []
    signal.signal(signal.SIGALRM, _alarm)
    for stage in (0, 1):
        g = seq[stage]
        assert g is not None
        row = {
            "id": index,
            "stage": stage,
            "n": len(g),
            "m": g.number_of_edges(),
            "simply_connected": pt.connected_link_order(g) is not None,
        }
        start = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, 2.5)
        try:
            v = pt.homotopy_type_with_verdict(
                g,
                max_vertices=80,
                max_simplices=25000,
                recursive_betti_states=4096,
                dong_seed=0,
            )
            row["status"] = v.kind.name
            row["spheres"] = v.wedge.spheres if not v.is_limited else None
            row["methods"] = [
                name for name, explanation in vars(pt.Theorem).items()
                if name.isupper() and isinstance(explanation, str)
                and explanation and explanation in v.reason
            ]
        except CaseTimeout:
            row["status"] = "TIMEOUT"
        except Exception as exc:
            row["status"] = "ERROR"
            row["error"] = repr(exc)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        row["seconds"] = round(time.monotonic() - start, 4)
        rows.append(row)
    return rows


def test_research_cubic14_census(capsys):
    rows = []
    workers = min(4, os.cpu_count() or 1)
    with capsys.disabled():
        print("CUBIC14_START", workers, flush=True)
        with mp.get_context("fork").Pool(workers, maxtasksperchild=8) as pool:
            for pair in pool.imap_unordered(_classify, _inputs(), chunksize=1):
                for row in pair:
                    rows.append(row)
                    print("CUBIC14_ROW " + json.dumps(row, sort_keys=True), flush=True)
        for stage in (0, 1):
            counts = Counter(r["status"] for r in rows if r["stage"] == stage)
            print("CUBIC14_SUMMARY", stage, json.dumps(counts, sort_keys=True), flush=True)
    assert len(rows) == 1080
    assert len({(r["id"], r["stage"]) for r in rows}) == 1080
    assert not any(r["status"] == "ERROR" for r in rows)

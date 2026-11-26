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


# All 233 cases not classified by the post-58 production run.
UNRESOLVED = ["2","4","7","12","13","15","17","18","20","21","22","24","26","28","29","30","31","33","34","35","37","38","39","40","43","45","46","48","49","50","53","54","55","56","59","60","62","63","64","65","67","68","69","70","72","74","75","76","77","78","79","80","81","83","84","86","87","91","92","93","95","97","98","99","106","109","110","111","115","116","118","119","120","121","122","125","130","133","136","138","143","145","146","148","149","151","154","155","159","160","161","163","165","166","168","171","173","174","175","177","178","179","180","184","187","190","198","200","201","202","207","208","209","211","213","217","218","221","222","223","224","225","227","233","235","236","238","243","244","245","249","250","251","255","256","257","260","265","266","267","268","274","275","276","278","279","280","281","284","288","289","290","293","294","295","296","297","301","303","304","305","306","310","312","313","314","315","316","318","321","322","324","325","327","328","330","332","334","335","336","337","339","340","341","342","343","345","347","350","351","352","353","354","355","357","359","360","365","366","367","370","371","372","382","385","386","387","388","389","398","399","458","460","480","488","492","496","498","501","503","504","4+10:1","4+10:10","4+10:12","4+10:13","4+10:18","4+10:2","4+10:3","4+10:4","4+10:6","4+10:9","4+4+6:2","6+8:2:5"]
# Replay earlier experimental successes first, then process the remainder.
PRIORITY = ["2", "15", "17", "18", "64", "138", "21", "24", "26", "28", "30", "31", "33", "35", "37", "38", "40", "45", "46", "48", "49", "50", "54", "55", "56", "59", "60", "62", "63"]


def _classify(item):
    index, cubic = item
    g = pc.CliqueSequence(
        nx.complement(cubic), bound=1000, relabel=True, pare_initial=True
    )[1]
    row = {"id": index, "n": len(g), "m": g.number_of_edges(), "attempts": []}
    assert pt.verify_connected_link_order(g, pt.connected_link_order(g))
    row["simple"] = True
    signal.signal(signal.SIGALRM, _alarm)
    start = time.monotonic()
    row["status"] = "LIMITED"
    natural = sorted(g)
    # First use the degree orders favored in earlier research; budget
    # each separately, retaining exact homology even when no type follows.
    orders = [
        sorted(natural, key=lambda v: (len(g[v]), v)),
        sorted(natural, key=lambda v: (-len(g[v]), v)),
        natural,
    ]
    seen = set()
    for order in orders:
        if tuple(order) in seen:
            continue
        seen.add(tuple(order))
        attempt = {"order": order}
        row["attempts"].append(attempt)
        signal.setitimer(signal.ITIMER_REAL, 4.0)
        t = time.monotonic()
        try:
            p = pt.symbolic_integral_morse_certificate(
                g, vertex_order=order, max_vertices=100,
                max_nodes=500000, max_critical=192,
                max_gradient_states=500000, max_sq2_states=3000000,
            )
            if p is None:
                attempt["status"] = "LIMITED"
            else:
                attempt.update(
                    status="HOMOLOGY", betti=p.betti, torsion=p.torsion,
                    critical=p.critical, faces=p.total_faces,
                    nodes=p.diagram_nodes, gradient_states=p.gradient_states,
                    sq2_limited=p.sq2_limited,
                )
                if p.sq2 is not None:
                    attempt["sq2"] = dict(
                        degree=p.sq2.degree, rank=p.sq2.rank,
                        source=p.sq2.source_rank, target=p.sq2.target_rank,
                        rows=p.sq2.rows, lifted=p.sq2.lifted_faces,
                        states=p.sq2.states,
                    )
                row["status"] = "HOMOLOGY"
                row["betti"] = p.betti
                row["torsion"] = p.torsion
                if p.homotopy_type is not None:
                    row["status"] = "EXACT_WEDGE" if p.wedge_spheres is not None else "EXACT_OTHER"
                    row["spheres"] = p.wedge_spheres
                    row["type"] = p.homotopy_type.to_latex()
                    row["method"] = "SQ2" if p.sq2_homotopy_type is not None else "INTEGRAL_MORSE"
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
    ids = list(dict.fromkeys(PRIORITY + UNRESOLVED))
    ids = [i for i in ids if i in UNRESOLVED]
    # Bound the first follow-up tranche; remaining inputs can use the
    # same driver without altering the production library.
    ids = ids[:100]
    rows = []
    workers = min(4, os.cpu_count() or 1)
    with capsys.disabled():
        print("POST63_START", len(ids), workers, flush=True)
        with mp.get_context("fork").Pool(workers, maxtasksperchild=5) as pool:
            for row in pool.imap_unordered(_classify, [(i, inputs[i]) for i in ids]):
                rows.append(row)
                print("POST63_ROW " + json.dumps(row, sort_keys=True), flush=True)
        print("POST63_SUMMARY", json.dumps(Counter(r["status"] for r in rows)), flush=True)
    assert len(rows) == len(ids)
    assert not any(a["status"] == "ERROR" for r in rows for a in r["attempts"])

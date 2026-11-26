"""Private, isolated DIMACS worker using the optional system Z3 library.

Run as a script, not through the package import machinery. The operating
system address-space limit covers formula loading as well as solving.
"""

from __future__ import annotations

import ctypes as c
import json
import os
import re
import resource
import sys


def solve(library, path, max_conflicts, memory_mb, x_count):
    """Load a CNF and return only its assignment variables and SAT status."""
    z3 = c.CDLL(library)
    ptr, string, uint = c.c_void_p, c.c_char_p, c.c_uint

    def bind(name, args, returns=None):
        function = getattr(z3, "Z3_" + name)
        function.argtypes = args
        function.restype = returns
        return function

    config = bind("mk_config", [], ptr)()
    context = bind("mk_context", [ptr], ptr)(config)
    bind("del_config", [ptr])(config)
    tactic = bind("mk_tactic", [ptr, string], ptr)(context, b"sat")
    bind("tactic_inc_ref", [ptr, ptr])(context, tactic)
    solver = bind("mk_solver_from_tactic", [ptr, ptr], ptr)(context, tactic)
    bind("solver_inc_ref", [ptr, ptr])(context, solver)
    try:
        params = bind("mk_params", [ptr], ptr)(context)
        bind("params_inc_ref", [ptr, ptr])(context, params)
        symbol = bind("mk_string_symbol", [ptr, string], ptr)
        set_uint = bind("params_set_uint", [ptr, ptr, ptr, uint])
        set_uint(context, params, symbol(context, b"max_conflicts"), max_conflicts)
        set_uint(context, params, symbol(context, b"max_memory"), memory_mb)
        bind("solver_set_params", [ptr, ptr, ptr])(context, solver, params)
        bind("params_dec_ref", [ptr, ptr])(context, params)
        bind("solver_from_file", [ptr, ptr, string])(context, solver, os.fsencode(path))
        code = bind("solver_check", [ptr, ptr], c.c_int)(context, solver)
        result = {
            "status": {1: "FOUND", -1: "EXHAUSTED", 0: "LIMIT_REACHED"}[code],
            "conflicts": 0,
            "selected": [],
        }
        stats = bind("solver_get_statistics", [ptr, ptr], ptr)(context, solver)
        bind("stats_inc_ref", [ptr, ptr])(context, stats)
        size = bind("stats_size", [ptr, ptr], uint)(context, stats)
        key = bind("stats_get_key", [ptr, ptr, uint], string)
        is_uint = bind("stats_is_uint", [ptr, ptr, uint], c.c_bool)
        value = bind("stats_get_uint_value", [ptr, ptr, uint], uint)
        for i in range(size):
            if key(context, stats, i) in (b"sat conflicts", b"conflicts") and is_uint(
                context, stats, i
            ):
                result["conflicts"] = value(context, stats, i)
        bind("stats_dec_ref", [ptr, ptr])(context, stats)
        if code == 1:
            model = bind("solver_get_model", [ptr, ptr], ptr)(context, solver)
            # DIMACS variables are named k!<positive index> by Z3.
            # The caller independently validates every decoded map.
            text = bind("model_to_string", [ptr, ptr], string)(context, model)
            result["selected"] = [
                int(match)
                for match in re.findall(rb"k!(\d+)\s*->\s*true", text)
                if int(match) <= x_count
            ]
        return result
    finally:
        bind("solver_dec_ref", [ptr, ptr])(context, solver)
        bind("tactic_dec_ref", [ptr, ptr])(context, tactic)
        bind("del_context", [ptr])(context)


def main():
    library, path, conflicts, memory, x_count = sys.argv[1:]
    # Set a hard process limit before loading libz3. Keep the parent's
    # memory space separate, including when the solver aborts.
    bound = int(memory) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (bound, bound))
    print(json.dumps(solve(library, path, int(conflicts), int(memory), int(x_count))))


if __name__ == "__main__":
    main()

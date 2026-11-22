# Accelerated unseeded octahedral retractions

The public function `find_octahedral_retraction_coaffine(G)` searches for
coaffine weak endomorphisms \(\phi:G\to G\). If the second iterate
\(\phi^2\) fixes its image pointwise and that image induces an octahedral
graph \(O_d\), the result contains a certified ordinary retraction onto
\(O_d\). Conversely, every ordinary octahedral retraction yields such a
coaffine weak endomorphism by swapping antipodes, so an **exhaustive**
unseeded search is complete for this target family.

## Optional C++17 backend

The original pure-Python algorithm remains available and requires no
compiler. A faster bitset implementation is included as C++17 *source*,
with a separate, explicit build command:

```bash
# From the pycliques-project monorepo root, after uv sync:
uv run python -m pycliques.octahedral_coaffine_native --build
```

Or from Python/Jupyter:

```python
from pycliques import build_coaffine_native
print(build_coaffine_native())
```

The build requires a C++17 compiler (for example, `g++` on Ubuntu).
It downloads nothing, stores the executable under the XDG cache
(default `~/.cache/pycliques/`) and hashes the source to avoid stale
binaries after an update. The wheel remains a pure-Python package.
There is **no compile-on-import** and **no compile-on-install**.

Choose the engine when calling the unchanged public function:

```python
import pycliques

result = pycliques.find_octahedral_retraction_coaffine(
    graph, backend="native", max_states=10000, time_limit=30
)
```

- `backend="auto"` (default) uses the native executable if present for
  hosts of order at most 2,048; otherwise it uses the existing Python solver
- `backend="python"` forces the established pure-Python implementation
- `backend="native"` requires an already built native executable and
  supports at most 2,048 vertices

The environment variable `PYCLIQUES_COAFFINE_NATIVE` can specify an
explicit executable path; otherwise the version-matched cached build is
used. Input graphs may have *arbitrary hashable NetworkX vertex labels*.
The wrapper serializes temporary integer indices and maps the result
back to original labels. Only the temporary graph is written to disk,
under an automatically removed temporary directory.

The C++ implementation propagates cached bitset neighborhood supports
rather than revisiting each image individually. Each search state
contains the current candidate domains and their support unions.
Arc consistency removes image candidates inconsistent with
neighboring domains. The solver branches exhaustively (until bounded)
on a minimum-domain variable and verifies the second-iterate condition
on each complete weak endomorphism.

### Exact support-change-driven acceleration

For any nonempty domain \\(D \\subseteq V(X)\\), the image values
supported by adjacency or equality in the original graph are

\\[
\\operatorname{supp}_X(D)=\\bigcup_{t\\in D}N_X[t]
=V(X)\\setminus\\bigcap_{t\\in D}N_{\\overline{X}}(t).
\\]

The updated native engine computes the intersection of *open*
complement neighborhoods and stops early if it becomes empty.
This is the exact same support union, with no approximation.

On each domain reduction the engine recomputes the support mask.
**Only a strict change of that support mask queues work for adjacent
source vertices.** A domain shrink that leaves its support unchanged
cannot affect an adjacent variable: the arc constraint sees only
the support mask, not the individual candidate identities.
The initial queue likewise includes only variables with nonfull
support. This is an exact arc-consistency fixed point, not a
heuristic pruning rule, and preserves the exhaustive search status
and the existing independently verified positive witness protocol.

The larger 2,048-vertex bitsets permit experimentation on
fourth pared clique iterates beyond the former 512-vertex limit.
Large dense host graphs can still be expensive; keep explicit
state/time budgets and interpret LIMIT_REACHED as inconclusive.
For regression coverage, the native test suite checks a
513-vertex true-twin enlargement of O3 with a genuine O3
retraction, a 2,048-vertex star with no possible coaffine image,
and exact outcome agreement with the Python reference on
small graphs.

**Every native FOUND result is checked again in Python**, including
coaffinity of the endomorphism, preservation of edges up to equality,
the equation \(\phi^2=r\), the induced octahedron, pointwise fixation of
the image, and the ordinary graph retraction using
`verify_retraction_witness`.

Only `EXHAUSTED` is a negative theorem. `LIMIT_REACHED` means the
state/time budget was exhausted or the subprocess deadline was hit;
it never proves that an octahedral retraction is absent.

## McKay 253650 from scratch

Once the optional executable is built, the following Jupyter-Python
code **does not include any target, antipodal pairs or archived map**:

```python
import networkx as nx
import pycliques as pc

p = pc.completely_pared_graph(
    nx.from_graph6_bytes(b"HEjer~{")
)
for _ in range(4):
    k = pc.clique_graph(p)
    assert k is not None
    p = pc.completely_pared_graph(k)

assert (len(p), p.number_of_edges()) == (259, 28890)
result = pc.find_octahedral_retraction_coaffine(
    p, backend="native", max_states=10000, time_limit=30
)
assert result.status is pc.RetractionSearchStatus.FOUND
assert result.dimension == 12
target = p.subgraph(v for pair in result.antipodal_pairs for v in pair)
assert pc.verify_retraction_witness(
    p, target, (result.retraction, result.inclusion)
)
print(f"P4 -> O_{result.dimension}")
```

In a local Linux test (C++17 compiled with `-O3`), a 259-vertex,
28,890-edge McKay 253650 host produced an O12 witness in **1,362
states** and approximately **1.2–1.6 seconds** of native CSP time.
These are descriptive measurements, not a guaranteed runtime or a
promise that every graph will be easy.

The previous solver's result statuses and `CoaffineOctahedronResult`
fields are retained. The native source has an explicit 2,048-vertex
capacity; with `backend="auto"`, a larger graph goes through the
unbounded Python implementation rather than generating a false negative.

# pycombtop

Build simplicial complexes from generating faces and use reductions, matchings, homology and covering graphs to study their topology. Install from the repository root with `uv sync` or `pip install ./packages/pycliques ./packages/pycombtop`.

```python
import networkx as nx
from pycombtop import homotopy_type_with_verdict

result = homotopy_type_with_verdict(nx.cycle_graph(5))
assert result.proved_wedge.spheres == {1: 1}
```

Use `kind` to distinguish a proved wedge, reduced Betti numbers over GF(2), and a limited calculation. Both homotopy entry points require a nonempty geometric realization. The void complex has no faces and the clique complex of the empty graph has the empty face alone.

[Certified reductions and references](https://github.com/rvf0068/pycliques-project/blob/main/docs/certified_homotopy.md)

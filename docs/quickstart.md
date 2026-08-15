# Quickstart

## Installation

Requires Python 3.10, 3.11, or 3.12.

### From PyPI

```bash
pip install groupoid
```

GROUPOID is an early development pre-release. See `STATUS.md`,
`LIMITATIONS.md`, and `CORRECTION_NOTICE.md` before relying on its scientific
semantics.

## Basic Usage

### Computing the Karcher Mean

```python
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from groupoid.manifold import karcher_mean

manifold = Hypersphere(dim=2)
points = manifold.random_point(n_samples=5)
mean = karcher_mean(manifold, points)
print(manifold.belongs(mean))
```

### Composing Matrix-Labelled Morphisms

```python
import numpy as np
from groupoid.groupoid import Morphism, compose, inverse

T_AB = np.array([[0.0, -1.0], [1.0, 0.0]])
T_BC = np.array([[0.0, 1.0], [-1.0, 0.0]])

f = Morphism(source="A", target="B", transport_map=T_AB)
g = Morphism(source="B", target="C", transport_map=T_BC)
h = compose(f, g)
h_inv = inverse(h)
```

`Morphism` supplies matrix composition/inversion. For point-valued aggregation,
`TransportGroupoidAggregator.register_transport` additionally requires that
explicitly registered opposite arrows be mutual numerical inverses and that the
exercised forward and inverse actions preserve the configured manifold
representation.

### Computing the Cycle-Basis Holonomy Defect

```python
import networkx as nx
import numpy as np
from groupoid.cohomology import cycle_basis_holonomy_defect

graph = nx.DiGraph()
graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])

g_A = np.eye(3)
g_B = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
g_C = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)

transport_maps = {
    ("A", "B"): g_B @ np.linalg.inv(g_A),
    ("B", "C"): g_C @ np.linalg.inv(g_B),
    ("A", "C"): g_C @ np.linalg.inv(g_A),
}

defect = cycle_basis_holonomy_defect(graph, transport_maps)
print(f"cycle-basis holonomy defect: {defect:.2e}")
```

Exact zero has a flatness interpretation only under the assumptions described
in `docs/theory.md`. The nonzero magnitude is basis-dependent and is not a
canonical H^1 norm.

### Tangent-Vector Parallel Transport

Use `pole_ladder` or `schild_ladder` directly for tangent-vector transport.
`compute_tangent_transport_matrix` is a convenience helper only for
vector-shaped point representations (1D coordinate arrays of equal shape). Do
not register its ambient tangent operator as a point-valued aggregation
morphism. `register_transport_from_points` is retained only as a fail-closed
compatibility stub.

## Running Tests

```bash
pytest tests/ -v
```

## Building Documentation

```bash
mkdocs serve
```

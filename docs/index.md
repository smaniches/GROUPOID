# GROUPOID

**Groupoid-based federated learning with Riemannian geometry.**

GROUPOID is a framework for federated learning that uses the mathematical
structure of groupoids, sheaves, and Riemannian geometry to aggregate
models across heterogeneous clients while preserving geometric invariants
and detecting inconsistencies via cohomological methods.

## Key Features

- **Riemannian Aggregation**: Compute Karcher (Frechet) means on manifolds
  instead of naive Euclidean averaging, respecting the geometry of parameter
  spaces.
- **Cohomological Obstruction Detection**: Use first cohomology H^1 of
  transport groupoids to detect when local models cannot be consistently
  aggregated.
- **Sheaf-Theoretic Consistency**: Model data flow between clients as
  sections of a cellular sheaf, with restriction maps enforcing local-to-global
  compatibility.
- **Sheaf Laplacian Spectral Analysis**: Analyze the algebraic connectivity
  of the federation and drive consensus via sheaf diffusion.
- **Parallel Transport**: Schild's ladder and pole ladder implementations
  for transporting gradients between client tangent spaces.
- **Riemannian Optimizers**: Manifold-aware SGD and Adam with curvature-adaptive
  learning rates.
- **Persistent Homology**: Track federation divergence across rounds using
  topological data analysis.
- **Differential Privacy**: Optional integration with Opacus and TenSEAL for
  differentially private and homomorphically encrypted federated learning.

## Architecture

```
Client A ──T_AB──► Client B
  │                   │
  T_AC              T_BD
  │                   │
  ▼                   ▼
Client C ──T_CD──► Client D
        ╲         ╱
         ╲       ╱
      Karcher Mean
     (on manifold)
         ╱       ╲
        ╱         ╲
   global model → local updates
   (via inverse transport)
```

## Installation

Install the core package:

```bash
pip install groupoid
```

For development (includes testing, linting, and benchmarking tools):

```bash
pip install -e ".[all]"
```

## Quick Example

```python
import networkx as nx
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from groupoid import TransportGroupoidAggregator

manifold = Hypersphere(dim=2)
graph = nx.DiGraph([("A", "B"), ("A", "C")])

aggregator = TransportGroupoidAggregator(
    manifold=manifold,
    graph=graph,
    base_node="A",
)

# Register rotation matrices as transport maps
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta),  np.cos(theta), 0],
              [0, 0, 1]])
aggregator.register_transport("A", "B", R)
aggregator.register_transport("A", "C", R.T)

# Each client has parameters on S^2
client_params = {
    "A": np.array([0.0, 0.0, 1.0]),
    "B": np.array([0.1, 0.0, 0.995]),
    "C": np.array([-0.1, 0.0, 0.995]),
}
# Normalize to manifold
client_params = {k: v / np.linalg.norm(v) for k, v in client_params.items()}

result = aggregator.aggregate(client_params)
print(f"H^1 = {result.h1_norm:.2e} (consistent: {result.is_consistent})")
print(f"Global model: {result.global_params}")
```

## Modules

| Module | Description |
|---|---|
| `groupoid.manifold` | Karcher mean computation via geomstats |
| `groupoid.groupoid` | Morphism composition, inverse, transport groupoid |
| `groupoid.cohomology` | First cohomology H^1 for obstruction detection |
| `groupoid.sheaf` | Cellular sheaf with restriction maps |
| `groupoid.aggregation` | Full federated aggregation pipeline |
| `groupoid.laplacian` | Sheaf Laplacian, spectral analysis, diffusion |
| `groupoid.transport` | Parallel transport (Schild's ladder, pole ladder) |
| `groupoid.optimizer` | Riemannian SGD, Adam, curvature-adaptive LR |
| `groupoid.persistence` | Persistent homology for divergence tracking |

## License

Copyright 2024 TOPOLOGICA LLC. Licensed under the Apache License, Version 2.0.

# GROUPOID

> **Pre-alpha research prototype.** Not production software.
> See [STATUS](https://github.com/smaniches/GROUPOID/blob/main/STATUS.md)
> and [LIMITATIONS](https://github.com/smaniches/GROUPOID/blob/main/LIMITATIONS.md).

**Groupoid-based aggregation for federated learning on Riemannian manifolds.**

GROUPOID explores using the mathematical structure of transport groupoids,
cellular sheaves, and Riemannian geometry to aggregate model parameters
across heterogeneous federated clients. The central hypothesis is that
geometry-aware aggregation (Karcher mean with parallel transport) can
outperform naive Euclidean averaging when client parameter spaces are
heterogeneous, and that cohomological invariants (H^1) can diagnose
irreconcilable model divergence before it degrades performance.

## Implemented and tested

| Module | Description | Test coverage |
|---|---|---|
| `groupoid.manifold` | Karcher mean via geomstats FrechetMean | Hypothesis (500 examples) |
| `groupoid.groupoid` | Morphism composition, inverse | Hypothesis (500 examples) |
| `groupoid.cohomology` | H^1 via cycle-basis holonomy | Hypothesis (500 examples) |
| `groupoid.sheaf` | Cellular sheaf, restriction maps | Hypothesis (500 examples) |
| `groupoid.laplacian` | Sheaf Laplacian, spectral analysis, diffusion | Integration tests |
| `groupoid.aggregation` | Transport-aware federated aggregation pipeline | Integration tests |

## Implemented, not yet tested

| Module | Description | Status |
|---|---|---|
| `groupoid.transport` | Schild's ladder, pole ladder parallel transport | Untested |
| `groupoid.optimizer` | Riemannian SGD, Adam, curvature-adaptive LR | Untested |
| `groupoid.persistence` | Vietoris-Rips persistent homology | Untested |

## Not yet implemented

- Differential privacy (Opacus, TenSEAL integration)
- Federated training loop with real neural networks
- Communication protocol for distributed deployment
- Convergence guarantees or formal proofs

## Architecture

```
Client A ---T_AB---> Client B
  |                    |
  T_AC               T_BD
  |                    |
  v                    v
Client C ---T_CD---> Client D
        \          /
         \        /
       Karcher Mean
      (on manifold)
         /        \
        /          \
   global model -> local updates
   (via inverse transport)
```

## Installation

From source (not published on PyPI):

```bash
git clone https://github.com/smaniches/GROUPOID.git
cd GROUPOID
pip install -e ".[dev]"
```

## Quick example

```python
import networkx as nx
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from groupoid import TransportGroupoidAggregator

manifold = Hypersphere(dim=2)
graph = nx.DiGraph([("A", "B"), ("A", "C")])

aggregator = TransportGroupoidAggregator(
    manifold=manifold, graph=graph, base_node="A"
)

theta = np.pi / 6
R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0, 0, 1],
])
aggregator.register_transport("A", "B", R)
aggregator.register_transport("A", "C", R.T)

client_params = {
    "A": np.array([0.0, 0.0, 1.0]),
    "B": np.array([0.1, 0.0, 0.995]),
    "C": np.array([-0.1, 0.0, 0.995]),
}
client_params = {k: v / np.linalg.norm(v) for k, v in client_params.items()}

result = aggregator.aggregate(client_params)
print(f"H^1 = {result.h1_norm:.2e} (consistent: {result.is_consistent})")
```

## License

Copyright 2024 TOPOLOGICA LLC. Licensed under the Apache License, Version 2.0.

# GROUPOID

[![CI](https://github.com/smaniches/GROUPOID/actions/workflows/ci.yml/badge.svg)](https://github.com/smaniches/GROUPOID/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#status)

> **Pre-alpha research prototype.** This is an early-stage exploration of
> groupoid-based aggregation for federated learning on Riemannian manifolds.
> It is not a production federated learning system. See
> [STATUS.md](STATUS.md) and [LIMITATIONS.md](LIMITATIONS.md).

## Overview

GROUPOID explores using transport groupoids, cellular sheaves, and
Riemannian geometry to aggregate model parameters across heterogeneous
federated clients. The core idea: instead of naive Euclidean averaging
(FedAvg), transport client parameters to a common frame via groupoid
morphisms, check cohomological consistency, and compute the intrinsic
Karcher mean on the parameter manifold.

## Implemented and tested

These components have working implementations with property-based and
integration tests:

- **Karcher mean** on Riemannian manifolds via geomstats (`groupoid.manifold`)
- **Transport groupoid**: morphism composition, inverse, composition
  associativity verified by Hypothesis (`groupoid.groupoid`)
- **First cohomology H^1**: holonomy-based obstruction detection on
  cycle basis; coboundary vanishing tested (`groupoid.cohomology`)
- **Cellular sheaf**: restriction maps with functoriality tested
  (`groupoid.sheaf`)
- **Sheaf Laplacian**: spectral analysis, algebraic connectivity,
  diffusion convergence tested (`groupoid.laplacian`)
- **Federated aggregation pipeline**: transport-aware aggregation with
  H^1 consistency checking, multi-round convergence tested
  (`groupoid.aggregation`)

## Implemented, not yet tested

These modules have implementations but lack test coverage:

- **Parallel transport**: Schild's ladder and pole ladder
  (`groupoid.transport`)
- **Riemannian optimizers**: SGD and Adam with exponential map retraction
  (`groupoid.optimizer`)
- **Persistent homology**: Vietoris-Rips filtration for divergence
  tracking (`groupoid.persistence`)

## Status

Pre-alpha. See [STATUS.md](STATUS.md) for details.

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

# Register rotation matrices as transport maps
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

## Running tests

```bash
pytest tests/ -v
```

## Documentation

[smaniches.github.io/GROUPOID](https://smaniches.github.io/GROUPOID)

## License

Copyright 2026 TOPOLOGICA LLC. Licensed under the
[Apache License, Version 2.0](LICENSE).

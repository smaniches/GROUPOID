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
- **Differential Privacy**: Optional integration with Opacus and TenSEAL for
  differentially private and homomorphically encrypted federated learning.

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
from geomstats.geometry.hypersphere import Hypersphere
from groupoid.manifold import karcher_mean

manifold = Hypersphere(dim=2)
client_params = manifold.random_point(n_samples=5)
global_model = karcher_mean(manifold, client_params)
```

## License

Copyright 2024 TOPOLOGICA LLC. Licensed under the Apache License, Version 2.0.

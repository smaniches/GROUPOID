# GROUPOID

[![CI](https://github.com/smaniches/GROUPOID/actions/workflows/ci.yml/badge.svg)](https://github.com/smaniches/GROUPOID/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Groupoid-based federated learning with Riemannian geometry.**

GROUPOID is a framework for federated learning that uses groupoids, sheaves,
and Riemannian geometry to aggregate models across heterogeneous clients while
preserving geometric invariants and detecting inconsistencies via cohomological
methods.

## Features

- **Riemannian Aggregation**: Karcher (Frechet) means on manifolds instead of
  naive Euclidean averaging
- **Cohomological Obstruction Detection**: First cohomology H^1 of transport
  groupoids detects when local models cannot be consistently aggregated
- **Sheaf-Theoretic Consistency**: Cellular sheaves with restriction maps
  enforce local-to-global compatibility
- **Differential Privacy**: Optional integration with Opacus and TenSEAL

## Installation

```bash
pip install groupoid
```

For development:

```bash
pip install -e ".[all]"
pre-commit install
```

## Quick Example

```python
from geomstats.geometry.hypersphere import Hypersphere
from groupoid.manifold import karcher_mean

manifold = Hypersphere(dim=2)
client_params = manifold.random_point(n_samples=5)
global_model = karcher_mean(manifold, client_params)
```

## Documentation

Full documentation is available at [smaniches.github.io/GROUPOID](https://smaniches.github.io/GROUPOID).

## License

Copyright 2024 TOPOLOGICA LLC. Licensed under the
[Apache License, Version 2.0](LICENSE).

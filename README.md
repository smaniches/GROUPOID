# GROUPOID

[![CI](https://github.com/smaniches/GROUPOID/actions/workflows/ci.yml/badge.svg)](https://github.com/smaniches/GROUPOID/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
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
- **Sheaf Laplacian**: connection Laplacian L = delta^T delta; PSD and
  delta^T-delta equality verified on non-orthogonal restriction maps, plus
  spectral analysis, algebraic connectivity, and diffusion convergence
  tested (`groupoid.laplacian`)
- **Federated aggregation pipeline**: transport-aware aggregation with
  H^1 consistency checking, multi-round convergence tested
  (`groupoid.aggregation`)

## Implemented, validated against ground truth, not yet integrated

These modules have unit tests that validate behavior against a
known-correct reference, but are not yet wired into the main aggregation
pipeline:

- **Parallel transport**: Schild's ladder and pole ladder
  (`groupoid.transport`). The pole ladder is validated against geomstats'
  analytic parallel transport on S^2 -- it matches in direction
  (cosine > 0.999) and magnitude. Schild's ladder is a coarser
  first-order approximation and is asserted as such. See
  [LIMITATIONS.md](LIMITATIONS.md) for the convergence caveat.
- **Persistent homology**: Vietoris-Rips filtration for divergence
  tracking (`groupoid.persistence`). Unit-tested against point clouds of
  known topology: a circle's dominant 1-cycle (via maximum persistence),
  two-cluster component counting (`betti_0 == 2` at a finite filtration),
  and a translation-invariant bottleneck distance. The Betti numbers are
  degenerate under the default `thresh=inf` filtration; see
  [LIMITATIONS.md](LIMITATIONS.md).

## Implemented, smoke-tested, not yet integrated

This module has implementation with smoke-test coverage only -- the tests
check coarse sanity (steps stay on the manifold), not core correctness --
and it is not yet wired into the main aggregation pipeline:

- **Riemannian optimizers**: SGD and Adam with exponential map
  retraction; smoke-tested to stay on the manifold after a step, with the
  curvature-adaptive learning rate covered for both its damping and
  fallback branches. Core descent/convergence behavior is not validated
  (`groupoid.optimizer`)

## Status

Pre-alpha. See [STATUS.md](STATUS.md) for details.

## Related work / why not just use X?

GROUPOID sits at the intersection of three existing toolchains and is not a
replacement for any of them. It is an exploratory prototype of one specific
idea -- transport-groupoid aggregation with cohomological consistency checking
-- not a federated learning framework.

- **Flower / FedML / TensorFlow Federated** -- mature federated learning
  frameworks providing the client/server communication, orchestration, and
  real training loops that GROUPOID deliberately does **not** implement (see
  [LIMITATIONS.md](LIMITATIONS.md): "Not a federated learning framework").
  GROUPOID is about the *aggregation operator*, not the FL plumbing; in
  principle a transport-aware aggregator like this one would be dropped into
  such a framework, not used instead of it.
- **geomstats / pymanopt** -- Riemannian-geometry libraries. GROUPOID *uses*
  geomstats for the manifold primitives (the Karcher mean delegates to
  geomstats `FrechetMean`). What GROUPOID adds on top is the transport
  groupoid, the H^1 holonomy/consistency check, and the cellular-sheaf
  Laplacian wiring -- not the manifold geometry itself.
- **Cellular-sheaf spectral methods** (the sheaf-Laplacian line of work,
  e.g. Hansen and Ghrist's spectral theory of cellular sheaves, and sheaf
  neural networks) -- GROUPOID's sheaf Laplacian follows this line and is the
  geometric machinery for detecting inconsistency across clients. The
  contribution here is applying it to the federated-aggregation setting, not
  the sheaf-Laplacian construction in the abstract.

In short: use Flower/FedML/TFF for the FL system, use geomstats/pymanopt for
manifold math; GROUPOID is a research prototype testing whether combining a
transport groupoid with sheaf-cohomological consistency yields a better
aggregation operator than Euclidean FedAvg. That hypothesis is **not yet
validated** (see [STATUS.md](STATUS.md)).

## Installation

Requires **Python 3.10, 3.11, or 3.12**. Python 3.13+ is not supported: the
`numpy<2.0` / `scipy<1.14` pins (needed for geomstats compatibility, see
[LIMITATIONS.md](LIMITATIONS.md)) have no wheels there, so `pip` will refuse
with a `Requires-Python` message rather than attempt a source build.

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

The suite reaches 100% line and branch coverage of the `groupoid` package
on Python 3.10-3.12, enforced in CI:

```bash
pytest tests/ --cov=groupoid --cov-branch --cov-fail-under=100
```

Coverage measures which lines run, not whether behavior is correct. See
[STATUS.md](STATUS.md) for the per-component validation depth, which
coverage alone does not capture.

## Documentation

[smaniches.github.io/GROUPOID](https://smaniches.github.io/GROUPOID)

## License

Copyright 2026 TOPOLOGICA LLC. Licensed under the
[Apache License, Version 2.0](LICENSE).

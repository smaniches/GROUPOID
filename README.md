# GROUPOID

[![CI](https://github.com/smaniches/GROUPOID/actions/workflows/ci.yml/badge.svg)](https://github.com/smaniches/GROUPOID/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/groupoid.svg)](https://pypi.org/project/groupoid/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#status)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20563974.svg)](https://doi.org/10.5281/zenodo.20563974)

> **Pre-alpha research prototype.** This is an early-stage exploration of
> groupoid-based aggregation for federated learning on Riemannian manifolds.
> It is not a production federated learning system. See
> [STATUS.md](STATUS.md), [LIMITATIONS.md](LIMITATIONS.md), and
> [CORRECTION_NOTICE.md](CORRECTION_NOTICE.md).

## Overview

GROUPOID explores using transport groupoids, cellular sheaves, and
Riemannian geometry to aggregate model parameters across heterogeneous
federated clients. The core idea: instead of naive Euclidean averaging
(FedAvg), transport client parameters to a common frame via explicit
point-action morphisms, evaluate cycle-holonomy consistency, and compute the
intrinsic Karcher mean on the parameter manifold.

## Implemented and tested

These components have working implementations with property-based and
integration tests:

- **Karcher mean** on Riemannian manifolds via geomstats (`groupoid.manifold`)
- **Transport groupoid**: morphism composition, inverse, composition
  associativity verified by Hypothesis (`groupoid.groupoid`)
- **Cycle-basis holonomy defect**: `cycle_basis_holonomy_defect` computes
  `max_{gamma in B} ||Hol(gamma) - I||_F` over the NetworkX cycle basis.
  Exact vanishing has a flatness interpretation under explicit assumptions,
  while the nonzero magnitude is basis- and representation-dependent. The
  historical `compute_h1` name is retained as a deprecated compatibility
  alias; the scalar is not a canonical H^1 norm. An incompletely specified
  basis cycle raises `IncompleteCocycleError`; conflicting explicitly supplied
  opposite arrows raise `NonReciprocalTransportError` (`groupoid.cohomology`)
- **Cellular sheaf**: restriction maps with functoriality tested
  (`groupoid.sheaf`)
- **Sheaf Laplacian**: connection Laplacian L = delta^T delta; PSD and
  delta^T-delta equality verified on non-orthogonal restriction maps, plus
  spectral analysis, algebraic connectivity, and diffusion convergence
  tested (`groupoid.laplacian`)
- **Federated aggregation pipeline**: point-valued aggregation through
  explicit invertible point actions, with a cycle-basis holonomy-defect
  threshold diagnostic and multi-round convergence tested. The current S^2
  evidence validates explicit SO(3) point actions (`groupoid.aggregation`)
- **Parallel transport**: Schild's ladder and pole ladder
  (`groupoid.transport`) are tangent-vector transport utilities. Pole ladder
  is validated against geomstats' analytic tangent-vector parallel transport
  on S^2 -- it matches in direction (cosine > 0.999) and magnitude. Schild's
  ladder is a coarser first-order approximation and is asserted as such. The
  former `register_transport_from_points` point-valued integration is
  withdrawn because tangent parallel transport does not by itself define the
  invertible point action required by the aggregator. The ambient tangent
  matrix helper is restricted to 1D coordinate-array point representations. See
  [CORRECTION_NOTICE.md](CORRECTION_NOTICE.md) and
  [LIMITATIONS.md](LIMITATIONS.md)
- **Persistent homology**: Vietoris-Rips filtration for divergence
  tracking (`groupoid.persistence`), wired into the pipeline via the
  aggregator's opt-in `track_divergence` flag (per-round H0-vs-H0
  bottleneck distance on the transported parameters, exposed as
  `FederatedRound.divergence`). Unit-tested against point clouds of
  known topology: a circle's dominant 1-cycle (via maximum persistence),
  two-cluster component counting (`betti_0 == 2` at a finite filtration),
  and a translation-invariant bottleneck distance. The persistence diagram
  retains a homology-dimension label, and `track_divergence` compares H0
  against H0 only (it does not pool features across dimensions); this is
  verified against an independent minimum-spanning-tree reconstruction of
  the H0 diagram. The Betti numbers are degenerate under the default
  `thresh=inf` filtration; see [LIMITATIONS.md](LIMITATIONS.md).

## Implemented and validated, not yet integrated

This module is validated against known-correct references but not yet
wired into the main aggregation pipeline:

- **Riemannian optimizers**: SGD and Adam with exponential map
  retraction; the momentum velocity and Adam first moment are
  parallel-transported between iterates (with a projection fallback for
  metrics without parallel transport). Validated: descent to a known
  target on S^2 (geodesic-distance objective) for SGD, momentum SGD, and
  Adam; transported moments preserve norm exactly where projection would
  annihilate them; the curvature-adaptive learning rate is covered for
  both its damping and fallback branches. No general convergence-rate
  guarantees are established (`groupoid.optimizer`)

## Status

Pre-alpha. See [STATUS.md](STATUS.md) for details.

## Related work / why not just use X?

GROUPOID sits at the intersection of three existing toolchains and is not a
replacement for any of them. It is an exploratory prototype of one specific
idea -- transport-groupoid aggregation with a cycle-holonomy consistency
diagnostic -- not a federated learning framework.

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
  groupoid, the cycle-basis holonomy diagnostic, and the cellular-sheaf
  Laplacian wiring -- not the manifold geometry itself.
- **Cellular-sheaf spectral methods** (the sheaf-Laplacian line of work,
  e.g. Hansen and Ghrist's spectral theory of cellular sheaves, and sheaf
  neural networks) -- GROUPOID's sheaf Laplacian follows this line and is the
  geometric machinery for detecting inconsistency across clients. The
  contribution here is applying it to the federated-aggregation setting, not
  the sheaf-Laplacian construction in the abstract.

In short: use Flower/FedML/TFF for the FL system, use geomstats/pymanopt for
manifold math; GROUPOID is a research prototype testing a transport-groupoid
aggregation operator against Euclidean FedAvg. A preregistered synthetic
benchmark ([experiments/](experiments/)) supports the transport benefit under
frame misalignment -- an effect largely built into the synthetic setup -- and
shows that the fixed NetworkX cycle-basis holonomy defect is associated with
aggregation error when the two corruption levels are pooled (Spearman rho =
0.587, permutation p = 1e-4). The within-level correlations are near zero, so
the statistic does not rank runs by error within a fixed corruption level. The
hypothesis remains **unvalidated on real federated learning tasks** (see
[STATUS.md](STATUS.md)).

## Installation

Requires **Python 3.10, 3.11, or 3.12**. Python 3.13+ is not supported: the
`numpy<2.0` / `scipy<1.14` pins (needed for geomstats compatibility, see
[LIMITATIONS.md](LIMITATIONS.md)) have no wheels there, so `pip` will refuse
with a `Requires-Python` message rather than attempt a source build.

From PyPI:

```bash
pip install groupoid
```

From source (for development):

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

# Register SO(3) point actions.
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
print(
    f"cycle-basis defect = {result.cycle_basis_holonomy_defect:.2e} "
    f"(below configured threshold: {result.passes_consistency_threshold})"
)
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

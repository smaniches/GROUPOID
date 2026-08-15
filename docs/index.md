# GROUPOID

> **Pre-alpha research prototype.** Not production software.
> See [STATUS](https://github.com/smaniches/GROUPOID/blob/main/STATUS.md),
> [LIMITATIONS](https://github.com/smaniches/GROUPOID/blob/main/LIMITATIONS.md),
> and [CORRECTION_NOTICE](https://github.com/smaniches/GROUPOID/blob/main/CORRECTION_NOTICE.md).

**Groupoid-based aggregation for federated learning on Riemannian manifolds.**

GROUPOID explores using the mathematical structure of transport groupoids,
cellular sheaves, and Riemannian geometry to aggregate model parameters
across heterogeneous federated clients. The current supported aggregation
path uses explicit invertible point actions and a Karcher mean. Its transport
consistency diagnostic is a cycle-basis holonomy defect, not a canonical H^1
norm; see the correction notice for the exact construct and assumptions.

## Implemented and tested

| Module | Description | Test coverage |
|---|---|---|
| `groupoid.manifold` | Karcher mean via geomstats FrechetMean | Hypothesis (500 examples) |
| `groupoid.groupoid` | Matrix-labelled morphism composition, inverse | Hypothesis (500 examples) |
| `groupoid.cohomology` | Cycle-basis holonomy defect; deprecated `compute_h1` compatibility alias | Coboundary zero, analytic nonzero cases, same-basis reference, basis/gauge regressions, reciprocal-edge invariant |
| `groupoid.sheaf` | Cellular sheaf, restriction maps | Hypothesis (500 examples) |
| `groupoid.laplacian` | Sheaf Laplacian, spectral analysis, diffusion | Unit: PSD + delta^T-delta equality on non-orthogonal maps, transport-consistent kernel; Integration: spectral analysis, diffusion |
| `groupoid.aggregation` | Point-valued aggregation using explicit invertible point actions | Integration: explicit SO(3) forward/return actions and manifold preservation |
| `groupoid.transport` | Schild and pole ladder tangent-vector parallel transport | Unit: pole ladder matches geomstats analytic tangent transport in direction (cosine > 0.999) and magnitude on S^2; ambient matrix helper restricted to vector-shaped points; point-valued `register_transport_from_points` integration withdrawn |
| `groupoid.persistence` | Vietoris-Rips persistent homology; wired into the pipeline via the aggregator's opt-in `track_divergence` flag | Unit: circle 1-cycle via max persistence, two-cluster component count at a finite filtration, translation-invariant bottleneck. Dimension-aware: diagram retains an H0/H1 label and `track_divergence` compares H0-vs-H0 only, verified against an independent MST reconstruction of the H0 diagram and shown not to leak an H1-only change into the H0 divergence. Integration: zero bottleneck on identical rounds, positive on a client jump. Betti degeneracy at thresh=inf documented in LIMITATIONS |

## Implemented and validated, not yet integrated

| Module | Description | Test coverage |
|---|---|---|
| `groupoid.optimizer` | Riemannian SGD, Adam (moments parallel-transported between iterates), curvature-adaptive LR | Unit: SGD, momentum SGD, and Adam descend to a known target on S^2; transported moments preserve norm exactly where projection would annihilate them; curvature-adaptive LR damps/falls back. No general convergence-rate analysis |

## Not yet implemented

- Differential privacy (Opacus, TenSEAL integration)
- Federated training loop with real neural networks
- Communication protocol for distributed deployment
- Convergence guarantees or formal proofs

## Test coverage

The committed suite reaches 100% line and branch coverage of the
`groupoid` package on Python 3.10-3.12, enforced in CI
(`--cov-branch --cov-fail-under=100`). Coverage measures which lines run,
not whether behavior is correct; the tables above record the validation
depth per module, which coverage alone does not capture.

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
   (via inverse point action)
```

## Installation

From PyPI (published as an early development pre-release; the API is
unstable, see [STATUS](https://github.com/smaniches/GROUPOID/blob/main/STATUS.md)):

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
    f"(below threshold: {result.passes_consistency_threshold})"
)
```

## License

Copyright 2026 TOPOLOGICA LLC. Licensed under the Apache License, Version 2.0.

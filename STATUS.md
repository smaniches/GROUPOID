# Status

**Pre-alpha research prototype.**

## What this means

- The mathematical primitives (groupoid composition, Karcher mean, H^1
  cohomology, sheaf Laplacian) are implemented and tested with
  property-based tests (Hypothesis, 500 examples per property).
- The aggregation pipeline connects these primitives into a working
  federated round on synthetic data (tested on S^2 with rotation
  transport maps).
- The transport and optimizer modules have smoke-test coverage
  (parallel transport preserves tangent-vector norm; Riemannian SGD/Adam
  steps stay on the manifold) but are not yet integrated into the main
  pipeline. The persistence module is implemented but untested and
  likewise not integrated.
- No federated training loop with real neural networks exists yet.
- No differential privacy mechanism is implemented.
- No formal convergence analysis or proofs exist.
- The package is not published on PyPI.

## Validation status

| Component | Status | Evidence |
|---|---|---|
| Karcher mean | Tested | Hypothesis: mean of identical points = that point (500 examples) |
| Morphism composition | Tested | Hypothesis: associativity verified (500 examples) |
| H^1 cohomology | Tested | Hypothesis: vanishes on coboundaries (500 examples) |
| Sheaf restriction maps | Tested | Hypothesis: functoriality verified (500 examples) |
| Sheaf Laplacian | Tested | Unit: delta^T-delta equality, PSD, kernel content on non-orthogonal maps; Integration: spectral analysis, diffusion convergence |
| Aggregation pipeline | Tested | Integration: multi-round convergence on S^2, consistency check |
| Parallel transport | Smoke-tested | Smoke: Schild's ladder and pole ladder preserve tangent-vector norm on S^2 |
| Riemannian optimizers | Smoke-tested | Smoke: SGD and Adam steps stay on S^2 (belongs after step) |
| Persistent homology | Implemented | Not tested |
| Differential privacy | Not implemented | Listed as dependency only |
| Real FL training | Not implemented | |
| Convergence proofs | Not available | |

## Versioning

This project uses `0.1.0.dev0` to indicate pre-release development.
The API is unstable and will change without notice.

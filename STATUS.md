# Status

**Pre-alpha research prototype.**

## What this means

- The mathematical primitives (groupoid composition, Karcher mean, H^1
  cohomology, sheaf Laplacian) are implemented and tested with
  property-based tests (Hypothesis, 500 examples per property).
- The aggregation pipeline connects these primitives into a working
  federated round on synthetic data (tested on S^2 with rotation
  transport maps).
- The parallel-transport module is validated against ground truth: the
  pole ladder matches geomstats' analytic parallel transport in direction
  (cosine > 0.999) and magnitude on S^2; Schild's ladder is a coarser
  first-order approximation. It is not yet integrated into the main
  pipeline.
- The persistence module is unit-tested against point clouds of known
  topology (a circle's dominant 1-cycle, two-cluster component counting,
  translation-invariant bottleneck distance). The persistence diagram
  retains a homology-dimension label, and `track_divergence` compares H0
  against H0 only; this is verified against an independent
  minimum-spanning-tree reconstruction of the H0 diagram (the finite H0
  death times equal the MST edge weights), and an H1-only change is shown
  not to leak into the H0 divergence. It is not yet integrated into the
  main pipeline. See the Betti-degeneracy caveat in LIMITATIONS.md.
- The optimizer module has smoke-test coverage only (Riemannian SGD/Adam
  steps stay on the manifold; curvature-adaptive learning rate damps in
  positive curvature and falls back gracefully). Its core descent and
  convergence behavior is not validated, and it is not yet integrated
  into the main pipeline.
- No federated training loop with real neural networks exists yet.
- No differential privacy mechanism is implemented.
- No formal convergence analysis or proofs exist.
- The package is published on PyPI only as an early development pre-release
  (`groupoid 0.1.0.dev2`); no stable release exists yet.

## Validation status

| Component | Status | Evidence |
|---|---|---|
| Karcher mean | Tested | Hypothesis: mean of identical points = that point (500 examples) |
| Morphism composition | Tested | Hypothesis: associativity verified (500 examples) |
| H^1 cohomology | Tested | Hypothesis: vanishes on coboundaries (500 examples); Unit: identity holonomy on a complete coboundary, a nonzero H^1 matched against a closed-form analytic value (2*sqrt(1-cos(angle sum)) for commuting same-axis rotations), agreement with an independent holonomy-product recomputation on a two-triangle multi-cycle graph, and an incomplete cocycle (missing edge map) raises IncompleteCocycleError naming the edge |
| Sheaf restriction maps | Tested | Hypothesis: functoriality verified (500 examples) |
| Sheaf Laplacian | Tested | Unit: delta^T-delta equality, PSD, kernel content on non-orthogonal maps; Integration: spectral analysis, diffusion convergence |
| Aggregation pipeline | Tested | Integration: multi-round convergence on S^2, consistency check |
| Parallel transport | Tested (not integrated) | Unit: pole ladder matches geomstats analytic parallel transport in direction (cosine > 0.999) and magnitude on S^2; Schild's ladder asserted as a coarser approximation; transport-matrix constructor is norm-preserving |
| Riemannian optimizers | Smoke-tested | Smoke: SGD and Adam steps stay on S^2; curvature-adaptive LR damps in positive curvature and falls back without curvature. Core descent/convergence not validated |
| Persistent homology | Tested (not integrated) | Unit: circle's dominant 1-cycle via max persistence; two-cluster component count (betti_0 = 2) at a finite filtration; translation-invariant bottleneck distance. Dimension-aware: diagram retains an H0/H1 label, `track_divergence` compares H0-vs-H0 only, verified against an independent MST reconstruction of the H0 diagram and shown to not leak an H1-only change into the H0 divergence. Betti degeneracy at thresh=inf documented in LIMITATIONS.md |
| Differential privacy | Not implemented | Listed as dependency only |
| Real FL training | Not implemented | |
| Convergence proofs | Not available | |

"Smoke-tested" means the tests exercise the code and check coarse sanity
properties (e.g. an optimizer step stays on the manifold) but do not
validate correctness against ground truth. "Tested" means the tests check
behavior against a known-correct reference or analytic result.

## Test coverage

The committed test suite reaches 100% line and branch coverage of the
`groupoid` package on the supported interpreters (Python 3.10-3.12),
enforced in CI with `--cov-branch --cov-fail-under=100`. The only excluded
lines are two provably-unreachable defensive guards in `aggregation.py` and
a `TYPE_CHECKING`-only import in `manifold.py`, each marked with
`# pragma: no cover` and a one-line justification. Coverage measures which
lines run, not whether behavior is correct; the validation-status table
above records correctness depth per component, which coverage alone does
not capture.

## Versioning

This project uses `0.1.0.dev2` to indicate pre-release development.
The API is unstable and will change without notice.

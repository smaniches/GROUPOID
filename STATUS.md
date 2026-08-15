# Status

**Pre-alpha research prototype.**

## What this means

- The mathematical primitives (groupoid composition, Karcher mean, the
  cycle-basis holonomy defect, and the sheaf Laplacian) are implemented and
  tested. The historical `compute_h1` name is retained only as a deprecated
  compatibility alias; the scalar is not a canonical H^1 norm.
- The aggregation pipeline connects these primitives into a working
  federated round on synthetic data (tested on S^2 with rotation
  transport maps).
- The parallel-transport module is validated against ground truth at the
  tangent-vector level: the pole ladder matches geomstats' analytic parallel
  transport in direction (cosine > 0.999) and magnitude on S^2; Schild's
  ladder is a coarser approximation. The former
  `register_transport_from_points` point-valued integration is withdrawn:
  tangent parallel transport does not by itself define the invertible point
  action required by the aggregation pipeline.
- The persistence module is unit-tested against point clouds of known
  topology (a circle's dominant 1-cycle, two-cluster component counting,
  translation-invariant bottleneck distance). The persistence diagram
  retains a homology-dimension label, and `track_divergence` compares H0
  against H0 only; this is verified against an independent
  minimum-spanning-tree reconstruction of the H0 diagram (the finite H0
  death times equal the MST edge weights), and an H1-only change is shown
  not to leak into the H0 divergence. It is wired into the aggregation
  pipeline via the aggregator's opt-in `track_divergence` flag. See the
  Betti-degeneracy caveat in LIMITATIONS.md.
- The optimizer module is validated against known-correct references:
  Riemannian SGD (with and without momentum) and Adam descend to a known
  target on S^2 (geodesic-distance objective), and the momentum/first-
  moment accumulators are parallel-transported between iterates with
  exact norm preservation (tangent projection is the fallback for
  metrics without parallel transport). No general convergence-rate
  analysis exists, and it is not yet integrated into the main pipeline.
- A preregistered synthetic benchmark (`experiments/`) tests the central
  hypothesis: the transport benefit under frame misalignment is supported
  in all preregistered cells (an effect largely built into the synthetic
  setup), the ablation shows transport -- not the intrinsic mean --
  accounts for essentially all of it, and the fixed NetworkX cycle-basis
  holonomy defect is positively associated with aggregation error across the
  two corruption levels (Spearman rho = 0.587, permutation p = 1e-4) while
  not ranking runs within a level. See
  `experiments/RESULTS.md`, including the failed null check and the
  documented deviation. The hypothesis remains unvalidated on real
  federated learning tasks.
- No federated training loop with real neural networks exists yet.
- No differential privacy mechanism is implemented.
- No formal convergence analysis or proofs exist.
- The package is distributed only as early development pre-releases; no stable
  release exists yet.

## Validation status

| Component | Status | Evidence |
|---|---|---|
| Karcher mean | Tested | Hypothesis: mean of identical points = that point (500 examples) |
| Morphism composition | Tested | Hypothesis: associativity verified (500 examples) |
| Cycle-basis holonomy defect | Tested with scope qualification | Exact zero on complete coboundary transports; closed-form nonzero holonomy checks; same-basis numerical recomputation; basis/gauge regressions; incomplete basis cycles and nonreciprocal opposite registrations fail closed |
| Sheaf restriction maps | Tested | Hypothesis: functoriality verified (500 examples) |
| Sheaf Laplacian | Tested | Unit: delta^T-delta equality, PSD, kernel content on non-orthogonal maps; Integration: spectral analysis, diffusion convergence |
| Aggregation pipeline | Tested on supported point actions | Integration with explicit SO(3)/identity point actions on S^2; forward/return actions remain on the manifold; explicitly registered opposite arrows must be reciprocal. Arbitrary square matrices are not claimed as generic geometric transports |
| Parallel transport | Tangent-vector transport tested; point-valued integration withdrawn | Unit: pole ladder matches geomstats analytic tangent transport in direction (cosine > 0.999) and magnitude on S^2; Schild's ladder is coarser. The ambient matrix helper supports only vector-shaped points and is not a generic point action. `register_transport_from_points` fails closed |
| Riemannian optimizers | Tested (not integrated) | Unit: SGD, momentum SGD, and Adam descend to a known target on S^2 (final geodesic distance < 1e-6 / 1e-3 from a 60-degree start); moment accumulators parallel-transported between iterates with exact norm preservation where projection would annihilate them (both fallback branches covered); curvature-adaptive LR damps in positive curvature and falls back without curvature. No general convergence-rate analysis |
| Persistent homology | Tested, integrated | Unit: circle's dominant 1-cycle via max persistence; two-cluster component count (betti_0 = 2) at a finite filtration; translation-invariant bottleneck distance. Dimension-aware: diagram retains an H0/H1 label, `track_divergence` compares H0-vs-H0 only, verified against an independent MST reconstruction of the H0 diagram and shown to not leak an H1-only change into the H0 divergence. Integration: opt-in `track_divergence` flag on the aggregator (zero bottleneck on identical rounds, positive on a client jump). Betti degeneracy at thresh=inf documented in LIMITATIONS.md |
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

This project uses `0.1.0.dev5` to indicate pre-release development.
The API is unstable and will change without notice.

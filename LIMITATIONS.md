# Limitations

## Scope

GROUPOID is a research prototype, not a production federated learning
system. The following limitations apply.

## Not a federated learning framework

- No communication protocol between clients and server.
- No support for distributed training across machines.
- No data partitioning or client simulation beyond test fixtures.
- The aggregation pipeline operates on numpy arrays, not neural
  network parameters in a training loop.

## Not privacy-preserving

- No differential privacy mechanism is implemented.
- Opacus and TenSEAL are listed as optional dependencies for future
  work but are not integrated into any code path.
- No privacy guarantees of any kind are provided.

## Mathematical limitations

- The Karcher mean computation delegates to geomstats FrechetMean.
  Convergence depends on the manifold and point distribution.
- H^1 computation uses cycle basis of the undirected graph. This is
  correct for 1-dimensional nerve complexes but does not generalize
  to higher-dimensional simplicial complexes without modification.
- H^1 requires a fully specified cocycle: every edge of every basis
  cycle must have a transport map in one direction or the other (the
  reverse direction is inverted). A cycle's holonomy is the ordered
  product over all of its edges, so if any edge map is missing the
  holonomy is undefined. `compute_h1` raises `IncompleteCocycleError`
  naming the missing edge rather than forming a meaningless partial
  product (which would otherwise be reported as a false (in)consistency).
- Sheaf Laplacian construction assumes uniform stalk dimension.
- Parallel transport approximations (Schild's ladder, pole ladder)
  are discrete approximations. The pole ladder matches geomstats'
  analytic parallel transport closely in direction (cosine > 0.999 on a
  60-degree S^2 hop) but does not converge to zero error as rungs
  increase; it plateaus at a small residual (~0.02 here) and drifts
  slightly off the endpoint tangent plane. Schild's ladder is markedly
  coarser (cosine ~0.98 on the same hop).
- Persistent homology Betti numbers are degenerate under the default
  filtration. `compute_persistence` counts only bars that die at
  infinity, so under the default `thresh=inf` (used by
  `track_divergence`) the Vietoris-Rips complex is fully connected and
  `betti_0 == 1`, `betti_1 == 0` regardless of the underlying topology.
  The informative loop signal lives in `max_persistence` / the finite
  bars; meaningful component counts require passing a finite
  `max_edge_length` between the intra- and inter-cluster scales.
- No formal convergence rate analysis exists for the groupoid
  aggregation method.

## Dependency constraints

- Requires `numpy < 2.0` due to geomstats compatibility.
- Requires `scipy < 1.14` for the same reason.
- The `ripser` and `persim` packages may have build issues on
  some platforms (C++ compilation required).

## Test coverage

- All 9 modules have test coverage. The committed suite reaches 100% line
  and branch coverage of the `groupoid` package on Python 3.10-3.12,
  enforced in CI (`--cov-branch --cov-fail-under=100`). Two unreachable
  defensive guards and a `TYPE_CHECKING` import are excluded via
  `# pragma: no cover` with justifications.
- Coverage is not validation. The transport and persistence modules are
  now validated against ground truth (analytic parallel transport;
  known-topology point clouds) but remain unintegrated into the pipeline.
  The optimizer module has smoke-test coverage only: its steps stay on
  the manifold and the curvature-adaptive learning rate behaves sensibly,
  but its core descent/convergence behavior is not validated.
- No end-to-end test with real neural network training exists.
- Property-based tests use 500 examples per property, which provides
  reasonable but not exhaustive coverage of edge cases.

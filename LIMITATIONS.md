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
- Sheaf Laplacian construction assumes uniform stalk dimension.
- Parallel transport approximations (Schild's ladder, pole ladder)
  are first-order accurate per rung. Accuracy depends on step count
  and geodesic distance.
- No formal convergence rate analysis exists for the groupoid
  aggregation method.

## Dependency constraints

- Requires `numpy < 2.0` due to geomstats compatibility.
- Requires `scipy < 1.14` for the same reason.
- The `ripser` and `persim` packages may have build issues on
  some platforms (C++ compilation required).

## Test coverage

- 8 of 9 modules have test coverage.
- The transport and optimizer modules have smoke-test coverage
  (norm-preserving transport; optimizer steps stay on the manifold).
- The persistence module is untested.
- No end-to-end test with real neural network training exists.
- Property-based tests use 500 examples per property, which provides
  reasonable but not exhaustive coverage of edge cases.

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
- The quantity formerly described as an `H^1` norm is the maximum
  Frobenius deviation from identity over the particular cycle basis returned by
  `networkx.cycle_basis` on the undirected graph. Its exact zero set can detect
  flat transport under connectedness, complete invertible underlying-edge
  transports, reciprocal opposite registrations when both directions are
  supplied, and the spanning-tree fundamental-cycle construction used by
  NetworkX. Its
  nonzero magnitude is basis-dependent and is not invariant under arbitrary
  invertible changes of frame. A finite `consistency_threshold` is therefore a
  representation-dependent diagnostic, not a canonical cohomological verdict.
- The cycle-basis defect checks only edges that lie on selected cycles. A bridge
  lies on no cycle, so a zero defect does not certify that every graph edge has
  a registered transport or that the graph is connected.
- Basis-cycle holonomy must be fully specified: every edge of every selected
  cycle needs a transport map in one direction or the other. A missing
  basis-cycle edge raises `IncompleteCocycleError` rather than forming a partial
  product. If both directions of one underlying edge are supplied, they must be
  mutual numerical inverses; a conflicting pair raises
  `NonReciprocalTransportError` before the directed data are reduced to an
  undirected cycle basis.
- `register_transport_from_points` is withdrawn from supported point-valued
  aggregation. The ladder routines transport tangent vectors. The ambient
  square operator assembled from projected tangent basis vectors is not
  generally an invertible action on manifold points; in the exact embedded
  geometry it can be rank-deficient. `compute_tangent_transport_matrix` is also
  restricted to vector-shaped point representations (1D coordinate arrays of
  equal shape); structured points must use the ladder functions directly. The
  compatibility method now fails closed.
- Direct `register_transport` matrices are supported only when they define the
  invertible point action required by the chosen manifold representation. The
  pipeline requires a finite square matrix with a finite numerical inverse at
  registration. If both directions of an edge are registered explicitly, they
  must be mutual numerical inverses. The pipeline also checks the actual forward
  and return images for manifold membership during aggregation.
  This does not prove that an arbitrary matrix preserves the entire manifold.
  The current S^2 evidence validates explicit SO(3) rotations.
- `FederatedRound.transport_residuals` is a compatibility field whose values are
  specifically `||T T^T - I||_F` for the composite forward maps. The primary
  semantic name is `orthogonality_residuals`; this quantity is not a generic
  transport-error metric.
- Sheaf Laplacian construction assumes uniform stalk dimension.
- Parallel transport approximations (Schild's ladder, pole ladder)
  are discrete approximations. The pole ladder matches geomstats'
  analytic tangent-vector parallel transport closely in direction (cosine >
  0.999 on a 60-degree S^2 hop) but does not converge to zero error as rungs
  increase; it plateaus at a small residual (~0.02 here) and drifts slightly
  off the endpoint tangent plane. Schild's ladder is markedly coarser (cosine
  ~0.98 on the same hop).
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
- `torch`, `pymanopt`, `POT`, and `einops` are declared as required
  runtime dependencies but are not imported by any current code path (the full
  test suite passes with all four uninstalled). Installing `groupoid` therefore
  pulls in packages, most notably `torch`, that the current code does not use.
  The `torch` advisory tracked in SECURITY.md applies to the installed
  dependency, not to any code path GROUPOID exercises.

## Test coverage

- Releases are gated on the repository's hard line and branch coverage checks
  on Python 3.10-3.12. Coverage is not validation: the scientific correction
  adds explicit tests for basis dependence, general non-orthogonal gauge
  dependence of the defect magnitude, the tangent/point representation
  boundary, and the supported SO(3) point-action round trip.
- The transport and persistence modules retain their ground-truth validations
  at the scopes stated above. The optimizer module remains validated for descent
  to a known target on S^2 and has no general convergence-rate analysis.
- No end-to-end test with real neural network training exists.
- Property-based tests use 500 examples per property, which provides
  reasonable but not exhaustive coverage of edge cases.

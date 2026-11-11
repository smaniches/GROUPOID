# Changelog

All notable changes to GROUPOID are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(with PEP 440 identifiers for Python pre-releases).

## [Unreleased]

## [0.1.0.dev10] - 2026-11-11

Pre-alpha corrective prerelease. It distributes one already-merged fix to
transport registration (#95), corrects the Karcher-mean default tolerance, and
corrects how the benchmark's preregistration, its H2 result and the SciPy
requirement are described. Stored benchmark results are unchanged; the only
edit to a historical experiment file is an explicit tolerance argument in
`experiments/run_benchmark.py` that leaves its output unchanged. See
`CORRECTION_NOTICE.md`, F03 to F06.

### Changed

- **Karcher mean default tolerance is `1e-12` (was `1e-6`):** `karcher_mean`
  passes `tol` to geomstats, whose gradient criterion stops the iteration once
  the squared Riemannian gradient norm of the Frechet function is at most
  `tol * dim`. At such a stop the old default allowed a gradient norm of about
  `1.41e-3` on S^2, and it left errors of up to `1.4e-3` rad on the benchmark's
  oracle problems; with the new default the criterion corresponds to a gradient
  norm of about `1.41e-6` (errors up to `1.7e-6` rad on all benchmark Karcher
  problems, at most 13 iterations, about 1.3 to 1.6 times the time per call).
  The stopping contract, including the near-zero-variance and iteration-limit
  exits that do not establish this bound, is documented in the docstring, and
  `TransportGroupoidAggregator` has a new `karcher_tol` field, with the same
  default, that it forwards to `karcher_mean`. Results differ from
  v0.1.0.dev9 by up to the old solver error; pass `tol=1e-6` or
  `karcher_tol=1e-6` to reproduce earlier outputs.

### Fixed

- **Transport registration stores its own copy of the matrix (#95):**
  `TransportGroupoidAggregator.register_transport` kept a reference to the
  caller's array when it was already a float64 NumPy array (or a view of
  one). Changing that array after registration changed the registered point
  action without repeating the finiteness, invertibility and reciprocity
  checks, and a later reverse registration was checked against the changed
  matrix. Registration now copies the matrix before validating it and stores
  the copy. Every earlier release stored the caller's array; the checks that
  such a change bypassed exist from v0.1.0.dev5. Code that changed an array
  after registering it, expecting the registered transport to follow, must
  register the new matrix instead.

### Documentation and scientific record

- **Benchmark solver accuracy (F03):** stored benchmark values of about
  `1e-3` rad or less include Karcher solver error of their own order. A
  sensitivity rerun of the unmodified runner and analysis with `tol=1e-12`
  leaves every verdict unchanged; at `sigma = 0.05` the extrinsic-versus-Karcher
  gaps are two to three and a half times smaller than stored.
  `experiments/run_benchmark.py` now passes the historical tolerance explicitly
  so that it keeps reproducing `results.json`.
- **Preregistration provenance (F04):** the analysis plan was committed before
  the runner and results within pull request #49 (`ea7cf5f`, `43fcad5`,
  `ff7f461`); it was not registered externally, and the push order is the
  author's account. README, STATUS, LIMITATIONS and the experiments index now
  say so; `PREREGISTRATION.md` is unchanged.
- **H2 interpretation (F05):** the pooled criterion was met as stored
  (rho = 0.587, p = 1e-4), but within each corruption level the association is
  absent (rho = -0.030 and 0.004; partial rho given the level -0.033), so H2 is
  not evidence of within-level prediction.
- **SciPy requirement (F06):** README, REPRODUCE and LIMITATIONS now state the
  `pyproject.toml` requirement `scipy>=1.10,<1.18` instead of `scipy<1.16`,
  attribute the Python 3.13 exclusion to `numpy<2.0`, and record the
  historical SciPy versions (1.13.1 for the REPRODUCE receipt, 1.15.3 for the
  benchmark).

### Validation and scope

- New regression tests: Karcher accuracy against a symmetric ring's exact mean
  and an independent numpy minimizer, and aggregator forwarding of
  `karcher_tol` (these fail with the old default); the gradient-criterion
  bound at several tolerances on test inputs; all 600
  stored benchmark runs re-executed against `results.json` (fails without the
  historical tolerance in the runner); the stored H2 statistics and the
  within-level and partial associations recomputed from `results.json`; and
  documented NumPy/SciPy upper bounds checked against `pyproject.toml` (failed
  on the previous documents).
- The regression test added in #95 fails on the v0.1.0.dev9 source and on
  the published v0.1.0.dev9 wheel, and passes on this release.
- The archived neural experiment replay tests pass unchanged. Python 3.11 and
  3.12 support and the pre-alpha constraints are unchanged.

## [0.1.0.dev9] - 2026-11-04

Pre-alpha corrective prerelease distributing already-merged numerical and
transactional fixes. The release is **not** evidence of a federated-learning
performance improvement or a complete training system.

### Fixed

- **Riemannian Adam second moment (#84):** use the manifold metric's
  squared gradient norm rather than ambient Euclidean `sum(grad**2)`.
  This matters for non-Euclidean metrics; matching closed-form and
  coordinate-equivariance checks are included. The caller must supply a
  true Riemannian gradient, not an arbitrary ambient covector.
- **Explicit-Euler sheaf diffusion stability (#85):** reject steps violating
  a sufficient spectral-norm stability bound; a previous legal input
  amplified disagreement energy. This conservative check may reject some
  otherwise stable steps. It is not a general convergence theorem.
- **Federated-round state integrity:** commit state only after a successful
  aggregation rather than exposing partial state on failure. Transaction
  regression tests are included.

### Validation and scope

- Includes a runnable five-client SO(3) aggregation illustration with an
  analytic reference, recorded standalone six-seed hidden-unit permutation
  experiments, and corresponding replay/invariant tests.
- The illustration uses constructed data. The archived neural experiment
  is not an end-to-end training experiment through the package's
  aggregator, does not infer unknown alignment, and does not demonstrate
  superiority to standard federated-learning systems.
- No historical experimental result or previous immutable release is
  modified. Python 3.11–3.12 and pre-alpha constraints remain unchanged.


## [0.1.0.dev8] - 2026-10-30

### Changed

- **Retired end-of-life Python 3.10 support.** Python 3.11 and 3.12 are the
  supported and tested interpreter versions; `Requires-Python`, package
  classifiers, installation instructions, test matrix, and citation metadata
  now agree. Python 3.10 reached upstream end of life on October 1, 2026.
  Existing users of Python 3.10 should migrate to 3.11 or 3.12 before
  upgrading this package. Python 3.13+ remains excluded for the separately
  documented numerical dependency constraints.
- This release contains no algorithm or scientific-method changes; historical
  experimental artifacts and scientific qualification statements are unchanged.

## [0.1.0.dev7] - 2026-10-29

Pre-alpha corrective release distributing the October 29 fixes already merged
on `main` (#77, #78 and #81). No new experiments, measured federated-learning
performance gains, or general convergence claims are introduced. Archived
preregistration and historical benchmark outputs remain unchanged.

### Fixed

- **Finite H0 bottleneck versus an empty diagram (#77).** The previous
  `track_divergence` fallback returned `0.0` when either finite H0 diagram
  was empty. The mathematically correct cost is half the largest unmatched
  persistence interval. For a two-point cloud separated by 4 versus a
  singleton, the result is now **2.0** in either direction, not zero;
  empty-versus-empty remains zero. Regression tests cover both directions.
  The metric remains an H0-only geometric diagnostic, **not** a certificate
  of federated model convergence or predictive stability.

### Changed

- **Removed an unused mandatory dependency (#81).** `torch` was declared as
  required but is not imported by the implemented library. New installations
  no longer require PyTorch or its dependency graph. The security CI installs
  the actual supported runtime closure, requires patched `setuptools`, checks
  `pip check`, and runs `pip-audit` without advisory ignores. The full
  Python 3.10–3.12 tests, docs, and code-quality checks passed on `main`.
  **Compatibility note:** PyTorch is no longer installed automatically;
  callers requiring it for their own workflows must install it separately.
  Upgrading an existing environment does not uninstall previously installed
  PyTorch or other packages; independently audit those environments.
- Updated SHA-pinned `actions/setup-python` and PyPI publishing action (#78).
  No algorithmic behavior was changed by this CI maintenance.

## [0.1.0.dev6]

Pre-alpha development release distributing the transport and validation fixes
merged after 0.1.0.dev5 (#68). The research maturity is unchanged (pre-alpha);
this is a prerelease. The preregistration and stored benchmark results are not
modified: the benchmark uses explicit transport maps, not the ladders, and a
rerun of all 600 runs under this candidate reproduced the stored per-run values
exactly.

**Who is affected.** In 0.1.0.dev5 and earlier, `pole_ladder` with
`n_rungs > 1`, every `schild_ladder` call, and `compute_tangent_transport_matrix`
(default `method="pole"`, `n_rungs=2`) can return incorrect tangent-vector
transport. On the tested S^2 hops the released dev5 two-rung result deviated
from closed-form Levi-Civita transport by up to 0.26 for vectors of norm 0.5.
Single-rung pole-ladder output is unchanged. A corrected one-rung Schild result
is not promised to be closer to the analytic answer than the previous
(accidentally exact, pole-ladder) single-rung output; the correction is to the
construction, and its error decreases as O(1 / n_rungs).

**Behavior changes.** Unknown transport `method`, `n_rungs < 1`, negative,
non-finite or zero-sum aggregation weights, and NaN/Inf registered holonomy maps
now raise `ValueError` instead of returning a value.

### Fixed

- **`pole_ladder` step schedule.** Each rung advanced by `log(end, base) / n_rungs`
  taken once in the *initial* tangent space, so for `n_rungs > 1` the ladder
  walked a geodesic that did not end at `end_point` and the result drifted off
  the endpoint tangent plane on multi-rung calls. The single-rung
  calculation was already exact on the tested S^2 configuration. The step is now `log(end, current_base) / (n_rungs - k)`, re-expressed
  at every rung. On S^2 the pole ladder now matches geomstats' analytic parallel
  transport to round-off (< 2e-15) for every rung count; `n_rungs=1` output is
  unchanged. Pinned by `tests/test_audit_regressions.py`.
- **`schild_ladder` was not Schild's ladder.** Its body was the pole-ladder
  reflection with a differently broken step schedule (the direction was
  recomputed on each rung but still divided by the original rung count), which
  is why it appeared not to converge. It now implements the geodesic-parallelogram construction (Kheyfets,
  Miller & Newton 2000) with the tangent vector rescaled by `1 / n_rungs`, the
  standard device that makes the first-order scheme converge (Guigui & Pennec
  2021). Measured on the S^2 test hop: error halves with every doubling of rungs
  (0.214 at 1 rung → 0.00327 at 64 on the 30-degree test hop), endpoint tangency to round-off, norm
  preserved to 1e-6. Outputs can change even at `n_rungs = 1` for
  vectors with components off the hop plane; the former single-rung function
  was pole ladder, not Schild's ladder. `pole_ladder` remains the recommended
  method. Both ladders now reject `n_rungs < 1` with `ValueError`.
- `compute_tangent_transport_matrix` raises `ValueError` for an unknown `method`
  instead of silently falling back to Schild's ladder.
- `cycle_basis_holonomy_defect` raises `ValueError` when a registered transport
  map contains NaN or infinity instead of propagating a non-finite defect.
- `TransportGroupoidAggregator.aggregate` validates caller-supplied weights
  (non-finite, negative, or zero-sum weights raise `ValueError`).
- `compute_persistence` silences only ripser's spurious "input matrix is
  square" `UserWarning` (emitted when `n_points == n_features` for a point
  cloud that is not a distance matrix); previously the warning leaked to every
  caller with a square point cloud. No other warning is suppressed.

### Changed

- `pyproject.toml` comment corrected to the actual `scipy<1.16` pin (it said
  `<1.14`); the coverage-exclusion comment now also names the `TYPE_CHECKING`
  import in `manifold.py`. `LIMITATIONS.md`, `PROBLEMS.md`, `STATUS.md`, and the README
  now record the measured ladder behaviour above instead of the plateau
  narrative that described the bug. `PROBLEMS.md` gains item 9.
- `tests/test_transport_validation.py` tightens the pole-ladder tangency bound
  from 0.03 to 1e-12 and compares the ladders on a generic off-plane vector
  (for a vector in the hop plane both ladders are exact on S^2 by symmetry).


## [0.1.0.dev5]

Scientific correction release. See `CORRECTION_NOTICE.md` for the evidence and
scope. Historical preregistration and result artifacts are preserved unchanged.

### Fixed
- **The former `H^1` scalar is now identified by the quantity actually
  computed.** `cycle_basis_holonomy_defect` returns
  `max_{gamma in B} ||Hol(gamma) - I||_F` over the NetworkX cycle basis. The
  historical `compute_h1` name remains as a deprecated compatibility alias.
  The exact zero set has a flatness interpretation under explicit connected,
  complete, invertible edge-transport assumptions with every cycle emitted by
  `networkx.cycle_basis` evaluated -- an argument specific to the emission
  order of the NetworkX Paton implementation exercised here, not to arbitrary
  cycle bases -- but the nonzero magnitude is cycle-basis-dependent and is not
  invariant under arbitrary invertible changes of frame. Finite threshold
  decisions are therefore representation-dependent diagnostics, not canonical
  cohomological verdicts.
- **Tangent-vector transport is no longer promoted into an unsupported
  point-valued morphism.** `compute_tangent_transport_matrix` now names the
  ambient-coordinate tangent operator explicitly. The old
  `compute_transport_matrix` name remains as a deprecated alias.
  `register_transport_from_points` is retained only as a compatibility stub
  and fails closed because tangent parallel transport does not by itself define
  the invertible point action required by the current aggregator.
- **Direct transport registration now states and checks the point-action
  contract exercised by aggregation.** Registered matrices must be finite,
  square and invertible, and actual forward and return images must remain on
  the configured manifold. This validates exercised point actions, not
  arbitrary matrices as generic geometric transports. The supported S^2
  evidence continues to use explicit SO(3) rotations.
- **Benchmark H2 interpretation corrected without changing its historical
  numerical result.** The committed rho=0.587 and permutation p=0.00010 are
  retained as the association between aggregation error and the fixed
  cycle-basis holonomy-defect statistic across pooled corruption levels; the
  near-zero within-level correlations remain an explicit caveat.
- **Preregistration provenance language calibrated.** Current explanatory
  documentation distinguishes Git commit-order evidence from the separate
  project-record assertion that the preregistration was pushed before local
  benchmark execution. The locked preregistration is unchanged.
- **Release automation now explicitly dispatches the release workflow after
  creating a tag.** The initial 0.1.0.dev4 auto-tag path used the workflow
  `GITHUB_TOKEN`; the resulting tag push did not trigger the tag-push release
  workflow. The 0.1.0.dev4 GitHub Release was subsequently produced after a
  manual workflow run. The current auto-tag workflow explicitly dispatches
  `release.yml` so future automated release cuts do not rely on a suppressed
  bot-authored push event.

## [0.1.0.dev4] - 2026-07-06

`v0.1.0.dev3` was cut in source control (below) but never tagged or
released: no `v0.1.0.dev3` tag was ever pushed, so PyPI, the GitHub
Release, and the Zenodo deposit all skip directly from `0.1.0.dev2` to
this version, which carries both the `0.1.0.dev3` changes and the ones
below. See the "Releasing" section of `CONTRIBUTING.md` for the
automation added in this version that prevents this gap from recurring.

### Fixed
- **Optimizer moment accumulators are now parallel-transported between
  iterates**: Adam's first moment was previously carried across steps by
  tangent projection (which can annihilate a geodesic-aligned moment) and
  the SGD momentum velocity was not moved at all (leaving it non-tangent
  at the new iterate). Both now use the metric's parallel transport (the
  Becigneul-Ganea construction), with projection kept as the fallback for
  metrics without parallel transport. Regression tests discriminate
  transport from projection by exact norm preservation (#49).

### Added
- `TransportGroupoidAggregator.register_transport_from_points`: computes
  and registers the transport matrix from two client base points via the
  pole or Schild ladder, wiring `groupoid.transport` into the pipeline
  (#49).
- Opt-in `track_divergence` flag on the aggregator: each round computes
  the persistent-homology summary of the transported parameters with the
  H0-vs-H0 bottleneck distance to the previous round, exposed as
  `FederatedRound.divergence` (default off; existing behavior unchanged),
  wiring `groupoid.persistence` into the pipeline (#49).
- Descent validation for the Riemannian optimizers: SGD, momentum SGD,
  and Adam descend to a known target on S^2; the optimizer status labels
  in STATUS.md and the docs are upgraded accordingly (general
  convergence-rate analysis still does not exist and is not claimed)
  (#49).
- A preregistered synthetic benchmark (`experiments/`): preregistration
  pushed before execution, 600 seeded runs, 2x2 transport/mean ablation
  against an oracle estimand, corrupted-cocycle conditions, and a
  committed preregistered analysis (`experiments/RESULTS.md`). Supports
  the transport benefit under frame misalignment (largely by
  construction) and the pooled H^1-error correlation under corruption,
  with the within-level caveat, a failed null check reported as failed,
  and one documented seeding deviation (#49).
- Release automation: pushing a version bump to `main` now tags and
  releases automatically, removing the manual tag-push step.

## [0.1.0.dev3] - 2026-07-06

Cut in source control but never tagged or released; see `0.1.0.dev4` above.

### Fixed
- **`Morphism` `str()`/`format()` output**: pydantic's `BaseModel` defines
  `__str__` separately from `__repr__`, so `print()`, f-strings, and loguru's
  `{}` formatting dumped the full transport matrix instead of the compact
  `Morphism(A -> B)` form the `__repr__` override implements. `__str__` now
  aliases `__repr__`, with a regression test (#47).

### Changed
- Docs-site installation pages corrected to state the package is published on
  PyPI as a development pre-release, aligning `docs/index.md` and
  `docs/quickstart.md` with the README and `STATUS.md` corrections from #37
  and #39 (#47).
- `curvature_adaptive_lr` docstring corrected to describe the implemented
  behavior: the rate is damped in positive curvature and returned unchanged
  otherwise; no enlargement is applied in negative curvature (#47).
- `LIMITATIONS.md` now records that `torch`, `pymanopt`, `POT`, and `einops`
  are declared runtime dependencies with no current code path (the suite
  passes with all four uninstalled) (#47).

## [0.1.0.dev2] - 2026-06-21

### Fixed
- **RiemannianAdam first-moment bias initialization**: the first moment was
  seeded as `grad` while the zero-init bias correction `m / (1 - beta1**t)` was
  still applied, leaving `1/(1 - beta1)` uncancelled and inflating the first
  optimization step ~10x (beta1 = 0.9). Now seeded as `(1 - beta1) * grad` so the
  bias correction recovers `grad`, matching standard Adam. The optimizer is not
  integrated into the aggregation pipeline, so no published result changes (#39).

### Changed
- `STATUS.md` corrected to state the package is published on PyPI as a development
  pre-release (`groupoid 0.1.0.dev2`) rather than "not published on PyPI" (#39).

## [0.1.0.dev1] - 2026-06-16

First tag cut on mathematically corrected code. The earlier `v0.1.0.dev0` tag and
its Zenodo version DOI (`10.5281/zenodo.20563975`) predate the correctness fixes
below and are **superseded**; that snapshot must not be cited for results. The
concept DOI (`10.5281/zenodo.20563974`) resolves to this version.

### Fixed
- **Sheaf (connection) Laplacian** corrected to a positive-semidefinite operator
  `L = delta^T delta` with a transport-consistent kernel, verified against an
  independently constructed coboundary (#13, closes #8).
- **Persistence diagrams** now retain the homology-dimension label, so H0 and H1
  bars are no longer conflated; `track_divergence` compares H0 against H0 only,
  verified against an independent scipy minimum-spanning-tree reconstruction
  (#21, closes #10).
- **Cohomology**: `compute_h1` raises `IncompleteCocycleError` naming the missing
  edge instead of forming a meaningless partial holonomy, and disconnected-graph
  transport is re-raised as an explicit domain error (#20).
- Benchmark CI regression check soft-skips on a missing JSON baseline instead of
  failing the job (#15).

### Added
- 100% line + branch test coverage of the `groupoid` package on Python 3.10-3.12,
  enforced in CI via `--cov-branch --cov-fail-under=100` (#18).
- Cold-clone reproducibility: a seeded determinism receipt and reproduction guide
  (`REPRODUCE.md`), install cap, stub clarity, and a real CI gate (#16).
- Aggregation tests covering forward-edge traversal, weighted mean, and the
  inconsistency flag (#17).
- Zenodo concept-DOI badge and `CITATION.cff` DOI metadata (#11).
- A `Documentation` project URL and keywords aligned across `pyproject.toml`,
  `codemeta.json`, and `CITATION.cff` (#19).
- A consolidated `PROBLEMS.md` self-audit recording corrected bugs, known
  limitations, and deferred work (#23).
- A `CHANGELOG.md` linked from package metadata (#24).
- A `.zenodo.json` so archival deposits carry curated metadata, a supersession
  note for `v0.1.0.dev0`, and version-chain links to the concept and prior DOIs.
- Dependabot configuration for weekly `pip` and `github-actions` updates (#25).

### Changed
- Numpy array annotations parameterized to `NDArray[np.float64]`, with
  `mypy --strict` enforced as a hard, gating CI step (#26).

### Security
- Replaced the deprecated soft `safety check` with a hard, gating `pip-audit`
  step; the fix-less torch advisory CVE-2025-3000 is documented and explicitly
  ignored with its rationale in `SECURITY.md` (#22).

## [0.1.0.dev0] - 2026-06-05

Initial pre-alpha research prototype. Core implementation of groupoid-based
aggregation for federated learning on Riemannian manifolds: the transport groupoid
(morphism composition and inverse), the Karcher mean via geomstats, H^1
cohomological consistency, the sheaf Laplacian, parallel transport (Schild's and
pole ladders), and persistence-based divergence tracking. Ships machine-readable
citation metadata (`CITATION.cff`, `codemeta.json` with ORCID). Pre-alpha: APIs
may change and the central hypothesis is not yet validated (see `STATUS.md` and
`LIMITATIONS.md`).

Date shown is the tag commit (`v0.1.0.dev0` -> 2a02954). `CITATION.cff` records
`date-released: 2026-05-25`, when the core implementation landed; the tag was
later placed on the metadata commit.

[Unreleased]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev7...HEAD
[0.1.0.dev7]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev6...v0.1.0.dev7
[0.1.0.dev6]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev5...v0.1.0.dev6
[0.1.0.dev5]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev4...v0.1.0.dev5
[0.1.0.dev4]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev2...v0.1.0.dev4
[0.1.0.dev3]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev2...af42cb4
[0.1.0.dev2]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev1...v0.1.0.dev2
[0.1.0.dev1]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev0...v0.1.0.dev1
[0.1.0.dev0]: https://github.com/smaniches/GROUPOID/releases/tag/v0.1.0.dev0

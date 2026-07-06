# Changelog

All notable changes to GROUPOID are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(with PEP 440 identifiers for Python pre-releases).

## [Unreleased]

Nothing yet.

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

[Unreleased]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev4...HEAD
[0.1.0.dev4]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev2...v0.1.0.dev4
[0.1.0.dev3]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev2...af42cb4
[0.1.0.dev2]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev1...v0.1.0.dev2
[0.1.0.dev1]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev0...v0.1.0.dev1
[0.1.0.dev0]: https://github.com/smaniches/GROUPOID/releases/tag/v0.1.0.dev0

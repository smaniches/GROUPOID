# Changelog

All notable changes to GROUPOID are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(with PEP 440 identifiers for Python pre-releases).

## [Unreleased]

Nothing yet.

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

[Unreleased]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev2...HEAD
[0.1.0.dev2]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev1...v0.1.0.dev2
[0.1.0.dev1]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev0...v0.1.0.dev1
[0.1.0.dev0]: https://github.com/smaniches/GROUPOID/releases/tag/v0.1.0.dev0

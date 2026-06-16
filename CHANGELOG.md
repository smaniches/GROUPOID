# Changelog

All notable changes to GROUPOID are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(with PEP 440 identifiers for Python pre-releases).

## [Unreleased]

Merged on `main` but not yet part of a tagged release. The current tag
`v0.1.0.dev0` predates the mathematical-correctness fixes below; a corrected tag
will be cut before the next archival (Zenodo) deposit so the version DOI points at
corrected code.

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

[Unreleased]: https://github.com/smaniches/GROUPOID/compare/v0.1.0.dev0...HEAD
[0.1.0.dev0]: https://github.com/smaniches/GROUPOID/releases/tag/v0.1.0.dev0

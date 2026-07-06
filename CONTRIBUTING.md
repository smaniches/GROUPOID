# Contributing to GROUPOID

Thank you for your interest in contributing to GROUPOID. This document
describes the development workflow and guidelines for contributions.

## Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/<your-username>/GROUPOID.git
cd GROUPOID
```

2. Install all dependencies:

```bash
pip install -e ".[all]"
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

## Code Style

All code is formatted and checked by the following tools (enforced via
pre-commit hooks):

- **ruff**: Linting and import sorting
- **black**: Code formatting (line length 99)
- **mypy**: Static type checking
- **bandit**: Security analysis

Run all checks manually:

```bash
pre-commit run --all-files
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Property-based tests only
pytest tests/theory/ -v

# Benchmarks
pytest benchmarks/ -v --benchmark-only
```

## Mathematical Contributions

Contributions involving mathematical claims (new algorithms, convergence
proofs, complexity bounds) must include:

- A proof or a citation to a peer-reviewed reference
- Property-based tests (using Hypothesis) that verify the claimed invariants
- Clear documentation of assumptions and limitations

If you find a mathematical error in the existing code or documentation,
please open an issue using the "Mathematical Correction" template.

## Pull Request Process

1. Create a feature branch from `main`.
2. Make your changes, ensuring all pre-commit hooks pass.
3. Add or update tests as appropriate.
4. Update documentation if your changes affect the public API.
5. Open a pull request against `main`.
6. Ensure CI passes (lint, test, security, docs build).
7. Address all review comments before merging.

## Releasing

1. Open a release-cut PR that bumps the version across every metadata
   file that carries it: `pyproject.toml`, `groupoid/__init__.py`,
   `codemeta.json`, `CITATION.cff` (version and `date-released`),
   `STATUS.md` (both mentions), and `.zenodo.json` (version, the erratum
   description, and `isNewVersionOf` advanced to the prior version's
   Zenodo DOI). Roll `CHANGELOG.md`'s `Unreleased` section into a new
   version heading with compare links. See PRs #40 or #48 for the
   template.
2. Merge the PR to `main`.

That's it — merging is the release trigger. `.github/workflows/release.yml`
runs on every push to `main`; its `detect` job reads the version out of
`pyproject.toml`, and if no `v<version>` tag exists yet, it creates and
pushes one automatically (using the standard `github-actions[bot]`
identity) and the rest of the workflow proceeds exactly as it would from
a manual tag push: build, provenance and SBOM attestation, publish to
PyPI via Trusted Publishing, Sigstore signing, and a GitHub Release
(which in turn triggers the Zenodo deposit for the new version DOI). A
push to `main` with no version change is a no-op — the tag already
exists, so `detect` skips the rest of the workflow.

The actual PyPI publish step still runs under the `pypi` GitHub
Environment; if that environment has required reviewers configured, the
publish waits for that approval exactly as before. Automating the tag
does not remove that gate.

If you need to re-publish an existing tag to PyPI only (no new GitHub
Release, no new Zenodo deposit), use the `workflow_dispatch` trigger on
`release.yml` with the `ref` input set to the existing tag.

## Issue Templates

When reporting bugs or requesting features, please use the appropriate
issue template:

- **Bug Report**: For reproducible bugs with steps, expected behavior,
  and environment details.
- **Feature Request**: For new functionality with motivation and
  proposed approach.
- **Mathematical Correction**: For errors in theorems, proofs, or
  algorithms.

## License

By contributing to GROUPOID, you agree that your contributions will be
licensed under the Apache License, Version 2.0.

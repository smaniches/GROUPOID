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

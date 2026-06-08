# Reproduce

A one-command determinism receipt for a **tested** module: the sheaf
(connection) Laplacian. The output below was produced by an actual seeded run
(numpy `default_rng(0)` / `default_rng(2)`), is bit-for-bit identical across
repeated runs, and matches the invariants asserted in
`tests/test_sheaf_laplacian.py` (the general, non-orthogonal case).

## Environment

- **Python 3.10, 3.11, or 3.12** (not 3.13+; the `numpy<2.0` / `scipy<1.14`
  pins required by geomstats have no wheels there -- see
  [LIMITATIONS.md](LIMITATIONS.md)).
- Pins used to produce the receipt below: `numpy 1.26.4`, `scipy 1.13.1`,
  `geomstats 2.8.0` (resolved by the constraints in `pyproject.toml`).

## Install

```bash
git clone https://github.com/smaniches/GROUPOID.git
cd GROUPOID
pip install -e ".[dev]"
```

## Run

```bash
pip install -e ".[dev]"          # one-time, if not already installed
python scripts/reproduce_sheaf_laplacian.py
```

`groupoid` logs at DEBUG by default. To see only the receipt, set the log
level (the receipt itself prints on stdout regardless):

```bash
# Linux / macOS
LOGURU_LEVEL=WARNING python scripts/reproduce_sheaf_laplacian.py
# Windows PowerShell
$env:LOGURU_LEVEL="WARNING"; python scripts/reproduce_sheaf_laplacian.py
```

## Expected output (deterministic, exact)

```
seed                     : 0
graph                    : nodes=['A', 'B', 'C', 'D'] edges=[('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')]
stalk_dim                : 3
L shape                  : (12, 12)
max|L - deltaT_delta|    : 4.441e-16   (expect < 1e-12)
max|L - L^T| (symmetry)  : 0.000e+00   (expect < 1e-12)
min eigenvalue (PSD)     : 2.813e-02   (expect >= -1e-9)
||L @ consistent_section|| : 4.981e-16   (expect < 1e-9, in kernel)
||L @ constant_section||   : 3.368e+00   (expect > 1e-3, NOT in kernel)
dim ker(L)                 : 3   (expect == stalk_dim = 3)
RESULT: sheaf-Laplacian invariants hold (deltaT-delta, PSD, kernel).
```

The floating-point residuals (`4.441e-16`, `4.981e-16`) are at machine epsilon
and may differ in the last digit on a different BLAS/CPU; the bracketed
tolerances (`< 1e-12`, `< 1e-9`, `> 1e-3`, `== 3`) are the contract and hold
across platforms. The integer `dim ker(L) = 3` and `min eigenvalue` sign are
exact invariants, not approximations.

## What this proves

- `L == delta^T delta` against an **independently** constructed coboundary
  `delta` (not reusing the builder's block formula).
- `L` is symmetric and positive-semidefinite (`min eigenvalue >= 0`).
- For orthogonal transport `R_uv = g_v g_u^{-1}`, transport-consistent global
  sections lie in `ker(L)`, constant sections do not, and `dim ker(L)` equals
  the stalk dimension (no spurious kernel).

To run the full property-based test suite (Hypothesis, 500 examples per
property) instead of this single receipt:

```bash
pytest tests/ -v
```

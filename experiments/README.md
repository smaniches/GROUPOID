# Experiments

Preregistered synthetic benchmark of the central GROUPOID hypothesis:
that transport-groupoid aggregation with the intrinsic Karcher mean
improves on Euclidean averaging for frame-heterogeneous clients, and
that the H^1 obstruction diagnoses inconsistent transports.

## Protocol

- [`PREREGISTRATION.md`](PREREGISTRATION.md) — hypotheses, data-generating
  process, factor grid, seeds, metrics, analysis plan, and decision
  criteria, committed and pushed **before** any benchmark execution (the
  git history is the verification mechanism).
- [`run_benchmark.py`](run_benchmark.py) — the runner: 600 seeded runs on
  S^2 (4 misalignment levels x 2 noise levels x 3 corruption levels x 25
  seeds), a 2x2 method ablation (transport on/off x extrinsic/Karcher
  mean) against an oracle estimand, with manipulation checks asserted at
  runtime.
- [`analyze.py`](analyze.py) — the preregistered analysis only; writes
  `RESULTS.md` from `results.json`.
- [`results.json`](results.json) / [`RESULTS.md`](RESULTS.md) — the
  committed run record and analysis.

## Reproduce

```bash
pip install -e ".[dev]"
LOGURU_LEVEL=WARNING python experiments/run_benchmark.py   # ~2 minutes
python experiments/analyze.py
```

Seeds are fixed; `results.json` is bit-reproducible up to floating-point
differences across BLAS/CPU (compare the analysis tables, not raw bytes).

## Summary of findings (see RESULTS.md for the full record)

- **H1 supported (6/6 cells):** under frame misalignment with exact
  transports, the full pipeline beats Euclidean averaging (median error
  differences 6e-2 to 3e-1 rad, all bootstrap CIs positive). This effect
  is largely built into the data-generating process — clients are placed
  in rotated frames — so it demonstrates the operator does what it says,
  not that it wins on any real task.
- **H2 supported with a caveat:** pooled over corrupted-transport runs,
  the H^1 norm correlates with aggregation error (Spearman rho = 0.587,
  permutation p = 1e-4). The within-cell correlations are near zero: H^1
  tracks the magnitude of cocycle corruption (a gross inconsistency
  detector), but does not rank runs by error within a fixed corruption
  level.
- **H3 supported (4/4 cells):** the improvement decomposes as
  preregistered — transport accounts for essentially all of it; the
  Karcher-vs-extrinsic mean margin is 2 to 4 orders of magnitude smaller
  on this data.
- **Null check (a) failed as specified** — the preregistered check
  wrongly assumed all methods tie at the null, but Karcher-based methods
  compute the oracle estimand exactly there; reported as failed with the
  diagnostic interpretation in RESULTS.md.

## What this does not show

No real federated learning task, no neural networks, no real datasets,
no claim about wall-clock or communication cost. The benchmark validates
the aggregation operator's intended behavior on synthetic
frame-misaligned data with a known ground truth, and quantifies the H^1
diagnostic's usefulness and its limits.

# Experiments

Preregistered synthetic benchmark of the central GROUPOID hypothesis: that
explicit frame transport plus intrinsic Karcher aggregation improves on
Euclidean averaging for frame-heterogeneous clients, and that the implemented
cycle-basis holonomy defect responds to corrupted transports.

## Protocol

- [`PREREGISTRATION.md`](PREREGISTRATION.md): hypotheses, data-generating
  process, factor grid, seeds, metrics, analysis plan, and decision criteria.
  The preserved Git DAG establishes that the preregistration commit precedes the
  runner and result commits. The project record states that the preregistration
  was also pushed before benchmark execution; public Git commit history alone
  does not independently prove the push time relative to local computation.
- [`run_benchmark.py`](run_benchmark.py): the historical runner, 600 seeded runs
  on S^2, a 2x2 method ablation against an oracle estimand, with manipulation
  checks asserted at runtime.
- [`analyze.py`](analyze.py): the historical preregistered analysis; writes
  `RESULTS.md` from `results.json`.
- [`results.json`](results.json) / [`RESULTS.md`](RESULTS.md): preserved
  historical run record and analysis.

The historical preregistration, runner, raw result JSON, and `RESULTS.md` retain
their original terminology. They are not rewritten by the later scientific
correction. See [`../CORRECTION_NOTICE.md`](../CORRECTION_NOTICE.md) for the
current interpretation.

## Reproduce

```bash
pip install -e ".[dev]"
LOGURU_LEVEL=WARNING python experiments/run_benchmark.py
python experiments/analyze.py
```

Seeds are fixed; `results.json` is bit-reproducible up to floating-point
differences across BLAS/CPU. Compare the analysis tables, not raw bytes.

## Summary of findings under the corrected construct definition

- **H1 supported (6/6 cells):** under frame misalignment with exact SO(3)
  transports, the full pipeline beats Euclidean averaging (median error
  differences 6e-2 to 3e-1 rad, all bootstrap CIs positive). This effect is
  largely built into the data-generating process, so it validates the synthetic
  operator behavior rather than a real federated-learning task.
- **Historical H2 criterion met, interpretation corrected:** pooled over the
  400 corrupted-transport runs, the fixed NetworkX cycle-basis maximum
  Frobenius holonomy defect has Spearman rho = 0.587 with aggregation error and
  permutation p = 1e-4. The within-cell correlations are near zero. The pooled
  relationship is therefore primarily a between-corruption-level association.
  The stored statistic is unchanged, but it must not be described as evidence
  that a canonical H^1 norm predicts error.
- **H3 supported (4/4 cells):** the preregistered ablation shows transport
  accounts for essentially all of the improvement; the Karcher-vs-extrinsic
  mean margin is 2 to 4 orders of magnitude smaller on this synthetic data.
- **Null check (a) failed as specified:** the preregistered check assumed all
  methods tie at the null, while Karcher-based methods compute the oracle
  estimand exactly there. The failure remains reported as a failure.

## What this does not show

No real federated learning task, no neural networks, no real datasets, and no
claim about wall-clock or communication cost. The benchmark uses explicit SO(3)
rotations and does not exercise the later-withdrawn
`register_transport_from_points` path. It supports the transport operator on the
specified synthetic frame model and provides an empirical characterization of
the fixed cycle-basis holonomy defect used in that experiment.

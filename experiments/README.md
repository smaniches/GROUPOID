# Experiments

Preregistered synthetic benchmark of the central GROUPOID hypothesis: that
explicit frame transport plus intrinsic Karcher aggregation improves on
Euclidean averaging for frame-heterogeneous clients, and that the implemented
cycle-basis holonomy defect responds to corrupted transports.

## Protocol

- [`PREREGISTRATION.md`](PREREGISTRATION.md): hypotheses, data-generating
  process, factor grid, seeds, metrics, analysis plan, and decision criteria.
  "Preregistered" here means committed first, not externally registered: in
  pull request #49 (preserved on GitHub as `refs/pull/49/head`) the plan
  (`ea7cf5f`) was committed before the runner (`43fcad5`) and the results
  (`ff7f461`), within about five minutes. Git does not record push times, and
  no external registry or timestamp exists, so the author's account that the
  plan was pushed before execution is not independently verifiable. See
  [`../CORRECTION_NOTICE.md`](../CORRECTION_NOTICE.md), F04.
- [`run_benchmark.py`](run_benchmark.py): the historical runner, 600 seeded runs
  on S^2, a 2x2 method ablation against an oracle estimand, with manipulation
  checks asserted at runtime. Since v0.1.0.dev10 it passes the historical Karcher
  tolerance (`1e-6`) explicitly, because the library default changed (F03).
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
`run_benchmark.py` overwrites `results.json`; `tests/test_frozen_benchmark.py`
instead re-executes all 600 runs and compares them with the stored records.

## Summary of findings under the corrected construct definition

- **H1 supported (6/6 cells):** under frame misalignment with exact SO(3)
  transports, the full pipeline beats Euclidean averaging (median error
  differences 6e-2 to 3e-1 rad, all bootstrap CIs positive). This effect is
  largely built into the data-generating process, so it validates the synthetic
  operator behavior rather than a real federated-learning task.
- **Historical H2 criterion met, interpretation corrected:** pooled over the
  400 corrupted-transport runs, the fixed NetworkX cycle-basis maximum
  Frobenius holonomy defect has Spearman rho = 0.587 with aggregation error and
  permutation p = 1e-4, which meets the prespecified criterion. Within each
  corruption level the association is absent (rho = -0.030 at epsilon = 0.1
  and 0.004 at epsilon = 0.3, n = 200 each; partial rho given the level
  -0.033), so the pooled value reflects that defect and error both grow with
  the corruption level. The stored statistic is unchanged, but it is not
  evidence that the defect, or a canonical H^1 norm, predicts error among runs
  at the same corruption level (CORRECTION_NOTICE.md, F05).
- **H3 supported (4/4 cells):** the preregistered ablation shows transport
  accounts for essentially all of the improvement; the Karcher-vs-extrinsic
  mean margin is 2 to 4 orders of magnitude smaller on this synthetic data.
  At sigma = 0.05 the stored margins (4.3e-5 and 4.2e-5) are mostly solver
  error; with an accurate solver they are 2.0e-5 and 1.2e-5 (F03).
- **Null check (a) failed as specified:** the preregistered check assumed all
  methods tie at the null, while Karcher-based methods compute the oracle
  estimand exactly there. The failure remains reported as a failure. Its
  stored magnitude at sigma = 0.05 (3.0e-5) is about three times the
  solver-accurate extrinsic-versus-Karcher gap (9.7e-6); see F03.
- **Solver accuracy (retrospective):** the stored results used the former
  Karcher default `tol = 1e-6`, which left oracle errors of up to 1.4e-3 rad
  (median 3e-5 to 7e-5). Rerunning the unmodified runner and analysis with
  `tol = 1e-12` leaves every verdict unchanged (CORRECTION_NOTICE.md, F03);
  stored values of about 1e-3 rad or less should not be read as measured
  geometric gaps.

## What this does not show

No real federated learning task, no neural networks, no real datasets, and no
claim about wall-clock or communication cost. The benchmark uses explicit SO(3)
rotations and does not exercise the later-withdrawn
`register_transport_from_points` path. It supports the transport operator on the
specified synthetic frame model and provides an empirical characterization of
the fixed cycle-basis holonomy defect used in that experiment.

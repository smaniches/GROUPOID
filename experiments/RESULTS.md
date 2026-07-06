# Benchmark results (preregistered analysis)

Analysis of `results.json` following `PREREGISTRATION.md` exactly;
deviations are listed at the end. Primary error metric: geodesic
distance to the oracle estimand (Karcher mean of the de-framed
truths). Effect sizes are medians of per-seed paired differences
with 10,000-resample percentile-bootstrap 95% CIs.

## Environment

- python: 3.11.15
- numpy: 1.26.4
- scipy: 1.15.3
- geomstats: 2.8.0
- platform: Linux-6.18.5-x86_64-with-glibc2.39
- n_runs: 600
- Karcher non-convergence warnings (runs, not excluded): 0

## Median primary error per cell

| alpha | sigma | epsilon | M1 euclidean | M2 karcher | M3 transport+eucl | M4 transport+karcher | median H^1 |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.05 | 0.0 | 3.00e-05 | 0.00e+00 | 3.00e-05 | 0.00e+00 | 0.00e+00 |
| 0.0 | 0.05 | 0.1 | 3.00e-05 | 0.00e+00 | 1.36e-02 | 1.35e-02 | 2.47e-01 |
| 0.0 | 0.05 | 0.3 | 3.00e-05 | 0.00e+00 | 5.69e-02 | 5.66e-02 | 6.95e-01 |
| 0.0 | 0.2 | 0.0 | 6.31e-04 | 0.00e+00 | 6.31e-04 | 0.00e+00 | 0.00e+00 |
| 0.0 | 0.2 | 0.1 | 6.31e-04 | 0.00e+00 | 1.31e-02 | 1.36e-02 | 2.25e-01 |
| 0.0 | 0.2 | 0.3 | 6.31e-04 | 0.00e+00 | 3.45e-02 | 3.37e-02 | 6.85e-01 |
| 0.3 | 0.05 | 0.0 | 7.44e-02 | 7.33e-02 | 3.16e-05 | 0.00e+00 | 4.85e-16 |
| 0.3 | 0.05 | 0.1 | 7.44e-02 | 7.33e-02 | 1.30e-02 | 1.31e-02 | 2.51e-01 |
| 0.3 | 0.05 | 0.3 | 7.44e-02 | 7.33e-02 | 3.74e-02 | 3.65e-02 | 6.89e-01 |
| 0.3 | 0.2 | 0.0 | 6.19e-02 | 6.24e-02 | 8.51e-04 | 0.00e+00 | 5.12e-16 |
| 0.3 | 0.2 | 0.1 | 6.19e-02 | 6.24e-02 | 1.33e-02 | 1.40e-02 | 2.40e-01 |
| 0.3 | 0.2 | 0.3 | 6.19e-02 | 6.24e-02 | 3.87e-02 | 3.82e-02 | 6.32e-01 |
| 0.7 | 0.05 | 0.0 | 1.53e-01 | 1.43e-01 | 4.32e-05 | 0.00e+00 | 5.67e-16 |
| 0.7 | 0.05 | 0.1 | 1.53e-01 | 1.43e-01 | 1.08e-02 | 1.07e-02 | 2.46e-01 |
| 0.7 | 0.05 | 0.3 | 1.53e-01 | 1.43e-01 | 3.20e-02 | 3.22e-02 | 7.39e-01 |
| 0.7 | 0.2 | 0.0 | 1.77e-01 | 1.77e-01 | 8.89e-04 | 0.00e+00 | 5.66e-16 |
| 0.7 | 0.2 | 0.1 | 1.77e-01 | 1.77e-01 | 1.22e-02 | 1.32e-02 | 2.27e-01 |
| 0.7 | 0.2 | 0.3 | 1.77e-01 | 1.77e-01 | 3.64e-02 | 3.70e-02 | 7.07e-01 |
| 1.2 | 0.05 | 0.0 | 2.85e-01 | 2.73e-01 | 4.17e-05 | 0.00e+00 | 6.16e-16 |
| 1.2 | 0.05 | 0.1 | 2.85e-01 | 2.73e-01 | 1.35e-02 | 1.35e-02 | 2.30e-01 |
| 1.2 | 0.05 | 0.3 | 2.85e-01 | 2.73e-01 | 4.82e-02 | 4.81e-02 | 7.36e-01 |
| 1.2 | 0.2 | 0.0 | 2.93e-01 | 2.76e-01 | 1.01e-03 | 0.00e+00 | 6.04e-16 |
| 1.2 | 0.2 | 0.1 | 2.93e-01 | 2.76e-01 | 1.37e-02 | 1.33e-02 | 2.55e-01 |
| 1.2 | 0.2 | 0.3 | 2.93e-01 | 2.76e-01 | 3.87e-02 | 3.88e-02 | 7.38e-01 |

## H1: transport benefit under misalignment (primary)

Criterion: in every cell with alpha > 0 and epsilon = 0, the 95% CI
of median(err(M1) - err(M4)) is entirely positive.

| alpha | sigma | median diff | 95% CI | positive? |
|---|---|---|---|---|
| 0.3 | 0.05 | 7.44e-02 | [4.02e-02, 8.71e-02] | yes |
| 0.3 | 0.2 | 6.19e-02 | [3.40e-02, 7.89e-02] | yes |
| 0.7 | 0.05 | 1.53e-01 | [9.09e-02, 2.03e-01] | yes |
| 0.7 | 0.2 | 1.77e-01 | [1.41e-01, 2.37e-01] | yes |
| 1.2 | 0.05 | 2.85e-01 | [1.69e-01, 4.53e-01] | yes |
| 1.2 | 0.2 | 2.93e-01 | [2.57e-01, 3.71e-01] | yes |

**H1 verdict: SUPPORTED** (6/6 cells positive).

## H2: H^1 predicts aggregation error under corruption (non-trivial)

- Pooled runs with epsilon > 0: n = 400
- Spearman rho(H^1, err(M4)) = 0.587
- Permutation p-value (10000 permutations): 0.00010
- Criterion rho >= 0.5 and p < 0.01: met

**H2 verdict: SUPPORTED**

Descriptive (no criterion): within-cell Spearman correlations:

| alpha | sigma | epsilon | rho |
|---|---|---|---|
| 0.0 | 0.05 | 0.1 | -0.232 |
| 0.0 | 0.05 | 0.3 | -0.014 |
| 0.0 | 0.2 | 0.1 | -0.129 |
| 0.0 | 0.2 | 0.3 | 0.009 |
| 0.3 | 0.05 | 0.1 | -0.032 |
| 0.3 | 0.05 | 0.3 | 0.270 |
| 0.3 | 0.2 | 0.1 | 0.031 |
| 0.3 | 0.2 | 0.3 | -0.196 |
| 0.7 | 0.05 | 0.1 | 0.238 |
| 0.7 | 0.05 | 0.3 | -0.070 |
| 0.7 | 0.2 | 0.1 | -0.152 |
| 0.7 | 0.2 | 0.3 | -0.201 |
| 1.2 | 0.05 | 0.1 | 0.058 |
| 1.2 | 0.05 | 0.3 | 0.089 |
| 1.2 | 0.2 | 0.1 | 0.052 |
| 1.2 | 0.2 | 0.3 | 0.149 |

Descriptive caveat (does not alter the preregistered verdict): the
within-cell correlations are near zero, so the pooled correlation is
driven by between-level variation (epsilon = 0.1 vs 0.3). On this
evidence H^1 tracks the magnitude of cocycle corruption -- it works
as a gross inconsistency detector -- but does not rank runs by error
within a fixed corruption level.

## H3: ablation decomposition

Criterion: in every cell with alpha >= 0.7 and epsilon = 0, the
transport margin median(err(M1) - err(M3)) exceeds the intrinsic-
mean margin median(err(M3) - err(M4)).

| alpha | sigma | transport margin | intrinsic-mean margin | transport larger? |
|---|---|---|---|---|
| 0.7 | 0.05 | 1.53e-01 | 4.32e-05 | yes |
| 0.7 | 0.2 | 1.76e-01 | 8.89e-04 | yes |
| 1.2 | 0.05 | 2.85e-01 | 4.17e-05 | yes |
| 1.2 | 0.2 | 2.91e-01 | 1.01e-03 | yes |

**H3 verdict: SUPPORTED** (4/4 cells).

## Null and manipulation checks

(a) alpha = 0, epsilon = 0 cells: all pairwise 95% CIs include 0.

| sigma | pair | median diff | 95% CI | includes 0? |
|---|---|---|---|---|
| 0.05 | euclidean - karcher | 3.00e-05 | [1.68e-05, 4.50e-05] | NO |
| 0.05 | euclidean - transport_euclidean | 0.00e+00 | [0.00e+00, 0.00e+00] | yes |
| 0.05 | euclidean - transport_karcher | 3.00e-05 | [1.68e-05, 4.50e-05] | NO |
| 0.05 | karcher - transport_euclidean | -3.00e-05 | [-4.50e-05, -1.68e-05] | NO |
| 0.05 | karcher - transport_karcher | 0.00e+00 | [0.00e+00, 0.00e+00] | yes |
| 0.05 | transport_euclidean - transport_karcher | 3.00e-05 | [1.68e-05, 4.50e-05] | NO |
| 0.2 | euclidean - karcher | 6.31e-04 | [4.53e-04, 1.54e-03] | NO |
| 0.2 | euclidean - transport_euclidean | 0.00e+00 | [0.00e+00, 0.00e+00] | yes |
| 0.2 | euclidean - transport_karcher | 6.31e-04 | [4.53e-04, 1.54e-03] | NO |
| 0.2 | karcher - transport_euclidean | -6.31e-04 | [-1.54e-03, -4.53e-04] | NO |
| 0.2 | karcher - transport_karcher | 0.00e+00 | [0.00e+00, 0.00e+00] | yes |
| 0.2 | transport_euclidean - transport_karcher | 6.31e-04 | [4.53e-04, 1.54e-03] | NO |

Check (a): FAILED.

Descriptive interpretation (the check is reported as failed above
and stays failed): the failing pairs are exactly those comparing an
extrinsic (renormalized Euclidean) mean against a Karcher-based
method. At alpha = 0, epsilon = 0 the Karcher-based methods compute
the oracle estimand exactly by construction (their error is 0),
while the extrinsic mean differs from the intrinsic mean by a small
systematic amount, so its median difference cannot include 0. The
check as preregistered wrongly assumed all methods are equivalent
at the null; it is diagnostic of the estimand choice, not of an
implementation error. The observed magnitudes (~3e-5 at sigma =
0.05, ~6e-4 at sigma = 0.2) are 2 to 4 orders of magnitude below
the H1 effect sizes (~6e-2 to 3e-1).

(b) max H^1 at epsilon = 0: 8.31e-16 (< 1e-8, asserted at runtime): passed.
(c) min H^1 at epsilon > 0: 1.32e-01 (> 0, asserted at runtime): passed.
(d) M1/M2 errors bit-identical across epsilon for fixed (alpha, sigma,
seed): asserted at runtime by the runner: passed.

## Deviations from preregistration

1. Seeding. The preregistration specified that run k uses the k-th
   child of SeedSequence(20260706), but that conflicts with its own
   manipulation check (d), which requires the client data to be
   identical across corruption levels for a fixed (alpha, sigma,
   seed). The runner therefore keys the data stream by
   (alpha_index, sigma_index, seed_index) -- independent of epsilon --
   and the corruption stream by the full cell. No other deviation.

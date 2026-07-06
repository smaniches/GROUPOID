# Preregistration: transport-groupoid aggregation benchmark

**Date:** 2026-07-06
**Status:** Registered before any benchmark execution. At the time this
document is committed, no benchmark comparison between the methods below
has been computed, on this or any other grid. The only prior execution
was a feasibility pilot (timing single `karcher_mean` calls and checking
the geomstats `parallel_transport` API); it produced no comparative
numbers. The commit and push timestamps of this file, which precede the
commit that adds `results.json`, are the verification mechanism.

**Amendments:** none may be made to this file after results exist. Any
deviation forced during implementation is documented in the "Deviations
from preregistration" section of `RESULTS.md`, with the reason.

## Background

GROUPOID's central hypothesis (stated as unvalidated in `STATUS.md`) is
that transporting client parameters to a common frame via groupoid
morphisms and aggregating with the intrinsic Karcher mean improves on
naive Euclidean averaging when clients are frame-heterogeneous, and that
the H^1 cohomological obstruction diagnoses inconsistent transports.
This experiment tests that hypothesis on synthetic data with a known
ground truth. The frame-misalignment benefit (H1) is expected largely by
construction — the data-generating process puts clients in rotated
frames — so the interesting preregistered quantities are the effect
sizes, the ablation decomposition (H3), and whether H^1 actually
predicts aggregation error when the cocycle is corrupted (H2), which is
not true by construction.

## Data-generating process

Manifold: unit sphere S^2 (geomstats `Hypersphere(dim=2)`).

Per run, with all randomness drawn from that run's dedicated generator:

1. Consensus point `p* = (0, 0, 1)`.
2. For each client `i` in `0..n-1` (n = 8): tangent noise
   `xi_i ~ Normal(0, sigma^2 I)` in `T_{p*}S^2` (sampled in R^3 and
   projected to the tangent space), de-framed truth
   `y_i = Exp_{p*}(xi_i)`.
3. Client frames: `Q_0 = I` (base client). For `i >= 1`, `Q_i` is the
   rotation by angle `alpha` about an axis drawn uniformly on S^2.
   Local parameter `x_i = Q_i y_i`.
4. Client graph: complete graph K_8 (its cycle basis is nonempty, so
   H^1 is informative). For each unordered pair `{i, j}` with `i < j`,
   the registered transport is `T_{i->j} = C_{ij} (Q_j Q_i^T)` where the
   corruption `C_{ij}` is a rotation by angle `delta_ij ~ Uniform[0,
   epsilon]` about an axis drawn uniformly on S^2. With `epsilon = 0`
   the cocycle is an exact coboundary and H^1 = 0 up to machine
   precision.

## Methods compared (2x2 ablation + oracle)

- **M1 `euclidean`** (FedAvg analog): arithmetic mean of the raw local
  parameters `x_i`, renormalized to S^2 (extrinsic mean). No transport,
  no intrinsic mean.
- **M2 `karcher`**: Karcher mean of the raw local parameters. Intrinsic
  mean, no transport.
- **M3 `transport_euclidean`**: transport every `x_i` to the base frame
  along the direct edge (`T_{i->0} x_i`), then renormalized arithmetic
  mean. Transport, no intrinsic mean.
- **M4 `transport_karcher`** (the GROUPOID pipeline):
  `TransportGroupoidAggregator.aggregate` — transport to base, H^1
  consistency check, Karcher mean. Transport and intrinsic mean.
- **Oracle estimand**: the Karcher mean of the de-framed truths
  `{y_i}` — what a perfect aggregator with exact frame knowledge would
  compute. All errors are measured against this estimand.

M1 and M2 never see the transports, so the corruption factor cannot
affect them; their per-cell results are expected to be constant across
`epsilon` (this serves as an implementation sanity check).

## Factor grid and replication

| Factor | Levels |
|---|---|
| Frame misalignment `alpha` (rad) | 0, 0.3, 0.7, 1.2 |
| Tangent noise `sigma` | 0.05, 0.2 |
| Transport corruption `epsilon` (rad) | 0, 0.1, 0.3 |
| Clients `n` | 8 (fixed) |
| Replicates (seeds) per cell | 25 |

Total: 4 x 2 x 3 x 25 = 600 runs. Runs are enumerated in a fixed order
(sorted grid, then seed index); run k uses the k-th child of
`numpy.random.SeedSequence(20260706)`. The analysis bootstrap uses
`numpy.random.default_rng(12345)`; the permutation test uses
`numpy.random.default_rng(54321)`.

## Recorded per run

- Geodesic distance (via `manifold.metric.dist`) from each method's
  output to the oracle estimand — the **primary error metric**.
- Secondary: geodesic distance from each method's output to `p*`.
- The H^1 norm reported by the aggregator (`FederatedRound.h1_norm`).
- The `is_consistent` flag, and whether any Karcher-mean call emitted a
  non-convergence warning (runs are **not excluded** for warnings; the
  count is reported).

## Hypotheses and preregistered decision criteria

Statistics: for paired method comparisons within a cell, the statistic
is the median over the 25 seeds of the per-seed difference in primary
error; its 95% CI is the percentile bootstrap over seeds with 10,000
resamples. "Positive CI" means the entire interval is above 0.

- **H1 (transport benefit; primary).** In every cell with `alpha > 0`
  and `epsilon = 0`: the CI of `err(M1) - err(M4)` is positive.
  *Supported* if this holds in all 6 such cells; *partially supported*
  if in at least 4; otherwise *not supported*. (Expected largely by
  construction; the reported effect sizes are the contribution.)
- **H2 (H^1 predicts error under corruption; the non-trivial claim).**
  Across all runs with `epsilon > 0` pooled (400 runs): Spearman rank
  correlation between the run's H^1 norm and the run's `err(M4)` is
  `rho >= 0.5`, with a permutation p-value < 0.01 (10,000 permutations
  of the error vector). *Supported / not supported* accordingly; the
  within-cell Spearman correlations are also reported descriptively
  (no criterion attached).
- **H3 (ablation decomposition).** In every cell with `alpha >= 0.7`
  and `epsilon = 0`: the transport margin exceeds the intrinsic-mean
  margin, i.e. `median(err(M1) - err(M3)) > median(err(M3) - err(M4))`.
  We additionally expect (descriptively, no criterion) the Karcher-vs-
  extrinsic margin `err(M3) - err(M4)` to be small at `sigma = 0.05`,
  since extrinsic and intrinsic means nearly coincide for concentrated
  data.
- **Null / manipulation checks.**
  (a) In the `alpha = 0, epsilon = 0` cells, the CIs of all pairwise
  method differences include 0 (no method should win when frames are
  aligned and transports exact).
  (b) At `epsilon = 0`, every run's H^1 norm is below 1e-8 (coboundary;
  asserted at runtime).
  (c) At `epsilon > 0`, every run's H^1 norm is positive.
  (d) M1 and M2 per-cell error distributions are identical across
  `epsilon` levels for fixed `(alpha, sigma)` and seed (they never see
  the transports; asserted at runtime).

## Reporting commitments

Results are committed and reported whatever they show, including a
falsified H1/H2/H3. `RESULTS.md` will contain: the preregistered
analyses exactly as specified above (tables of medians and CIs per
cell), the H2 correlation and p-value, the manipulation-check outcomes,
the non-convergence warning count, environment versions, and a
"Deviations from preregistration" section (empty if none). No metric,
factor level, or criterion may be added, dropped, or reinterpreted
after results exist without being listed as a deviation.

## Feasibility pilot disclosure

Prior to registration, single `karcher_mean` calls were timed (warm
calls ~2-20 ms for 8 points on S^2) to confirm the grid is tractable,
and the geomstats `parallel_transport` signature was checked. No method
comparison, no aggregation error, and no H^1-vs-error relationship was
computed.

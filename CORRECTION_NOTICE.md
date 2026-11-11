# Scientific correction notice

This notice documents the scientific corrections made to GROUPOID. F01 and F02
are mathematical-semantic corrections prepared against frozen repository state
`d4ab3f4c8439d5102c75a59f176965a7957c620a` (released in v0.1.0.dev5). F03 to
F06 were prepared against `5bfbee91a80355ebb97ea82e718a669c52426b3d` for
v0.1.0.dev10: one numerical correction to the Karcher mean default and three
corrections to how the benchmark record and the dependency requirements are
described. Historical preregistration and result artifacts are preserved
unchanged; the only edit to a historical file, an explicit tolerance argument in
`experiments/run_benchmark.py` that keeps it reproducing `results.json`, is
described under F03.

## F01: cycle-basis holonomy defect was misidentified as an H^1 norm

Earlier releases described `compute_h1` and `FederatedRound.h1_norm` as a
first-cohomology norm or invariant. The implementation instead evaluates

\[
D_{\mathcal B}(T)
= \max_{\gamma \in \mathcal B}
\left\|\operatorname{Hol}_T(\gamma)-I\right\|_F,
\]

where `B` is the cycle basis returned by `networkx.cycle_basis` on the
undirected graph.

The corrected primary API name is `cycle_basis_holonomy_defect`. The legacy
`compute_h1` and `h1_norm` names are retained only for compatibility and are
explicitly documented as historical names for the same scalar.

The correction distinguishes three properties that were previously conflated:

1. Under a connected graph with one complete invertible connection map per
   underlying undirected edge, represented by one orientation or a reciprocal
   pair, identity holonomy on every cycle emitted by `networkx.cycle_basis` is
   sufficient for flat transport. Thus exact zero of the implemented defect has
   a valid flatness interpretation under explicit assumptions. That supporting
   argument is specific to the NetworkX Paton implementation exercised by
   GROUPOID: the emitted list is not in general the fundamental-cycle basis of
   one fixed spanning tree, but in emission order each cycle contributes exactly
   one chord absent from every earlier emitted cycle, so the induced constraint
   system is triangular. The argument is not claimed for arbitrary cycle bases
   or for future NetworkX implementations whose emission order may differ.
2. The nonzero magnitude is not independent of the selected cycle basis.
3. The Frobenius magnitude is invariant under orthogonal conjugation and
   reversal in the orthogonal domain, but is not invariant under general
   invertible changes of frame. Consequently a finite threshold on this
   magnitude is representation-dependent and is not a canonical
   cohomological verdict.

The preregistered synthetic benchmark uses SO(3) rotation transports. Its
stored numerical results are therefore not invalidated by the general
non-orthogonal gauge counterexample. The H1 transport-benefit and H3 ablation
results remain unchanged. The historical H2 statistic also remains unchanged,
but its interpretation is corrected: it is an association between aggregation
error and the fixed NetworkX cycle-basis holonomy defect used by the benchmark,
not evidence that a canonical H^1 norm predicts error. The existing within-cell
caveat remains important because the pooled association is driven primarily by
between-corruption-level variation.

No 600-run benchmark rerun is required for this correction.

## F02: tangent-vector transport was promoted into an unsupported point action

`compute_transport_matrix` historically projected ambient coordinate vectors
into a tangent space, transported those tangent vectors, and stored them as the
columns of a square ambient array. That construction can be meaningful as an
ambient-coordinate representation of a tangent-vector operator. It does not
thereby define an invertible action on manifold points.

For an embedded manifold such as S^2, the exact projector extension has the
form

\[
A_{p\to q}=L_{p\to q}P_p,
\]

with `P_p` the tangent projector and `L` Levi-Civita parallel transport. Since
`P_p p = 0`, the exact ambient operator is rank-deficient and sends the base
normal direction to zero. Numerical ladder error can perturb that rank, but
accidental ambient invertibility created by approximation error has no geometric
meaning.

Earlier `register_transport_from_points` code nevertheless registered this
array as the same `Morphism` abstraction used by the point-valued aggregation
pipeline. The pipeline then applied it directly to manifold points and later
used an ordinary matrix inverse for return transport. That integration claim is
withdrawn.

The ladder routines remain supported as tangent-vector transport utilities at
the level established by their existing validation. `compute_tangent_transport_matrix`
is the corrected primary name for the ambient-coordinate tangent operator and
is restricted to vector-shaped point representations (1D coordinate arrays of
equal shape). Structured points must use the ladder functions directly on
native-shaped tangent objects.
`compute_transport_matrix` remains only as a deprecated compatibility alias.
`register_transport_from_points` remains discoverable but fails closed instead
of silently constructing an unsupported point action.

The direct `register_transport` path remains supported only under an explicit
point-action contract: a registered matrix must be finite with a finite numerical
inverse in the chosen representation. If both orientations of one underlying
edge are registered explicitly, they must be mutual numerical inverses. The
forward and return actions actually exercised by aggregation must preserve the
configured manifold domain. The current S^2
scientific evidence validates explicit SO(3) rotations. Arbitrary square
matrices are not validated as geometric transports merely because they are
numerically invertible.

The historical `transport_residuals` field is also retained for compatibility,
but its stored quantity is specifically the composite-map orthogonality defect
`||T T^T - I||_F`. The clearer `orthogonality_residuals` property exposes the
same values without implying a generic transport-error metric.

The preregistered benchmark is not contaminated by F02 because it constructs
and registers explicit Rodrigues rotation matrices directly. It never calls
`compute_transport_matrix` or `register_transport_from_points`.

## F03: the Karcher tolerance did not control the accuracy of the mean

`karcher_mean` forwards `tol` to the geomstats gradient descent, whose
gradient criterion stops the iteration at the first iterate `m` where the
weighted Frechet function `F(m) = sum_i w_i d(m, x_i)^2 / (2 sum_i w_i)`
satisfies `||grad F(m)||^2 <= tol * dim`. The iteration can also end when the
weighted variance is numerically zero or after `max_iter` steps; only the
gradient criterion establishes `||grad F(m)|| <= sqrt(tol * dim)`. The former
default `tol = 1e-6` therefore allowed a gradient norm of about `1.41e-3` on S^2
at a gradient-criterion stop, while the parameter was documented only as a
"convergence tolerance".

Measured against an independent numpy minimizer (unit-step Riemannian gradient
iteration run to round-off), the `tol = 1e-6` oracle estimates of the
benchmark's 200 distinct oracle problems were off by a median of `3.3e-5` rad
(maximum `2.2e-4`) at `sigma = 0.05` and a median of `7.1e-5` rad (maximum
`1.4e-3`) at `sigma = 0.2`. At `sigma = 0.05` geomstats stopped after a single
step on every oracle problem.

Correction in v0.1.0.dev10: the default is `tol = 1e-12`, so a stop on the
gradient criterion bounds the returned gradient norm by about `1.41e-6` on S^2.
The stopping contract, including the two exits that do not establish this
bound, is documented in the `karcher_mean` docstring, and
`TransportGroupoidAggregator(karcher_tol=...)` forwards the tolerance. On all
Karcher problems the benchmark solves (oracle, M2 and M4) the largest error
against the independent minimizer is `1.7e-6` rad, reached in at most 13 of
the 100 allowed iterations, with no iteration-limit stops and with every
returned point within the gradient bound; a call takes about 1.3 to 1.6 times
as long. Passing `tol = 1e-6` (or `karcher_tol = 1e-6`)
reproduces earlier outputs. Regression tests compare the default against a
symmetric ring whose mean is known exactly and against the independent
minimizer, and check the gradient-criterion bound on test inputs.

The stored benchmark is not changed. `experiments/run_benchmark.py` now passes
`HISTORICAL_KARCHER_TOL = 1e-6`, the default in force when `results.json` was
produced, so it keeps reproducing that file; a test re-executes all 600 runs
against the stored records, and all 600 records were bit-identical on the
machine used for this correction. As a sensitivity analysis, the unmodified
historical runner and `analyze.py` were rerun with `tol = 1e-12`:

- H1: 6/6 cells supported; median differences unchanged at the reported
  precision apart from `7.44e-02` becoming `7.43e-02`.
- H2: pooled `rho = 0.588` (stored `0.587`), permutation `p = 1e-4`; the
  criterion is met.
- H3: 4/4 cells supported. The intrinsic-mean margins at `sigma = 0.05` are
  `1.95e-05` and `1.19e-05` instead of the stored `4.32e-05` and `4.17e-05`;
  at `sigma = 0.2` they are `5.04e-04` and `1.02e-03` instead of `8.89e-04`
  and `1.01e-03`.
- Null check (a) still fails. The extrinsic-versus-Karcher gap at `alpha = 0`
  is `9.71e-06` (95% CI `[8.81e-06, 1.54e-05]`) at `sigma = 0.05` instead of
  the stored `3.00e-05` (`[1.68e-05, 4.50e-05]`), and `5.74e-04`
  (`[3.72e-04, 1.04e-03]`) at `sigma = 0.2` instead of `6.31e-04`
  (`[4.53e-04, 1.54e-03]`).

Stored quantities of about `1e-3` rad or less (the M1 error at `alpha = 0`, the
M3 errors and intrinsic-mean margins at `epsilon = 0`, and the null-check (a)
magnitudes) therefore contain solver error of their own order and should not
be quoted as measured extrinsic-versus-intrinsic gaps. At `sigma = 0.05` the
stored values are two to three and a half times the solver-accurate ones, so the
descriptive "~3e-5 at sigma = 0.05" in `RESULTS.md` overstates that gap about
threefold. The zero M2 error at `alpha = 0` and the zero M4 error at
`epsilon = 0` remain zero because those methods solve the same problem as the
oracle with the same solver. Effects of `6e-2` rad and larger (H1 and the H3
transport margins) are not affected, and the H2 statistics change only in the
third decimal.

The stored "Karcher non-convergence warnings: 0" was counted with a detector
that listens for Python warnings, while geomstats reports the iteration limit
through `logging`, so that count could not have registered a limit stop.
Instrumenting the solver shows that no historical solve used more than 5 of its
100 iterations, so the count is correct, but the stored zero is not evidence of
convergence by itself.

## F04: the preregistration is a commit-order record, not an external registration

`experiments/PREREGISTRATION.md` describes itself as "Registered before any
benchmark execution" and names "the commit and push timestamps of this file" as
the verification mechanism. The file is left unchanged under its own
no-amendment rule. What the public record establishes is narrower:

- The benchmark was developed in pull request #49, whose commits are preserved
  on GitHub as `refs/pull/49/head`. On 2026-07-06 (UTC): `ea7cf5f` (16:41:40)
  adds `PREREGISTRATION.md`; `43fcad5` (16:44:23) adds `run_benchmark.py` and
  `analyze.py`; `ff7f461` (16:46:52) adds `results.json`, `RESULTS.md` and
  `experiments/README.md` and adds two labelled descriptive notes to
  `analyze.py`, without changing its decision rules; `21865c5` (16:53:33)
  replaces the runner's assert-based manipulation checks with explicit errors,
  without changing any computed value.
- `PREREGISTRATION.md` has not changed since `ea7cf5f`, and `results.json`,
  `RESULTS.md` and `analyze.py` have not changed since `ff7f461`.
- `main` contains the pull request as one squash commit, `ddb3877`, whose tree
  equals that of `21865c5`; `main`'s own history does not show the order.
- Git commit times are set by the committing machine, and Git does not record
  push times. The plan was not registered with an external registry or
  timestamping service.

The record therefore shows that the analysis plan was committed before the
runner and the results, minutes apart, within one pull request. That is
consistent with the author's account that the plan was pushed before the
benchmark was executed, but it does not independently establish it, and it
cannot rule out executions before the plan commit beyond the timing pilot that
`PREREGISTRATION.md` itself discloses. Current documents use "preregistered"
in this internal, commit-order sense. Future experiments should use an
external, timestamped registration.

## F05: H2 was met as a pooled criterion, not as within-level prediction

The prespecified H2 criterion (Spearman `rho >= 0.5` and permutation
`p < 0.01` over the 400 corrupted-transport runs) was met as stored:
`rho = 0.587`, `p = 1e-4`. That record and its "SUPPORTED" verdict are
unchanged. The association is between the two corruption levels: the defect
and the error both increase from `epsilon = 0.1` to `epsilon = 0.3` (Spearman
correlations with `epsilon` of `0.866` and `0.691`). Recomputed from
`results.json`:

- within `epsilon = 0.1` (n = 200, pooled over `alpha` and `sigma`):
  `rho = -0.030` (two-sided `p = 0.68`);
- within `epsilon = 0.3` (n = 200): `rho = 0.004` (`p = 0.95`);
- partial Spearman correlation given the corruption level (rank residuals):
  `-0.033`;
- within the 16 individual cells (n = 25 each): from `-0.23` to `0.27`,
  median `-0.002`.

"H2 SUPPORTED" therefore records that the pooled criterion was met. It is not
evidence that the defect predicts aggregation error, or ranks runs by error,
among runs at the same corruption level, and with only two corruption levels
the benchmark cannot characterize a dose-response relationship. The supported
conclusion is narrower: in this synthetic design both the fixed cycle-basis
defect and the aggregation error increase with the corruption level. The
sensitivity rerun of F03 shows the same pattern (pooled `0.588`; within-level
`-0.027` and `0.006`; partial `-0.030`). A test recomputes these statistics
from the stored results.

## F06: SciPy requirement statements disagreed with the package metadata

`README.md`, `REPRODUCE.md` and `LIMITATIONS.md` stated `scipy<1.16` and
partly attributed the Python 3.13 exclusion to it, while `pyproject.toml` has
allowed `scipy>=1.10,<1.18` since the dependency update `e86091c`
(2026-11-02). Corrections:

- The supported SciPy range is the one in `pyproject.toml`,
  `scipy>=1.10,<1.18`; CI installs the newest release it allows. The upper
  bound caps the tested range and does not mark a known incompatibility. The
  test suite and the 600-run benchmark reproduction pass with SciPy 1.17.1.
- Python 3.13 and later are excluded by `numpy<2.0`: NumPy 1.26.4 publishes
  wheels only up to CPython 3.12, while SciPy 1.15.3 and 1.17.1 publish
  CPython 3.13 wheels.
- The historical environments are recorded separately. The `REPRODUCE.md`
  receipt was produced with SciPy 1.13.1, when the requirement was
  `scipy<1.14`; its computation uses NumPy only, and it gives the same output
  with SciPy 1.13.1 and 1.17.1. The benchmark `results.json` records
  SciPy 1.15.3.
- A test checks that every documented NumPy and SciPy upper bound matches
  `pyproject.toml`.

## Provenance and historical record

The following historical artifacts are intentionally not rewritten:

- `experiments/PREREGISTRATION.md`
- `experiments/results.json`
- `experiments/RESULTS.md`
- `experiments/analyze.py`
- `experiments/run_benchmark.py`, apart from the explicit historical Karcher
  tolerance added for v0.1.0.dev10 (F03), which leaves its output unchanged

Current explanatory surfaces may quote or describe those artifacts with the
corrected construct terminology, but historical wording remains part of the
original record.

The order of the preregistration, runner and result commits, and what that
order does and does not establish, is set out in F04.

## Effect on conclusions

F01 and F02:

- Stored benchmark numerical values changed: **No**.
- H1 transport-benefit conclusion changed: **No**.
- H2 stored rho or permutation p-value changed: **No**.
- H2 interpretation changed: **Yes**.
- H3 ablation conclusion changed: **No**.
- General finite-threshold consistency semantics changed: **Yes**.
- `register_transport_from_points` supported-integration claim withdrawn: **Yes**.
- Tangent-vector ladder validation withdrawn: **No**.
- Retraction warranted: **No**.

F01 and F02 correct mathematical construct identity and transport
representation semantics. They preserve the empirical observations that remain
supported while removing claims that the implementation does not establish.

F03 to F06:

- Library default output changed: **Yes**. With the default, a stop on the
  gradient criterion now means a gradient norm of at most about `1.4e-6` on
  S^2 instead of `1.4e-3`. On the benchmark problems, where every solve met
  that bound, errors are up to `1.7e-6` rad instead of `1.8e-3` rad;
  `tol = 1e-6` restores the earlier output.
- Stored benchmark numerical values changed: **No**.
- H1, H2 and H3 verdicts and the null-check outcomes changed: **No**, also under
  the `tol = 1e-12` sensitivity rerun.
- Interpretation of stored values below about `1e-3` rad changed: **Yes**
  (F03).
- Description of the preregistration's provenance changed: **Yes** (F04).
- H2 interpretation narrowed: **Yes** (F05).
- Documented SciPy requirement changed: **Yes**, to match `pyproject.toml`
  (F06); the installed requirement itself is unchanged.
- Retraction warranted: **No**.

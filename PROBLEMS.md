# Known Problems, Things We Got Wrong, Things Deferred

A persistent, honest record of mathematical bugs caught and corrected, known
limitations of the current implementation, and work deliberately deferred. This
file complements the per-module validation table in `STATUS.md` and the detailed
caveats in `LIMITATIONS.md`: those describe what the code does and how far it is
trusted; this file is the consolidated self-audit. Modeled on the companion
`TopoGeoML/PROBLEMS.md` and `homology-cliff/PROBLEMS.md`.

## Mathematical bugs caught and corrected (before any tagged release of corrected code)

Each entry names the commit/PR that fixed it; each fix is covered by a regression
test that fails on the pre-fix code.

1. **Sheaf (connection) Laplacian was not PSD and used an inconsistent kernel
   (fixed in PR #13, commit `0da333a`, closes #8).** The earlier construction did
   not yield a positive-semidefinite operator and its kernel did not match the
   transport-consistent agreement space. The corrected operator is
   `L = δ^T δ` for the cellular-sheaf coboundary `δ`; it is now verified PSD and
   verified against an *independently constructed* coboundary (the test does not
   reuse the builder's block formula), with the kernel dimension and
   transport-consistency asserted as invariants. This is the bug that motivated the
   project-wide rule never to archive/DOI a snapshot carrying an unproven Laplacian.

2. **Persistence diagrams mixed homology dimensions, contaminating divergence
   (fixed in PR #21, commit `1aa3d40`, closes #10).** `compute_persistence`
   flattened ripser's per-dimension diagrams via `np.vstack` into a `(k, 2)` array,
   discarding the homology-dimension label, so H0 and H1 bars became
   indistinguishable; `track_divergence` then fed a mixed H0+H1 pool to
   `persim.bottleneck`, silently contaminating the H0 divergence with loop (H1)
   structure. The stored diagram now carries the dimension as a third column
   (`(k, 3)`) with a `diagram_for_dim` accessor, and `track_divergence` compares
   H0-vs-H0 only. Verified against an independent scipy minimum-spanning-tree
   reconstruction of the H0 diagram (finite H0 deaths equal MST edge weights), plus
   a regression showing an H1-only change no longer leaks into the H0 divergence.

3. **Cohomology silently formed meaningless partial holonomy on incomplete cocycles
   (fixed in PR #20, commit `71be9bf`).** A cycle's holonomy is the ordered product
   over all its edges; if any edge transport map was missing, the earlier code
   formed a partial product and could report a false (in)consistency. `compute_h1`
   now raises `IncompleteCocycleError` naming the missing edge, and a
   disconnected-graph transport request is re-raised as an explicit domain error
   rather than producing a silent wrong answer. The later dev5 correction below
   supersedes the interpretation of `compute_h1` itself: the scalar is a
   cycle-basis holonomy defect, not a canonical H^1 norm.

4. **Schild's ladder implementation was incorrect; test tolerances were too loose
   (fixed in commit `21e782c`).** The parallel-transport approximation was corrected
   and tolerances tightened so the tests validate against analytic transport rather
   than passing trivially. The measured residuals are now documented honestly in
   `LIMITATIONS.md` (and summarized below) rather than asserted as exact.

5. **Karcher-mean weights were not honored (fixed in commit `f0bfa80`).** A
   pre-public audit pass found the weighted Karcher mean ignored caller-supplied
   weights; it now honors them, with a dead import removed and docs corrected.

6. **Optimizer moment accumulators were carried across iterates by tangent
   projection, not parallel transport (fixed in commit `5371c20`).** Adam's first
   moment was projected onto each new tangent space and the SGD momentum velocity
   was not moved at all. Projection loses the component normal to the new tangent
   space, and an untransported moment is not tangent at the new iterate. Both
   accumulators are now parallel-transported, with tangent projection kept only as
   a fallback for metrics lacking parallel transport.

7. **The cycle-basis holonomy defect was misidentified as an H^1 cohomology norm
   (corrected in the dev5 scientific correction).** The implementation computes
   `max ||Hol(gamma)-I||_F` over the specific `networkx.cycle_basis` returned for
   the undirected graph. Exact zero has a valid flatness interpretation under
   explicit connectedness, complete invertible underlying-edge transport,
   reciprocal opposite registrations when both directions are supplied, and
   complete evaluation of every cycle emitted by `networkx.cycle_basis`
   (justified for the emission order of the NetworkX Paton implementation
   exercised here, not for arbitrary cycle bases; the emitted list is not in
   general one spanning tree's fundamental-cycle basis), but the nonzero
   magnitude depends on the chosen cycle basis and
   is not invariant under arbitrary invertible changes of frame. The historical
   benchmark numbers are retained; H2 is reinterpreted as a correlation with this
   fixed cycle-basis defect rather than with a canonical cohomology norm.

8. **A tangent-vector transport operator was incorrectly promoted into an
   invertible point action (corrected in the dev5 scientific correction).**
   `compute_transport_matrix` assembled an ambient square array by projecting
   ambient basis vectors into a tangent space and transporting those tangent
   vectors. `register_transport_from_points` then registered that array as the
   same point-valued morphism used by aggregation. For an embedded manifold such
   as S^2, the exact projector extension is rank-deficient in ambient coordinates
   and does not define the required point action. The tangent ladder utilities
   remain valid at their tested scope; the from-points integration is withdrawn
   and now fails closed. Direct registered matrices are supported only under an
   explicit invertible manifold point-action contract, including reciprocal
   opposite arrows when both directions are stored. The ambient tangent matrix
   helper is supported only for vector-shaped point representations. The
   preregistered benchmark is unaffected because it registers explicit SO(3)
   rotations directly.

9. **`schild_ladder` was not Schild's ladder, and its step schedule never
   reached the endpoint (corrected in this release).** The function body was
   the pole-ladder midpoint reflection, so the two "different" ladders were the
   same algorithm, and it advanced `direction / n_rungs` while re-reading
   `direction` from the current base every rung, so for `n_rungs > 1` it
   stopped short of `end_point` and returned a vector that was not tangent
   there. The recorded "does not converge as rungs increase" behaviour was this
   bug, not a property of Schild's ladder. The function now implements the
   geodesic parallelogram of Kheyfets, Miller and Newton (2000), with the
   tangent vector rescaled by `1 / n_rungs` so the first-order scheme actually
   converges (Guigui and Pennec, 2021). Measured on the S^2 test hop: error
   halves with every doubling of rungs from 1 to 64 (ratio 2.00), endpoint
   tangency to round-off, norm preserved to 1e-6. `pole_ladder` is exact on the
   same hop and remains the recommended method; the accuracy ordering asserted
   by the tests is unchanged. The corrected `pole_ladder` step schedule from the
   same audit (`log(end, current_base) / (n_rungs - k)`) is applied here too.
   The preregistered benchmark is unaffected because it registers explicit
   SO(3) rotations directly and never calls either ladder.

## Known mathematical limitations (not bugs — documented honestly)

From `LIMITATIONS.md`; recorded here so absence of a guarantee is never mistaken
for an oversight:

- **Pole ladder exactness is established only on S^2.** With the per-rung
  step schedule (each rung re-expresses the remaining geodesic in the current
  tangent space) `pole_ladder` matches geomstats' analytic tangent-vector
  parallel transport to round-off on a 60-degree S^2 hop for every rung count;
  the earlier "plateau at ~0.02 / drifts off the tangent plane" behaviour was a
  step-schedule bug, not an inherent limitation. S^2 is a symmetric space, where
  the pole ladder is known to be exact; no accuracy claim is made for
  non-symmetric manifolds. Schild's ladder is a first-order scheme: its error
  halves with every doubling of `n_rungs` (about 0.214 -> 0.00327 from 1 to 64 rungs
  on a 30-degree test hop with a unit off-plane vector) but it is not exact where the pole ladder is.
- **Persistent-homology Betti numbers are degenerate under the default filtration.**
  `compute_persistence` counts only bars dying at infinity, so under the default
  `thresh=inf` (used by `track_divergence`) the Vietoris-Rips complex is fully
  connected and `betti_0 == 1`, `betti_1 == 0` regardless of underlying topology.
  The informative signal lives in `max_persistence` / the finite bars; meaningful
  component counts require a finite `max_edge_length` between the intra- and
  inter-cluster scales.
- **Sheaf Laplacian assumes uniform stalk dimension.**
- **The cycle-basis holonomy-defect magnitude is noncanonical.** It depends on the
  selected basis and, outside orthogonal/unitary representations, on the chosen
  frame. The finite consistency threshold is therefore representation-dependent.
- **A zero cycle-basis defect is not a completeness check.** Bridges lie on no
  cycles, and disconnected components require separate handling.

## Not implemented at this stage (by design)

From `STATUS.md`; stated plainly so the scope is unambiguous:

- No federated training loop with real neural networks exists yet.
- No differential privacy mechanism is implemented (Opacus / TenSEAL are listed as
  optional dependencies for future work but are integrated into no code path).
- No formal convergence analysis or proofs exist for the groupoid aggregation
  method.
- The Riemannian optimizers are validated for **descent to a known target on
  S^2** (SGD, momentum SGD, Adam) with parallel-transported moment accumulators,
  but no general convergence-rate analysis exists and the module is not yet
  integrated into the aggregation pipeline. (100% line+branch coverage is not the
  same as validation; see `STATUS.md`.)
- No generic construction is currently supplied that turns Levi-Civita tangent
  parallel transport on an arbitrary Riemannian manifold into the invertible point
  action required by the current point-valued aggregator.

## Honest epistemic statement

GROUPOID is a pre-alpha research prototype. Its validated claims are deliberately
narrow. Groupoid matrix composition, the Karcher mean, the sheaf Laplacian, the
cycle-basis holonomy defect at its stated scope, and tangent-vector transport
utilities have direct tests against algebraic, analytic, or constructed reference
cases. The point-valued aggregation pipeline is supported for explicit point
actions that satisfy its runtime manifold contract; the current S^2 evidence uses
SO(3) rotations. The central scientific hypothesis remains supported only by the
preregistered synthetic benchmark and remains **unvalidated on real federated
learning tasks**. Historical preregistration and result artifacts are preserved;
current interpretation corrections are documented in `CORRECTION_NOTICE.md`.

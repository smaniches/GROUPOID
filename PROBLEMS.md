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
   rather than producing a silent wrong answer.

4. **Schild's ladder implementation was incorrect; test tolerances were too loose
   (fixed in commit `21e782c`).** The parallel-transport approximation was corrected
   and tolerances tightened so the tests validate against analytic transport rather
   than passing trivially. The measured residuals are now documented honestly in
   `LIMITATIONS.md` (and summarized below) rather than asserted as exact.

5. **Karcher-mean weights were not honored (fixed in commit `f0bfa80`).** A
   pre-public audit pass found the weighted Karcher mean ignored caller-supplied
   weights; it now honors them, with a dead import removed and docs corrected.

## Known mathematical limitations (not bugs — documented honestly)

From `LIMITATIONS.md`; recorded here so absence of a guarantee is never mistaken
for an oversight:

- **Pole ladder does not converge to zero error.** It matches geomstats' analytic
  parallel transport closely in direction (cosine > 0.999 on a 60-degree S^2 hop)
  but plateaus at a small residual (~0.02 here) and drifts slightly off the
  endpoint tangent plane as rungs increase. Schild's ladder is markedly coarser
  (cosine ~0.98 on the same hop).
- **Persistent-homology Betti numbers are degenerate under the default filtration.**
  `compute_persistence` counts only bars dying at infinity, so under the default
  `thresh=inf` (used by `track_divergence`) the Vietoris-Rips complex is fully
  connected and `betti_0 == 1`, `betti_1 == 0` regardless of underlying topology.
  The informative signal lives in `max_persistence` / the finite bars; meaningful
  component counts require a finite `max_edge_length` between the intra- and
  inter-cluster scales.
- **Sheaf Laplacian assumes uniform stalk dimension.**
- **H^1 uses the cycle basis of the undirected graph.** Correct for 1-dimensional
  nerve complexes; does not generalize to higher-dimensional simplicial complexes
  without modification.

## Not implemented at this stage (by design)

From `STATUS.md`; stated plainly so the scope is unambiguous:

- No federated training loop with real neural networks exists yet.
- No differential privacy mechanism is implemented (Opacus / TenSEAL are listed as
  optional dependencies for future work but are integrated into no code path).
- No formal convergence analysis or proofs exist for the groupoid aggregation
  method.
- The Riemannian optimizers are **smoke-tested only**: their steps stay on the
  manifold and the curvature-adaptive learning rate behaves sensibly, but core
  descent/convergence behavior is not validated, and the module is not yet
  integrated into the aggregation pipeline. (100% line+branch coverage is not the
  same as validation — see `STATUS.md`.)

## Honest epistemic statement

GROUPOID is a pre-alpha research prototype. Its tested mathematical primitives
(groupoid composition, Karcher mean, H^1 cohomology, the sheaf Laplacian) are
validated against independent ground truth and covered by a hard, CI-enforced 100%
line+branch test gate; the parallel-transport and persistence modules are validated
against analytic / known-topology references but not yet integrated into the main
pipeline. The central scientific hypothesis — that groupoid transport + cohomological
consistency + the intrinsic Karcher mean improve on Euclidean averaging for
heterogeneous federated clients — is **not yet validated**, and `README.md` and
`STATUS.md` say so. The mathematical bugs above were caught before the corrected code was
archived, and each is guarded by a regression test. The repository aims to be exact
about what has and has not been demonstrated.

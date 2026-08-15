# Scientific correction notice

This notice documents two mathematical-semantic corrections prepared against
frozen repository state `d4ab3f4c8439d5102c75a59f176965a7957c620a`.
Historical preregistration and result artifacts are preserved unchanged.

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

## Provenance and historical record

The following historical artifacts are intentionally not rewritten:

- `experiments/PREREGISTRATION.md`
- `experiments/results.json`
- `experiments/RESULTS.md`
- `experiments/run_benchmark.py`
- `experiments/analyze.py`

Current explanatory surfaces may quote or describe those artifacts with the
corrected construct terminology, but historical wording remains part of the
original record.

Git commit ancestry establishes that the preregistration commit precedes the
runner and result commits in the preserved branch history. Git commit history
alone does not independently establish when a commit was first pushed to a
public remote relative to local computation. Current documentation therefore
distinguishes commit-order evidence from the author-reported push chronology.

## Effect on conclusions

- Stored benchmark numerical values changed: **No**.
- H1 transport-benefit conclusion changed: **No**.
- H2 stored rho or permutation p-value changed: **No**.
- H2 interpretation changed: **Yes**.
- H3 ablation conclusion changed: **No**.
- General finite-threshold consistency semantics changed: **Yes**.
- `register_transport_from_points` supported-integration claim withdrawn: **Yes**.
- Tangent-vector ladder validation withdrawn: **No**.
- Retraction warranted: **No**.

This is a correction of mathematical construct identity and transport
representation semantics. It preserves the empirical observations that remain
supported while removing claims that the implementation does not establish.

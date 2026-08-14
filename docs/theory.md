# Mathematical Theory

This page provides the mathematical background for GROUPOID. We assume
familiarity with basic differential geometry and category theory.

## Transport Groupoids and Point Actions

A **groupoid** is a category in which every morphism is invertible. In the
current aggregation pipeline, client parameters are represented as points of a
manifold embedded or coordinatized in a vector space. A registered transport
matrix is therefore required to define an invertible action on those represented
points, not merely an arbitrary square linear map.

Given a network of clients, let a directed graph `G = (V, E)` carry a matrix
`T_ij` on each registered edge. The aggregation path uses these matrices only
where their forward and inverse actions preserve the manifold-valued points
actually transported. The preregistered S^2 benchmark uses explicit SO(3)
rotations, which satisfy this contract.

The algebraic groupoid relations are:

- **Identity**: `T_ii = Id`
- **Inverse**: reverse traversal uses `T_ij^{-1}`. If both orientations of one
  underlying edge are explicitly registered, they must be mutual numerical
  inverses; registration and holonomy evaluation reject a conflicting pair.
- **Composition**: path actions compose by matrix multiplication

The `Morphism` container implements matrix composition and inversion. Its
construction alone does not prove that a matrix is a geometrically valid point
action for an arbitrary manifold representation.

## Karcher Mean on Riemannian Manifolds

When model parameters live on a Riemannian manifold `(M, g)`, the standard
Euclidean average need not remain on the manifold. GROUPOID uses the Karcher
mean (Frechet mean), defined as a minimizer of the weighted sum of squared
geodesic distances:

\[
\bar{x} = \arg\min_{y \in M} \sum_{i=1}^{n} w_i d_g(y, x_i)^2.
\]

The implementation delegates this computation to geomstats.

## Cycle-Basis Holonomy Defect

Earlier releases described the transport diagnostic as a first-cohomology
`H^1` norm. That terminology was too strong. The implementation computes a
specific cycle-holonomy statistic.

Let

\[
\mathcal B = \operatorname{cycle\_basis}(G_{\mathrm{undirected}})
\]

be the basis returned by NetworkX. For a cycle `gamma`, define the ordered
holonomy

\[
\operatorname{Hol}_T(\gamma)
= T_{v_n v_1} T_{v_{n-1}v_n}\cdots T_{v_1v_2}.
\]

The implemented scalar is

\[
D_{\mathcal B}(T)
= \max_{\gamma \in \mathcal B}
\left\|\operatorname{Hol}_T(\gamma)-I\right\|_F.
\]

The primary API name is `cycle_basis_holonomy_defect`.

### What exact zero means

For a connected graph whose every underlying undirected edge carries one
invertible connection map, represented either by one orientation or by a
reciprocal pair, the spanning-tree fundamental cycles constructed by the
NetworkX/Paton procedure are sufficient to test flatness. Choose a root and
define a frame at each vertex by composing transports along the unique tree
path. Identity holonomy on every fundamental cycle forces each non-tree edge to
agree with the tree-induced transport. Consequently every closed loop has
identity holonomy.

Under those assumptions,

\[
D_{\mathcal B}(T)=0
\quad\Longleftrightarrow\quad
\text{the represented graph connection is flat}.
\]

This is a statement about exact flatness. A zero cycle defect does not certify
that the graph is connected, that bridge-edge transports were supplied, or that
the matrices define valid actions on manifold points.

### What the magnitude does not mean

The nonzero value is not a canonical cohomology norm.

- Different valid cycle bases can produce different maxima for the same graph
  and transport assignment.
- Under a vertexwise frame change `T'_uv = G_v T_uv G_u^{-1}`, loop holonomy
  changes by similarity. Exact identity is preserved, but the Frobenius distance
  to identity is not preserved under a general invertible similarity transform.
- In an orthogonal representation, orthogonal conjugation and reversal preserve
  the Frobenius magnitude. This is the domain used by the synthetic SO(3)
  benchmark.

Accordingly, the finite `consistency_threshold` in the aggregator is a
representation-dependent operational threshold. The legacy `is_consistent`
field means only that the measured defect is below that configured threshold.

### Incomplete cycle transport

A cycle holonomy requires every edge of that cycle. If a selected basis cycle
contains an edge without a transport map in either direction,
`cycle_basis_holonomy_defect` raises `IncompleteCocycleError`. If both
orientations are supplied for one underlying edge but are not mutual numerical
inverses, it raises `NonReciprocalTransportError` before reducing the graph to
its undirected cycle basis. Bridges are not part of any cycle and are therefore
outside this diagnostic; the aggregation path checks path availability
separately.

## Tangent Parallel Transport Is Not a Point Action

Levi-Civita parallel transport maps tangent vectors:

\[
L_{p\to q}:T_pM\rightarrow T_qM.
\]

The ladder utilities in `groupoid.transport` approximate this tangent-vector
map. `compute_tangent_transport_matrix` forms an ambient coordinate array by
projecting each ambient coordinate vector into `T_pM`, transporting the tangent
vector, and storing the transported vector as a column. This helper is defined
only for vector-shaped point representations, where both points are 1D
coordinate arrays of the same shape. Matrix-valued or otherwise structured
points must use the ladder functions directly on native-shaped tangent objects.

For an embedded manifold, the exact ambient projector extension has the form

\[
A_{p\to q}=L_{p\to q}P_p.
\]

When the tangent dimension is smaller than the ambient dimension, this operator
is rank-deficient. On `S^2` in `R^3`, `P_p p = 0`, so the exact operator cannot
be an invertible 3 by 3 point action. Numerical ladder error can perturb that
rank, but approximation error does not supply the missing geometric contract.

For this reason `register_transport_from_points` is withdrawn from the supported
point-valued aggregation path and now fails closed. Tangent-vector transport
remains available independently. A generic construction of a point action from
tangent parallel transport on an arbitrary Riemannian manifold is not claimed.

## Sheaf Theory and Restriction Maps

A **cellular sheaf** `F` on `G` assigns a vector space to each node and linear
restriction maps to incidences. A global section is an assignment whose
restrictions agree on shared edges.

The implemented sheaf Laplacian is

\[
L_F = \delta^T\delta,
\]

so it is positive semidefinite and its kernel is the space of sections satisfying
the represented restriction constraints.

## Connection to the Current Prototype

| Mathematical object | Current implementation meaning |
|---|---|
| Node | Client |
| Explicit point action `T_ij` | Caller-supplied invertible map used to move represented manifold points |
| Karcher mean | Geometric aggregation of transported points |
| `D_B(T) = 0` under stated assumptions | Flat represented transport connection |
| `D_B(T) > 0` | Nontrivial holonomy detected on at least one selected basis cycle |
| Finite defect threshold | Representation-dependent operational diagnostic |
| Tangent parallel transport | Separate tangent-vector utility, not automatically a point action |
| Global sheaf section | Assignment satisfying the sheaf restriction constraints |

See `CORRECTION_NOTICE.md` for the correction history and the exact distinction
between the historical `H^1` terminology and the implemented statistic.

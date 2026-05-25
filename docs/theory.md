# Mathematical Theory

This page provides the mathematical background for GROUPOID. We assume
familiarity with basic differential geometry and category theory.

## Transport Groupoids

A **groupoid** is a category in which every morphism is invertible. In the
federated learning context, the transport groupoid captures how model
parameters are related across clients.

Given a network of $n$ clients, define a directed graph $G = (V, E)$ where
each node $v_i \in V$ represents a client and each edge $(v_i, v_j) \in E$
carries a **transport map** $T_{ij}$ that translates parameters from client
$i$'s space to client $j$'s space.

The groupoid axioms require:

- **Identity**: $T_{ii} = \text{Id}$ for all $i$
- **Inverse**: $T_{ji} = T_{ij}^{-1}$ for all $(i, j) \in E$
- **Composition**: $T_{ik} = T_{jk} \circ T_{ij}$ whenever $(i, j)$ and $(j, k)$ are in $E$

This structure generalizes the parallel transport groupoid of a Riemannian
manifold (see Kobayashi and Nomizu, *Foundations of Differential Geometry*,
1963) to the discrete setting of federated networks.

## Karcher Mean on Riemannian Manifolds

When model parameters live on a Riemannian manifold $(M, g)$, the standard
Euclidean average is not well-defined. Instead, we use the **Karcher mean**
(also called the Frechet mean), defined as the point minimizing the sum of
squared geodesic distances (Karcher, 1977; Kendall, 1990):

$$
\bar{x} = \arg\min_{y \in M} \sum_{i=1}^{n} w_i \, d_g(y, x_i)^2
$$

where $d_g$ is the geodesic distance and $w_i$ are non-negative weights
summing to 1.

The Karcher mean is computed iteratively via Riemannian gradient descent
(Pennec, 2006; Moakher, 2005):

1. Initialize $\bar{x}_0$ (e.g., to $x_1$)
2. Compute $v = \sum_i w_i \, \text{Log}_{\bar{x}_k}(x_i)$
3. Update $\bar{x}_{k+1} = \text{Exp}_{\bar{x}_k}(\epsilon \, v)$
4. Repeat until $\|v\| < \text{tol}$

Here $\text{Log}$ and $\text{Exp}$ denote the Riemannian logarithmic and
exponential maps. Convergence is guaranteed when all points lie within a
geodesic ball of radius less than $\pi / (2\sqrt{K})$ where $K$ is the
maximum sectional curvature (Afsari, 2011).

## First Cohomology and Global Consistency

The **first cohomology** $H^1$ of a transport cocycle measures the obstruction
to global consistency (see Bott and Tu, *Differential Forms in Algebraic
Topology*, 1982). Given transport maps $\{T_{ij}\}$ on a graph $G$:

- A **cocycle** is an assignment of transport maps to edges satisfying
  $T_{ik} = T_{jk} \circ T_{ij}$ around every triangle.
- A **coboundary** is a cocycle that can be written as $T_{ij} = g_j \circ g_i^{-1}$
  for some gauge transformations $\{g_i\}$ at each node.
- $H^1 = 0$ if and only if every cocycle is a coboundary.

In practice, we compute $H^1$ by checking the **holonomy** around each cycle
in a cycle basis of $G$:

$$
\text{Hol}(\gamma) = T_{v_n v_1} \circ T_{v_{n-1} v_n} \circ \cdots \circ T_{v_1 v_2}
$$

If $\text{Hol}(\gamma) = \text{Id}$ for all basis cycles $\gamma$, then
$H^1 = 0$ and the local models are globally consistent.

## Sheaf Theory and Restriction Maps

A **cellular sheaf** $\mathcal{F}$ on $G$ assigns (Hansen and Ghrist, 2019):

- A vector space $\mathcal{F}(v)$ to each node $v$ (the stalk)
- A linear map $\mathcal{F}_{v \to e} : \mathcal{F}(v) \to \mathcal{F}(e)$
  to each incidence (the restriction map)

A **global section** is an assignment $s(v) \in \mathcal{F}(v)$ for each node
such that the restriction maps agree on shared edges:

$$
\mathcal{F}_{u \to e}(s(u)) = \mathcal{F}_{v \to e}(s(v))
$$

for every edge $e = (u, v)$.

The **sheaf Laplacian** $L_{\mathcal{F}}$ generalizes the graph Laplacian and
its kernel equals the space of global sections. A nonzero kernel indicates
that the local data can be consistently glued into a global model. The
smallest nonzero eigenvalue (spectral gap) controls the convergence rate of
sheaf diffusion toward consensus (Hansen and Ghrist, 2019).

## Connection to Federated Learning

| Mathematical Concept | Federated Learning Analogue |
|---|---|
| Node $v_i$ | Client $i$ |
| Stalk $\mathcal{F}(v_i)$ | Client $i$'s parameter space |
| Transport map $T_{ij}$ | Parallel transport between parameter spaces |
| Karcher mean | Geometric model aggregation |
| $H^1 = 0$ | Local models are globally consistent |
| $H^1 \neq 0$ | Obstruction to consistent aggregation |
| Global section | A successfully aggregated global model |
| Spectral gap of $L_{\mathcal{F}}$ | Convergence rate of federation |

The connection between federated averaging and Riemannian optimization was
explored by (Li et al., 2020) in the context of heterogeneous data. Our
framework extends this by providing cohomological diagnostics (H^1) that
can detect irreconcilable model divergence before it degrades global
performance.

## References

- Afsari, B. (2011). Riemannian $L^p$ center of mass: existence, uniqueness,
  and convexity. *Proceedings of the AMS*, 139(2), 655-673.
- Bott, R., Tu, L.W. (1982). *Differential Forms in Algebraic Topology*.
  Springer.
- Hansen, J., Ghrist, R. (2019). Toward a spectral theory of cellular sheaves.
  *Journal of Applied and Computational Topology*, 3(4), 315-358.
- Karcher, H. (1977). Riemannian center of mass and mollifier smoothing.
  *Communications on Pure and Applied Mathematics*, 30(5), 509-541.
- Kobayashi, S., Nomizu, K. (1963). *Foundations of Differential Geometry,
  Vol. 1*. Wiley.
- Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., Smith, V.
  (2020). Federated optimization in heterogeneous networks. *MLSys*.
- McMahan, B., Moore, E., Ramage, D., Hampson, S., y Arcas, B.A. (2017).
  Communication-efficient learning of deep networks from decentralized data.
  *AISTATS*.
- Moakher, M. (2005). A differential geometric approach to the geometric mean
  of symmetric positive-definite matrices. *SIAM Journal on Matrix Analysis
  and Applications*, 26(3), 735-747.
- Pennec, X. (2006). Intrinsic statistics on Riemannian manifolds: basic tools
  for geometric measurements. *Journal of Mathematical Imaging and Vision*,
  25(1), 127-154.

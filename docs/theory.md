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

## Karcher Mean on Riemannian Manifolds

When model parameters live on a Riemannian manifold $(M, g)$, the standard
Euclidean average is not well-defined. Instead, we use the **Karcher mean**
(also called the Frechet mean), defined as the point minimizing the sum of
squared geodesic distances:

$$
\bar{x} = \arg\min_{y \in M} \sum_{i=1}^{n} w_i \, d_g(y, x_i)^2
$$

where $d_g$ is the geodesic distance and $w_i$ are non-negative weights
summing to 1.

The Karcher mean is computed iteratively:

1. Initialize $\bar{x}_0$ (e.g., to $x_1$)
2. Compute $v = \sum_i w_i \, \text{Log}_{\bar{x}_k}(x_i)$
3. Update $\bar{x}_{k+1} = \text{Exp}_{\bar{x}_k}(\epsilon \, v)$
4. Repeat until $\|v\| < \text{tol}$

Here $\text{Log}$ and $\text{Exp}$ denote the Riemannian logarithmic and
exponential maps.

## First Cohomology and Global Consistency

The **first cohomology** $H^1$ of a transport cocycle measures the obstruction
to global consistency. Given transport maps $\{T_{ij}\}$ on a graph $G$:

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

A **cellular sheaf** $\mathcal{F}$ on $G$ assigns:

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
that the local data can be consistently glued into a global model.

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

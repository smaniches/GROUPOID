"""Federated aggregation via groupoid transport and Karcher mean.

This module ties together the transport groupoid, sheaf consistency,
and Riemannian aggregation into a complete federated learning round.
The key insight is that naive parameter averaging fails when client
models live on different tangent spaces. Instead, we:

1. Transport all client parameters to a common base point via the
   groupoid morphisms.
2. Check cohomological consistency (H^1) to detect irreconcilable
   disagreements before aggregation.
3. Compute the Karcher mean on the manifold of the transported
   parameters.
4. Transport the aggregated model back to each client's local frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from loguru import logger

from groupoid.cohomology import compute_h1
from groupoid.groupoid import Morphism, compose, inverse
from groupoid.manifold import karcher_mean


@dataclass
class FederatedRound:
    """Result of a single federated aggregation round."""

    global_params: np.ndarray
    local_updates: dict[str, np.ndarray]
    h1_norm: float
    is_consistent: bool
    transport_residuals: dict[str, float]
    round_idx: int = 0


@dataclass
class TransportGroupoidAggregator:
    """Federated aggregator using groupoid transport and Riemannian geometry.

    Given a network of clients with local model parameters on a Riemannian
    manifold, this aggregator:
    - Uses parallel transport (groupoid morphisms) to align parameters
    - Checks cohomological consistency before aggregation
    - Computes intrinsic Karcher mean on the parameter manifold
    - Distributes the result back via inverse transport

    Parameters
    ----------
    manifold
        A geomstats manifold instance for the parameter space.
    graph
        Directed graph of the client network.
    base_node
        The node to which all parameters are transported for aggregation.
    consistency_threshold
        Maximum H^1 norm before raising a consistency warning.
    """

    manifold: object
    graph: nx.DiGraph
    base_node: str
    consistency_threshold: float = 1e-6
    morphisms: dict[tuple[str, str], Morphism] = field(default_factory=dict)
    _round_idx: int = field(default=0, init=False)

    def register_transport(self, source: str, target: str, matrix: np.ndarray) -> None:
        """Register a transport map between two clients."""
        self.morphisms[(source, target)] = Morphism(
            source=source,
            target=target,
            transport_map=matrix,
        )
        logger.debug("Registered transport {} -> {}", source, target)

    def _get_transport_to_base(self, node: str) -> np.ndarray | None:
        """Compute the composite transport map from node to base_node.

        Uses shortest path in the graph and composes morphisms along it.
        Returns None if node == base_node (identity transport).
        """
        if node == self.base_node:
            return None

        path = nx.shortest_path(self.graph.to_undirected(), node, self.base_node)
        composite: Morphism | None = None

        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            if (src, tgt) in self.morphisms:
                m = self.morphisms[(src, tgt)]
            elif (tgt, src) in self.morphisms:
                m = inverse(self.morphisms[(tgt, src)])
            else:
                raise ValueError(f"No transport map for edge ({src}, {tgt})")

            composite = m if composite is None else compose(composite, m)

        return composite.transport_map if composite is not None else None

    def check_consistency(self, client_params: dict[str, np.ndarray]) -> float:
        """Check cohomological consistency of current transport maps.

        Returns the H^1 norm. A value near zero indicates that the
        local models can be consistently aggregated.
        """
        transport_maps = {(m.source, m.target): m.transport_map for m in self.morphisms.values()}
        h1 = compute_h1(self.graph, transport_maps)
        logger.info("Cohomological consistency check: H^1 = {:.2e}", h1)
        return h1

    def aggregate(
        self,
        client_params: dict[str, np.ndarray],
        weights: dict[str, float] | None = None,
    ) -> FederatedRound:
        """Run one round of groupoid-aware federated aggregation.

        Parameters
        ----------
        client_params
            Map from client node ID to local model parameters.
        weights
            Optional per-client weights for the Karcher mean.

        Returns
        -------
        FederatedRound
            The aggregation result including global params, consistency
            metrics, and per-client local updates.
        """
        self._round_idx += 1
        logger.info("Starting aggregation round {}", self._round_idx)

        h1 = self.check_consistency(client_params)
        is_consistent = h1 < self.consistency_threshold

        if not is_consistent:
            logger.warning(
                "H^1 = {:.2e} exceeds threshold {:.2e}, "
                "aggregation may produce inconsistent results",
                h1,
                self.consistency_threshold,
            )

        # Transport all client params to base node frame
        transported = {}
        transport_residuals = {}
        for node, params in client_params.items():
            if node == self.base_node:
                transported[node] = params
                transport_residuals[node] = 0.0
            else:
                # _get_transport_to_base returns None only for node == base_node
                # (excluded by this else); a disconnected graph raises
                # NetworkXNoPath. This guard is therefore defensively
                # unreachable.
                T = self._get_transport_to_base(node)
                if T is None:  # pragma: no cover - unreachable defensive guard (see above)
                    raise ValueError(f"No transport path from {node} to {self.base_node}")
                transported_params = T @ params
                transported[node] = transported_params
                transport_residuals[node] = float(
                    np.linalg.norm(T @ T.T - np.eye(T.shape[0]), "fro")
                )

        # Stack transported params and compute Karcher mean
        nodes = sorted(transported.keys())
        param_stack = np.stack([transported[n] for n in nodes])

        if weights is not None:
            w = np.array([weights.get(n, 1.0) for n in nodes])
            w = w / w.sum()
        else:
            w = None

        global_params = karcher_mean(self.manifold, param_stack, weights=w)

        # Transport global params back to each client's local frame
        local_updates = {}
        for node in client_params:
            if node == self.base_node:
                local_updates[node] = global_params
            else:
                # Same defensively-unreachable guard as in the forward
                # transport loop above.
                T = self._get_transport_to_base(node)
                if T is None:  # pragma: no cover - unreachable defensive guard (see above)
                    raise ValueError(f"No transport path from {node} to {self.base_node}")
                T_inv = np.linalg.inv(T)
                local_updates[node] = T_inv @ global_params

        result = FederatedRound(
            global_params=global_params,
            local_updates=local_updates,
            h1_norm=h1,
            is_consistent=is_consistent,
            transport_residuals=transport_residuals,
            round_idx=self._round_idx,
        )

        logger.info(
            "Round {} complete: H^1={:.2e}, consistent={}",
            self._round_idx,
            h1,
            is_consistent,
        )
        return result

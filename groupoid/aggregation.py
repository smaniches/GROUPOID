"""Federated aggregation via explicit point actions and a Karcher mean.

The aggregation pipeline treats ``client_params`` as manifold-valued points.
Caller-supplied transport matrices must therefore define invertible linear
point actions in the chosen representation and must preserve the admissible
manifold points that the pipeline actually transports.  Tangent-vector parallel
transport is a different mathematical object and is not promoted into this
point-action contract automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import numpy.typing as npt
from loguru import logger

from groupoid import persistence as _persistence
from groupoid.cohomology import cycle_basis_holonomy_defect
from groupoid.groupoid import (
    Morphism,
    compose,
    inverse,
    validate_reciprocal_transports,
)
from groupoid.manifold import karcher_mean


class DisconnectedClientGraphError(Exception):
    """Raised when the client graph has no transport path to the base node."""


class InvalidPointTransportError(ValueError):
    """Raised when a registered matrix cannot serve as the required point action."""


class UnsupportedTransportRepresentationError(RuntimeError):
    """Raised when tangent transport is requested as a point-valued morphism."""


@dataclass
class FederatedRound:
    """Result of a single federated aggregation round.

    ``h1_norm``, ``is_consistent``, and ``transport_residuals`` are retained
    as compatibility field names. ``h1_norm`` stores the cycle-basis holonomy
    Frobenius defect, not a canonical H^1 norm. ``is_consistent`` means only
    that this representation-dependent defect is below the configured numerical
    threshold. ``transport_residuals`` stores ``||T T^T - I||_F`` for the
    composite forward maps, so its precise meaning is an orthogonality defect.
    """

    global_params: npt.NDArray[np.float64]
    local_updates: dict[str, npt.NDArray[np.float64]]
    h1_norm: float
    is_consistent: bool
    transport_residuals: dict[str, float]
    round_idx: int = 0
    divergence: _persistence.PersistenceSummary | None = None

    @property
    def cycle_basis_holonomy_defect(self) -> float:
        """Primary name for the scalar stored in the legacy ``h1_norm`` field."""
        return self.h1_norm

    @property
    def passes_consistency_threshold(self) -> bool:
        """Whether the defect is below the configured representation-specific threshold."""
        return self.is_consistent

    @property
    def orthogonality_residuals(self) -> dict[str, float]:
        """Primary name for the legacy ``transport_residuals`` values."""
        return self.transport_residuals


@dataclass
class TransportGroupoidAggregator:
    """Federated aggregator using explicit invertible point actions.

    The current point-valued aggregation path is scientifically supported for
    caller-supplied matrices that act invertibly on the chosen point
    representation and preserve the manifold domain used by the Karcher mean.
    The preregistered S^2 benchmark exercises this contract with explicit
    SO(3) rotations.  Arbitrary square matrices are not thereby validated as
    geometric transport morphisms.

    ``consistency_threshold`` is a threshold on the basis-dependent holonomy
    defect.  A finite threshold decision is not invariant under general
    non-orthogonal changes of frame; it must not be interpreted as a canonical
    cohomological verdict.
    """

    manifold: object
    graph: nx.DiGraph
    base_node: str
    consistency_threshold: float = 1e-6
    track_divergence: bool = False
    morphisms: dict[tuple[str, str], Morphism] = field(default_factory=dict)
    _round_idx: int = field(default=0, init=False)
    _prev_divergence: _persistence.PersistenceSummary | None = field(
        default=None, init=False, repr=False
    )

    def register_transport(
        self, source: str, target: str, matrix: npt.NDArray[np.float64]
    ) -> None:
        """Register an invertible candidate point action between two clients.

        Registration establishes only the algebraic prerequisites that can be
        checked without seeing a point: a finite square matrix with a finite
        inverse. If the opposite orientation is already registered, the two
        matrices must also satisfy the groupoid inverse law numerically. During
        aggregation the actual forward and return actions are required to map
        the transported points back onto ``self.manifold``. Passing these checks
        validates the exercised point actions, not every possible manifold point.
        """
        candidate = np.asarray(matrix, dtype=float)
        if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
            raise InvalidPointTransportError(
                f"transport {source}->{target} must be a square matrix; "
                f"got shape {candidate.shape}"
            )
        if not np.all(np.isfinite(candidate)):
            raise InvalidPointTransportError(
                f"transport {source}->{target} contains non-finite values"
            )
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                candidate_inverse = np.linalg.inv(candidate)
        except np.linalg.LinAlgError as exc:
            raise InvalidPointTransportError(
                f"transport {source}->{target} is singular and cannot define "
                "the inverse point action required by aggregation"
            ) from exc
        if not np.all(np.isfinite(candidate_inverse)):
            raise InvalidPointTransportError(
                f"transport {source}->{target} has a non-finite numerical inverse "
                "and cannot define the return point action required by aggregation"
            )

        reverse = self.morphisms.get((target, source))
        if reverse is not None:
            validate_reciprocal_transports(
                candidate,
                reverse.transport_map,
                source=source,
                target=target,
            )

        self.morphisms[(source, target)] = Morphism(
            source=source,
            target=target,
            transport_map=candidate,
        )
        logger.debug("Registered candidate point action {} -> {}", source, target)

    def register_transport_from_points(
        self,
        source: str,
        target: str,
        source_point: npt.NDArray[np.float64],
        target_point: npt.NDArray[np.float64],
        method: str = "pole",
        n_rungs: int = 2,
    ) -> npt.NDArray[np.float64]:
        """Deprecated compatibility stub; tangent transport is not a point action.

        Earlier releases assembled a square ambient array from transported
        tangent basis vectors and silently registered it as an invertible
        point-valued morphism.  On an embedded manifold such as S^2, the exact
        projector extension of tangent parallel transport is rank-deficient in
        the ambient representation and sends the base point's normal direction
        to zero.  It therefore cannot satisfy the point-action and inverse
        contract used by :meth:`aggregate`.

        Use :meth:`register_transport` with an explicitly justified point action
        (for example, the SO(3) rotations used by the S^2 benchmark).  Tangent-
        vector utilities remain available in :mod:`groupoid.transport`.
        """
        raise UnsupportedTransportRepresentationError(
            "register_transport_from_points() is disabled because tangent-vector "
            "parallel transport does not by itself define the invertible point "
            "action required by this aggregator. Register an explicit, "
            "representation-correct point action instead."
        )

    def _require_manifold_point(
        self,
        point: npt.NDArray[np.float64],
        *,
        context: str,
    ) -> None:
        """Fail closed unless ``point`` belongs to the configured manifold."""
        belongs = getattr(self.manifold, "belongs", None)
        if not callable(belongs):
            raise InvalidPointTransportError(
                "the point-valued aggregation contract requires a manifold.belongs() check"
            )
        if not bool(np.all(np.asarray(belongs(point)))):
            raise InvalidPointTransportError(f"{context} is outside the configured manifold")

    def _apply_point_action(
        self,
        matrix: npt.NDArray[np.float64],
        point: npt.NDArray[np.float64],
        *,
        context: str,
    ) -> npt.NDArray[np.float64]:
        """Apply a registered linear point action and verify its exercised image."""
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                mapped = matrix @ point
        except ValueError as exc:
            raise InvalidPointTransportError(
                f"{context} has incompatible matrix/point dimensions"
            ) from exc
        result = np.asarray(mapped, dtype=float)
        if not np.all(np.isfinite(result)):
            raise InvalidPointTransportError(f"{context} produced non-finite coordinates")
        self._require_manifold_point(result, context=context)
        return result

    def _get_transport_to_base(self, node: str) -> npt.NDArray[np.float64] | None:
        """Compute the composite registered point action from ``node`` to base."""
        if node == self.base_node:
            return None

        try:
            path = nx.shortest_path(self.graph.to_undirected(), node, self.base_node)
        except nx.NetworkXNoPath as exc:
            raise DisconnectedClientGraphError(
                f"client graph is disconnected: no transport path from "
                f"{node} to base {self.base_node}"
            ) from exc
        composite: Morphism | None = None

        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            if (src, tgt) in self.morphisms:
                morphism = self.morphisms[(src, tgt)]
            elif (tgt, src) in self.morphisms:
                morphism = inverse(self.morphisms[(tgt, src)])
            else:
                raise ValueError(f"No transport map for edge ({src}, {tgt})")

            composite = morphism if composite is None else compose(composite, morphism)

        return composite.transport_map if composite is not None else None

    def check_consistency(
        self, client_params: dict[str, npt.NDArray[np.float64]]
    ) -> float:
        """Return the current cycle-basis holonomy defect.

        ``client_params`` is retained in the signature for API compatibility;
        the defect depends only on the graph and registered matrices.  A value
        near zero is not, by itself, a proof that the graph is connected, that
        bridge transports are present, or that the matrices define valid point
        actions.
        """
        transport_maps = {
            (m.source, m.target): m.transport_map for m in self.morphisms.values()
        }
        defect = cycle_basis_holonomy_defect(self.graph, transport_maps)
        logger.info("Cycle-basis holonomy defect = {:.2e}", defect)
        return defect

    def aggregate(
        self,
        client_params: dict[str, npt.NDArray[np.float64]],
        weights: dict[str, float] | None = None,
    ) -> FederatedRound:
        """Run one point-valued aggregation round under the explicit transport contract."""
        self._round_idx += 1
        logger.info("Starting aggregation round {}", self._round_idx)

        for node, params in client_params.items():
            self._require_manifold_point(params, context=f"client {node} input")

        defect = self.check_consistency(client_params)
        passes_threshold = defect < self.consistency_threshold

        if not passes_threshold:
            logger.warning(
                "Cycle-basis holonomy defect {:.2e} exceeds configured threshold {:.2e}; "
                "this is a representation-dependent diagnostic, not a canonical verdict",
                defect,
                self.consistency_threshold,
            )

        transported: dict[str, npt.NDArray[np.float64]] = {}
        orthogonality_residuals: dict[str, float] = {}
        for node, params in client_params.items():
            if node == self.base_node:
                transported[node] = params
                orthogonality_residuals[node] = 0.0
            else:
                transform = self._get_transport_to_base(node)
                if transform is None:  # pragma: no cover
                    raise ValueError(f"No transport path from {node} to {self.base_node}")
                transported[node] = self._apply_point_action(
                    transform,
                    params,
                    context=f"forward point action {node}->{self.base_node}",
                )
                orthogonality_residuals[node] = float(
                    np.linalg.norm(transform @ transform.T - np.eye(transform.shape[0]), "fro")
                )

        nodes = sorted(transported.keys())
        param_stack = np.stack([transported[node] for node in nodes])

        divergence: _persistence.PersistenceSummary | None = None
        if self.track_divergence:
            divergence = _persistence.track_divergence(
                param_stack, previous_summary=self._prev_divergence
            )
            self._prev_divergence = divergence

        if weights is not None:
            normalized_weights = np.array([weights.get(node, 1.0) for node in nodes])
            normalized_weights = normalized_weights / normalized_weights.sum()
        else:
            normalized_weights = None

        global_params = karcher_mean(self.manifold, param_stack, weights=normalized_weights)
        self._require_manifold_point(global_params, context="Karcher mean output")

        local_updates: dict[str, npt.NDArray[np.float64]] = {}
        for node in client_params:
            if node == self.base_node:
                local_updates[node] = global_params
            else:
                transform = self._get_transport_to_base(node)
                if transform is None:  # pragma: no cover
                    raise ValueError(f"No transport path from {node} to {self.base_node}")
                try:
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        inverse_transform = np.linalg.inv(transform)
                except np.linalg.LinAlgError as exc:  # pragma: no cover
                    raise InvalidPointTransportError(
                        f"composite transport for {node}->{self.base_node} became singular"
                    ) from exc
                if not np.all(np.isfinite(inverse_transform)):
                    raise InvalidPointTransportError(
                        f"return point action {self.base_node}->{node} has a non-finite "
                        "numerical inverse"
                    )
                local_updates[node] = self._apply_point_action(
                    inverse_transform,
                    global_params,
                    context=f"return point action {self.base_node}->{node}",
                )

        result = FederatedRound(
            global_params=global_params,
            local_updates=local_updates,
            h1_norm=defect,
            is_consistent=passes_threshold,
            transport_residuals=orthogonality_residuals,
            round_idx=self._round_idx,
            divergence=divergence,
        )

        logger.info(
            "Round {} complete: cycle-basis defect={:.2e}, passes_threshold={}",
            self._round_idx,
            defect,
            passes_threshold,
        )
        return result

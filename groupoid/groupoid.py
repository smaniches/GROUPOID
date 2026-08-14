"""Algebraic transport-map composition for GROUPOID."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict


class Morphism(BaseModel):
    """A matrix-labelled arrow between two nodes.

    This container implements algebraic composition and inversion.  Its mere
    construction does not certify that ``transport_map`` is a geometrically
    valid action on a particular manifold representation.  The point-valued
    aggregation pipeline imposes that stronger contract when matrices are
    registered and exercised.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    target: str
    transport_map: npt.NDArray[np.float64]

    def __repr__(self) -> str:
        return f"Morphism({self.source} -> {self.target})"

    __str__ = __repr__


class CompositionError(Exception):
    """Raised when morphism composition is not defined."""


class NonReciprocalTransportError(ValueError):
    """Raised when opposite registered arrows violate the groupoid inverse law."""


def validate_reciprocal_transports(
    forward: npt.NDArray[np.float64],
    reverse: npt.NDArray[np.float64],
    *,
    source: str,
    target: str,
    rtol: float = 1e-9,
    atol: float = 1e-10,
) -> None:
    """Require two explicitly supplied opposite arrows to be numerical inverses.

    GROUPOID permits storing either orientation of an underlying connection
    edge. If callers explicitly store both ``source -> target`` and
    ``target -> source``, they must represent the same groupoid edge and hence
    satisfy the inverse law in both multiplication orders.
    """
    forward_array = np.asarray(forward, dtype=float)
    reverse_array = np.asarray(reverse, dtype=float)
    if (
        forward_array.ndim != 2
        or reverse_array.ndim != 2
        or forward_array.shape[0] != forward_array.shape[1]
        or reverse_array.shape != forward_array.shape
        or not np.all(np.isfinite(forward_array))
        or not np.all(np.isfinite(reverse_array))
    ):
        raise NonReciprocalTransportError(
            f"opposite transports {source}->{target} and {target}->{source} "
            "must be finite square matrices of the same shape"
        )

    identity = np.eye(forward_array.shape[0])
    with np.errstate(over="ignore", invalid="ignore"):
        reverse_forward = reverse_array @ forward_array
        forward_reverse = forward_array @ reverse_array
    reciprocal = (
        np.all(np.isfinite(reverse_forward))
        and np.all(np.isfinite(forward_reverse))
        and np.allclose(reverse_forward, identity, rtol=rtol, atol=atol)
        and np.allclose(forward_reverse, identity, rtol=rtol, atol=atol)
    )
    if not reciprocal:
        residual = max(
            float(np.linalg.norm(reverse_forward - identity, ord="fro")),
            float(np.linalg.norm(forward_reverse - identity, ord="fro")),
        )
        raise NonReciprocalTransportError(
            f"opposite transports {source}->{target} and {target}->{source} "
            f"are not mutual numerical inverses (max Frobenius residual={residual:.3e})"
        )


def compose(f: Morphism, g: Morphism) -> Morphism:
    """Compose two morphisms ``f`` then ``g``."""
    if f.target != g.source:
        raise CompositionError(f"Cannot compose: {f.target} != {g.source}")

    logger.debug("Composing {} with {}", f, g)
    composed_map = g.transport_map @ f.transport_map
    return Morphism(
        source=f.source,
        target=g.target,
        transport_map=composed_map,
    )


def inverse(f: Morphism) -> Morphism:
    """Return the matrix inverse of a morphism.

    ``numpy.linalg.LinAlgError`` is raised if the stored matrix is singular.
    Geometric validity of the inverse as a manifold point action is a separate
    contract enforced by the aggregation layer when such an action is used.
    """
    logger.debug("Inverting {}", f)
    inv_map = np.linalg.inv(f.transport_map)
    return Morphism(
        source=f.target,
        target=f.source,
        transport_map=inv_map,
    )

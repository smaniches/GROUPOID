"""Groupoid structure for federated learning transport maps."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict


class Morphism(BaseModel):
    """A morphism in the transport groupoid.

    Represents a transport map between two nodes (clients) in the
    federated learning network.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    target: str
    transport_map: npt.NDArray[np.float64]

    def __repr__(self) -> str:
        return f"Morphism({self.source} -> {self.target})"


class CompositionError(Exception):
    """Raised when morphism composition is not defined."""


def compose(f: Morphism, g: Morphism) -> Morphism:
    """Compose two morphisms f and g (f followed by g).

    Requires f.target == g.source (category composition law).

    Parameters
    ----------
    f : Morphism
        First morphism (applied first).
    g : Morphism
        Second morphism (applied second).

    Returns
    -------
    Morphism
        The composed morphism from f.source to g.target.

    Raises
    ------
    CompositionError
        If f.target != g.source.
    """
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
    """Compute the inverse of a morphism.

    Parameters
    ----------
    f : Morphism
        The morphism to invert.

    Returns
    -------
    Morphism
        The inverse morphism from f.target to f.source.
    """
    logger.debug("Inverting {}", f)
    inv_map = np.linalg.inv(f.transport_map)
    return Morphism(
        source=f.target,
        target=f.source,
        transport_map=inv_map,
    )

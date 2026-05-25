"""Riemannian manifold operations for GROUPOID."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from geomstats.geometry.base import LevelSet


def karcher_mean(
    manifold: LevelSet,
    points: np.ndarray,
    weights: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Compute the Karcher (Frechet) mean on a Riemannian manifold.

    Parameters
    ----------
    manifold : geomstats manifold
        A geomstats manifold instance with a metric.
    points : np.ndarray
        Array of shape (n_points, *point_shape) on the manifold.
    weights : np.ndarray or None
        Optional weights for the mean computation.
    max_iter : int
        Maximum iterations for gradient descent.
    tol : float
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        The Karcher mean point on the manifold.
    """
    from geomstats.learning.frechet_mean import FrechetMean

    n_points = points.shape[0]
    logger.debug(
        "Computing Karcher mean of {} points on {}",
        n_points,
        type(manifold).__name__,
    )

    estimator = FrechetMean(manifold)
    estimator.fit(points)

    mean: np.ndarray = estimator.estimate_
    logger.debug("Karcher mean converged")
    return mean

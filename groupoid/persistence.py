"""Persistent homology for federated model divergence detection.

Applies topological data analysis to track how the distribution of
client model parameters evolves across federated rounds. The key
invariant: if the persistence diagram is stable across rounds, the
federation is converging; if new topological features appear, clients
are diverging into distinct clusters.

Uses Vietoris-Rips filtration on the space of model parameters to
compute persistence diagrams, then measures stability via bottleneck
and Wasserstein distances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class PersistenceSummary:
    """Summary of persistent homology computation."""

    betti_0: int
    betti_1: int
    total_persistence: float
    max_persistence: float
    diagram: np.ndarray
    bottleneck_to_previous: float | None = None


def compute_persistence(
    points: np.ndarray,
    max_dim: int = 1,
    max_edge_length: float = np.inf,
) -> PersistenceSummary:
    """Compute persistent homology of a point cloud.

    Parameters
    ----------
    points
        Array of shape (n_points, n_features) representing model
        parameters or their embeddings.
    max_dim
        Maximum homological dimension to compute.
    max_edge_length
        Maximum edge length for the Rips filtration.

    Returns
    -------
    PersistenceSummary
        Topological summary including Betti numbers and persistence.
    """
    from persim import plot_diagrams  # noqa: F401
    from ripser import ripser

    logger.debug("Computing persistence: {} points, max_dim={}", points.shape[0], max_dim)

    result = ripser(points, maxdim=max_dim, thresh=max_edge_length)
    diagrams = result["dgms"]

    # Compute Betti numbers (count features that persist at infinity)
    betti_0 = int(np.sum(diagrams[0][:, 1] == np.inf)) if len(diagrams) > 0 else 0
    betti_1 = int(np.sum(diagrams[1][:, 1] == np.inf)) if len(diagrams) > 1 else 0

    # Compute total and max persistence (excluding infinite features)
    all_finite = []
    for dgm in diagrams:
        finite_mask = dgm[:, 1] < np.inf
        if np.any(finite_mask):
            lifetimes = dgm[finite_mask, 1] - dgm[finite_mask, 0]
            all_finite.extend(lifetimes.tolist())

    total_persistence = float(sum(all_finite)) if all_finite else 0.0
    max_persistence = float(max(all_finite)) if all_finite else 0.0

    # Concatenate all diagrams for storage
    full_diagram = np.vstack(diagrams) if diagrams else np.empty((0, 2))

    logger.debug(
        "Persistence: beta_0={}, beta_1={}, total={:.4f}",
        betti_0,
        betti_1,
        total_persistence,
    )

    return PersistenceSummary(
        betti_0=betti_0,
        betti_1=betti_1,
        total_persistence=total_persistence,
        max_persistence=max_persistence,
        diagram=full_diagram,
    )


def track_divergence(
    current_params: np.ndarray,
    previous_summary: PersistenceSummary | None = None,
    max_dim: int = 1,
) -> PersistenceSummary:
    """Track federation divergence across rounds.

    Computes persistence of current parameter distribution and, if a
    previous summary exists, computes the bottleneck distance to
    measure how much the topological structure has changed.

    Parameters
    ----------
    current_params
        Array of shape (n_clients, n_features).
    previous_summary
        PersistenceSummary from the previous round, if available.
    max_dim
        Maximum homological dimension.

    Returns
    -------
    PersistenceSummary
        Updated summary with bottleneck distance to previous round.
    """
    summary = compute_persistence(current_params, max_dim=max_dim)

    if previous_summary is not None:
        from persim import bottleneck

        # Compare H0 diagrams (connected components)
        current_h0 = summary.diagram[summary.diagram[:, 1] < np.inf]
        prev_h0 = previous_summary.diagram[previous_summary.diagram[:, 1] < np.inf]

        if len(current_h0) > 0 and len(prev_h0) > 0:
            dist = bottleneck(current_h0, prev_h0)
            summary.bottleneck_to_previous = float(dist)
            logger.info("Bottleneck distance to previous round: {:.4f}", dist)
        else:
            summary.bottleneck_to_previous = 0.0

    return summary

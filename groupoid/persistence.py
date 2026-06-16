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
import numpy.typing as npt
from loguru import logger


@dataclass
class PersistenceSummary:
    """Summary of persistent homology computation.

    The ``diagram`` array is dimension-labelled: it has shape ``(k, 3)``
    where column 0 is birth, column 1 is death, and column 2 is the
    homology dimension (0 for H0/connected components, 1 for H1/loops,
    ...). Retaining the dimension label is what lets ``track_divergence``
    compare like-with-like (H0 against H0) instead of pooling features
    from different homology dimensions into one bottleneck computation.
    Use :meth:`diagram_for_dim` to extract a single dimension's bars.
    """

    betti_0: int
    betti_1: int
    total_persistence: float
    max_persistence: float
    diagram: npt.NDArray[np.float64]
    bottleneck_to_previous: float | None = None

    def diagram_for_dim(self, dim: int, *, finite_only: bool = False) -> npt.NDArray[np.float64]:
        """Return the ``(birth, death)`` bars of a single homology dimension.

        Parameters
        ----------
        dim
            Homology dimension to select (0 = H0, 1 = H1, ...).
        finite_only
            If True, drop bars that die at infinity (the essential
            classes). H0 always carries exactly one infinite bar (the
            whole space's single surviving component under ``thresh=inf``);
            excluding it leaves the finite bars, which for H0 are exactly
            the minimum-spanning-tree edge weights of the point cloud.

        Returns
        -------
        np.ndarray
            Array of shape ``(m, 2)`` with columns ``(birth, death)``.
            Empty diagrams return shape ``(0, 2)``.
        """
        if self.diagram.shape[0] == 0:
            return np.empty((0, 2))
        mask = self.diagram[:, 2] == dim
        if finite_only:
            mask = mask & (self.diagram[:, 1] < np.inf)
        # np.asarray pins the return type (numpy fancy-indexing is typed as
        # Any, which would trip mypy's warn_return_any); it is a no-copy view
        # here since the slice is already an ndarray.
        return np.asarray(self.diagram[mask][:, :2])


def compute_persistence(
    points: npt.NDArray[np.float64],
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

    # Concatenate all diagrams for storage, RETAINING the homology
    # dimension as a third column. ripser returns a per-dimension list
    # (diagrams[d] holds dimension d); a bare np.vstack would discard
    # that label and make H0 and H1 bars indistinguishable. We append a
    # dim column so downstream consumers (e.g. track_divergence) can
    # compare like dimensions instead of pooling them.
    labelled = []
    for dim, dgm in enumerate(diagrams):
        if dgm.shape[0] == 0:
            continue
        dim_col = np.full((dgm.shape[0], 1), float(dim))
        labelled.append(np.hstack([dgm, dim_col]))
    full_diagram = np.vstack(labelled) if labelled else np.empty((0, 3))

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
    current_params: npt.NDArray[np.float64],
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

        # Compare H0 diagrams (connected components) ONLY. We select bars
        # whose homology dimension is 0 and which are finite (the single
        # infinite H0 bar -- the whole space's surviving component -- is
        # excluded so persim never has to match infinities). Selecting on
        # the dimension label is essential: without it, H1 (loop) bars
        # leak into this pool and silently contaminate the "H0 divergence"
        # with loop-structure changes. See PersistenceSummary.diagram_for_dim.
        current_h0 = summary.diagram_for_dim(0, finite_only=True)
        prev_h0 = previous_summary.diagram_for_dim(0, finite_only=True)

        if len(current_h0) > 0 and len(prev_h0) > 0:
            dist = bottleneck(current_h0, prev_h0)
            summary.bottleneck_to_previous = float(dist)
            logger.info("Bottleneck distance to previous round: {:.4f}", dist)
        else:
            summary.bottleneck_to_previous = 0.0

    return summary

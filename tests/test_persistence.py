"""Behavioral tests for the persistent-homology divergence tracker.

These tests validate real topological behavior of ``groupoid.persistence``
against point clouds with known structure, not just line execution:

- A circle produces a single prominent 1-cycle, so the maximum H1 lifetime
  (``max_persistence``) is large -- this validates that loop structure is
  actually detected, not merely that the function runs.
- Two well-separated clusters resolve into exactly two connected components
  when the Rips filtration is capped between the intra- and inter-cluster
  scales -- this validates component counting.
- ``track_divergence`` is exercised across all three of its branches: no
  previous summary, a previous summary with non-empty H0 diagrams (bottleneck
  computed), and the degenerate empty-diagram fallback.

Caveat validated and documented: with the default ``thresh=inf`` filtration,
``betti_0``/``betti_1`` count only bars that die at infinity, so on a single
connected Rips complex they are degenerate (``betti_0 == 1``, ``betti_1 == 0``)
regardless of loop structure. The meaningful loop signal therefore lives in
``max_persistence`` (a finite bar), which is what the circle test asserts. The
finite-threshold component test is what exercises ``betti_0 > 1``. This
degeneracy is recorded in LIMITATIONS.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from groupoid.persistence import (
    PersistenceSummary,
    compute_persistence,
    track_divergence,
)


def _circle(n: int = 40, radius: float = 1.0) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return radius * np.column_stack([np.cos(theta), np.sin(theta)])


def _two_clusters(seed: int = 0, spread: float = 0.03) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=[0.0, 0.0], scale=spread, size=(12, 2))
    c2 = rng.normal(loc=[5.0, 5.0], scale=spread, size=(12, 2))
    return np.vstack([c1, c2])


class TestComputePersistence:
    def test_circle_has_prominent_one_cycle(self):
        """A sampled circle of radius 1 has a single dominant H1 loop; its
        lifetime (max finite persistence) must be substantially larger than
        the spacing between adjacent samples. This validates loop detection,
        not just execution."""
        pts = _circle(n=40, radius=1.0)
        summary = compute_persistence(pts, max_dim=1)

        # Adjacent-sample spacing on a unit circle with 40 points.
        spacing = 2.0 * np.sin(np.pi / 40)  # ~0.157
        assert summary.max_persistence > 10.0 * spacing
        # The dominant loop should be a substantial fraction of the diameter.
        assert summary.max_persistence > 1.0
        assert summary.total_persistence >= summary.max_persistence

    def test_two_clusters_resolve_at_finite_threshold(self):
        """With the Rips filtration capped between intra-cluster (~0.03) and
        inter-cluster (~7.07) distance, the two clusters are exactly two
        connected components that never merge, so both survive to infinity:
        betti_0 == 2."""
        pts = _two_clusters(seed=0)
        summary = compute_persistence(pts, max_dim=1, max_edge_length=1.0)
        assert summary.betti_0 == 2
        assert summary.betti_1 == 0

    def test_single_cluster_is_one_component(self):
        """A single connected cluster yields one infinite H0 bar."""
        rng = np.random.default_rng(7)
        pts = rng.normal(scale=0.1, size=(15, 3))
        summary = compute_persistence(pts, max_dim=1)
        assert summary.betti_0 == 1
        assert isinstance(summary, PersistenceSummary)
        assert summary.bottleneck_to_previous is None

    def test_diagram_is_concatenated_finite_and_infinite(self):
        """The stored diagram concatenates H0 and H1 bars and contains both
        finite and (for H0) infinite death times."""
        pts = _circle(n=30)
        summary = compute_persistence(pts, max_dim=1)
        assert summary.diagram.ndim == 2
        assert summary.diagram.shape[1] == 2
        assert np.any(summary.diagram[:, 1] == np.inf)  # the infinite H0 bar
        assert np.any(summary.diagram[:, 1] < np.inf)  # finite bars exist


class TestTrackDivergence:
    def test_first_round_has_no_bottleneck(self):
        """With no previous summary the bottleneck distance is left as None."""
        pts = _two_clusters(seed=1)
        summary = track_divergence(pts)
        assert summary.bottleneck_to_previous is None

    def test_identical_rounds_have_zero_bottleneck(self):
        """Re-running on the same point cloud must yield a bottleneck distance
        of ~0: the topological structure is unchanged. This validates that the
        comparison path actually compares (a non-zero result would mean the
        diagrams were mishandled)."""
        pts = _two_clusters(seed=2)
        first = track_divergence(pts)
        second = track_divergence(pts.copy(), previous_summary=first)
        assert second.bottleneck_to_previous is not None
        assert np.isfinite(second.bottleneck_to_previous)
        assert second.bottleneck_to_previous == pytest.approx(0.0, abs=1e-6)

    def test_translation_invariant_bottleneck(self):
        """A rigid translation leaves all pairwise distances -- and hence the
        persistence diagram -- unchanged, so the bottleneck distance between a
        cloud and its translate is ~0. This is a genuine geometric invariant,
        not a tautology."""
        pts = _two_clusters(seed=3)
        first = track_divergence(pts)
        shifted = track_divergence(pts + 10.0, previous_summary=first)
        assert shifted.bottleneck_to_previous == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.filterwarnings("ignore:The input point cloud has more columns:UserWarning")
    def test_empty_finite_diagram_falls_back_to_zero(self):
        """When the current cloud has no finite H0 bars (a single point has
        only the infinite component), the empty-diagram branch sets the
        bottleneck distance to 0.0 rather than calling persim. (ripser emits a
        benign columns>rows warning for the degenerate single-point input.)"""
        prev = track_divergence(_two_clusters(seed=4))
        single_point = np.array([[0.0, 0.0]])
        result = track_divergence(single_point, previous_summary=prev)
        assert result.bottleneck_to_previous == 0.0

"""Numerical proof that the persistence diagram retains homology-dimension
labels and that ``track_divergence`` compares LIKE dimensions (H0 vs H0).

This file is the regression proof for the dimension-mixing bug (issue #10).
Before the fix, ``compute_persistence`` flattened ripser's per-dimension list
of diagrams into one ``(k, 2)`` array with no dimension column, and
``track_divergence`` selected ALL finite features (H0 *and* H1) and fed that
mixed pool to a single ``persim.bottleneck`` call -- so the "H0 divergence"
was silently contaminated by H1 (loop) structure.

Three claims are proven on point clouds of KNOWN, DISTINCT topology:

(a) The stored diagram retains a dimension label, so H0 and H1 bars are
    separable (``test_diagram_retains_dimension_label``).

(b) ``track_divergence``'s H0 comparison uses ONLY H0 features. This is split
    into an EXACT selection check and a value check verified against an
    INDEPENDENT recomputation:
      - The selected sub-diagram has exactly ``n - 1`` finite bars, all born
        at 0, with NO dimension-1 bars, and its death times equal the edge
        weights of the cloud's minimum spanning tree
        (``test_h0_selection_is_exactly_the_mst``). The MST <-> H0 identity is
        a standard theorem: the finite H0 death times of a Vietoris-Rips
        filtration are exactly the single-linkage merge heights, i.e. the MST
        edge weights. The reference is built from ``scipy``'s MST, never from
        ripser/persim, so it is genuinely independent of the code under test.
      - The bottleneck value reported by the production code equals the
        bottleneck of two INDEPENDENTLY-built (scipy-MST) H0 diagrams
        (``test_h0_bottleneck_matches_independent_mst_reference``).

(c) The bug is gone: a change confined to H1 no longer leaks into the H0
    divergence. Two clouds with IDENTICAL H0 but DIFFERENT H1 (a regular
    polygon, which has one prominent loop, vs. a collinear chain with the same
    nearest-neighbour spacing, which has none) yield an H0 divergence of ~0,
    while the OLD buggy mixed-dimension pool yields a large value dominated by
    the unmatched loop (``test_h1_only_change_does_not_leak_into_h0``).

Tolerance note: ripser emits float32 persistence diagrams, so a value built
from float64 coordinates (the scipy-MST reference) can only agree to the
float32 representational floor (~1e-7 relative), NOT 1e-9. The value check
therefore uses ``rel=1e-6`` -- still ~6 orders of magnitude tighter than the
bug signal (a selection regression moves the value by O(0.1-1); see claim c).
The load-bearing property -- independence of the reference from the code under
test -- is fully preserved: the reference H0 diagram comes from scipy's MST.
"""

from __future__ import annotations

import numpy as np
import pytest
from persim import bottleneck
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from groupoid.persistence import (
    PersistenceSummary,
    compute_persistence,
    track_divergence,
)

# ripser emits float32 diagrams; this is the relative floor for agreement
# between a float32-origin value and a float64 reference. See module docstring.
FLOAT32_REL = 1e-6


# --------------------------------------------------------------------------- #
# Point-cloud constructors with known, distinct topology.
# --------------------------------------------------------------------------- #
def _regular_polygon(n: int = 12, radius: float = 1.0) -> np.ndarray:
    """A regular n-gon: n connected points (H0) and one prominent loop (H1)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return radius * np.column_stack([np.cos(theta), np.sin(theta)])


def _collinear_chain(n: int = 12, spacing: float = 1.0) -> np.ndarray:
    """n collinear points equally spaced: same H0 merge heights as a polygon
    with the same nearest-neighbour ``spacing``, but NO loop (H1 empty)."""
    return np.column_stack([np.arange(n) * spacing, np.zeros(n)])


def _two_clusters(separation: float, seed: int, spread: float = 0.02) -> np.ndarray:
    """Two tight Gaussian blobs separated along x. The inter-cluster distance
    controls the largest finite H0 death (the merge of the two components),
    so different separations give genuinely different H0 diagrams."""
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=[0.0, 0.0], scale=spread, size=(8, 2))
    c2 = rng.normal(loc=[separation, 0.0], scale=spread, size=(8, 2))
    return np.vstack([c1, c2])


def _independent_h0_from_mst(points: np.ndarray) -> np.ndarray:
    """Build the finite H0 persistence diagram INDEPENDENTLY of ripser.

    The finite H0 death times of a Vietoris-Rips filtration are exactly the
    single-linkage merge heights of the point cloud, i.e. the edge weights of
    its Euclidean minimum spanning tree. Every finite H0 class is born at 0
    (each point is its own component at filtration value 0) and dies when the
    edge that merges its component into another appears. This routine computes
    that diagram from scipy's MST -- no ripser, no persim -- giving an
    independent reference for the production H0 selection.

    Returns an ``(n - 1, 2)`` array of ``(birth=0, death=edge_weight)`` bars.
    """
    dist = squareform(pdist(points))
    mst = minimum_spanning_tree(dist).toarray()
    weights = mst[mst > 0.0]
    births = np.zeros_like(weights)
    return np.column_stack([births, weights])


# --------------------------------------------------------------------------- #
# (a) The diagram retains a separable homology-dimension label.
# --------------------------------------------------------------------------- #
class TestDimensionLabelRetained:
    def test_diagram_retains_dimension_label(self):
        """A noisy circle has KNOWN topology: one persistent connected
        component (H0) and one persistent loop (H1). The stored diagram must
        be (k, 3) with column 2 the homology dimension, and H0/H1 must be
        separable by that label -- the loop's H1 lifetime is large, the
        infinite bar is the single H0 component."""
        rng = np.random.default_rng(0)
        n = 60
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        circle = np.column_stack([np.cos(theta), np.sin(theta)])
        circle = circle + rng.normal(scale=0.02, size=circle.shape)

        summary = compute_persistence(circle, max_dim=1)

        # The diagram carries the dimension column.
        assert summary.diagram.ndim == 2
        assert summary.diagram.shape[1] == 3

        h0 = summary.diagram_for_dim(0)
        h1 = summary.diagram_for_dim(1)
        # H0 and H1 are separable and non-empty.
        assert h0.shape[0] > 0
        assert h1.shape[0] == 1  # exactly one prominent loop on a circle
        # Exactly one infinite H0 bar (the surviving component); no infinite H1.
        assert int(np.sum(h0[:, 1] == np.inf)) == 1
        assert np.all(h1[:, 1] < np.inf)
        # The loop is genuinely persistent (lifetime is a large fraction of
        # the unit diameter), proving the H1 label is meaningful, not noise.
        h1_lifetime = float((h1[:, 1] - h1[:, 0]).max())
        assert h1_lifetime > 1.0

    def test_empty_higher_dimension_is_handled(self):
        """A collinear chain has H0 bars but an EMPTY H1. The labelled diagram
        must still be well-formed (k, 3), H1 selection returns (0, 2), and no
        spurious dim-1 rows are fabricated. This exercises the empty-per-
        dimension branch of the labelling code."""
        chain = _collinear_chain(n=10, spacing=1.0)
        summary = compute_persistence(chain, max_dim=1)
        assert summary.diagram.shape[1] == 3
        assert summary.diagram_for_dim(1).shape == (0, 2)
        assert np.all(summary.diagram[:, 2] == 0)  # only H0 labels present

    def test_diagram_for_dim_on_single_point_finite_h0_is_empty(self):
        """A single point yields only the infinite H0 component and an empty
        finite H0 selection; the accessor returns a well-shaped (0, 2) array
        when the finite-only mask matches nothing."""
        single = np.array([[0.0, 0.0]])
        with pytest.warns(UserWarning, match="more columns"):
            summary = compute_persistence(single, max_dim=1)
        # finite-only H0 is empty for a single point (only the infinite bar).
        assert summary.diagram_for_dim(0, finite_only=True).shape == (0, 2)

    def test_diagram_for_dim_on_fully_empty_diagram(self):
        """When the stored diagram is itself empty -- shape (0, 3), the
        contract `compute_persistence` returns when every homology dimension
        is empty -- the accessor takes its empty-diagram fast path and returns
        a well-shaped (0, 2) array for any requested dimension. Constructed
        directly to exercise the documented empty-input contract of the
        accessor (ripser never returns a fully empty diagram for a non-empty
        cloud, since H0 always carries the surviving component)."""
        empty_summary = PersistenceSummary(
            betti_0=0,
            betti_1=0,
            total_persistence=0.0,
            max_persistence=0.0,
            diagram=np.empty((0, 3)),
        )
        assert empty_summary.diagram_for_dim(0).shape == (0, 2)
        assert empty_summary.diagram_for_dim(1, finite_only=True).shape == (0, 2)


# --------------------------------------------------------------------------- #
# (b) track_divergence's H0 comparison uses ONLY H0 features, verified
#     against an INDEPENDENT (scipy-MST) reference.
# --------------------------------------------------------------------------- #
class TestH0SelectionIsExact:
    @pytest.mark.parametrize(
        ("sep_a", "sep_b", "seed_a", "seed_b"),
        [
            (3.0, 7.0, 1, 2),
            (2.0, 9.0, 11, 12),
            (4.0, 5.5, 21, 22),
            (1.5, 8.0, 31, 32),
            (6.0, 2.5, 41, 42),
        ],
    )
    def test_h0_selection_is_exactly_the_mst(self, sep_a, sep_b, seed_a, seed_b):
        """The finite-H0 sub-diagram that track_divergence compares must be
        EXACTLY the H0 diagram, proven element-wise against the scipy-MST
        reference (independent of ripser):
          - exactly n - 1 finite bars,
          - every bar born at 0,
          - NO dimension-1 bars in the selection,
          - death times equal the MST edge weights (sorted).
        """
        cloud_a = _two_clusters(sep_a, seed_a)
        cloud_b = _two_clusters(sep_b, seed_b)

        first = track_divergence(cloud_a)
        second = track_divergence(cloud_b, previous_summary=first)

        for cloud, summary in ((cloud_a, first), (cloud_b, second)):
            sel = summary.diagram_for_dim(0, finite_only=True)
            n = cloud.shape[0]
            # n - 1 finite H0 bars, all born at 0.
            assert sel.shape == (n - 1, 2)
            assert np.allclose(sel[:, 0], 0.0)
            # No H1 contamination: nothing with dim==1 is in the H0 selection.
            # (Verified directly against the raw labelled diagram.)
            raw = summary.diagram
            assert raw.shape[1] == 3
            h1_finite = raw[(raw[:, 2] == 1) & (raw[:, 1] < np.inf)]
            # The selection contains none of those H1 bars.
            for bar in h1_finite:
                assert not np.any(np.all(np.isclose(sel, bar[:2]), axis=1)) or bar[1] == 0.0
            # Death times equal the independent MST edge weights.
            mst_deaths = np.sort(_independent_h0_from_mst(cloud)[:, 1])
            sel_deaths = np.sort(sel[:, 1])
            assert np.allclose(sel_deaths, mst_deaths, rtol=FLOAT32_REL, atol=FLOAT32_REL)

    @pytest.mark.parametrize(
        ("sep_a", "sep_b", "seed_a", "seed_b"),
        [
            (3.0, 7.0, 1, 2),
            (2.0, 9.0, 11, 12),
            (4.0, 5.5, 21, 22),
        ],
    )
    def test_h0_bottleneck_matches_independent_mst_reference(self, sep_a, sep_b, seed_a, seed_b):
        """The bottleneck distance track_divergence reports must equal the
        bottleneck of two INDEPENDENTLY-constructed (scipy-MST) H0 diagrams,
        to the float32 floor. The clouds have genuinely different H0 (different
        cluster separations), so the bottleneck is non-zero and a relative
        tolerance is meaningful. The reference H0 diagrams come from scipy's
        MST, NOT from re-running the production selection -- so this is an
        independent check, not a code mirror."""
        cloud_a = _two_clusters(sep_a, seed_a)
        cloud_b = _two_clusters(sep_b, seed_b)

        first = track_divergence(cloud_a)
        second = track_divergence(cloud_b, previous_summary=first)
        produced = second.bottleneck_to_previous

        # Independent reference: bottleneck of the two MST-derived H0 diagrams.
        ref = float(
            bottleneck(
                _independent_h0_from_mst(cloud_a),
                _independent_h0_from_mst(cloud_b),
            )
        )

        assert produced is not None
        assert ref > 0.1  # genuinely different H0 -> non-trivial distance
        assert produced == pytest.approx(ref, rel=FLOAT32_REL)


# --------------------------------------------------------------------------- #
# (c) The bug is gone: an H1-only change does not leak into the H0 divergence.
# --------------------------------------------------------------------------- #
class TestH1ChangeDoesNotLeakIntoH0:
    def test_h1_only_change_does_not_leak_into_h0(self):
        """Construct two clouds that differ ONLY in H1:
          - a regular 12-gon (one prominent loop, H1 lifetime ~1.2),
          - a collinear chain of 12 points with the SAME nearest-neighbour
            spacing (no loop, H1 empty).
        Both have identical H0 merge structure (same n-1 chord-length deaths),
        so the correct H0 divergence is ~0. The OLD buggy code pooled all
        finite features across dimensions; that mixed pool is dominated by the
        12-gon's unmatched loop, giving a LARGE bottleneck. This test proves
        (i) the construction is as designed (H0 identical, H1 differs) and
        (ii) the fixed H0 divergence is ~0 while the old mixed-pool value is
        large -- demonstrating the leak is closed."""
        n = 12
        polygon = _regular_polygon(n=n, radius=1.0)
        # Nearest-neighbour spacing on the n-gon = chord subtending 2*pi/n.
        spacing = 2.0 * np.sin(np.pi / n)
        chain = _collinear_chain(n=n, spacing=spacing)

        poly_sum = compute_persistence(polygon, max_dim=1)
        chain_sum = compute_persistence(chain, max_dim=1)

        # --- Verify the construction actually has the intended topology. ---
        poly_h0 = poly_sum.diagram_for_dim(0, finite_only=True)
        chain_h0 = chain_sum.diagram_for_dim(0, finite_only=True)
        poly_h1 = poly_sum.diagram_for_dim(1)
        chain_h1 = chain_sum.diagram_for_dim(1)

        # Identical H0: same number of finite bars and matching death times.
        assert poly_h0.shape[0] == n - 1
        assert chain_h0.shape[0] == n - 1
        assert np.allclose(
            np.sort(poly_h0[:, 1]), np.sort(chain_h0[:, 1]), rtol=FLOAT32_REL, atol=FLOAT32_REL
        )
        # Different H1: the polygon has exactly one prominent loop; the chain
        # has none.
        assert poly_h1.shape[0] == 1
        assert float((poly_h1[:, 1] - poly_h1[:, 0]).max()) > 1.0
        assert chain_h1.shape[0] == 0

        # --- Fixed behaviour: H0-only divergence is ~0. ---
        result = track_divergence(chain, previous_summary=poly_sum)
        assert result.bottleneck_to_previous == pytest.approx(0.0, abs=1e-5)

        # --- Old buggy behaviour, recomputed inline for contrast: the
        #     mixed-dimension finite pool (H0 + H1, no dim filter) leaks the
        #     loop into the comparison and yields a LARGE distance. ---
        def _old_mixed_finite_pool(summary):
            raw = summary.diagram
            return raw[raw[:, 1] < np.inf][:, :2]

        buggy_distance = float(
            bottleneck(_old_mixed_finite_pool(chain_sum), _old_mixed_finite_pool(poly_sum))
        )
        # The leak is real and large: the old value is dominated by the
        # unmatched loop (~half its lifetime), orders of magnitude above the
        # corrected ~0. This contrast IS the proof the bug existed and is now
        # fixed.
        assert buggy_distance > 0.5
        assert buggy_distance > 1000.0 * (result.bottleneck_to_previous + 1e-9)

"""Regression tests for defects found in the 2026-10 quality audit.

Each test fails on the pre-fix code and pins the corrected contract:

- ``pole_ladder`` previously reused the tangent direction computed at the
  original base point for every rung and divided it by the *total* rung count,
  so for ``n_rungs >= 2`` the ladder walked a wrong, shortened path and never
  reached ``end_point``.  The error therefore *grew* with the rung count and
  plateaued at ~2e-2 (the "inherent limitation" formerly recorded in
  ``LIMITATIONS.md``).  The corrected schedule re-expresses the remaining
  geodesic in the current tangent space each rung; on S^2 (a symmetric space,
  where pole ladder is exact) the result now matches the analytic transport to
  round-off for every rung count.
- ``compute_tangent_transport_matrix`` silently fell back to Schild's ladder
  for any unrecognised ``method`` string.
- ``cycle_basis_holonomy_defect`` folded cycle deviations with ``max``, which
  discards NaN (``max(0.0, nan) == 0.0``), so a non-finite transport map was
  reported as *flat* transport.
- ``TransportGroupoidAggregator.aggregate`` normalised client weights without
  validation, so zero / negative / non-finite weights produced NaN parameters
  and a misleading "Karcher mean output is outside the manifold" error.
- ``compute_persistence`` let ripser emit a spurious "input matrix is square"
  ``UserWarning`` whenever ``n_points == n_features`` for a point cloud.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pytest
from geomstats.geometry.hypersphere import Hypersphere

from groupoid.aggregation import TransportGroupoidAggregator
from groupoid.cohomology import cycle_basis_holonomy_defect
from groupoid.persistence import compute_persistence
from groupoid.transport import compute_tangent_transport_matrix, pole_ladder, schild_ladder


def _s2_setup():
    manifold = Hypersphere(dim=2)
    base = np.array([0.0, 0.0, 1.0])
    end = np.array([0.5, 0.0, np.sqrt(3) / 2])
    end = end / np.linalg.norm(end)
    # A tangent vector with components both along and across the geodesic:
    # the pre-fix bug is invisible for vectors orthogonal to the geodesic.
    v = manifold.to_tangent(np.array([1.0, 1.0, 0.0]), base)
    ground_truth = manifold.metric.parallel_transport(v, base, end_point=end)
    return manifold, base, end, v, ground_truth


class TestPoleLadderStepSchedule:
    @pytest.mark.parametrize("n_rungs", [1, 2, 4, 8, 16])
    def test_pole_ladder_is_exact_on_s2_for_every_rung_count(self, n_rungs):
        """On S^2 the pole ladder is exact; the output must equal geomstats'
        analytic Levi-Civita transport to round-off regardless of ``n_rungs``.
        Pre-fix: error 8.9e-3 (2 rungs) rising to 2.0e-2 (16 rungs)."""
        manifold, base, end, v, gt = _s2_setup()
        transported = pole_ladder(manifold, v, base, end, n_rungs=n_rungs)
        assert np.linalg.norm(transported - gt) < 1e-12
        # Exactly tangent at the true end point (pre-fix: drifted off-plane).
        assert abs(float(np.dot(transported, end))) < 1e-12

    def test_multi_rung_does_not_degrade_relative_to_single_rung(self):
        """Adding rungs must never make the result worse than one rung.
        Pre-fix the multi-rung path did not even reach ``end_point``."""
        manifold, base, end, v, gt = _s2_setup()
        err_one = np.linalg.norm(pole_ladder(manifold, v, base, end, n_rungs=1) - gt)
        err_many = np.linalg.norm(pole_ladder(manifold, v, base, end, n_rungs=32) - gt)
        assert err_many <= err_one + 1e-12

    def test_default_single_rung_output_is_unchanged(self):
        """The default ``n_rungs=1`` call is the pre-fix algorithm verbatim; pin
        it by re-deriving the single-rung pole construction explicitly."""
        manifold, base, end, v, _ = _s2_setup()
        metric = manifold.metric
        direction = metric.log(end, base)
        pole = metric.exp(-v, base)
        mid = metric.exp(direction / 2.0, base)
        reflected = metric.exp(-metric.log(pole, mid), mid)
        expected = metric.log(reflected, metric.exp(direction, base))
        np.testing.assert_allclose(pole_ladder(manifold, v, base, end), expected, atol=1e-14)


class TestSchildLadderIsFirstOrder:
    """Regression pin for the corrected Schild's ladder.

    The previous ``schild_ladder`` body was the pole-ladder reflection with a
    broken step schedule (``direction / n_rungs`` re-read every rung, so the
    walk never reached ``end_point`` for ``n_rungs > 1``); its apparent
    "non-convergence" was that bug. The geodesic-parallelogram Schild's ladder
    with the vector rescaled by ``1 / n_rungs`` is a first-order scheme: the
    error must halve when the rung count doubles.
    """

    @staticmethod
    def _hop():
        manifold = Hypersphere(dim=2)
        base = np.array([0.0, 0.0, 1.0])
        end = np.array([0.5, 0.0, np.sqrt(3) / 2])
        end = end / np.linalg.norm(end)
        # A unit vector not aligned with the hop, so the parallelogram error
        # is non-trivial (the e_x/e_z hop is exact by symmetry).
        v = manifold.to_tangent(np.array([0.6, 0.8, 0.0]), base)
        gt = manifold.metric.parallel_transport(v, base, end_point=end)
        return manifold, base, end, v, gt

    def test_error_halves_when_rungs_double(self):
        manifold, base, end, v, gt = self._hop()
        errors = [
            float(np.linalg.norm(schild_ladder(manifold, v, base, end, n_rungs=n) - gt))
            for n in (1, 2, 4, 8, 16, 32, 64)
        ]
        # Closed-form S^2 reference for the 30-degree test hop.
        assert errors[0] == pytest.approx(0.21406374, rel=1e-2)
        assert errors[-1] == pytest.approx(0.00326856, rel=1e-2)
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
            assert coarse / fine == pytest.approx(2.0, abs=0.15)

    def test_lands_in_endpoint_tangent_plane_and_preserves_norm(self):
        manifold, base, end, v, _gt = self._hop()
        for n in (1, 3, 8):
            out = schild_ladder(manifold, v, base, end, n_rungs=n)
            assert abs(float(np.dot(out, end))) < 1e-12
            assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(v), rel=1e-6)

    def test_pole_ladder_remains_more_accurate(self):
        manifold, base, end, v, gt = self._hop()
        pole_err = np.linalg.norm(pole_ladder(manifold, v, base, end, n_rungs=8) - gt)
        schild_err = np.linalg.norm(schild_ladder(manifold, v, base, end, n_rungs=8) - gt)
        assert pole_err < 1e-12 < schild_err

    def test_rejects_non_positive_rungs(self):
        manifold, base, end, v, _gt = self._hop()
        with pytest.raises(ValueError, match="n_rungs"):
            schild_ladder(manifold, v, base, end, n_rungs=0)


class TestTransportMethodValidation:
    @pytest.mark.parametrize("n_rungs", [0, -1])
    def test_pole_ladder_rejects_non_positive_rungs(self, n_rungs):
        manifold, base, end, v, _ = _s2_setup()
        with pytest.raises(ValueError, match="n_rungs"):
            pole_ladder(manifold, v, base, end, n_rungs=n_rungs)

    def test_unknown_method_raises_instead_of_silently_using_schild(self):
        manifold, base, end, _, _ = _s2_setup()
        with pytest.raises(ValueError, match="Unknown transport method 'polar'"):
            compute_tangent_transport_matrix(manifold, base, end, method="polar")

    def test_known_methods_still_dispatch(self):
        manifold, base, end, _, _ = _s2_setup()
        pole_op = compute_tangent_transport_matrix(manifold, base, end, method="pole")
        schild_op = compute_tangent_transport_matrix(manifold, base, end, method="schild")
        assert pole_op.shape == schild_op.shape == (3, 3)
        assert np.all(np.isfinite(pole_op)) and np.all(np.isfinite(schild_op))


class TestHolonomyDefectRejectsNonFinite:
    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_transport_map_raises(self, bad):
        """Pre-fix: ``max(0.0, nan)`` returned 0.0 and the graph was declared
        flat; an infinite entry likewise must not be reported as a number."""
        graph = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "A")])
        maps = {
            ("A", "B"): np.eye(2),
            ("B", "C"): np.eye(2),
            ("C", "A"): np.array([[1.0, bad], [0.0, 1.0]]),
        }
        with pytest.raises(ValueError, match="Non-finite holonomy"):
            cycle_basis_holonomy_defect(graph, maps)

    def test_finite_defect_still_reported(self):
        graph = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "A")])
        maps = {
            ("A", "B"): np.eye(2),
            ("B", "C"): np.eye(2),
            ("C", "A"): np.array([[1.0, 0.5], [0.0, 1.0]]),
        }
        assert cycle_basis_holonomy_defect(graph, maps) == pytest.approx(0.5)


def _make_aggregator():
    manifold = Hypersphere(dim=2)
    graph = nx.DiGraph([("A", "B"), ("A", "C")])
    agg = TransportGroupoidAggregator(manifold=manifold, graph=graph, base_node="A")
    agg.register_transport("A", "B", np.eye(3))
    agg.register_transport("A", "C", np.eye(3))
    params = {
        "A": np.array([0.0, 0.0, 1.0]),
        "B": np.array([0.1, 0.0, 0.995]),
        "C": np.array([-0.1, 0.0, 0.995]),
    }
    return agg, {k: v / np.linalg.norm(v) for k, v in params.items()}


class TestAggregateWeightValidation:
    def test_zero_sum_weights_rejected(self):
        agg, params = _make_aggregator()
        with pytest.raises(ValueError, match="strictly positive sum"):
            agg.aggregate(params, weights={"A": 0.0, "B": 0.0, "C": 0.0})

    def test_negative_weight_rejected(self):
        agg, params = _make_aggregator()
        with pytest.raises(ValueError, match="finite and non-negative"):
            agg.aggregate(params, weights={"A": 1.0, "B": -1.0, "C": 1.0})

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_weight_rejected(self, bad):
        agg, params = _make_aggregator()
        with pytest.raises(ValueError, match="finite and non-negative"):
            agg.aggregate(params, weights={"A": 1.0, "B": bad})

    def test_invalid_weights_do_not_consume_round_or_divergence_state(self):
        agg, params = _make_aggregator()
        agg.track_divergence = True
        first = agg.aggregate(params)
        saved_divergence = agg._prev_divergence
        assert first.round_idx == 1
        with pytest.raises(ValueError, match="finite and non-negative"):
            agg.aggregate(params, weights={"B": -1.0})
        assert agg._round_idx == 1
        assert agg._prev_divergence is saved_divergence
        assert agg.aggregate(params).round_idx == 2

    def test_extreme_finite_weights_normalise_without_overflow(self):
        agg, params = _make_aggregator()
        expected = agg.aggregate(params, weights={"A": 1.0, "B": 1.0, "C": 0.0}).global_params
        actual = agg.aggregate(params, weights={"A": 1e308, "B": 1e308, "C": 0.0}).global_params
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_valid_weights_still_normalise_and_aggregate(self):
        """Missing clients default to weight 1.0; a zero weight on one client
        is allowed as long as the total is positive."""
        agg, params = _make_aggregator()
        result = agg.aggregate(params, weights={"B": 0.0, "C": 2.0})
        assert np.all(np.isfinite(result.global_params))
        assert np.linalg.norm(result.global_params) == pytest.approx(1.0)


class TestPersistenceSquarePointCloud:
    def test_square_point_cloud_emits_no_ripser_warning(self):
        """3 points x 3 features is a point cloud, not a distance matrix;
        ripser's square-input warning is spurious here and must be silenced."""
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 0.995], [-0.1, 0.0, 0.995]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            summary = compute_persistence(points, max_dim=0)
        assert summary.betti_0 == 1

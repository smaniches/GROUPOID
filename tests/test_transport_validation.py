"""Ground-truth validation of the parallel-transport ladders.

Norm preservation (the existing smoke tests) only checks that transport is an
isometry -- a necessary but not sufficient condition. The identity map and any
wrong-angle rotation are also norm-preserving. These tests instead compare the
ladder output to geomstats' analytic ``metric.parallel_transport`` on S^2,
which is the actual Levi-Civita parallel transport, and check the *direction*
of the transported vector, not just its magnitude.

Empirical finding (recorded so the asserted tolerances are honest, not
reverse-engineered):

- ``pole_ladder`` matches the analytic transport to round-off on a 60 degree
  geodesic hop for every rung count (S^2 is a symmetric space, where the pole
  ladder is exact). The cosine-similarity check below is kept as the
  direction-level contract; ``tests/test_audit_regressions.py`` pins the
  round-off-level agreement and the per-rung step schedule.
- ``schild_ladder`` is the geodesic-parallelogram scheme of Kheyfets, Miller
  and Newton. It is first order: on the same hop with a unit tangent vector
  the error against the analytic transport halves with every doubling of
  ``n_rungs`` (0.277 at 1 rung, 0.0040 at 64 rungs; measured ratio 2.00).
  ``tests/test_audit_regressions.py`` pins that convergence order; here it
  is asserted to be in the right half-space, norm-preserving, and strictly
  less accurate than the pole ladder, consistent with its documented status
  as the coarser ladder.
"""

from __future__ import annotations

import numpy as np
import pytest
from geomstats.geometry.hypersphere import Hypersphere

from groupoid.transport import pole_ladder, schild_ladder


def _setup():
    manifold = Hypersphere(dim=2)
    base = np.array([0.0, 0.0, 1.0])
    end = np.array([0.5, 0.0, np.sqrt(3) / 2])
    end = end / np.linalg.norm(end)
    v = manifold.to_tangent(np.array([1.0, 0.0, 0.0]), base)
    ground_truth = manifold.metric.parallel_transport(v, base, end_point=end)
    return manifold, base, end, v, ground_truth


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestPoleLadderMatchesAnalyticTransport:
    def test_pole_ladder_direction_matches_geomstats(self):
        """pole_ladder must agree with geomstats' analytic parallel transport
        in direction (cosine > 0.999) and magnitude. This is a ground-truth
        correctness check, not merely an isometry check."""
        manifold, base, end, v, gt = _setup()
        transported = pole_ladder(manifold, v, base, end, n_rungs=8)

        assert _cosine(transported, gt) > 0.999
        assert np.linalg.norm(transported) == pytest.approx(np.linalg.norm(v), rel=1e-2)
        # The final rung lands exactly on ``end``, so the transported vector
        # must lie in the endpoint tangent plane to round-off (the pre-fix
        # ladder drifted off-plane by ~0.02 and this bound was 0.03).
        tangent_residual = abs(float(np.dot(transported, end)))
        assert tangent_residual < 1e-12

    def test_pole_ladder_beats_schild_in_direction(self):
        """The documented accuracy ordering must hold: pole ladder is closer
        to the analytic transport than Schild's ladder for the same rungs.

        The vector is taken off the plane of the hop geodesic: for a vector
        *in* that plane both ladders are exact on S^2 by symmetry, so the
        ordering is only observable on a generic tangent vector."""
        manifold, base, end, _v, _gt = _setup()
        v = manifold.to_tangent(np.array([0.6, 0.8, 0.0]), base)
        gt = manifold.metric.parallel_transport(v, base, end_point=end)
        pole = pole_ladder(manifold, v, base, end, n_rungs=8)
        schild = schild_ladder(manifold, v, base, end, n_rungs=8)
        assert _cosine(pole, gt) > _cosine(schild, gt)
        assert np.linalg.norm(pole - gt) < 1e-12 < np.linalg.norm(schild - gt)


class TestSchildLadderApproximate:
    def test_schild_ladder_is_in_right_half_space(self):
        """Schild's ladder is the coarser approximation; assert that its
        direction is not inverted (positive cosine with ground truth), that it
        preserves norm, and that it lands in the endpoint tangent plane."""
        manifold, base, end, v, gt = _setup()
        transported = schild_ladder(manifold, v, base, end, n_rungs=8)
        assert _cosine(transported, gt) > 0.9
        assert np.linalg.norm(transported) == pytest.approx(np.linalg.norm(v), rel=1e-2)
        assert abs(float(np.dot(transported, end))) < 1e-12

"""Validation tests for the Riemannian optimizers.

Two things are validated here, beyond the smoke tests in test_smoke.py:

1. Moment transport. The SGD momentum velocity and Adam's first moment are
   parallel-transported into each new iterate's tangent space. The tests
   discriminate transport from the two wrong alternatives: leaving the
   moment at the old point (it would not be tangent at the new iterate) and
   projecting it onto the new tangent space (projection annihilates the
   component normal to the new tangent space, so a geodesic-aligned moment
   would lose norm; parallel transport is an isometry and preserves it).

2. Descent. On S^2, minimizing f(x) = d(x, target)^2 / 2 via its Riemannian
   gradient -Log_x(target) must drive the iterate to the target. This is a
   known-correct reference (the unique minimizer on the geodesically convex
   ball), so it validates that the update rules actually descend, not just
   that iterates stay on the manifold.

The Euclidean-gradient argument is fed the Riemannian gradient directly:
for S^2 embedded in R^3 the tangent projection of a tangent vector is
itself, so to_tangent() leaves it unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from geomstats.geometry.hypersphere import Hypersphere

from groupoid.optimizer import RiemannianAdam, RiemannianSGD, _transport_moment


def _geodesic_grad(manifold, point, target):
    """Riemannian gradient of f(x) = d(x, target)^2 / 2, i.e. -Log_x(target)."""
    return -manifold.metric.log(target, point)


class TestMomentTransport:
    def test_sgd_velocity_parallel_transported_not_projected(self):
        # One quarter-turn step from the north pole: lr = pi/2 with unit
        # gradient along +x moves the iterate to (-1, 0, 0). The velocity
        # (1, 0, 0) is radial there: projection would send it to zero and
        # leaving it untransported would leave it non-tangent. Parallel
        # transport preserves its unit norm and lands it in the new
        # tangent space.
        manifold = Hypersphere(dim=2)
        point = np.array([0.0, 0.0, 1.0])
        grad = np.array([1.0, 0.0, 0.0])

        opt = RiemannianSGD(manifold=manifold, lr=np.pi / 2, momentum=0.9)
        new_point = opt.step(point, grad)

        np.testing.assert_allclose(new_point, [-1.0, 0.0, 0.0], atol=1e-12)
        assert opt._velocity is not None
        assert manifold.is_tangent(opt._velocity, new_point, atol=1e-10)
        # Isometry: norm preserved exactly; projection would give 0.
        assert np.linalg.norm(opt._velocity) == pytest.approx(1.0, abs=1e-10)

    def test_adam_first_moment_parallel_transported_not_projected(self):
        # Adam's first step moves by ~lr along the normalized gradient
        # (m_hat = grad, sqrt(v_hat) = |grad|). With lr = pi/2 the iterate
        # reaches (-1, 0, 0) where the stored first moment (1-beta1)*grad
        # is radial: transport preserves its norm, projection would zero it.
        manifold = Hypersphere(dim=2)
        point = np.array([0.0, 0.0, 1.0])
        grad = np.array([1.0, 0.0, 0.0])
        beta1 = 0.9

        opt = RiemannianAdam(manifold=manifold, lr=np.pi / 2, beta1=beta1, eps=0.0)
        new_point = opt.step(point, grad)

        np.testing.assert_allclose(new_point, [-1.0, 0.0, 0.0], atol=1e-12)
        assert opt._m is not None
        assert manifold.is_tangent(opt._m, new_point, atol=1e-10)
        assert np.linalg.norm(opt._m) == pytest.approx(1.0 - beta1, abs=1e-10)

    def test_transport_moment_projection_fallback_without_parallel_transport(self):
        # A metric without parallel_transport falls back to tangent
        # projection (the compatibility shim).
        class _Metric:
            pass

        class _Manifold:
            metric = _Metric()

            def to_tangent(self, vec, point):
                return 2.0 * vec  # marker so the fallback path is observable

        moment = np.array([1.0, 0.0, 0.0])
        out = _transport_moment(_Manifold(), moment, moment, moment)
        np.testing.assert_allclose(out, 2.0 * moment)

    def test_transport_moment_projection_fallback_on_not_implemented(self):
        # A metric that declares parallel_transport but raises
        # NotImplementedError must also fall back to projection.
        class _Metric:
            def parallel_transport(self, moment, base_point, end_point=None):
                raise NotImplementedError

        class _Manifold:
            metric = _Metric()

            def to_tangent(self, vec, point):
                return 3.0 * vec  # marker so the fallback path is observable

        moment = np.array([0.0, 1.0, 0.0])
        out = _transport_moment(_Manifold(), moment, moment, moment)
        np.testing.assert_allclose(out, 3.0 * moment)


class TestDescent:
    """The optimizers must descend f(x) = d(x, target)^2 / 2 to the target."""

    manifold = Hypersphere(dim=2)
    start = np.array([0.0, 0.0, 1.0])
    # 60 degrees away from the start, well inside the geodesically convex
    # ball where the objective has a unique minimizer.
    target = np.array([np.sin(np.pi / 3), 0.0, np.cos(np.pi / 3)])

    def _run(self, opt, n_steps):
        point = self.start
        initial = float(self.manifold.metric.dist(point, self.target))
        for _ in range(n_steps):
            point = opt.step(point, _geodesic_grad(self.manifold, point, self.target))
        final = float(self.manifold.metric.dist(point, self.target))
        return initial, final

    def test_sgd_descends_to_target(self):
        opt = RiemannianSGD(manifold=self.manifold, lr=0.1)
        initial, final = self._run(opt, n_steps=200)
        assert final < 1e-6 < initial

    def test_sgd_with_momentum_descends_to_target(self):
        opt = RiemannianSGD(manifold=self.manifold, lr=0.05, momentum=0.5)
        initial, final = self._run(opt, n_steps=300)
        assert final < 1e-6 < initial

    def test_adam_descends_to_target(self):
        opt = RiemannianAdam(manifold=self.manifold, lr=0.05)
        initial, final = self._run(opt, n_steps=400)
        assert final < 1e-3 < initial

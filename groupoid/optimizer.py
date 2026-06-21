"""Riemannian optimizers for federated learning on manifolds.

Standard SGD and Adam assume Euclidean parameter spaces. When model
parameters live on a Riemannian manifold (e.g., Stiefel manifold for
orthogonal layers, SPD manifold for covariance matrices), we need
Riemannian counterparts that respect the geometry.

This module provides:
- Riemannian SGD with retraction
- Riemannian Adam with exponential map updates
- Adaptive learning rate based on sectional curvature
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger


@dataclass
class RiemannianSGD:
    """Riemannian stochastic gradient descent.

    Updates parameters by computing the Riemannian gradient (projection
    of Euclidean gradient onto tangent space) and retracting back to
    the manifold via the exponential map.

    Parameters
    ----------
    manifold
        A geomstats manifold instance.
    lr
        Learning rate.
    momentum
        Momentum coefficient (0 = no momentum).
    """

    manifold: Any
    lr: float = 0.01
    momentum: float = 0.0
    _velocity: npt.NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def step(
        self, point: npt.NDArray[np.float64], euclidean_grad: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Perform one optimization step.

        Parameters
        ----------
        point
            Current point on the manifold.
        euclidean_grad
            Euclidean gradient (will be projected to tangent space).

        Returns
        -------
        np.ndarray
            Updated point on the manifold.
        """
        # Project gradient onto tangent space
        riemannian_grad = self.manifold.to_tangent(euclidean_grad, point)

        # Apply momentum
        vel: npt.NDArray[np.float64]
        if self.momentum > 0:
            if self._velocity is None:
                vel = riemannian_grad
            else:
                vel = self.manifold.to_tangent(
                    self.momentum * self._velocity + riemannian_grad, point
                )
            self._velocity = vel
            update = -self.lr * vel
        else:
            update = -self.lr * riemannian_grad

        # Retract to manifold via exponential map
        new_point: npt.NDArray[np.float64] = self.manifold.metric.exp(update, point)

        return new_point


@dataclass
class RiemannianAdam:
    """Riemannian Adam optimizer.

    Adapts the Adam optimizer to Riemannian manifolds by maintaining
    exponential moving averages of the Riemannian gradient and its
    squared norm, with updates via the exponential map.

    Parameters
    ----------
    manifold
        A geomstats manifold instance.
    lr
        Learning rate.
    beta1
        Exponential decay rate for first moment.
    beta2
        Exponential decay rate for second moment.
    eps
        Small constant for numerical stability.
    """

    manifold: Any
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    _m: npt.NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    _v: float = field(default=0.0, init=False, repr=False)
    _t: int = field(default=0, init=False, repr=False)

    def step(
        self, point: npt.NDArray[np.float64], euclidean_grad: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Perform one optimization step.

        Parameters
        ----------
        point
            Current point on the manifold.
        euclidean_grad
            Euclidean gradient.

        Returns
        -------
        np.ndarray
            Updated point on the manifold.
        """
        self._t += 1

        # Riemannian gradient
        grad = self.manifold.to_tangent(euclidean_grad, point)
        grad_norm_sq = float(np.sum(grad**2))

        # Update biased first moment (tangent vector). Seed from zero like the
        # second moment below: m_1 = (1 - beta1) * grad, so the bias correction
        # m_hat = m_1 / (1 - beta1) recovers `grad` on the first step. Seeding
        # m_1 = grad directly would leave the 1/(1-beta1) factor uncancelled and
        # inflate the first update by ~10x (beta1=0.9).
        first_moment: npt.NDArray[np.float64]
        if self._m is None:
            first_moment = (1 - self.beta1) * grad
        else:
            first_moment = self.manifold.to_tangent(
                self.beta1 * self._m + (1 - self.beta1) * grad, point
            )
        self._m = first_moment

        # Update biased second moment (scalar, norm-based)
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad_norm_sq

        # Bias correction
        m_hat = first_moment / (1 - self.beta1**self._t)
        v_hat = self._v / (1 - self.beta2**self._t)

        # Adaptive update
        update = -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        update = self.manifold.to_tangent(update, point)

        new_point: npt.NDArray[np.float64] = self.manifold.metric.exp(update, point)
        return new_point


def curvature_adaptive_lr(
    manifold: Any,  # geomstats manifold; no upstream type stubs
    point: npt.NDArray[np.float64],
    base_lr: float,
    tangent_vec: npt.NDArray[np.float64],
) -> float:
    """Adapt learning rate based on local sectional curvature.

    In regions of high positive curvature, geodesics converge and
    we should take smaller steps. In regions of negative curvature,
    geodesics diverge and we can take larger steps.

    Parameters
    ----------
    manifold
        A geomstats manifold with a curvature method.
    point
        Current point on the manifold.
    base_lr
        Base learning rate to adapt.
    tangent_vec
        Direction of the update.

    Returns
    -------
    float
        Adapted learning rate.
    """
    try:
        if hasattr(manifold.metric, "sectional_curvature"):
            # Use a random orthogonal vector for the plane
            random_vec = manifold.to_tangent(np.random.randn(*tangent_vec.shape), point)
            kappa = manifold.metric.sectional_curvature(tangent_vec, random_vec, point)
            kappa = float(np.mean(kappa)) if hasattr(kappa, "__len__") else float(kappa)

            # Scale: lr / (1 + max(kappa, 0)) damps in positive curvature
            adapted: float = base_lr / (1.0 + max(kappa, 0.0))
            logger.debug("Curvature-adapted LR: {:.6f} (kappa={:.4f})", adapted, kappa)
            return adapted
    except (AttributeError, NotImplementedError):
        pass

    return base_lr

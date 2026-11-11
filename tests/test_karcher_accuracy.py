"""Accuracy contract of ``karcher_mean`` (CORRECTION_NOTICE.md, F03).

``tol`` is forwarded to geomstats, whose gradient criterion stops the
iteration once the squared Riemannian gradient norm of the weighted Frechet
function is at most ``tol * dim``; its near-zero-variance and iteration-limit
exits do not establish that bound. The pre-v0.1.0.dev10 default ``tol=1e-6``
therefore allowed a gradient norm of about 1.4e-3 on S^2 at a gradient-criterion
stop, and geodesic errors of the same order. These tests fail
on that default and pass with the corrected one.

Every reference is independent of the implementation under test: a ring
centred on the pole has the pole as its Frechet mean by symmetry, and the
gradient norm and reference minimizer use closed-form S^2 logarithm and
exponential maps written here in numpy rather than geomstats.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
from geomstats.geometry.hypersphere import Hypersphere

import groupoid.aggregation as aggregation_module
from groupoid.aggregation import TransportGroupoidAggregator
from groupoid.manifold import DEFAULT_KARCHER_TOL, karcher_mean


def _pole(dim: int) -> np.ndarray:
    pole = np.zeros(dim + 1)
    pole[-1] = 1.0
    return pole


def _log(base: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Riemannian logarithms at ``base`` of the rows of ``points`` (unit sphere)."""
    cos = np.clip(points @ base, -1.0, 1.0)
    perp = points - np.outer(cos, base)
    sin = np.linalg.norm(perp, axis=1)
    angle = np.arctan2(sin, cos)
    scale = np.divide(angle, sin, out=np.ones_like(sin), where=sin > 0.0)
    return perp * scale[:, None]


def _exp(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(tangent))
    if norm == 0.0:
        return base
    point = math.cos(norm) * base + math.sin(norm) * tangent / norm
    return point / np.linalg.norm(point)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic distance via atan2, which stays accurate at small angles."""
    cos = float(a @ b)
    sin = float(np.linalg.norm(a - cos * b))
    return math.atan2(sin, cos)


def _grad_norm(m: np.ndarray, points: np.ndarray, weights: np.ndarray) -> float:
    """||grad F(m)|| for F(m) = sum_i w_i d(m, x_i)^2 / (2 sum_i w_i)."""
    mean_log = (weights[:, None] * _log(m, points)).sum(axis=0) / weights.sum()
    return float(np.linalg.norm(mean_log))


def _reference_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Unit-step Riemannian gradient (fixed-point) iteration run to round-off."""
    m = points.mean(axis=0)
    m = m / np.linalg.norm(m)
    for _ in range(1000):
        mean_log = (weights[:, None] * _log(m, points)).sum(axis=0) / weights.sum()
        if np.linalg.norm(mean_log) < 1e-15:
            break
        m = _exp(m, mean_log)
    assert _grad_norm(m, points, weights) < 1e-14
    return m


def _cluster(seed: int, sigma: float, n_points: int = 8, dim: int = 2) -> np.ndarray:
    """Benchmark-like data: isotropic tangent Gaussian at the pole, mapped by exp."""
    rng = np.random.default_rng(seed)
    pole = _pole(dim)
    tangents = np.zeros((n_points, dim + 1))
    tangents[:, :dim] = rng.normal(scale=sigma, size=(n_points, dim))
    return np.stack([_exp(pole, v) for v in tangents])


def test_default_tolerance_value() -> None:
    assert DEFAULT_KARCHER_TOL == 1e-12


def test_symmetric_ring_mean_is_the_pole() -> None:
    """Six points at geodesic radius 0.5 around the pole: the mean is the pole.

    With the former default ``tol=1e-6`` geomstats stopped about 9e-4 rad
    from the pole on this input.
    """
    manifold = Hypersphere(dim=2)
    pole = _pole(2)
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    ring = np.stack([_exp(pole, 0.5 * np.array([np.cos(a), np.sin(a), 0.0])) for a in angles])

    estimate = karcher_mean(manifold, ring)

    gradient_bound = math.sqrt(DEFAULT_KARCHER_TOL * manifold.dim)
    assert _dist(estimate, pole) <= 2.0 * gradient_bound


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("sigma", [0.05, 0.2])
def test_default_estimate_is_near_the_independent_minimizer(sigma: float, seed: int) -> None:
    manifold = Hypersphere(dim=2)
    points = _cluster(seed, sigma)
    ones = np.ones(len(points))

    estimate = karcher_mean(manifold, points)

    gradient_bound = math.sqrt(DEFAULT_KARCHER_TOL * manifold.dim)
    assert _grad_norm(estimate, points, ones) <= gradient_bound * (1.0 + 1e-6)
    # For these concentrated samples the Frechet function is close to
    # 1-strongly convex near its minimizer, so error <~ gradient norm; the
    # factor 2 leaves room for the curvature correction.
    assert _dist(estimate, _reference_mean(points, ones)) <= 2.0 * gradient_bound


@pytest.mark.parametrize("dim", [2, 4])
@pytest.mark.parametrize("tol", [1e-6, 1e-9, DEFAULT_KARCHER_TOL])
def test_returned_mean_satisfies_the_documented_gradient_bound(tol: float, dim: int) -> None:
    """The gradient-criterion bound ||grad F|| <= sqrt(tol * dim) holds at the
    returned point for these inputs, with and without weights."""
    manifold = Hypersphere(dim=dim)
    rng = np.random.default_rng(100 + dim)
    for seed in range(5):
        points = _cluster(1000 * dim + seed, 0.3, n_points=7, dim=dim)
        weights = rng.uniform(0.1, 2.0, size=len(points))
        ones = np.ones(len(points))
        for w, passed in ((ones, None), (weights, weights / weights.sum())):
            estimate = karcher_mean(manifold, points, weights=passed, tol=tol)
            assert _grad_norm(estimate, points, w) <= math.sqrt(tol * dim) * (1.0 + 1e-6)


def _frame(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)


def test_aggregator_forwards_karcher_tol_and_meets_the_contract(monkeypatch) -> None:
    """``karcher_tol`` reaches ``karcher_mean``; the default round is accurate."""
    manifold = Hypersphere(dim=2)
    latent = _cluster(7, 0.2, n_points=4)
    nodes = ["A", "B", "C", "D"]
    frames = {"A": np.eye(3)}
    frames.update(
        {
            node: _frame(np.array(axis), angle)
            for node, axis, angle in (
                ("B", (1.0, 0.2, 0.0), 0.7),
                ("C", (0.0, 1.0, 0.3), 1.1),
                ("D", (0.4, -0.5, 1.0), 0.4),
            )
        }
    )
    observed = {node: frames[node] @ latent[i] for i, node in enumerate(nodes)}

    seen: list[tuple[float, np.ndarray]] = []
    real_karcher_mean = aggregation_module.karcher_mean

    def recording_karcher_mean(*args, **kwargs):
        seen.append((kwargs["tol"], np.array(args[1], copy=True)))
        return real_karcher_mean(*args, **kwargs)

    monkeypatch.setattr(aggregation_module, "karcher_mean", recording_karcher_mean)

    def run(**kwargs):
        graph = nx.DiGraph([("A", node) for node in nodes[1:]])
        agg = TransportGroupoidAggregator(manifold=manifold, graph=graph, base_node="A", **kwargs)
        for node in nodes[1:]:
            agg.register_transport("A", node, frames[node] @ frames["A"].T)
        return agg.aggregate(observed).global_params

    default_mean = run()
    historical_mean = run(karcher_tol=1e-6)

    assert [tol for tol, _ in seen] == [DEFAULT_KARCHER_TOL, 1e-6]
    np.testing.assert_array_equal(
        historical_mean, real_karcher_mean(manifold, seen[1][1], tol=1e-6)
    )
    gradient_bound = math.sqrt(DEFAULT_KARCHER_TOL * manifold.dim)
    reference = _reference_mean(latent, np.ones(len(latent)))
    assert _dist(default_mean, reference) <= 2.0 * gradient_bound

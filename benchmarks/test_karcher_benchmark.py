"""Benchmark tests for Karcher mean computation."""

import pytest

from groupoid.manifold import karcher_mean


@pytest.mark.benchmark(group="karcher-mean")
def test_benchmark_karcher_mean_sphere(benchmark):
    """Benchmark Karcher mean on S^2 with 50 points."""
    from geomstats.geometry.hypersphere import Hypersphere

    manifold = Hypersphere(dim=2)
    points = manifold.random_point(n_samples=50)

    result = benchmark(karcher_mean, manifold, points)

    assert manifold.belongs(result, atol=1e-4)

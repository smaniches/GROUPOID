"""Tests for small but real branches the main suites did not reach.

Each test validates an actual behavioral contract, not just line execution:

- ``Morphism.__repr__`` renders the source -> target arrow.
- ``Sheaf`` section storage round-trips a value.
- ``karcher_mean`` forwards convergence controls when the geomstats estimator
  exposes an optimizer, and silently tolerates older estimators that do not
  (forward/backward-compatibility shim).
- ``curvature_adaptive_lr`` falls back to the base learning rate when a
  manifold declares ``sectional_curvature`` but raises ``NotImplementedError``.
- ``compute_h1`` raises ``IncompleteCocycleError`` for a cycle whose edges
  have no transport maps, because the holonomy product is then undefined.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from groupoid.cohomology import IncompleteCocycleError, compute_h1
from groupoid.groupoid import Morphism
from groupoid.manifold import karcher_mean
from groupoid.optimizer import curvature_adaptive_lr
from groupoid.sheaf import Sheaf


class TestMorphismRepr:
    def test_repr_shows_arrow(self):
        m = Morphism(source="A", target="B", transport_map=np.eye(2))
        assert repr(m) == "Morphism(A -> B)"


class TestSheafSections:
    def test_section_round_trip(self):
        sheaf = Sheaf(nx.DiGraph([("A", "B")]))
        value = np.array([1.0, 2.0, 3.0])
        sheaf.set_section("A", value)
        np.testing.assert_array_equal(sheaf.get_section("A"), value)


class _FakeOptimizer:
    """Mimics a geomstats gradient-descent optimizer with tunable knobs."""

    def __init__(self) -> None:
        self.max_iter = 0
        self.epsilon = 0.0


class _BareOptimizer:
    """An optimizer object exposing neither max_iter nor epsilon."""


class _FakeFrechetMean:
    """Minimal stand-in for geomstats FrechetMean used to exercise the
    optimizer-forwarding branches in karcher_mean without depending on a
    specific geomstats version's internals."""

    optimizer: object | None = None
    _next_optimizer: object | None = None

    def __init__(self, manifold) -> None:  # noqa: A002 - mirror real signature
        self.optimizer = type(self)._next_optimizer
        self.estimate_ = None

    def fit(self, points, weights=None):  # noqa: D401 - test helper
        self.estimate_ = points[0]
        return self


class TestKarcherMeanOptimizerForwarding:
    def _patch(self, monkeypatch, optimizer_obj):
        # karcher_mean imports FrechetMean lazily from
        # geomstats.learning.frechet_mean, so patching it there is what takes
        # effect.
        _FakeFrechetMean._next_optimizer = optimizer_obj
        import geomstats.learning.frechet_mean as gfm

        monkeypatch.setattr(gfm, "FrechetMean", _FakeFrechetMean, raising=False)

    def test_forwards_controls_to_optimizer(self, monkeypatch):
        opt = _FakeOptimizer()
        self._patch(monkeypatch, opt)
        pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        karcher_mean(object(), pts, max_iter=42, tol=1e-3)
        assert opt.max_iter == 42
        assert opt.epsilon == 1e-3

    def test_tolerates_optimizer_without_knobs(self, monkeypatch):
        self._patch(monkeypatch, _BareOptimizer())
        pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        # Must not raise even though the optimizer lacks max_iter/epsilon.
        out = karcher_mean(object(), pts, max_iter=5, tol=1e-2)
        np.testing.assert_array_equal(out, pts[0])

    def test_tolerates_missing_optimizer(self, monkeypatch):
        self._patch(monkeypatch, None)
        pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        out = karcher_mean(object(), pts)
        np.testing.assert_array_equal(out, pts[0])


class TestCurvatureAdaptiveLRError:
    def test_not_implemented_curvature_falls_back(self):
        """A manifold whose metric advertises sectional_curvature but raises
        NotImplementedError must return the base LR unchanged via the
        except branch, not propagate the error."""

        class _RaisingMetric:
            def sectional_curvature(self, *args, **kwargs):
                raise NotImplementedError

        class _Manifold:
            metric = _RaisingMetric()

            def to_tangent(self, vec, point):
                return vec

        m = _Manifold()
        point = np.array([0.0, 0.0, 1.0])
        tangent = np.array([1.0, 0.0, 0.0])
        assert curvature_adaptive_lr(m, point, 0.07, tangent) == 0.07


class TestCohomologyAllEdgesMissing:
    def test_cycle_with_no_transport_maps_raises(self):
        """A cycle for which no transport map exists in either direction for
        any edge leaves the holonomy undefined; compute_h1 must raise
        IncompleteCocycleError naming the first missing edge rather than
        silently returning a (false) 0.0 consistency value."""
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        with pytest.raises(IncompleteCocycleError, match="no transport map for edge"):
            compute_h1(graph, {})

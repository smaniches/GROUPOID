"""Diffusion-step energy stability for non-orthogonal sheaf transports."""

import networkx as nx
import numpy as np
import pytest

from groupoid.laplacian import build_sheaf_laplacian, sheaf_diffusion_step
from groupoid.sheaf import Sheaf


def _two_node_sheaf(scale: float = 10.0) -> Sheaf:
    graph = nx.DiGraph([("A", "B")])
    sheaf = Sheaf(graph)
    sheaf.set_restriction_map("A", "B", np.array([[scale]]))
    return sheaf


def _energy(sheaf: Sheaf, sections: dict[str, np.ndarray]) -> float:
    """Independently sum squared edge residuals, without using the L builder."""
    total = 0.0
    for u, v in sheaf.graph.edges():
        residual = sheaf.get_restriction_map(u, v) @ sections[u] - sections[v]
        total += float(residual @ residual)
    return total


def test_default_rejects_energy_amplification() -> None:
    sheaf = _two_node_sheaf()
    x = {"A": np.array([1.0]), "B": np.array([0.0])}
    L = build_sheaf_laplacian(sheaf, stalk_dim=1)
    stacked = np.concatenate([x["A"], x["B"]])

    # Prior implementation accepted eta=.1 and multiplied E by 82.81.
    previous = stacked - 0.1 * L @ stacked
    assert _energy(sheaf, x) == pytest.approx(100.0)
    assert _energy(sheaf, {"A": previous[:1], "B": previous[1:]}) == pytest.approx(8281.0)
    with pytest.raises(ValueError, match="conservative sheaf diffusion stability"):
        sheaf_diffusion_step(sheaf, x, stalk_dim=1)
    np.testing.assert_array_equal(x["A"], [1.0])
    np.testing.assert_array_equal(x["B"], [0.0])


@pytest.mark.parametrize("step", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_step_is_rejected(step: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        sheaf_diffusion_step(
            _two_node_sheaf(),
            {"A": np.array([1.0]), "B": np.array([0.0])},
            stalk_dim=1,
            step_size=step,
        )


def test_rejects_conservative_bound_boundary() -> None:
    sheaf = _two_node_sheaf()
    L = build_sheaf_laplacian(sheaf, stalk_dim=1)
    bound = float(np.linalg.norm(L, ord=np.inf))
    assert bound == pytest.approx(110.0)
    with pytest.raises(ValueError, match="stability bound"):
        sheaf_diffusion_step(
            sheaf,
            {"A": np.array([1.0]), "B": np.array([0.0])},
            stalk_dim=1,
            step_size=2.0 / bound,
        )


def test_safe_nonorthogonal_step_decreases_disagreement_energy() -> None:
    sheaf = _two_node_sheaf()
    x = {"A": np.array([1.0]), "B": np.array([0.0])}
    y = sheaf_diffusion_step(sheaf, x, stalk_dim=1, step_size=0.01)
    assert _energy(sheaf, y) == pytest.approx(0.01)
    assert _energy(sheaf, y) < _energy(sheaf, x)


def test_zero_step_and_edgeless_sheaf_are_noops() -> None:
    sheaf = _two_node_sheaf()
    x = {"A": np.array([1.0]), "B": np.array([0.0])}
    for node, value in sheaf_diffusion_step(sheaf, x, 1, step_size=0.0).items():
        np.testing.assert_array_equal(value, x[node])

    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "B"])
    no_edges = Sheaf(graph)
    for node, value in sheaf_diffusion_step(no_edges, x, 1, step_size=0.5).items():
        np.testing.assert_array_equal(value, x[node])


def test_nonfinite_laplacian_norm_is_rejected() -> None:
    sheaf = _two_node_sheaf(float("nan"))
    with pytest.raises(ValueError, match="finite norm bound"):
        sheaf_diffusion_step(
            sheaf,
            {"A": np.array([1.0]), "B": np.array([0.0])},
            stalk_dim=1,
            step_size=0.01,
        )


def test_random_nonorthogonal_sheaf_energy_cannot_increase() -> None:
    rng = np.random.default_rng(17)
    for n in (3, 5, 10):
        graph = nx.DiGraph((str(i), str(i + 1)) for i in range(n - 1))
        sheaf = Sheaf(graph)
        for u, v in graph.edges():
            sheaf.set_restriction_map(u, v, rng.normal(size=(2, 2)))
        for _ in range(20):
            x = {node: rng.normal(size=2) for node in graph.nodes()}
            L = build_sheaf_laplacian(sheaf, stalk_dim=2)
            step = 1.9 / np.linalg.norm(L, ord=np.inf)
            y = sheaf_diffusion_step(sheaf, x, stalk_dim=2, step_size=float(step))
            assert _energy(sheaf, y) <= _energy(sheaf, x) * (1.0 + 1e-12)

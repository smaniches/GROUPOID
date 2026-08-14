from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import groupoid.aggregation as aggregation_module
import groupoid.transport as transport_module
from groupoid.aggregation import (
    InvalidPointTransportError,
    TransportGroupoidAggregator,
    UnsupportedTransportRepresentationError,
)
from groupoid.cohomology import compute_h1, cycle_basis_holonomy_defect
from groupoid.groupoid import NonReciprocalTransportError
from groupoid.transport import compute_tangent_transport_matrix, compute_transport_matrix


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def cycle_defect(
    cycle: list[str],
    transport_maps: dict[tuple[str, str], np.ndarray],
) -> float:
    maps = []
    for i, u in enumerate(cycle):
        v = cycle[(i + 1) % len(cycle)]
        if (u, v) in transport_maps:
            maps.append(transport_maps[(u, v)])
        else:
            maps.append(np.linalg.inv(transport_maps[(v, u)]))
    hol = maps[0]
    for matrix in maps[1:]:
        hol = matrix @ hol
    return float(np.linalg.norm(hol - np.eye(hol.shape[0]), ord="fro"))


def test_primary_defect_zero_on_complete_coboundary_and_legacy_alias_matches():
    graph = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "C")])
    gauges = {
        "A": rotation_z(0.1),
        "B": rotation_z(0.4),
        "C": rotation_z(-0.2),
    }
    transports = {
        (u, v): gauges[v] @ np.linalg.inv(gauges[u]) for u, v in graph.edges()
    }

    defect = cycle_basis_holonomy_defect(graph, transports)
    assert defect < 1e-12

    with pytest.warns(DeprecationWarning, match="not a canonical H\\^1 norm"):
        legacy = compute_h1(graph, transports)
    assert legacy == pytest.approx(defect)


def test_cycle_basis_defect_magnitude_is_basis_dependent_even_in_so2():
    theta = np.pi / 6
    transports = {
        ("U", "A"): np.eye(2),
        ("A", "V"): np.eye(2),
        ("U", "B"): np.eye(2),
        ("B", "V"): rotation_z(theta),
        ("U", "C"): np.eye(2),
        ("C", "V"): rotation_z(2 * theta),
    }
    c12 = ["U", "A", "V", "B"]
    c23 = ["U", "B", "V", "C"]
    c13 = ["U", "A", "V", "C"]

    basis_one = max(cycle_defect(c12, transports), cycle_defect(c23, transports))
    basis_two = max(cycle_defect(c12, transports), cycle_defect(c13, transports))

    assert basis_one == pytest.approx(2 * np.sqrt(1 - np.cos(theta)))
    assert basis_two == pytest.approx(2 * np.sqrt(1 - np.cos(2 * theta)))
    assert basis_one != pytest.approx(basis_two)


def test_defect_magnitude_changes_under_general_similarity_but_zero_set_does_not():
    graph = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "C")])
    holonomy = np.array([[1.0, 1.0], [0.0, 1.0]])
    transports = {
        ("A", "B"): np.eye(2),
        ("B", "C"): np.eye(2),
        ("A", "C"): np.linalg.inv(holonomy),
    }
    original = cycle_basis_holonomy_defect(graph, transports)

    gauges = {
        "A": np.diag([4.0, 1.0]),
        "B": np.diag([4.0, 1.0]),
        "C": np.diag([4.0, 1.0]),
    }
    transformed = {
        (u, v): gauges[v] @ matrix @ np.linalg.inv(gauges[u])
        for (u, v), matrix in transports.items()
    }
    changed = cycle_basis_holonomy_defect(graph, transformed)

    assert original > 0
    assert changed > 0
    assert changed == pytest.approx(4.0 * original)

    coboundary = {(u, v): gauges[v] @ np.linalg.inv(gauges[u]) for u, v in graph.edges()}
    assert cycle_basis_holonomy_defect(graph, coboundary) < 1e-12


def test_zero_cycle_defect_does_not_certify_bridge_completeness():
    graph = nx.DiGraph([("A", "B"), ("B", "C")])
    transports = {("A", "B"): np.eye(2)}
    assert cycle_basis_holonomy_defect(graph, transports) == 0.0


def test_nonreciprocal_dual_edge_is_rejected_before_cycle_reduction():
    graph = nx.DiGraph(
        [("A", "B"), ("B", "A"), ("B", "C"), ("A", "C")]
    )
    transports = {
        ("A", "B"): np.eye(2),
        ("B", "A"): 2.0 * np.eye(2),
        ("B", "C"): np.eye(2),
        ("A", "C"): np.eye(2),
    }
    with pytest.raises(NonReciprocalTransportError, match="not mutual numerical inverses"):
        cycle_basis_holonomy_defect(graph, transports)


def test_reciprocity_validator_rejects_mismatched_opposite_shapes():
    graph = nx.DiGraph([("A", "B"), ("B", "A")])
    transports = {
        ("A", "B"): np.eye(2),
        ("B", "A"): np.eye(3),
    }
    with pytest.raises(NonReciprocalTransportError, match="finite square matrices"):
        cycle_basis_holonomy_defect(graph, transports)


def test_aggregator_rejects_nonreciprocal_reverse_registration():
    agg = make_sphere_aggregator()
    agg.register_transport("A", "B", np.eye(3))
    with pytest.raises(NonReciprocalTransportError, match="not mutual numerical inverses"):
        agg.register_transport("B", "A", 2.0 * np.eye(3))


def test_aggregator_accepts_reciprocal_reverse_registration():
    agg = make_sphere_aggregator()
    theta = np.pi / 5
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    agg.register_transport("A", "B", rotation)
    agg.register_transport("B", "A", rotation.T)
    np.testing.assert_allclose(
        agg.morphisms[("B", "A")].transport_map,
        np.linalg.inv(agg.morphisms[("A", "B")].transport_map),
        atol=1e-12,
    )


class ProjectionSphere:
    def belongs(self, point: np.ndarray) -> bool:
        return bool(np.isclose(np.linalg.norm(point), 1.0, atol=1e-10))

    def to_tangent(self, vector: np.ndarray, point: np.ndarray) -> np.ndarray:
        return vector - np.dot(vector, point) * point


def test_exact_tangent_projector_extension_is_singular(monkeypatch):
    manifold = ProjectionSphere()
    base = np.array([0.0, 0.0, 1.0])
    end = np.array([0.0, 0.0, 1.0])

    def identity_transport(_manifold, tangent, _base, _end, n_rungs=1):
        return tangent

    monkeypatch.setattr(transport_module, "pole_ladder", identity_transport)
    operator = compute_tangent_transport_matrix(manifold, base, end, method="pole")

    assert np.linalg.matrix_rank(operator) == 2
    np.testing.assert_allclose(operator @ base, np.zeros(3), atol=1e-12)
    assert np.linalg.det(operator) == pytest.approx(0.0)

    with pytest.warns(DeprecationWarning, match="generic transport matrix"):
        legacy = compute_transport_matrix(manifold, base, end, method="pole")
    np.testing.assert_allclose(legacy, operator)


def fake_sphere_mean(manifold, points, weights=None, **_kwargs):
    if weights is None:
        mean = points.mean(axis=0)
    else:
        mean = np.average(points, axis=0, weights=weights)
    return mean / np.linalg.norm(mean)


def make_sphere_aggregator(base_node: str = "A") -> TransportGroupoidAggregator:
    return TransportGroupoidAggregator(
        manifold=ProjectionSphere(),
        graph=nx.DiGraph([("A", "B")]),
        base_node=base_node,
    )


def test_from_points_path_fails_closed():
    agg = make_sphere_aggregator()
    with pytest.raises(UnsupportedTransportRepresentationError, match="tangent-vector"):
        agg.register_transport_from_points(
            "B",
            "A",
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
        )
    assert not agg.morphisms


@pytest.mark.parametrize(
    "matrix",
    [
        np.ones((2, 3)),
        np.array([[1.0, np.nan], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.diag([1e-320, 1.0]),
    ],
)
def test_register_transport_rejects_noninvertible_or_malformed_point_actions(matrix):
    agg = make_sphere_aggregator()
    with pytest.raises(InvalidPointTransportError):
        agg.register_transport("A", "B", matrix)


def test_explicit_rotation_point_action_round_trip_is_supported(monkeypatch):
    monkeypatch.setattr(aggregation_module, "karcher_mean", fake_sphere_mean)
    agg = make_sphere_aggregator()
    theta = np.pi / 3
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    agg.register_transport("A", "B", rotation)

    north = np.array([0.0, 0.0, 1.0])
    result = agg.aggregate({"A": north, "B": north})

    assert result.passes_consistency_threshold
    assert result.is_consistent
    assert result.cycle_basis_holonomy_defect == result.h1_norm
    assert result.orthogonality_residuals == result.transport_residuals
    assert agg.manifold.belongs(result.global_params)
    assert all(agg.manifold.belongs(point) for point in result.local_updates.values())

    forward = agg._get_transport_to_base("B")
    assert forward is not None
    np.testing.assert_allclose(np.linalg.inv(forward) @ (forward @ north), north, atol=1e-12)


def test_invertible_matrix_that_leaves_manifold_fails_at_exercised_point(monkeypatch):
    monkeypatch.setattr(aggregation_module, "karcher_mean", fake_sphere_mean)
    agg = make_sphere_aggregator()
    agg.register_transport("A", "B", np.diag([2.0, 1.0, 1.0]))

    x_axis = np.array([1.0, 0.0, 0.0])
    with pytest.raises(InvalidPointTransportError, match="outside the configured manifold"):
        agg.aggregate({"A": x_axis, "B": x_axis})


def test_composite_return_inverse_must_remain_finite(monkeypatch):
    monkeypatch.setattr(aggregation_module, "karcher_mean", fake_sphere_mean)
    agg = make_sphere_aggregator()
    unstable = np.diag([1e-320, 1.0, 1.0])

    monkeypatch.setattr(
        agg,
        "_get_transport_to_base",
        lambda node: unstable if node == "B" else None,
    )

    north = np.array([0.0, 0.0, 1.0])
    with pytest.raises(InvalidPointTransportError, match="non-finite numerical inverse"):
        agg.aggregate({"A": north, "B": north})


def test_non_manifold_client_input_fails_closed(monkeypatch):
    monkeypatch.setattr(aggregation_module, "karcher_mean", fake_sphere_mean)
    agg = make_sphere_aggregator()
    agg.register_transport("A", "B", np.eye(3))

    with pytest.raises(InvalidPointTransportError, match="client A input"):
        agg.aggregate(
            {
                "A": np.array([0.0, 0.0, 2.0]),
                "B": np.array([0.0, 0.0, 1.0]),
            }
        )


def test_incomplete_basis_cycle_raises_explicitly():
    graph = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "C")])
    transports = {("A", "B"): np.eye(2), ("B", "C"): np.eye(2)}
    from groupoid.cohomology import IncompleteCocycleError

    with pytest.raises(IncompleteCocycleError, match="no transport map for edge"):
        cycle_basis_holonomy_defect(graph, transports)


class AlwaysPointManifold:
    def belongs(self, point: np.ndarray) -> bool:
        return True


class NoBelongsManifold:
    pass


def test_point_contract_requires_manifold_membership_predicate(monkeypatch):
    monkeypatch.setattr(aggregation_module, "karcher_mean", fake_sphere_mean)
    agg = TransportGroupoidAggregator(
        manifold=NoBelongsManifold(),
        graph=nx.DiGraph([("A", "B")]),
        base_node="A",
    )
    agg.register_transport("A", "B", np.eye(2))
    with pytest.raises(InvalidPointTransportError, match=r"manifold\.belongs"):
        agg.aggregate({"A": np.array([1.0, 0.0]), "B": np.array([1.0, 0.0])})


def test_point_action_dimension_mismatch_fails_closed():
    agg = TransportGroupoidAggregator(
        manifold=AlwaysPointManifold(),
        graph=nx.DiGraph([("A", "B")]),
        base_node="A",
    )
    agg.register_transport("A", "B", np.eye(2))
    with pytest.raises(InvalidPointTransportError, match="incompatible matrix/point dimensions"):
        agg.aggregate({"A": np.ones(3), "B": np.ones(3)})


def test_point_action_nonfinite_result_fails_closed():
    agg = TransportGroupoidAggregator(
        manifold=AlwaysPointManifold(),
        graph=nx.DiGraph([("A", "B")]),
        base_node="A",
    )
    agg.register_transport("A", "B", 0.5 * np.eye(2))
    huge = np.array([1e308, 0.0])
    with pytest.raises(InvalidPointTransportError, match="non-finite coordinates"):
        agg.aggregate({"A": huge, "B": huge})


def test_transport_path_errors_and_forward_branch():
    manifold = AlwaysPointManifold()

    disconnected = TransportGroupoidAggregator(
        manifold=manifold,
        graph=nx.DiGraph([("A", "B")]),
        base_node="A",
    )
    disconnected.graph.add_node("C")
    with pytest.raises(aggregation_module.DisconnectedClientGraphError):
        disconnected._get_transport_to_base("C")

    missing = TransportGroupoidAggregator(
        manifold=manifold,
        graph=nx.DiGraph([("A", "B")]),
        base_node="A",
    )
    with pytest.raises(ValueError, match="No transport map"):
        missing._get_transport_to_base("B")

    forward = TransportGroupoidAggregator(
        manifold=manifold,
        graph=nx.DiGraph([("A", "B")]),
        base_node="B",
    )
    forward.register_transport("A", "B", np.eye(2))
    np.testing.assert_allclose(forward._get_transport_to_base("A"), np.eye(2))
    assert forward._get_transport_to_base("B") is None


def test_threshold_warning_weighting_and_divergence_paths(monkeypatch):
    monkeypatch.setattr(aggregation_module, "karcher_mean", fake_sphere_mean)

    class Summary:
        bottleneck_to_previous = None

    summary = Summary()
    monkeypatch.setattr(
        aggregation_module._persistence, "track_divergence", lambda *_a, **_k: summary
    )

    graph = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "C")])
    agg = TransportGroupoidAggregator(
        manifold=AlwaysPointManifold(),
        graph=graph,
        base_node="A",
        consistency_threshold=1e-12,
        track_divergence=True,
    )
    agg.register_transport("A", "B", np.diag([2.0, 1.0]))
    agg.register_transport("B", "C", np.eye(2))
    agg.register_transport("A", "C", np.eye(2))
    params = {"A": np.array([1.0, 0.0]), "B": np.array([1.0, 0.0]), "C": np.array([1.0, 0.0])}
    result = agg.aggregate(params, weights={"A": 2.0, "B": 1.0, "C": 1.0})
    assert not result.passes_consistency_threshold
    assert result.divergence is summary
    assert agg._prev_divergence is summary


def test_morphism_compose_and_inverse_contract():
    from groupoid.groupoid import CompositionError, Morphism, compose, inverse

    f = Morphism(source="A", target="B", transport_map=np.array([[2.0, 0.0], [0.0, 1.0]]))
    g = Morphism(source="B", target="C", transport_map=np.array([[1.0, 1.0], [0.0, 1.0]]))
    composed = compose(f, g)
    np.testing.assert_allclose(composed.transport_map, g.transport_map @ f.transport_map)
    np.testing.assert_allclose(inverse(f).transport_map, np.linalg.inv(f.transport_map))
    assert str(f) == "Morphism(A -> B)"

    bad = Morphism(source="X", target="Y", transport_map=np.eye(2))
    with pytest.raises(CompositionError):
        compose(f, bad)


class FlatMetric:
    def log(self, end_point: np.ndarray, base_point: np.ndarray) -> np.ndarray:
        return end_point - base_point

    def exp(self, tangent_vec: np.ndarray, base_point: np.ndarray) -> np.ndarray:
        return base_point + tangent_vec


class FlatManifold:
    metric = FlatMetric()

    def to_tangent(self, vector: np.ndarray, point: np.ndarray) -> np.ndarray:
        return vector


def test_ladder_branches_execute_as_tangent_vector_utilities():
    manifold = FlatManifold()
    base = np.array([0.0, 0.0])
    end = np.array([1.0, 0.0])
    tangent = np.array([0.0, 0.2])

    pole = transport_module.pole_ladder(manifold, tangent, base, end, n_rungs=2)
    schild = transport_module.schild_ladder(manifold, tangent, base, end, n_rungs=2)
    assert np.all(np.isfinite(pole))
    assert np.all(np.isfinite(schild))

    operator = compute_tangent_transport_matrix(
        manifold,
        base,
        end,
        method="schild",
        n_rungs=2,
    )
    assert operator.shape == (2, 2)

class MatrixPointManifold:
    def to_tangent(self, vector: np.ndarray, point: np.ndarray) -> np.ndarray:
        if vector.shape != point.shape:
            raise ValueError(f"shape mismatch {vector.shape} vs {point.shape}")
        return vector


def test_tangent_transport_matrix_rejects_structured_point_representations():
    manifold = MatrixPointManifold()
    base = np.eye(2)
    end = np.eye(2)
    with pytest.raises(ValueError, match="only vector-shaped point representations"):
        compute_tangent_transport_matrix(manifold, base, end)

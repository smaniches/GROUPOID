"""Run a deterministic, measured example of transport-aware aggregation on S^2.

The *inputs* are constructed analytically in five different coordinate frames;
the Karcher means and holonomy defects are computed by GROUPOID. This is not
federated training and is not a benchmark or evidence of generalization.

Run::

    python -m examples.transport_aggregation_demo
    python -m examples.transport_aggregation_demo --json
"""

from __future__ import annotations

import argparse
import json

import networkx as nx
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere

from groupoid.aggregation import TransportGroupoidAggregator
from groupoid.manifold import karcher_mean

NODES = ("A", "B", "C", "D", "E")
EDGES = (("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A"))
NORTH = np.array([0.0, 0.0, 1.0])


def _rotation_y(angle: float) -> np.ndarray:
    """An exact SO(3) point action up to floating-point roundoff."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def _latent_points() -> dict[str, np.ndarray]:
    """Symmetric S^2 points with a known intrinsic mean at the north pole."""
    a, b = 0.12, 0.10
    return {
        "A": NORTH.copy(),
        "B": np.array([np.sin(a), 0.0, np.cos(a)]),
        "C": np.array([-np.sin(a), 0.0, np.cos(a)]),
        "D": np.array([0.0, np.sin(b), np.cos(b)]),
        "E": np.array([0.0, -np.sin(b), np.cos(b)]),
    }


def _frames() -> dict[str, np.ndarray]:
    """Known, unequal local coordinate frames, never estimated from data."""
    return {
        "A": np.eye(3),
        "B": _rotation_y(0.82),
        "C": _rotation_y(0.70),
        "D": _rotation_y(0.66),
        "E": _rotation_y(0.78),
    }


def _aggregate_case(
    manifold: Hypersphere,
    observed: dict[str, np.ndarray],
    frames: dict[str, np.ndarray],
    *,
    corrupt_edge: bool,
) -> dict[str, float | bool]:
    graph = nx.DiGraph(EDGES)
    aggregator = TransportGroupoidAggregator(manifold, graph, base_node="A")
    for u, v in EDGES:
        transport = frames[v] @ frames[u].T
        if corrupt_edge and (u, v) == ("E", "A"):
            # The perturbation remains a valid SO(3) point action. It violates
            # cycle flatness, rather than making the transport singular.
            transport = _rotation_y(0.25) @ transport
        aggregator.register_transport(u, v, transport)

    result = aggregator.aggregate(observed)
    return {
        "error_rad": float(manifold.metric.dist(result.global_params, NORTH)),
        "cycle_defect": float(result.cycle_basis_holonomy_defect),
        "passes_threshold": bool(result.passes_consistency_threshold),
        "return_frame_max_error": max(
            float(np.linalg.norm(result.local_updates[node] - frames[node] @ result.global_params))
            for node in NODES
        ),
    }


def run_demo() -> dict[str, object]:
    """Execute two aggregation cases against an independent analytic oracle.

    Symmetric points all lie in the open northern hemisphere; their intrinsic
    Frechet mean is NORTH. The untransported comparator uses the same Karcher
    estimator to isolate the effect of coordinate-frame alignment.
    """
    manifold = Hypersphere(dim=2)
    latent = _latent_points()
    frames = _frames()
    observed = {node: frames[node] @ latent[node] for node in NODES}

    # This reference deliberately does not call GROUPOID's Karcher estimator:
    # symmetry provides the answer independently of the implementation.
    untransported = karcher_mean(manifold, np.stack([observed[node] for node in NODES]))
    return {
        "setting": "deterministic synthetic S2 inputs; real numerical aggregation; no training",
        "oracle": "north pole, by symmetry of five latent points",
        "untransported_error_rad": float(manifold.metric.dist(untransported, NORTH)),
        "flat": _aggregate_case(manifold, observed, frames, corrupt_edge=False),
        "corrupted": _aggregate_case(manifold, observed, frames, corrupt_edge=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="Print stable JSON for programmatic use"
    )
    args = parser.parse_args()
    results = run_demo()
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))
        return

    flat = results["flat"]
    corrupted = results["corrupted"]
    if not isinstance(flat, dict) or not isinstance(corrupted, dict):
        raise TypeError("Aggregation cases must be dictionaries")
    print("GROUPOID | deterministic S2 transport-aggregation demonstration")
    print("Analytic oracle: north pole (symmetric latent points)")
    print(f"Untransported Karcher error (radians): {results['untransported_error_rad']:.6f}")
    for label, case in (("Flat", flat), ("Corrupted edge E->A", corrupted)):
        print(
            f"{label}: geodesic error={case['error_rad']:.6f} rad, "
            f"cycle defect={case['cycle_defect']:.3e}, "
            f"passes threshold={case['passes_threshold']}, "
            f"return-frame residual={case['return_frame_max_error']:.3e}"
        )
    print("The inputs are synthetic; results are computed, not fabricated loss curves.")
    print("No neural-network training, real-world validation, or performance claim.")


if __name__ == "__main__":
    main()

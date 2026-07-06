"""Preregistered transport-groupoid aggregation benchmark (runner).

Implements experiments/PREREGISTRATION.md exactly, with one documented
deviation: the preregistration's "run k uses the k-th child of
SeedSequence(20260706)" conflicts with its own manipulation check (d),
which requires the client data to be identical across corruption levels
for a fixed (alpha, sigma, seed). The data stream is therefore keyed by
(alpha_index, sigma_index, seed_index) -- independent of epsilon -- and
only the corruption stream is keyed by the full cell. This is recorded
in the "Deviations from preregistration" section of RESULTS.md.

Run:
    python experiments/run_benchmark.py            # writes experiments/results.json
"""

from __future__ import annotations

import json
import platform
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np

BASE_SEED = 20260706
N_CLIENTS = 8
ALPHAS = [0.0, 0.3, 0.7, 1.2]
SIGMAS = [0.05, 0.2]
EPSILONS = [0.0, 0.1, 0.3]
N_SEEDS = 25
NORTH = np.array([0.0, 0.0, 1.0])


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis."""
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)


def random_axis(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def make_clients(manifold, alpha: float, sigma: float, rng: np.random.Generator):
    """De-framed truths y_i and framed local parameters x_i = Q_i y_i."""
    ys, xs, frames = [], [], []
    for i in range(N_CLIENTS):
        xi = manifold.to_tangent(rng.normal(scale=sigma, size=3), NORTH)
        y = manifold.metric.exp(xi, NORTH)
        q = np.eye(3) if i == 0 else rotation(random_axis(rng), alpha)
        ys.append(y)
        xs.append(q @ y)
        frames.append(q)
    return np.stack(ys), np.stack(xs), frames


def make_transports(frames, epsilon: float, rng: np.random.Generator):
    """Registered transports T_{i->j} = C_{ij} Q_j Q_i^T for i < j on K_n."""
    transports = {}
    for i in range(N_CLIENTS):
        for j in range(i + 1, N_CLIENTS):
            exact = frames[j] @ frames[i].T
            if epsilon > 0:
                delta = rng.uniform(0.0, epsilon)
                exact = rotation(random_axis(rng), delta) @ exact
            transports[(str(i), str(j))] = exact
    return transports


def karcher_with_warning_flag(manifold, points):
    from groupoid.manifold import karcher_mean

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est = karcher_mean(manifold, points)
    warned = any("Maximum number of iterations" in str(w.message) for w in caught)
    return est, warned


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        raise RuntimeError("degenerate extrinsic mean")
    return v / n


def run_one(manifold, alpha_i: int, sigma_i: int, eps_i: int, seed_i: int) -> dict:
    from groupoid.aggregation import TransportGroupoidAggregator

    alpha, sigma, epsilon = ALPHAS[alpha_i], SIGMAS[sigma_i], EPSILONS[eps_i]
    # Data stream is epsilon-independent (documented deviation, see module
    # docstring); the corruption stream is keyed by the full cell.
    data_rng = np.random.default_rng(np.random.SeedSequence((BASE_SEED, alpha_i, sigma_i, seed_i)))
    corr_rng = np.random.default_rng(
        np.random.SeedSequence((BASE_SEED, 999, alpha_i, sigma_i, eps_i, seed_i))
    )

    ys, xs, frames = make_clients(manifold, alpha, sigma, data_rng)
    transports = make_transports(frames, epsilon, corr_rng)

    oracle, oracle_warned = karcher_with_warning_flag(manifold, ys)

    # M1: extrinsic (renormalized Euclidean) mean of raw local params.
    m1 = normalize(xs.mean(axis=0))
    # M2: Karcher mean of raw local params (no transport).
    m2, m2_warned = karcher_with_warning_flag(manifold, xs)
    # M3: transport to base along the direct edge, extrinsic mean. The
    # registered direction for pair {0, i} is 0->i, so i->0 is its inverse,
    # matching the aggregator's reverse-edge handling.
    transported = [xs[0]]
    for i in range(1, N_CLIENTS):
        transported.append(np.linalg.inv(transports[("0", str(i))]) @ xs[i])
    m3 = normalize(np.stack(transported).mean(axis=0))
    # M4: the full GROUPOID pipeline.
    edges = [(str(i), str(j)) for i in range(N_CLIENTS) for j in range(i + 1, N_CLIENTS)]
    graph = nx.DiGraph(edges)
    agg = TransportGroupoidAggregator(manifold=manifold, graph=graph, base_node="0")
    for (src, tgt), matrix in transports.items():
        agg.register_transport(src, tgt, matrix)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        round_result = agg.aggregate({str(i): xs[i] for i in range(N_CLIENTS)})
    m4_warned = any("Maximum number of iterations" in str(w.message) for w in caught)
    m4 = round_result.global_params

    # Manipulation checks (b) and (c): H^1 vanishes exactly on the
    # coboundary and is positive under corruption.
    if epsilon == 0:
        if round_result.h1_norm >= 1e-8:
            raise RuntimeError(
                f"check (b) failed: H^1 = {round_result.h1_norm} on exact coboundary"
            )
    elif round_result.h1_norm <= 0.0:
        raise RuntimeError("check (c) failed: H^1 = 0 under corrupted transports")

    dist = manifold.metric.dist
    return {
        "alpha": alpha,
        "sigma": sigma,
        "epsilon": epsilon,
        "seed": seed_i,
        "err_oracle": {
            "euclidean": float(dist(m1, oracle)),
            "karcher": float(dist(m2, oracle)),
            "transport_euclidean": float(dist(m3, oracle)),
            "transport_karcher": float(dist(m4, oracle)),
        },
        "err_pstar": {
            "euclidean": float(dist(m1, NORTH)),
            "karcher": float(dist(m2, NORTH)),
            "transport_euclidean": float(dist(m3, NORTH)),
            "transport_karcher": float(dist(m4, NORTH)),
        },
        "h1_norm": float(round_result.h1_norm),
        "is_consistent": bool(round_result.is_consistent),
        "karcher_warned": bool(oracle_warned or m2_warned or m4_warned),
    }


def main() -> None:
    from geomstats.geometry.hypersphere import Hypersphere

    manifold = Hypersphere(dim=2)
    runs = []
    total = len(ALPHAS) * len(SIGMAS) * len(EPSILONS) * N_SEEDS
    for alpha_i in range(len(ALPHAS)):
        for sigma_i in range(len(SIGMAS)):
            for eps_i in range(len(EPSILONS)):
                for seed_i in range(N_SEEDS):
                    runs.append(run_one(manifold, alpha_i, sigma_i, eps_i, seed_i))
                    if len(runs) % 100 == 0:
                        print(f"{len(runs)}/{total} runs complete", file=sys.stderr)

    # Manipulation check (d): M1/M2 never see the transports, so their
    # errors must be bit-identical across epsilon for fixed (alpha, sigma,
    # seed).
    by_key: dict[tuple, list[dict]] = {}
    for r in runs:
        by_key.setdefault((r["alpha"], r["sigma"], r["seed"]), []).append(r)
    for group in by_key.values():
        for method in ("euclidean", "karcher"):
            vals = {g["err_oracle"][method] for g in group}
            if len(vals) != 1:
                raise RuntimeError(f"check (d) failed: {method} varies across epsilon")

    import geomstats
    import scipy

    out = {
        "meta": {
            "preregistration": "experiments/PREREGISTRATION.md",
            "base_seed": BASE_SEED,
            "n_runs": len(runs),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "geomstats": geomstats.__version__,
            "platform": platform.platform(),
            "karcher_warning_count": sum(r["karcher_warned"] for r in runs),
        },
        "runs": runs,
    }
    path = Path(__file__).parent / "results.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} ({len(runs)} runs)")


if __name__ == "__main__":
    main()

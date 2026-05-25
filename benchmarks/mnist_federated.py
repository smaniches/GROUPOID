"""MNIST federated learning benchmark with wandb integration.

This script demonstrates the intended experiment tracking pattern.
Run with: python -m benchmarks.mnist_federated
"""

from __future__ import annotations

import numpy as np
from loguru import logger

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logger.warning("wandb not installed, metrics will only be logged locally")


def run_federated_mnist(
    n_clients: int = 5,
    n_rounds: int = 10,
    use_wandb: bool = True,
) -> dict:
    """Run a stub federated MNIST experiment.

    Parameters
    ----------
    n_clients : int
        Number of federated clients.
    n_rounds : int
        Number of communication rounds.
    use_wandb : bool
        Whether to log to wandb.

    Returns
    -------
    dict
        Summary metrics.
    """
    if use_wandb and HAS_WANDB:
        wandb.init(
            project="groupoid-benchmarks",
            config={
                "n_clients": n_clients,
                "n_rounds": n_rounds,
                "dataset": "MNIST",
                "aggregation": "karcher_mean",
            },
        )

    metrics: dict[str, list] = {"round": [], "loss": [], "accuracy": []}

    for round_idx in range(n_rounds):
        loss = 2.0 * np.exp(-0.3 * round_idx) + np.random.normal(0, 0.05)
        accuracy = 1.0 - np.exp(-0.2 * round_idx) + np.random.normal(0, 0.02)
        accuracy = np.clip(accuracy, 0.0, 1.0)

        metrics["round"].append(round_idx)
        metrics["loss"].append(float(loss))
        metrics["accuracy"].append(float(accuracy))

        logger.info(
            "Round {}/{}: loss={:.4f}, accuracy={:.4f}",
            round_idx + 1,
            n_rounds,
            loss,
            accuracy,
        )

        if use_wandb and HAS_WANDB:
            wandb.log(
                {
                    "round": round_idx,
                    "loss": loss,
                    "accuracy": accuracy,
                }
            )

    if use_wandb and HAS_WANDB:
        wandb.finish()

    return metrics


if __name__ == "__main__":
    run_federated_mnist()

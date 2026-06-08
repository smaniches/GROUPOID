"""SYNTHETIC DEMO -- NOT A BENCHMARK. NO FEDERATED TRAINING OCCURS HERE.

This script does NOT train any model and does NOT touch MNIST. It generates
synthetic loss/accuracy curves from a closed-form exponential decay plus
Gaussian noise to illustrate the experiment-tracking pattern (logging shapes,
metric names) that a real federated benchmark would use.

The numbers it produces are fabricated by design. Do not cite, plot, or
compare them as results. A real federated learning training loop does not yet
exist in this repository (see STATUS.md and LIMITATIONS.md).

Run with:
    python -m examples.synthetic_curve_demo

To log the synthetic curves to Weights & Biases (off by default), pass an
explicit opt-in flag. Logging targets a clearly-named scratch project so a
placeholder can never be mistaken for, or land in, a real results project:
    python -m examples.synthetic_curve_demo --wandb
"""

from __future__ import annotations

import argparse

import numpy as np
from loguru import logger

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logger.warning("wandb not installed, synthetic metrics will only be logged locally")

# Deliberately NOT a "results" project: this is a scratch namespace so a
# fabricated curve can never be logged into a real benchmark results project.
WANDB_DEMO_PROJECT = "groupoid-synthetic-demo-scratch"


def make_synthetic_curve_demo(
    n_clients: int = 5,
    n_rounds: int = 10,
    log_to_wandb: bool = False,
) -> dict:
    """Generate SYNTHETIC (fabricated) loss/accuracy curves. No training occurs.

    The returned curves come from ``2.0 * exp(-0.3 * round) + noise`` (loss) and
    ``1.0 - exp(-0.2 * round) + noise`` (accuracy). They are placeholders that
    illustrate logging shape only. They are not measurements and must not be
    treated as results.

    Parameters
    ----------
    n_clients : int
        Number of nominal federated clients (used only as a logged config value;
        no client actually trains).
    n_rounds : int
        Number of synthetic communication rounds to generate.
    log_to_wandb : bool
        Opt-in flag, default ``False``. When ``True`` and wandb is installed, the
        synthetic curves are logged to the scratch project
        ``groupoid-synthetic-demo-scratch`` (never a results project).

    Returns
    -------
    dict
        The synthetic curves: ``{"round": [...], "loss": [...], "accuracy": [...]}``.
    """
    logger.warning(
        "make_synthetic_curve_demo produces FABRICATED curves; no model is trained "
        "and no MNIST data is used. Do not treat the output as results."
    )

    if log_to_wandb and HAS_WANDB:
        wandb.init(
            project=WANDB_DEMO_PROJECT,
            config={
                "n_clients": n_clients,
                "n_rounds": n_rounds,
                "dataset": "NONE (synthetic demo, no MNIST)",
                "aggregation": "NONE (synthetic demo, no training)",
                "synthetic": True,
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
            "[SYNTHETIC] Round {}/{}: loss={:.4f}, accuracy={:.4f}",
            round_idx + 1,
            n_rounds,
            loss,
            accuracy,
        )

        if log_to_wandb and HAS_WANDB:
            wandb.log(
                {
                    "round": round_idx,
                    "synthetic_loss": loss,
                    "synthetic_accuracy": accuracy,
                }
            )

    if log_to_wandb and HAS_WANDB:
        wandb.finish()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SYNTHETIC demo curves (no training, no MNIST)."
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help=(
            "Opt in to logging the synthetic curves to the wandb scratch project "
            f"'{WANDB_DEMO_PROJECT}'. Off by default."
        ),
    )
    parser.add_argument("--n-clients", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=10)
    args = parser.parse_args()

    make_synthetic_curve_demo(
        n_clients=args.n_clients,
        n_rounds=args.n_rounds,
        log_to_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()

"""Compare a fresh real-digits gauge run with an immutable archived trace.

Only scientific fields are compared: wall-clock times are deliberately ignored.
This utility requires the Python standard library, not PyTorch or geomstats.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ARMS = ("canonical_fedavg", "scrambled_raw_fedavg", "transported_fedavg")


def compare(recorded: dict, fresh: dict, *, loss_atol: float = 1e-8) -> dict:
    """Fail closed on protocol drift or any per-round accuracy/loss discrepancy."""
    for field in (
        "schema_version",
        "experiment",
        "status",
        "scope",
        "contract",
        "model",
        "dataset",
    ):
        if recorded[field] != fresh[field]:
            raise AssertionError(f"Protocol mismatch in {field}")
    if recorded["status"] != "COMPLETED":
        raise AssertionError("Archived result is not complete")

    max_loss_delta = 0.0
    for arm in ARMS:
        lhs = recorded["per_round"][arm]
        rhs = fresh["per_round"][arm]
        if len(lhs) != recorded["contract"]["rounds"] or len(rhs) != len(lhs):
            raise AssertionError(f"Incorrect round count for {arm}")
        if recorded["final"][arm] != lhs[-1] or fresh["final"][arm] != rhs[-1]:
            raise AssertionError(f"Final result mismatch for {arm}")
        for old, new in zip(lhs, rhs, strict=True):
            if old["round"] != new["round"]:
                raise AssertionError(f"Round index drift in {arm}")
            if old["heldout_accuracy"] != new["heldout_accuracy"]:
                raise AssertionError(f"Held-out accuracy drift in {arm}")
            delta = abs(old["heldout_loss"] - new["heldout_loss"])
            if not math.isfinite(delta) or delta > loss_atol:
                raise AssertionError(f"Held-out loss drift in {arm}: {delta}")
            max_loss_delta = max(max_loss_delta, delta)
    for label, result in (("archived", recorded), ("fresh", fresh)):
        discrepancy = result["max_gauge_equivariance_state_l2"]
        if not math.isfinite(discrepancy) or discrepancy > 5e-11:
            raise AssertionError(f"Known-gauge equivalence failed in {label} run")
    return {
        "status": "PASS",
        "seed": recorded["contract"]["seed"],
        "rounds": recorded["contract"]["rounds"],
        "arms_checked": len(ARMS),
        "max_abs_loss_delta": max_loss_delta,
        "checked_fields": "protocol, per-round heldout accuracy/loss, gauge equivalence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--loss-atol", type=float, default=1e-8)
    args = parser.parse_args()
    if not math.isfinite(args.loss_atol) or args.loss_atol < 0:
        parser.error("--loss-atol must be finite and non-negative")
    archived = json.loads(args.archive.read_text(encoding="utf-8"))
    fresh = json.loads(args.fresh.read_text(encoding="utf-8"))
    print(json.dumps(compare(archived, fresh, loss_atol=args.loss_atol), sort_keys=True))


if __name__ == "__main__":
    main()

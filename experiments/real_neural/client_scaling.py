"""Run the archived real-digits experiment with additional client counts.

This is exploratory known-permutation replay, not unknown-gauge recovery.
The archived script and six original traces remain byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

ARCHIVE = Path(__file__).with_name("groupoid_real_neural.py.txt")
ARCHIVE_SHA256 = "4e852ddc63c0443079464cab853efc276e916e6412ca22436aef0eda73401492"
DEFAULT_SEEDS = (11, 23, 41, 59, 20261103)


def _strict_int(value: object) -> bool:
    """Exclude booleans; bool is a subclass of int."""
    return isinstance(value, int) and not isinstance(value, bool)


def run_extension(clients: int, seeds: tuple[int, ...], rounds: int = 18) -> dict:
    """Run the pinned original and preserve invalid partitions as outcomes."""
    if not _strict_int(clients) or clients < 2:
        raise ValueError("clients must be an integer >= 2")
    if not _strict_int(rounds) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if not seeds or any(not _strict_int(seed) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonempty nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds are not independent replications")

    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("archived experiment source changed: stop")
    # run_name cannot execute the archived CLI or overwrite historical files.
    original = runpy.run_path(str(ARCHIVE), run_name="__archived_experiment__")
    contract = original["Contract"]
    execute = original["main"]
    outcomes = []

    for seed in seeds:
        try:
            output = execute(contract(seed=seed, rounds=rounds, clients=clients))
        except ValueError as exc:
            # In particular, fewer than 32 samples in one non-IID partition.
            # Do not silently replace a seed that fails the frozen guard.
            outcomes.append(
                {
                    "seed": seed,
                    "status": "INVALID_PARTITION",
                    "error_type": type(exc).__name__,
                }
            )
            continue

        if output.get("status") != "COMPLETED":
            raise RuntimeError("archived experiment returned nonterminal status")
        if output["contract"]["clients"] != clients or output["contract"]["seed"] != seed:
            raise RuntimeError("execution contract mismatch")
        if len(output["per_round"]["canonical_fedavg"]) != rounds:
            raise RuntimeError("incomplete communication rounds")

        final = output["final"]
        canonical = final["canonical_fedavg"]["heldout_accuracy"]
        aligned = final["transported_fedavg"]["heldout_accuracy"]
        raw = final["scrambled_raw_fedavg"]["heldout_accuracy"]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (canonical, aligned, raw)):
            raise RuntimeError("invalid heldout accuracy")
        if abs(canonical - aligned) > 1e-12:
            raise RuntimeError("exact-known-gauge reference mismatch")
        residual = output["max_gauge_equivariance_state_l2"]
        if not math.isfinite(residual) or residual > 5e-11:
            raise RuntimeError("gauge equivariance invariant violated")

        outcomes.append(
            {
                "seed": seed,
                "status": "PASS",
                "canonical": canonical,
                "known_inverse": aligned,
                "raw_misaligned": raw,
                "max_state_l2": residual,
                "partition_sizes": output["dataset"]["partition_counts"],
            }
        )

    return {
        "schema_version": 1,
        "status": "EXPLORATORY",
        "clients": clients,
        "rounds": rounds,
        "seeds": list(seeds),
        "archived_source_sha256": digest,
        "scope": "known_exact_permutations; real digits; not unknown alignment",
        "outcomes": outcomes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=18)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_extension(args.clients, tuple(args.seeds), args.rounds)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"

    with args.out.open("x", encoding="utf-8") as file:
        file.write(payload)

    completed = sum(item["status"] == "PASS" for item in result["outcomes"])
    invalid = len(result["outcomes"]) - completed
    print(
        json.dumps(
            {
                "status": result["status"],
                "clients": args.clients,
                "completed": completed,
                "invalid": invalid,
            },
            sort_keys=True,
        )
    )
    return 0 if invalid == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

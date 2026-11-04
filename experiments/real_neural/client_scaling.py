"""Extend the archived real-digits experiment to declared client counts.

Run the *original byte-preserved* experiment implementation. No new metric
is introduced: client count is an exploratory factor, not a preregistered arm.
This does not infer unknown gauges or demonstrate an improvement over
canonical FedAvg. All outputs are local, create-only and not sent to GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

ARCHIVE = Path(__file__).with_name("groupoid_real_neural.py.txt")
ARCHIVE_SHA256 = "4e852ddc63c0443079464cab853efc276e916e6412ca22436aef0eda73401492"
DEFAULT_SEEDS = (11, 23, 41, 59, 20261103)


def run_extension(clients: int, seeds: tuple[int, ...], rounds: int = 18) -> dict:
    """Run actual archived Python, including its exact seeded partition guard."""
    if type(clients) is not int or clients < 2:
        raise ValueError("clients must be an integer >= 2")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if not seeds or any(type(x) is not int or x < 0 for x in seeds):
        raise ValueError("seeds must be nonempty nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds are not independent replications")
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("archived experiment source changed: stop")
    # run_name never equals '__main__': importing the original cannot invoke
    # its CLI and overwrite any archived evidence.
    original = runpy.run_path(str(ARCHIVE), run_name="__archived_experiment__")
    contract = original["Contract"]
    execute = original["main"]
    outcomes = []
    for seed in seeds:
        try:
            output = execute(contract(seed=seed, rounds=rounds, clients=clients))
        except ValueError as exc:
            # The archive refuses clients with <32 examples after Dirichlet
            # allocation. Preserve this validity boundary, do not retry.
            outcomes.append(dict(seed=seed, status="INVALID_PARTITION",
                                 error_type=type(exc).__name__))
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
        outcomes.append(dict(seed=seed, status="PASS", canonical=canonical,
                             known_inverse=aligned, raw_misaligned=raw,
                             max_state_l2=residual,
                             partition_sizes=output["dataset"]["partition_counts"]))
    return dict(schema_version=1, status="EXPLORATORY", clients=clients, rounds=rounds,
                seeds=list(seeds), archived_source_sha256=digest,
                scope="known_exact_permutations; real digits; not unknown alignment",
                outcomes=outcomes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=18)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_extension(args.clients, tuple(args.seeds), args.rounds)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    # Refuse overwrites. A failed run cannot masquerade as the saved result.
    with args.out.open("x", encoding="utf-8") as file:
        file.write(payload)
    print(json.dumps(dict(status=result["status"], clients=args.clients,
                          completed=sum(x["status"] == "PASS" for x in result["outcomes"]),
                          invalid=sum(x["status"] != "PASS" for x in result["outcomes"])),
                     sort_keys=True))
    return 0 if all(x["status"] == "PASS" for x in result["outcomes"]) else 2


if __name__ == "__main__":
    sys.exit(main())

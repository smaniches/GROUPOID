"""Behavioral regression for the runnable, numerical S2 aggregation example."""

import json
import subprocess
import sys
from pathlib import Path

from examples.transport_aggregation_demo import run_demo


def test_independent_oracle_and_corrupted_cycle() -> None:
    result = run_demo()
    flat = result["flat"]
    corrupted = result["corrupted"]
    assert isinstance(flat, dict) and isinstance(corrupted, dict)

    # Only the coordinate frame differs; the underlying latent mean is NORTH.
    assert result["untransported_error_rad"] > 0.1
    assert flat["error_rad"] < 1e-4
    assert flat["cycle_defect"] < 1e-10
    assert flat["passes_threshold"] is True
    assert flat["return_frame_max_error"] < 1e-10

    # The extra rotation remains a valid point action but violates flatness.
    assert corrupted["cycle_defect"] > 0.1
    assert corrupted["passes_threshold"] is False
    assert corrupted["error_rad"] > flat["error_rad"] + 1e-3
    assert corrupted["return_frame_max_error"] > 0.1


def test_json_cli_matches_library_example() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "examples.transport_aggregation_demo", "--json"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["oracle"] == "north pole, by symmetry of five latent points"
    assert result["flat"]["passes_threshold"] is True
    assert result["corrupted"]["passes_threshold"] is False
    assert result["flat"]["cycle_defect"] < 1e-10

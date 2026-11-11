"""Bound the exploratory client-scaling runner without editing original archives."""

import ast
import runpy
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "experiments/real_neural/client_scaling.py"


def _load_namespace():
    return runpy.run_path(str(RUNNER), run_name="__test__")


def test_scaling_script_parses_without_gpu_backend():
    ast.parse(RUNNER.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "clients,seeds,rounds",
    [
        (1, (11,), 18),
        (7, (), 18),
        (7, (11, 11), 18),
        (7, (11,), 0),
        (7.0, (11,), 18),
    ],
)
def test_invalid_contracts_are_refused_before_loading_archive(clients, seeds, rounds):
    namespace = _load_namespace()
    with pytest.raises(ValueError):
        namespace["run_extension"](clients, seeds, rounds)


def test_real_digits_seven_clients_one_round_when_optional_torch_exists():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    namespace = _load_namespace()
    evidence = namespace["run_extension"](7, (11,), 1)
    assert evidence["clients"] == 7
    assert evidence["outcomes"][0]["status"] == "PASS"
    assert evidence["outcomes"][0]["canonical"] == pytest.approx(
        evidence["outcomes"][0]["known_inverse"], abs=1e-12
    )

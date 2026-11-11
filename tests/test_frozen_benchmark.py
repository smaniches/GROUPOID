"""The historical benchmark runner still reproduces the stored ``results.json``.

``experiments/run_benchmark.py`` passes the Karcher tolerance that was the
library default when ``results.json`` was produced (``tol=1e-6``), because the
default became ``1e-12`` in v0.1.0.dev10 (CORRECTION_NOTICE.md, F03). This test
re-executes all 600 preregistered runs through ``run_one`` (never ``main``,
which would overwrite ``results.json``) and compares every stored record.

The tolerances admit only platform-level floating-point differences; geomstats
reports distances through ``arccos``, which resolves angles near zero only to
about 1.5e-8. Without the pinned tolerance the oracle estimates move by up to
about 1e-3 rad and this test fails.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from geomstats.geometry.hypersphere import Hypersphere

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "historical_run_benchmark", EXPERIMENTS / "run_benchmark.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_reproduces_every_stored_run() -> None:
    runner = _load_runner()
    assert runner.HISTORICAL_KARCHER_TOL == 1e-6
    stored = json.loads((EXPERIMENTS / "results.json").read_text())["runs"]
    manifold = Hypersphere(dim=2)

    keys = [
        (a, s, e, k)
        for a in range(len(runner.ALPHAS))
        for s in range(len(runner.SIGMAS))
        for e in range(len(runner.EPSILONS))
        for k in range(runner.N_SEEDS)
    ]
    assert len(keys) == len(stored) == 600

    for key, expected in zip(keys, stored, strict=True):
        record = runner.run_one(manifold, *key)
        assert record.keys() == expected.keys()
        for field in ("alpha", "sigma", "epsilon", "seed", "is_consistent", "karcher_warned"):
            assert record[field] == expected[field], (key, field)
        assert record["h1_norm"] == pytest.approx(expected["h1_norm"], rel=1e-9, abs=1e-14)
        for group in ("err_oracle", "err_pstar"):
            assert record[group].keys() == expected[group].keys()
            for method, value in expected[group].items():
                assert record[group][method] == pytest.approx(value, rel=1e-9, abs=5e-8), (
                    key,
                    group,
                    method,
                )

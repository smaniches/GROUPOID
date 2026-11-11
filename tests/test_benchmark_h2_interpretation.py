"""Stored H2 result and its corrected interpretation (CORRECTION_NOTICE.md, F05).

The statistics are recomputed from the preserved ``experiments/results.json``
with the historical definition in ``experiments/analyze.py``: the Spearman
correlation between ``h1_norm`` (the fixed cycle-basis holonomy defect) and
``err_oracle.transport_karcher`` over the 400 runs with ``epsilon > 0``.

The prespecified pooled criterion (rho >= 0.5, one-sided permutation
p < 0.01) is met as stored. The association is between the two corruption
levels: within each level, and after removing the level, it is near zero, so
the record does not support within-level prediction of aggregation error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


@pytest.fixture(scope="module")
def corrupted() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = json.loads((EXPERIMENTS / "results.json").read_text())["runs"]
    pooled = [r for r in runs if r["epsilon"] > 0]
    defect = np.array([r["h1_norm"] for r in pooled])
    error = np.array([r["err_oracle"]["transport_karcher"] for r in pooled])
    level = np.array([r["epsilon"] for r in pooled])
    return defect, error, level


def test_stored_pooled_criterion_is_reproduced(corrupted) -> None:
    defect, error, _ = corrupted
    assert len(defect) == 400
    rho = float(spearmanr(defect, error).statistic)
    assert round(rho, 3) == 0.587
    # Same permutation procedure and generator as experiments/analyze.py.
    rng = np.random.default_rng(54321)
    exceed = sum(
        float(spearmanr(defect, rng.permutation(error)).statistic) >= rho for _ in range(10_000)
    )
    pvalue = (1 + exceed) / (1 + 10_000)
    assert f"{pvalue:.5f}" == "0.00010"
    assert rho >= 0.5 and pvalue < 0.01
    text = (EXPERIMENTS / "RESULTS.md").read_text()
    assert "- Spearman rho(H^1, err(M4)) = 0.587" in text
    assert "**H2 verdict: SUPPORTED**" in text


@pytest.mark.parametrize(("level", "expected_rho"), [(0.1, -0.030), (0.3, 0.004)])
def test_no_association_within_a_corruption_level(corrupted, level, expected_rho) -> None:
    defect, error, levels = corrupted
    within = levels == level
    assert within.sum() == 200
    result = spearmanr(defect[within], error[within])
    assert float(result.statistic) == pytest.approx(expected_rho, abs=5e-4)
    assert float(result.pvalue) > 0.5


def test_partial_association_given_the_level_is_near_zero(corrupted) -> None:
    """Rank-residual partial Spearman correlation controlling for epsilon."""
    defect, error, levels = corrupted

    def residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        slope, intercept = np.polyfit(x, y, 1)
        return y - (slope * x + intercept)

    z = rankdata(levels)
    partial = float(np.corrcoef(residual(rankdata(defect), z), residual(rankdata(error), z))[0, 1])
    assert partial == pytest.approx(-0.033, abs=5e-4)
    # The pooled association comes from the shared dependence on the level.
    assert float(spearmanr(defect, levels).statistic) == pytest.approx(0.866, abs=5e-4)
    assert float(spearmanr(error, levels).statistic) == pytest.approx(0.691, abs=5e-4)

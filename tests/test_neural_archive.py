"""Machine-checkable custody and internal consistency of six neural-gauge traces.

This lightweight archive gate does not rerun the experiment or require PyTorch.
It proves source/trace bytes are unchanged and checks the recorded contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

ARCHIVE = Path(__file__).resolve().parents[1] / "experiments" / "real_neural"
SOURCE_SHA256 = "4e852ddc63c0443079464cab853efc276e916e6412ca22436aef0eda73401492"
RECORDS = {
    "groupoid_run_seed1.json": (
        20261103,
        "20df0a3cdd442f76ba8277b687ee04ae78b41a57e44a3806d8660d663f48f1cc",
    ),
    "groupoid_seed_11.json": (
        11,
        "e9d73f63d6babdf1ef496530b108ac2e7e2969bea87ea4de605a3c63c1cf09ee",
    ),
    "groupoid_seed_23.json": (
        23,
        "5a198e11fd1b41e739eb1341fc53a0953138507851fa93f0af36ecd16038b1e4",
    ),
    "groupoid_seed_37.json": (
        37,
        "e61cfe34536739a798ba7fa3f7ec7cf1fe4775ab136f39330e923f9d9ec45fb4",
    ),
    "groupoid_seed_41.json": (
        41,
        "e64987fcd0e92338f099898c687435e8f552535d1116afcdb96d56be02fe017b",
    ),
    "groupoid_seed_59.json": (
        59,
        "557921a4e1cc51418f397df826c27e3dee34dfa475c347faa8197415db3f9864",
    ),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_script_hash() -> None:
    assert _digest(ARCHIVE / "groupoid_real_neural.py.txt") == SOURCE_SHA256


@pytest.mark.parametrize("name,seed_digest", RECORDS.items())
def test_original_trace_is_intact(name: str, seed_digest: tuple[int, str]) -> None:
    seed, digest = seed_digest
    trace_path = ARCHIVE / "archives" / name
    assert _digest(trace_path) == digest
    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["status"] == "COMPLETED"
    assert record["experiment"] == "real_digits_mlp_hidden_permutation_transport"
    assert record["contract"]["seed"] == seed
    assert record["contract"]["rounds"] == 18
    assert record["contract"]["clients"] == 4
    assert record["dataset"]["examples"] == 1797
    assert record["dataset"]["test_examples"] == 540
    assert sum(record["dataset"]["partition_counts"]) == record["dataset"]["train_examples"]
    assert len(record["dataset"]["partition_counts"]) == 4
    assert record["model"]["parameter_count"] == 460
    for arm in ("canonical_fedavg", "scrambled_raw_fedavg", "transported_fedavg"):
        assert len(record["per_round"][arm]) == 18
        assert record["per_round"][arm][-1] == record["final"][arm]
        assert [x["round"] for x in record["per_round"][arm]] == list(range(1, 19))
        assert all(math.isfinite(x["heldout_loss"]) for x in record["per_round"][arm])
        assert all(0 <= x["heldout_accuracy"] <= 1 for x in record["per_round"][arm])

    # Known exact hidden-unit maps recover the canonical pointwise aggregation.
    canonical = record["final"]["canonical_fedavg"]["heldout_accuracy"]
    transported = record["final"]["transported_fedavg"]["heldout_accuracy"]
    scrambled = record["final"]["scrambled_raw_fedavg"]["heldout_accuracy"]
    assert canonical == pytest.approx(transported, abs=1e-12)
    assert scrambled < canonical
    assert record["max_gauge_equivariance_state_l2"] < 5e-11

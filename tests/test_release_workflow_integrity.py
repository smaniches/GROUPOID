"""Falsifiable shape tests for the release workflow's SBOM privilege separation.

The wheel-derived SBOM must be generated in an unprivileged job, the
privileged build job must never resolve the built wheel's runtime dependency
graph, the SBOM attestation must target the wheel subject only, and
publication must wait for the whole integrity chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"


def _jobs() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _needs(job: dict[str, Any]) -> set[str]:
    needs = job.get("needs", [])
    return {needs} if isinstance(needs, str) else set(needs)


def _run_text(job: dict[str, Any]) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _attest_sbom_steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    # The trailing "@" is load-bearing: it must not match the build job's
    # actions/attest-build-provenance@ step.
    return [
        step for step in job["steps"] if str(step.get("uses", "")).startswith("actions/attest@")
    ]


def test_sbom_job_permissions_are_exactly_contents_read() -> None:
    assert _jobs()["sbom"]["permissions"] == {"contents": "read"}


def test_attest_sbom_job_permissions_are_only_those_required() -> None:
    assert _jobs()["attest-sbom"]["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }


def test_privileged_build_job_performs_no_runtime_dependency_resolution() -> None:
    build = _jobs()["build"]
    text = _run_text(build)
    assert ".sbom-runtime" not in text
    assert "cyclonedx" not in text.lower()
    assert "bind_release_sbom" not in text
    assert "--python" not in text  # pip's install-into-another-environment mode
    assert not _attest_sbom_steps(build)


def test_sbom_generation_lives_in_the_unprivileged_sbom_job() -> None:
    sbom = _jobs()["sbom"]
    text = _run_text(sbom)
    assert "build" in _needs(sbom)
    assert ".sbom-runtime" in text
    assert "cyclonedx-py environment" in text
    assert "--verify-only" in text
    assert not _attest_sbom_steps(sbom)


def test_sbom_attestation_subject_is_wheel_only_never_sdist() -> None:
    jobs = _jobs()
    all_attest_steps = [step for job in jobs.values() for step in _attest_sbom_steps(job)]
    attest_job_steps = _attest_sbom_steps(jobs["attest-sbom"])
    assert all_attest_steps == attest_job_steps
    assert len(attest_job_steps) == 1
    step_inputs = attest_job_steps[0]["with"]
    assert step_inputs["subject-path"] == "dist/*.whl"
    # Custom predicate mode. sbom-path must stay absent: it takes precedence in
    # the action's mode detection, and that detector rejects the reproducible
    # CycloneDX document for omitting the spec-optional serialNumber.
    assert step_inputs["predicate-type"] == "https://cyclonedx.org/bom"
    assert step_inputs["predicate-path"] == "sbom.cdx.json"
    assert "sbom-path" not in step_inputs


def test_attest_sbom_job_installs_nothing_and_runs_no_code() -> None:
    job = _jobs()["attest-sbom"]
    assert _needs(job) == {"build", "sbom"}
    for step in job["steps"]:
        assert "run" not in step
        assert str(step.get("uses", "")).startswith(
            ("actions/download-artifact@", "actions/attest@")
        )


def test_publication_waits_for_full_integrity_chain() -> None:
    jobs = _jobs()
    assert any(
        str(step.get("uses", "")).startswith("actions/attest-build-provenance@")
        for step in jobs["build"]["steps"]
    )
    for publisher in ("publish-pypi", "sign-and-release"):
        assert {"build", "sbom", "attest-sbom"} <= _needs(jobs[publisher])

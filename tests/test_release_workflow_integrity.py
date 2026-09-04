"""Falsifiable shape tests for the release workflow's SBOM privilege separation.

The wheel-derived SBOM must be generated in an unprivileged job, the
privileged build job must never resolve the built wheel's runtime dependency
graph, the SBOM attestation must target the wheel subject only, and
publication must wait for the whole integrity chain.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest import mock

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


# --------------------------------------------------------------------------
# Behavioural coverage of the PyPI reconciliation the release workflow ships.
#
# These tests do not assert on the presence of strings. They extract the exact
# script the workflow will execute, run it against a stubbed PyPI index and a
# real temporary dist/, and assert on what it does to the bytes on disk, on the
# step output it writes, and on its exit status. A guard that is gutted while
# keeping its wording therefore fails these tests.
# --------------------------------------------------------------------------

WHEEL = "groupoid-9.9.9-py3-none-any.whl"
SDIST = "groupoid-9.9.9.tar.gz"

ORIGINAL = {WHEEL: b"original wheel bytes", SDIST: b"original sdist bytes"}
REBUILD = {WHEEL: b"rebuilt wheel bytes!", SDIST: b"rebuilt sdist bytes!"}


def _heredoc(step: dict[str, Any]) -> str:
    """The python program a `python - <<'PY' ... PY` step actually executes."""
    body = step["run"]
    _, _, rest = body.partition("<<'PY'\n")
    program, _, _ = rest.rpartition("\nPY")
    assert program.strip(), f"no PY heredoc found in step {step.get('name')!r}"
    return program


def _step(job: str, predicate) -> dict[str, Any]:
    return next(step for step in _jobs()[job]["steps"] if predicate(step))


class _FakePyPI:
    """Minimal stand-in for the two urllib calls the scripts make."""

    def __init__(self, published: dict[str, bytes], *, digests: dict[str, str] | None = None):
        self.published = published
        self.digests = digests or {}
        self.file_reads: list[str] = []

    def urlopen(self, url: str, timeout: int = 0) -> Any:  # noqa: ARG002
        if url.startswith("https://pypi.org/pypi/"):
            if not self.published:
                raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
            urls = [
                {
                    "filename": name,
                    "digests": {
                        "sha256": self.digests.get(name, hashlib.sha256(blob).hexdigest())
                    },
                    "url": f"https://files.example.invalid/{name}",
                }
                for name, blob in sorted(self.published.items())
            ]
            return io.BytesIO(json.dumps({"urls": urls}).encode())
        name = url.rsplit("/", 1)[-1]
        self.file_reads.append(name)
        return io.BytesIO(self.published[name])


def _exec_reconcile(
    tmp_path: Path, dist_files: dict[str, bytes], pypi: _FakePyPI
) -> tuple[int, str, dict[str, bytes], dict[str, str]]:
    """Run the shipped reconcile script; return exit code, output, dist, step outputs."""
    program = _heredoc(_step("build", lambda s: s.get("id") == "pypi"))

    dist = tmp_path / "dist"
    dist.mkdir()
    for name, blob in dist_files.items():
        (dist / name).write_bytes(blob)
    github_output = tmp_path / "github_output"
    github_output.write_text("", encoding="utf-8")

    stdout = io.StringIO()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    os.environ["GITHUB_OUTPUT"] = str(github_output)
    try:
        with (
            mock.patch.object(urllib.request, "urlopen", pypi.urlopen),
            contextlib.redirect_stdout(stdout),
        ):
            try:
                exec(compile(program, "reconcile", "exec"), {"__name__": "__main__"})  # noqa: S102
                code = 0
            except SystemExit as exit_:
                code = int(exit_.code or 0)
    finally:
        os.chdir(cwd)
        os.environ.pop("GITHUB_OUTPUT", None)

    final = {p.name: p.read_bytes() for p in sorted(dist.iterdir())}
    outputs = dict(
        line.split("=", 1)
        for line in github_output.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    return code, stdout.getvalue(), final, outputs


def test_state_absent_publishes_this_build_and_still_attests_it(tmp_path: Path) -> None:
    """Nothing on PyPI: the first-release path is untouched."""
    code, out, dist, outputs = _exec_reconcile(tmp_path, dict(REBUILD), _FakePyPI({}))

    assert code == 0
    assert "ABSENT" in out
    assert dist == REBUILD, "a first release must publish exactly what it built"
    assert outputs["adopted"] == "false"


def test_state_complete_adopts_the_published_bytes_verbatim(tmp_path: Path) -> None:
    """Fully published: the rebuild is discarded for PyPI's immutable bytes."""
    pypi = _FakePyPI(dict(ORIGINAL))
    code, out, dist, outputs = _exec_reconcile(tmp_path, dict(REBUILD), pypi)

    assert code == 0
    assert "COMPLETE" in out
    assert dist == ORIGINAL, "the release must carry PyPI's bytes, not the rebuild"
    assert outputs["adopted"] == "true"
    assert sorted(pypi.file_reads) == sorted(ORIGINAL)


def test_state_complete_fails_closed_if_a_download_does_not_match_its_digest(
    tmp_path: Path,
) -> None:
    pypi = _FakePyPI(dict(ORIGINAL), digests={WHEEL: "0" * 64})
    code, out, dist, _ = _exec_reconcile(tmp_path, dict(REBUILD), pypi)

    assert code == 1
    assert "!= PyPI" in out
    assert dist[WHEEL] != ORIGINAL[WHEEL]


def test_state_partial_fresh_dispatch_fails_and_never_completes_from_a_rebuild(
    tmp_path: Path,
) -> None:
    """The invariant this whole guard exists for.

    PyPI holds the original wheel but not the sdist. A fresh dispatch has only
    a *rebuilt* sdist to offer. Pairing it with the published wheel would
    publish, attest and release a mixture of two builds, so the run must stop.
    """
    pypi = _FakePyPI({WHEEL: ORIGINAL[WHEEL]})
    code, out, dist, outputs = _exec_reconcile(tmp_path, dict(REBUILD), pypi)

    assert code == 1, "a partially published version must fail closed on a rebuild"
    assert "PARTIAL" in out
    # It must not silently converge: no adoption, no publishable mixture, and
    # nothing downstream may treat this as an adopted run.
    assert "adopted" not in outputs
    assert dist == REBUILD, "the guard must not stage a mixed file set"
    # And it must name the only safe recovery, which reuses the ORIGINAL
    # release-dist rather than rebuilding the missing member.
    assert "FAILED JOBS" in out
    assert "Re-run all jobs" in out
    assert pypi.file_reads == [], "a partial state must not download anything"


def test_state_conflict_fails_closed_when_pypi_holds_a_file_this_build_did_not_make(
    tmp_path: Path,
) -> None:
    """COMPLETE is exact set equality, not `built` being a subset of `published`.

    PyPI holding an extra filename means the published set is not the one this
    workflow builds. Adopting it would attach a file to the release that no
    build here produced, so the run stops before downloading anything.
    """
    extra = "groupoid-9.9.9-py3-none-manylinux1_x86_64.whl"
    pypi = _FakePyPI({**ORIGINAL, extra: b"a wheel this build never made"})
    code, out, dist, outputs = _exec_reconcile(tmp_path, dict(REBUILD), pypi)

    assert code == 1
    assert "CONFLICT" in out
    assert extra in out
    assert "adopted" not in outputs
    assert pypi.file_reads == [], "a conflicting file set must not be downloaded"
    assert dist == REBUILD, "dist/ must be left exactly as built"


def test_partial_recovery_by_same_run_retry_verifies_then_uploads_only_the_gap(
    tmp_path: Path,
) -> None:
    """The supported PARTIAL recovery, as seen by publish-pypi.

    Re-running only the failed jobs does not re-execute build, so dist/ is the
    ORIGINAL release-dist. The pre-upload check must confirm the already
    published member byte-for-byte and mark only the genuinely missing one for
    upload.
    """
    program = _heredoc(
        _step("publish-pypi", lambda s: "already on PyPI, sha256" in s.get("run", ""))
    )
    code, out = _exec_publish_check(
        tmp_path, program, dict(ORIGINAL), _FakePyPI({WHEEL: ORIGINAL[WHEEL]})
    )

    assert code == 0
    assert f"{WHEEL}: already on PyPI" in out
    assert f"{SDIST}: not yet on PyPI; this run will upload it" in out


def test_publish_refuses_a_conflicting_file_set_on_the_same_run_retry_path(
    tmp_path: Path,
) -> None:
    """The build job's CONFLICT guard is bypassed by "Re-run failed jobs".

    That retry does not re-execute build, so dist/ is the retained original
    release-dist and this pre-upload check is the only thing standing between
    an unexpected published filename and a release. It must reject before the
    local gap is reported, and long before twine sees anything.
    """
    extra = "groupoid-9.9.9-py3-none-manylinux1_x86_64.whl"
    steps = _jobs()["publish-pypi"]["steps"]
    guard_index = _step_index(steps, lambda s: "already on PyPI, sha256" in s.get("run", ""))
    publish_index = _step_index(
        steps, lambda s: str(s.get("uses", "")).startswith("pypa/gh-action-pypi-publish@")
    )
    # A non-zero exit here fails the job, and the upload is a later step.
    assert guard_index < publish_index

    # dist/ is the retained original release-dist; PyPI has the wheel (matching
    # bytes) plus a file that build never produced, and lacks the sdist.
    pypi = _FakePyPI({WHEEL: ORIGINAL[WHEEL], extra: b"a wheel this build never made"})
    code, out = _exec_publish_check(tmp_path, _heredoc(steps[guard_index]), dict(ORIGINAL), pypi)

    assert code == 1
    assert "CONFLICT" in out
    assert extra in out
    # It must stop before deciding to publish the gap, not after.
    assert "this run will upload it" not in out
    assert SDIST not in out


def test_publish_refuses_when_a_published_filename_holds_different_bytes(
    tmp_path: Path,
) -> None:
    """Reproduces run 32427167884, which skip-existing alone reported as success."""
    program = _heredoc(
        _step("publish-pypi", lambda s: "already on PyPI, sha256" in s.get("run", ""))
    )
    code, out = _exec_publish_check(tmp_path, program, dict(REBUILD), _FakePyPI(dict(ORIGINAL)))

    assert code == 1
    assert "PyPI files are immutable" in out


def _exec_publish_check(
    tmp_path: Path, program: str, dist_files: dict[str, bytes], pypi: _FakePyPI
) -> tuple[int, str]:
    dist = tmp_path / "dist"
    dist.mkdir()
    for name, blob in dist_files.items():
        (dist / name).write_bytes(blob)
    stdout = io.StringIO()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with (
            mock.patch.object(urllib.request, "urlopen", pypi.urlopen),
            contextlib.redirect_stdout(stdout),
        ):
            try:
                exec(
                    compile(program, "publish-check", "exec"), {"__name__": "__main__"}
                )  # noqa: S102
                code = 0
            except SystemExit as exit_:
                code = int(exit_.code or 0)
    finally:
        os.chdir(cwd)
    return code, stdout.getvalue()


# --------------------------------------------------------------------------
# Shape guards for the parts that are wiring rather than logic.
# --------------------------------------------------------------------------


ADOPTED_CLAUSE = "needs.build.outputs.adopted != 'true'"


def _gate_admits(expression: str, adopted: str) -> bool:
    """Would this `if:` let the step/job run, given that build output value?

    Asserts the expression actually gates on the reconciliation result, then
    evaluates that clause. `adopted` is '' when the output is unset, which is
    how GitHub renders a job whose output was never written.
    """
    assert ADOPTED_CLAUSE in expression, f"{expression!r} does not gate on the adoption result"
    return adopted != "true"


def _publish_action_step() -> dict[str, Any]:
    return next(
        s
        for s in _jobs()["publish-pypi"]["steps"]
        if str(s.get("uses", "")).startswith("pypa/gh-action-pypi-publish@")
    )


def test_build_exposes_the_reconciliation_result_to_downstream_jobs() -> None:
    assert _jobs()["build"]["outputs"] == {"adopted": "${{ steps.pypi.outputs.adopted }}"}


def test_complete_adoption_skips_the_pypi_publishing_action() -> None:
    """dist/ IS what PyPI serves, so uploading would only mint orphan attestations."""
    assert not _gate_admits(_publish_action_step()["if"], "true")


def test_complete_adoption_does_not_run_sign_and_release() -> None:
    """Re-signing would overwrite the existing release's asset set."""
    assert not _gate_admits(_jobs()["sign-and-release"]["if"], "true")


def test_complete_adoption_still_verifies_pypi_end_to_end() -> None:
    """Adoption is verification, so both hash checks must run ungated."""
    steps = _jobs()["publish-pypi"]["steps"]
    for marker in ("already on PyPI, sha256", "verified against PyPI"):
        step = next(s for s in steps if marker in s.get("run", ""))
        assert "if" not in step, f"{step['name']!r} must run on the adoption path too"


def test_absent_still_publishes_and_releases() -> None:
    """A first publication is unaffected by either gate."""
    assert _gate_admits(_publish_action_step()["if"], "false")
    assert _gate_admits(_jobs()["sign-and-release"]["if"], "false")
    # The release gate keeps its original trigger condition alongside the new one.
    gate = _jobs()["sign-and-release"]["if"]
    assert "github.event_name == 'push'" in gate
    assert "github.event.inputs.create_github_release == 'true'" in gate


def test_same_run_failed_job_recovery_can_still_publish_and_release() -> None:
    """Re-running failed jobs reuses the original build's output, not a new one.

    That build saw ABSENT and emitted 'false' (or, if it never got that far,
    nothing at all), so the missing PyPI member and the GitHub Release can
    still be completed from the retained original release-dist.
    """
    for adopted in ("false", ""):
        assert _gate_admits(_publish_action_step()["if"], adopted)
        assert _gate_admits(_jobs()["sign-and-release"]["if"], adopted)


def _step_index(steps: list[dict[str, Any]], predicate) -> int:
    return next(i for i, step in enumerate(steps) if predicate(step))


def test_reconciliation_runs_before_anything_consumes_dist() -> None:
    steps = _jobs()["build"]["steps"]
    reconcile = _step_index(steps, lambda s: s.get("id") == "pypi")
    attest = _step_index(
        steps, lambda s: str(s.get("uses", "")).startswith("actions/attest-build-provenance@")
    )
    upload = _step_index(
        steps, lambda s: str(s.get("uses", "")).startswith("actions/upload-artifact@")
    )
    assert reconcile < attest < upload


def test_build_provenance_is_not_claimed_for_adopted_bytes() -> None:
    steps = _jobs()["build"]["steps"]
    attest = steps[
        _step_index(
            steps, lambda s: str(s.get("uses", "")).startswith("actions/attest-build-provenance@")
        )
    ]
    assert attest["if"] == "steps.pypi.outputs.adopted != 'true'"


def test_hashes_are_checked_on_both_sides_of_the_upload() -> None:
    steps = _jobs()["publish-pypi"]["steps"]
    pre = _step_index(steps, lambda s: "already on PyPI, sha256" in s.get("run", ""))
    publish = _step_index(
        steps, lambda s: str(s.get("uses", "")).startswith("pypa/gh-action-pypi-publish@")
    )
    post = _step_index(steps, lambda s: "verified against PyPI" in s.get("run", ""))
    assert pre < publish < post


def test_the_github_release_cannot_precede_the_publication_it_mirrors() -> None:
    jobs = _jobs()
    assert "publish-pypi" in _needs(jobs["sign-and-release"])
    guard = next(
        s for s in jobs["sign-and-release"]["steps"] if "not a v<version> tag" in s.get("run", "")
    )
    assert guard["env"]["RELEASE_TAG"] == "${{ github.event.inputs.ref || github.ref_name }}"

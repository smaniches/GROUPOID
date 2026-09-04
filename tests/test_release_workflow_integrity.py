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
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _jobs() -> dict[str, Any]:
    return _workflow()["jobs"]


def _on() -> dict[str, Any]:
    # PyYAML resolves the bare `on:` key to the boolean True.
    return _workflow()[True]


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


CONCURRENCY_EXPRESSION = "${{ github.event.inputs.ref || github.ref_name }}"


def _concurrency_key(event: str, ref_name: str, inputs_ref: str | None = None) -> str:
    """Render the workflow's concurrency group for one concrete event context.

    Pins the shipped expression, then evaluates it. GitHub's `a || b` yields b
    when a is null or the empty string, and `github.event.inputs` exists only on
    a workflow_dispatch, so on a push the input side is always absent.
    """
    group = _workflow()["concurrency"]["group"]
    prefix, marker, rest = group.partition("${{")
    assert marker, f"concurrency group {group!r} interpolates nothing"
    assert f"{marker}{rest}" == CONCURRENCY_EXPRESSION, f"unexpected expression in {group!r}"
    supplied = (inputs_ref or "") if event == "workflow_dispatch" else ""
    return prefix + (supplied or ref_name)


def test_the_dispatch_ref_input_is_the_optional_string_this_key_assumes() -> None:
    """The key is only correct if `ref` really is an optional, defaulted string."""
    ref_input = _on()["workflow_dispatch"]["inputs"]["ref"]
    assert ref_input["required"] is False
    assert ref_input["default"] == ""
    # No `type:` means string; a boolean or choice input would render differently.
    assert "type" not in ref_input


def test_same_version_serializes_across_tag_push_and_dispatch() -> None:
    """The race: two runs for one version must never publish concurrently."""
    from_push = _concurrency_key("push", ref_name="v0.1.0.dev5")
    from_dispatch = _concurrency_key(
        "workflow_dispatch", ref_name="main", inputs_ref="v0.1.0.dev5"
    )
    assert from_push == from_dispatch == "groupoid-release-v0.1.0.dev5"


def _release_target_guard() -> dict[str, Any]:
    return next(s for s in _jobs()["build"]["steps"] if "not a v<version> tag" in s.get("run", ""))


def _guard_admits(target: str) -> bool:
    """Evaluate the build guard's shell `case` for one effective target."""
    body = _release_target_guard()["run"]
    assert "v[0-9]*)" in body, f"guard does not match on a version tag: {body}"
    completed = subprocess.run(
        ["bash", "-c", body],
        env={**os.environ, "RELEASE_TARGET": target},
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def test_the_guard_runs_before_anything_is_checked_out_or_built() -> None:
    """The key is only trustworthy if non-tag targets die before publication."""
    steps = _jobs()["build"]["steps"]
    assert steps[0]["name"] == _release_target_guard()["name"]
    assert str(steps[1].get("uses", "")).startswith("actions/checkout@")
    assert _release_target_guard()["env"]["RELEASE_TARGET"] == (
        "${{ github.event.inputs.ref || github.ref_name }}"
    )


def test_a_blank_or_branch_dispatch_ref_cannot_reach_publication() -> None:
    """The residual race: a blank ref keys on the branch but builds the tag's version.

    main's pyproject.toml carries the version auto-tag-release turns into the
    tag, so such a run would publish the same version as the tag-keyed run
    while sitting in a different concurrency group. It must not get that far.
    """
    assert _concurrency_key("workflow_dispatch", ref_name="main", inputs_ref="") != (
        _concurrency_key("push", ref_name="v0.1.0.dev5")
    )
    for target in ("main", "", "refs/tags/v0.1.0.dev5", "0.1.0.dev5"):
        assert not _guard_admits(target), f"guard admitted {target!r}"


def test_the_guard_admits_exactly_the_targets_the_key_serializes() -> None:
    for target in ("v0.1.0.dev5", "v0.1.0.dev6", "v1.0.0"):
        assert _guard_admits(target), f"guard rejected {target!r}"


def test_different_versions_do_not_block_each_other() -> None:
    keys = {
        _concurrency_key("push", ref_name="v0.1.0.dev5"),
        _concurrency_key("push", ref_name="v0.1.0.dev6"),
        _concurrency_key("workflow_dispatch", ref_name="main", inputs_ref="v0.1.0.dev6"),
    }
    assert len(keys) == 2, keys


def test_the_group_is_namespaced_to_this_workflow() -> None:
    group = _workflow()["concurrency"]["group"]
    assert group.startswith("groupoid-release-")


def test_a_waiting_run_never_cancels_one_that_may_be_publishing() -> None:
    assert _workflow()["concurrency"]["cancel-in-progress"] is False


def test_the_lock_is_workflow_level_not_job_level() -> None:
    """Job-level locks would leave reconciliation outside the critical section."""
    workflow = _workflow()
    assert "concurrency" in workflow
    for name, job in workflow["jobs"].items():
        assert "concurrency" not in job, f"job {name} narrows the lock"


def test_the_four_reconciliation_states_are_unchanged(tmp_path: Path) -> None:
    """Serializing runs must not alter what a single run decides."""
    extra = "groupoid-9.9.9-py3-none-manylinux1_x86_64.whl"
    cases = {
        "ABSENT": (_FakePyPI({}), 0, dict(REBUILD), "false"),
        "COMPLETE": (_FakePyPI(dict(ORIGINAL)), 0, dict(ORIGINAL), "true"),
        "PARTIAL": (_FakePyPI({WHEEL: ORIGINAL[WHEEL]}), 1, dict(REBUILD), None),
        "CONFLICT": (
            _FakePyPI({**ORIGINAL, extra: b"never built here"}),
            1,
            dict(REBUILD),
            None,
        ),
    }
    for index, (state, (pypi, want_code, want_dist, want_adopted)) in enumerate(cases.items()):
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        code, out, dist, outputs = _exec_reconcile(case_dir, dict(REBUILD), pypi)
        assert code == want_code, f"{state}: exit {code}"
        assert state in out, f"{state} not reported: {out}"
        assert dist == want_dist, f"{state}: wrong dist/"
        assert outputs.get("adopted") == want_adopted, f"{state}: adopted={outputs.get('adopted')}"


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


def test_complete_adoption_does_not_mint_a_new_sbom_attestation() -> None:
    """The SBOM is regenerated with today's resolution, so it can name

    different transitive versions than the closure the released wheel shipped
    with. Attesting it on an adoption run would record permanent evidence for
    a graph that was never released.
    """
    attest = _attest_sbom_steps(_jobs()["attest-sbom"])[0]
    assert not _gate_admits(attest["if"], "true")


def test_the_attest_sbom_job_itself_is_never_gated() -> None:
    """publish-pypi needs this job to complete for its verification to run."""
    assert "if" not in _jobs()["attest-sbom"]


def test_absent_and_recovery_still_mint_the_sbom_attestation() -> None:
    attest = _attest_sbom_steps(_jobs()["attest-sbom"])[0]
    for adopted in ("false", ""):
        assert _gate_admits(attest["if"], adopted)


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

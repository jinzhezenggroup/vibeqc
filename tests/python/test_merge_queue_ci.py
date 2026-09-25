"""Keep merge-queue CI from spending runners on orphaned synthetic commits."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _job(source: str, name: str) -> str:
    section = source.split(f"\n  {name}:\n", 1)[1]
    next_job = section.find("\n  ")
    return section if next_job < 0 else section[:next_job]


def test_required_merge_group_jobs_use_the_liveness_gate() -> None:
    expected = {
        "ci.yml": ("cpu", "cuda-compile", "python"),
        "cumetal-cuda.yml": ("cuda-tests",),
    }
    for filename, jobs in expected.items():
        source = (WORKFLOWS / filename).read_text(encoding="utf-8")
        gate = _job(source, "merge_queue_liveness")
        assert "/git/ref/${ref}" in gate
        assert "running CI fail-open" in gate
        assert "active=false" in gate
        for job in jobs:
            section = _job(source, job)
            assert "needs: merge_queue_liveness" in section
            assert "needs.merge_queue_liveness.outputs.active == 'true'" in section


def test_merge_group_concurrency_cancels_superseded_same_ref_runs() -> None:
    for filename in ("ci.yml", "cumetal-cuda.yml"):
        source = (WORKFLOWS / filename).read_text(encoding="utf-8")
        concurrency = source.split("concurrency:\n", 1)[1].split("\njobs:\n", 1)[0]
        assert "github.event_name == 'merge_group'" in concurrency
        assert "&& github.ref" in concurrency
        assert "cancel-in-progress:" in concurrency


def test_ci_aggregate_accepts_only_liveness_confirmed_orphans() -> None:
    source = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    section = _job(source, "pass")
    assert "merge_queue_liveness" in section
    assert "Accept an orphaned merge-group run" in section
    assert "needs.merge_queue_liveness.outputs.active != 'true'" in section
    assert "needs.merge_queue_liveness.outputs.active == 'true'" in section


def test_dequeue_cleanup_cancels_only_runs_with_deleted_queue_refs() -> None:
    source = (WORKFLOWS / "merge-queue-cleanup.yml").read_text(encoding="utf-8")
    assert "pull_request_target:" in source
    assert "types: [dequeued]" in source
    assert "actions: write" in source
    assert "-f event=merge_group" in source
    assert "gh-readonly-queue/" in source
    assert "/git/ref/heads/${head_branch}" in source
    assert "404)" in source
    assert "/actions/runs/${run_id}/cancel" in source
    assert "Could not verify" in source

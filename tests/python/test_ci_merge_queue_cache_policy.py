"""Static regression gates for merge-queue CUDA cache provenance and fallback."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github" / "workflows" / "ci.yml").read_text()


def _step(name: str) -> str:
    marker = f"      - name: {name}\n"
    start = WORKFLOW.index(marker)
    end = WORKFLOW.find("      - name: ", start + len(marker))
    return WORKFLOW[start:] if end < 0 else WORKFLOW[start:end]


def test_trusted_cuda_snapshot_is_scoped_to_successful_master_ci_pushes() -> None:
    discovery = _step("Locate trusted master CUDA ccache snapshot")
    assert "/actions/workflows/ci.yml/runs" in discovery
    assert "-f branch=master -f event=push -f status=success" in discovery
    assert '.event == "push"' in discovery
    assert '.head_branch == "master"' in discovery
    assert '.conclusion == "success"' in discovery
    assert "/actions/runs/${candidate_run_id}/artifacts" in discovery
    assert "/actions/artifacts" not in discovery


def test_trusted_cuda_snapshot_failures_fall_back_instead_of_gating_ci() -> None:
    discovery = _step("Locate trusted master CUDA ccache snapshot")
    download = _step("Restore trusted master CUDA ccache snapshot")
    fallback = _step("Restore CUDA ccache")
    publish = _step("Publish trusted master CUDA ccache snapshot")

    assert 'if runs_payload="$(gh api --method GET' in discovery
    assert 'if ! artifacts_payload="$(gh api --method GET' in discovery
    assert "falling back to actions/cache or a cold compile" in discovery
    assert "continue-on-error: true" in download
    assert "steps.trusted_cuda_ccache_download.outcome != 'success'" in fallback
    assert "continue-on-error: true" in publish

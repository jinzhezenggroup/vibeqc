"""Exercise the workflow's actual shell body with a controlled GitHub API stub."""

import os
import subprocess
import textwrap
from pathlib import Path


def run_report(
    tmp_path: Path, mode: str
) -> tuple[subprocess.CompletedProcess[str], str]:
    root = Path(__file__).resolve().parents[2]
    body = (
        (root / ".github/workflows/pr-overlap.yml")
        .read_text()
        .split("        run: |\n", 1)[1]
    )
    script = tmp_path / "report.sh"
    script.write_text(textwrap.dedent(body))
    gh = tmp_path / "gh"
    gh.write_text("""#!/bin/bash
case "$*" in
  *"pulls?state=open"*)
    if [ "$TEST_MODE" = fail ]; then echo 'API unavailable' >&2; exit 42; fi
    if [ "$TEST_MODE" = empty ]; then echo 632; else printf '632\\n633\\n'; fi;;
  *"pulls/632/files"*) printf 'docs/cuda_ownership.json\\npython/example.py\\n';;
  *"pulls/633/files"*) printf 'docs/cuda_ownership.json\\n';;
  *) exit 99;;
esac
""")
    gh.chmod(0o755)
    summary = tmp_path / "summary.md"
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "REPO": "owner/repo",
        "PR_NUMBER": "632",
        "GITHUB_STEP_SUMMARY": str(summary),
        "TEST_MODE": mode,
    }
    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result, summary.read_text() if summary.exists() else ""


def test_failed_pr_listing_cannot_report_no_overlap(tmp_path: Path) -> None:
    result, summary = run_report(tmp_path, "fail")
    assert result.returncode == 0
    assert "overlap status is unknown" in result.stdout
    assert "Overlap status is unknown" in summary
    assert "No changed-file overlap" not in summary


def test_confirmed_empty_listing_reports_no_overlap(tmp_path: Path) -> None:
    result, summary = run_report(tmp_path, "empty")
    assert result.returncode == 0, result.stderr
    assert "No changed-file overlap" in summary


def test_shared_hotspot_is_reported(tmp_path: Path) -> None:
    result, summary = run_report(tmp_path, "overlap")
    assert result.returncode == 0, result.stderr
    assert "#633" in summary and "**HOTSPOT**" in summary
    assert "::warning title=High-conflict overlap with PR #633" in result.stdout

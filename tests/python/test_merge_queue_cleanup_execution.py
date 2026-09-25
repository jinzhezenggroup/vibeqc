"""Run the actual privileged cleanup shell against local API fixtures only."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40
OTHER = "b" * 40

GH = r'''
import json, os, sys
from pathlib import Path
if "POST" in sys.argv:
    endpoint = next(arg for arg in sys.argv if arg.endswith("/cancel"))
    with open(os.environ["CANCEL_LOG"], "a") as log:
        log.write(endpoint.split("/")[-2] + "\n")
else:
    page = int(next(arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("page=")))
    rows = json.loads(Path(os.environ["RUN_FIXTURE"]).read_text())
    print(json.dumps({"workflow_runs": rows[(page-1)*100:page*100]}))
'''
CURL = r'''
import os, sys
from pathlib import Path
output = sys.argv[sys.argv.index("--output") + 1]
Path(output).write_text(os.environ["REF_BODY"])
print(os.environ["REF_STATUS"], end="")
'''


def _run(tmp_path: Path, rows: list[dict], status: str, body: str) -> list[int]:
    if shutil.which("bash") is None or shutil.which("jq") is None:
        pytest.skip("requires bash and jq")
    workflow = (ROOT / ".github/workflows/merge-queue-cleanup.yml").read_text()
    assert "uses: actions/checkout" not in workflow
    script = textwrap.dedent(workflow.split("        run: |\n", 1)[1])
    for name, source in (("gh", GH), ("curl", CURL)):
        executable = tmp_path / name
        executable.write_text(f"#!{sys.executable}\n" + source)
        executable.chmod(0o755)
    fixture, log = tmp_path / "runs.json", tmp_path / "cancelled.txt"
    fixture.write_text(json.dumps(rows))
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "GITHUB_REPOSITORY": "fixture/repository",
        "GH_TOKEN": "test-only-no-credential",
        "RUN_FIXTURE": str(fixture),
        "CANCEL_LOG": str(log),
        "REF_STATUS": status,
        "REF_BODY": body,
    }
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if not log.exists():
        return []
    return [int(line) for line in log.read_text().splitlines()]


def _row(
    run: int = 42,
    *,
    head: str = SHA,
    branch: str = "gh-readonly-queue/master/pr-1",
    state: str = "in_progress",
) -> dict:
    return {
        "id": run,
        "status": state,
        "head_branch": branch,
        "head_sha": head,
        "html_url": "https://example.invalid/run",
    }


@pytest.mark.parametrize(
    "status,body,head,expected",
    [
        ("200", json.dumps({"object": {"sha": SHA}}), SHA, []),
        ("200", json.dumps({"object": {"sha": OTHER}}), SHA, [42]),
        ("404", "{}", SHA, [42]),
        ("403", "{}", SHA, []),
        ("500", "{}", SHA, []),
        ("000", "", SHA, []),
        ("200", "not-json", SHA, []),
        ("200", "{}", SHA, []),
        ("200", json.dumps({"object": {"sha": OTHER}}), "", []),
        ("200", json.dumps({"object": {"sha": "short"}}), SHA, []),
    ],
)
def test_cleanup_proves_ref_or_commit_is_obsolete(
    tmp_path: Path, status: str, body: str, head: str, expected: list[int]
) -> None:
    assert _run(tmp_path, [_row(head=head)], status, body) == expected


def test_cleanup_never_cancels_nonqueue_or_terminal_runs(tmp_path: Path) -> None:
    rows = [
        _row(1, branch="master"),
        _row(2, state="completed"),
        _row(3, state="queued"),
    ]
    assert _run(tmp_path, rows, "404", "{}") == [3]


def test_cleanup_follows_more_than_one_run_page(tmp_path: Path) -> None:
    # Only a single stale queued run needs cancellation after 100 terminal rows.
    rows = [_row(i, state="completed") for i in range(100)] + [
        _row(101, state="queued")
    ]
    assert _run(tmp_path, rows, "200", json.dumps({"object": {"sha": OTHER}})) == [101]

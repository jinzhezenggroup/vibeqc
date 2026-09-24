"""Execute the actual PR ratchet on small real Git histories."""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "case", ["shrinking", "regrowth", "wrong-head", "linear", "missing-base"]
)
def test_pr_ratchet_uses_the_verified_merge_parent(tmp_path: Path, case: str) -> None:
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("requires Git and Bash")
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="test",
        GIT_AUTHOR_EMAIL="test@example.invalid",
        GIT_COMMITTER_NAME="test",
        GIT_COMMITTER_EMAIL="test@example.invalid",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        RUNNER_TEMP=str(tmp_path),
    )

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    legacy = tmp_path / "tests/python/test_codegen.py"
    legacy.parent.mkdir(parents=True)
    (tmp_path / "initial").write_text("baseline\n", encoding="utf-8")
    git("init", "-b", "base")
    if case != "missing-base":
        legacy.write_text("# baseline\n" * 3, encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "baseline")
    old_base = git("rev-parse", "HEAD")
    git("checkout", "-b", "candidate")
    legacy.write_text(
        "# candidate\n" * (4 if case == "regrowth" else 2), encoding="utf-8"
    )
    git("add", ".")
    git("commit", "-m", "candidate")
    head = git("rev-parse", "HEAD")
    git("checkout", "base")
    (tmp_path / "unrelated").write_text("new base work\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "advance target branch")
    if case != "linear":
        git("merge", "--no-ff", "candidate", "-m", "synthetic PR merge")
    tools = tmp_path / "tools"
    tools.mkdir()
    shutil.copy2(ROOT / "tools/check_codegen_test_ownership.py", tools)
    workflow = (ROOT / ".github/workflows/pre-commit.yml").read_text(encoding="utf-8")
    step = workflow.split("      - name: Prevent legacy codegen test regrowth\n", 1)[1]
    step = step.split("      - name:", 1)[0]
    command = textwrap.dedent(step.split("        run: |\n", 1)[1])
    env["OWNERSHIP_HEAD"] = old_base if case == "wrong-head" else head
    result = subprocess.run(
        ["bash", "-eu", "-c", command],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if case == "shrinking":
        assert result.returncode == 0, result.stderr
        assert "tested-base ratchet passed" in result.stdout
    else:
        assert result.returncode != 0, result.stdout
    if case == "regrowth":
        assert "grew to 4 lines (budget 3)" in result.stderr
    if case in ("wrong-head", "linear"):
        assert "Expected a two-parent PR test merge" in result.stderr

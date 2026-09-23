"""Bind evidence limits to the base actually included in the tested merge."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _evidence_step() -> str:
    workflow = (ROOT / ".github/workflows/pre-commit.yml").read_text(encoding="utf-8")
    step = workflow.split(
        "- name: Check new benchmark evidence against the PR base", 1
    )[1]
    step = step.split("      - name:", 1)[0]
    command = step.split("        run: ", 1)[1]
    if command.startswith("|\n"):
        return textwrap.dedent(command[2:])
    return command.strip()


@pytest.mark.parametrize("case", ("updated-base", "wrong-head", "linear-checkout"))
def test_evidence_uses_verified_test_merge_parent(tmp_path: Path, case: str) -> None:
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("requires git and bash")
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="test",
        GIT_AUTHOR_EMAIL="test@example.invalid",
        GIT_COMMITTER_NAME="test",
        GIT_COMMITTER_EMAIL="test@example.invalid",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
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

    git("init", "-b", "base")
    (tmp_path / "initial").write_text("baseline\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "baseline")
    stale = git("rev-parse", "HEAD")
    git("checkout", "-b", "candidate")
    (tmp_path / "candidate").write_text("PR change\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "candidate")
    head = git("rev-parse", "HEAD")
    git("checkout", "base")
    (tmp_path / "upstream").write_text("independent base change\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "updated base")
    base = git("rev-parse", "HEAD")
    if case != "linear-checkout":
        git("merge", "--no-ff", "candidate", "-m", "test merge")

    tools = tmp_path / "tools"
    tools.mkdir()
    # Observe only the selected comparison base; production evidence accounting
    # is unchanged and has its own tests. This stub does not approve evidence.
    (tools / "evidence.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "Path('invocation.json').write_text(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    env.update(
        EVIDENCE_BASE=stale, EVIDENCE_HEAD=stale if case == "wrong-head" else head
    )
    result = subprocess.run(
        ["bash", "-eu", "-c", _evidence_step()],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    invocation = tmp_path / "invocation.json"
    if case == "updated-base":
        assert result.returncode == 0, result.stderr
        assert json.loads(invocation.read_text()) == ["check-change", "--base", base]
        changed = git("diff", "--name-only", base, "HEAD").splitlines()
        assert changed == ["candidate"]
    else:
        assert result.returncode != 0
        assert not invocation.exists()

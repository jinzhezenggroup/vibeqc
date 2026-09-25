"""Execute the workflow's actual selector against small local Git histories."""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STEP = "      - name: Select change-aware PR CodSpeed coverage\n"


def _selector() -> str:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  cpu-benchmark:\n", 1)[1].split(
        "\n  upload-coverage:\n", 1
    )[0]
    assert workflow.count(STEP) == 1
    assert STEP in job
    assert job.index("actions/checkout@") < job.index(STEP)
    assert job.index(STEP) < job.index("Run CPU CodSpeed endpoint suite")
    step = job.split(STEP, 1)[1].split("      - name:", 1)[0]
    assert "if: github.event_name == 'pull_request'" in step
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("requires Git and Bash")
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _git(tmp_path, "commit", "--allow-empty", "-m", "base")
    return tmp_path


def _commit_file(repository: Path, path: str) -> str:
    target = repository / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("fixture\n", encoding="utf-8")
    _git(repository, "add", "--", path)
    _git(repository, "commit", "-m", "fixture change")
    return _git(repository, "rev-parse", "HEAD")


def _select(repository: Path, base: str, head: str) -> str:
    script = _selector().replace("${{ github.event.pull_request.base.sha }}", base)
    script = script.replace("${{ github.event.pull_request.head.sha }}", head)
    environment_file = repository / ".git" / "selector-env"
    environment_file.write_text("", encoding="utf-8")
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=repository,
        env={**os.environ, "GITHUB_ENV": str(environment_file)},
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return environment_file.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("path", "extra"),
    [
        ("src/dft/xc.cpp", "wb97mv"),
        ("src/scf/solver.cpp", "wb97mv"),
        ("src/methods/rks.cpp", "wb97mv"),
        ("src/integrals/eri.cpp", "wb97mv"),
        ("include/vibeqc.h", "wb97mv"),
        ("python/vibeqc/calculator.py", "wb97mv"),
        ("python/vibeqc_compiler/method/spec.py", "wb97mv"),
        ("python/vibeqc_compiler/xc/spec.py", "wb97mv"),
        ("python/vibeqc_compiler/integral/eri.py", "wb97mv"),
        ("manifests/method.json", "wb97mv"),
        ("upstream/libxc/functional.c", "wb97mv"),
        ("tools/libxc_metadata.py", "wb97mv"),
        ("benchmarks/test_cpu_codspeed.py", "wb97mv"),
        (".github/workflows/ci.yml", "wb97mv"),
        ("docs/README.md", ""),
        ("src/xtb/native/runtime.cpp", ""),
        ("tools/unrelated.py", ""),
        ("tests/python/test_unrelated.py", ""),
    ],
)
def test_changed_path_selects_bounded_extras(
    repository: Path, path: str, extra: str
) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    head = _commit_file(repository, path)
    assert _select(repository, base, head) == f"VIBEQC_CODSPEED_EXTRA_CASES={extra}\n"


def test_base_only_change_does_not_expand_pr_tier(repository: Path) -> None:
    _git(repository, "checkout", "-b", "candidate")
    head = _commit_file(repository, "docs/change.md")
    _git(repository, "checkout", "main")
    base = _commit_file(repository, "src/dft/base-only.cpp")
    assert _select(repository, base, head) == "VIBEQC_CODSPEED_EXTRA_CASES=\n"


def test_sensitive_rename_still_selects_advanced_endpoint(repository: Path) -> None:
    base = _commit_file(repository, "src/dft/removed.cpp")
    _git(repository, "mv", "src/dft/removed.cpp", "relocated.txt")
    _git(repository, "commit", "-m", "rename")
    head = _git(repository, "rev-parse", "HEAD")
    assert _select(repository, base, head) == "VIBEQC_CODSPEED_EXTRA_CASES=wb97mv\n"

"""Independent reference inputs are CI test data, never native build inputs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _preparation() -> tuple[str, str]:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    name = "      - name: Prepare pinned GFN1 and D3 reference inputs\n"
    assert workflow.count(name) == 1
    step = workflow.split(name, 1)[1].split("      - name:", 1)[0]
    assert "if: matrix.shard == 'core' && " in step
    assert "steps.gfn1_reference_sources.outputs.cache-hit != 'true'" in step
    commands = step.split("        run: |\n", 1)[1]
    return workflow, "\n".join(line[10:] for line in commands.splitlines())


def test_reference_setup_is_after_build_and_before_core_tests() -> None:
    workflow, commands = _preparation()
    python_job = workflow.split("\n  python:\n", 1)[1].split("\n  cpu-benchmark:\n", 1)[0]
    names = re.findall(r"^      - name: (.+)$", python_job, flags=re.MULTILINE)
    assert names.index("Build the CPU native library") < names.index(
        "Prepare pinned GFN1 and D3 reference inputs"
    ) < names.index("Run Python tests with coverage")
    assert commands.splitlines() == [
        ".venv/bin/python tools/source_registry.py sync xtbloom-gfn1-parameters",
        ".venv/bin/python tools/source_registry.py sync xtbloom-gfn1-d3",
    ]
    cache = python_job.split("      - name: Cache pinned GFN1 and D3 reference inputs\n", 1)[1].split("      - name:", 1)[0]
    assert "if: matrix.shard == 'core'" in cache
    assert "upstream/manifest.json" in cache and "tools/source_registry.py" in cache
    assert ".cache/vibeqc-sources/xtbloom-gfn1-parameters" in cache
    assert ".cache/vibeqc-sources/xtbloom-gfn1-d3" in cache
    assert "restore-keys:" not in cache
    other_jobs = workflow.replace(python_job, "")
    assert "source_registry.py sync" not in other_jobs


@pytest.mark.parametrize("failed_source", ["", "xtbloom-gfn1-parameters", "xtbloom-gfn1-d3"])
def test_sync_commands_propagate_failure_before_testing(
    tmp_path: Path, failed_source: str
) -> None:
    _, commands = _preparation()
    executable = tmp_path / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "source_registry.py").write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "assert sys.argv[1] == 'sync'\n"
        "with Path('calls.txt').open('a') as stream: stream.write(sys.argv[2] + '\\n')\n"
        "raise SystemExit(23 if sys.argv[2] == os.environ['FAIL_SOURCE'] else 0)\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["bash", "-e", "-c", commands + "\nprintf ready > ready.txt\n"],
        cwd=tmp_path,
        env={**os.environ, "FAIL_SOURCE": failed_source},
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == (23 if failed_source else 0)
    expected = ["xtbloom-gfn1-parameters"]
    if failed_source != "xtbloom-gfn1-parameters":
        expected.append("xtbloom-gfn1-d3")
    assert (tmp_path / "calls.txt").read_text().splitlines() == expected
    assert (tmp_path / "ready.txt").exists() == (not failed_source)

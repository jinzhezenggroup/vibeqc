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
    assert (
        "if: (matrix.shard == 'core' || matrix.shard == 'compiler-heavy') && " in step
    )
    assert "steps.gfn1_reference_sources.outputs.cache-hit != 'true'" in step
    commands = step.split("        run: |\n", 1)[1]
    return workflow, "\n".join(line[10:] for line in commands.splitlines())


def test_reference_setup_is_after_build_and_before_reference_tests() -> None:
    workflow, commands = _preparation()
    python_job = workflow.split("\n  python:\n", 1)[1].split("\n  cpu-benchmark:\n", 1)[
        0
    ]
    names = re.findall(r"^      - name: (.+)$", python_job, flags=re.MULTILINE)
    assert (
        names.index("Build the CPU native library")
        < names.index("Prepare pinned GFN1 and D3 reference inputs")
        < names.index("Run Python tests with coverage")
    )
    assert commands.splitlines() == [
        ".venv/bin/python tools/source_registry.py sync xtbloom-gfn1-parameters",
        ".venv/bin/python tools/source_registry.py sync xtbloom-gfn1-d3",
    ]
    cache = python_job.split(
        "      - name: Cache pinned GFN1 and D3 reference inputs\n", 1
    )[1].split("      - name:", 1)[0]
    assert "if: matrix.shard == 'core' || matrix.shard == 'compiler-heavy'" in cache
    assert "upstream/manifest.json" in cache and "tools/source_registry.py" in cache
    assert ".cache/vibeqc-sources/xtbloom-gfn1-parameters" in cache
    assert ".cache/vibeqc-sources/xtbloom-gfn1-d3" in cache
    assert "restore-keys:" not in cache
    other_jobs = workflow.replace(python_job, "")
    assert "source_registry.py sync" not in other_jobs


@pytest.mark.parametrize(
    "failed_source", ["", "xtbloom-gfn1-parameters", "xtbloom-gfn1-d3"]
)
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


@pytest.mark.parametrize(
    "shard", ["core", "compiler-heavy", "posthf", "runtime-heavy", "ecp-forces"]
)
@pytest.mark.parametrize("cache_hit", ["true", "false", ""])
def test_reference_guards_cover_both_consumers_and_cache_states(
    shard: str, cache_hit: str
) -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for name, expected in (
        (
            "Cache pinned GFN1 and D3 reference inputs",
            shard in {"core", "compiler-heavy"},
        ),
        (
            "Prepare pinned GFN1 and D3 reference inputs",
            shard in {"core", "compiler-heavy"} and cache_hit != "true",
        ),
    ):
        step = workflow.split(f"      - name: {name}\n", 1)[1].split(
            "      - name:", 1
        )[0]
        expression = step.split("        if: ", 1)[1].splitlines()[0]
        # These guards use the shared ==/!=/&&/|| boolean subset of Actions
        # and Bash. Evaluate the actual checked-in expressions, not a copy.
        expression = expression.replace("matrix.shard", '"$SHARD"').replace(
            "steps.gfn1_reference_sources.outputs.cache-hit", '"$CACHE_HIT"'
        )
        completed = subprocess.run(
            [
                "bash",
                "-c",
                f"if [[ {expression} ]]; then printf run; else printf skip; fi",
            ],
            env={**os.environ, "SHARD": shard, "CACHE_HIT": cache_hit},
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert completed.stdout == ("run" if expected else "skip")

"""Exercise the real CI shell selection without building or running workloads."""

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHARDS = ("core", "runtime-heavy", "posthf", "ecp-forces", "compiler-heavy")


def _selection(event: str, shard: str) -> tuple[str, list[str], list[str]]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("CI selection uses Bash")
    source = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    step = source.split("- name: Run Python tests with coverage", 1)[1]
    body = step.split("        run: |\n", 1)[1]
    body = body.split("          .venv/bin/python -m pytest", 1)[0]
    script = textwrap.dedent(body)
    script = script.replace("${{ github.event_name }}", event)
    script = script.replace("${{ matrix.shard }}", shard)
    script += '\nprintf "%s\\0" "$dist_mode" "${test_targets[@]}" --extra-- "${extra_args[@]}"\n'
    result = subprocess.run(
        [bash, "-eu", "-c", script], check=True, capture_output=True, cwd=ROOT
    )
    values = result.stdout.decode().split("\0")[:-1]
    boundary = values.index("--extra--")
    return values[0], values[1:boundary], values[boundary + 1 :]


@pytest.mark.parametrize(
    "event", ("pull_request", "push", "schedule", "workflow_dispatch")
)
def test_every_python_test_file_has_exactly_one_ci_shard(event: str) -> None:
    selections = {shard: _selection(event, shard) for shard in SHARDS}
    for test in sorted((ROOT / "tests/python").glob("test_*.py")):
        relative = test.relative_to(ROOT).as_posix()
        owners = []
        for shard, (_, targets, arguments) in selections.items():
            selected = "tests/python" in targets or relative in targets
            ignored = f"--ignore={relative}" in arguments
            if selected and not ignored:
                owners.append(shard)
        assert len(owners) == 1, (relative, owners)


@pytest.mark.parametrize("event", ("schedule", "workflow_dispatch"))
@pytest.mark.parametrize("shard", SHARDS)
def test_full_qualification_is_not_deselected(event: str, shard: str) -> None:
    distribution, _, arguments = _selection(event, shard)
    assert "--deselect" not in arguments
    assert distribution == ("worksteal" if shard == "core" else "loadfile")


def test_python_ci_preserves_per_test_timings_with_debug_artifacts() -> None:
    source = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert '--junitxml=".artifacts/pytest-${{ matrix.shard }}.xml"' in source
    assert "path: .artifacts/" in source

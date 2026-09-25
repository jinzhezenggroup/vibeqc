"""Execute the CI compile commands with a recording compiler, not a GPU.

The native-AOT D4 image must select its own wide-range arithmetic mode. A
runtime environment variable cannot change a precompiled Metal library.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _compile_commands() -> str:
    workflow = (ROOT / ".github/workflows/cumetal-cuda.yml").read_text()
    marker = "      - name: Build CuMetal CodSpeed benchmarks\n        run: |\n"
    assert workflow.count(marker) == 1
    body = workflow.split(marker, 1)[1].split("\n      - name:", 1)[0]
    return textwrap.dedent(body)


@pytest.mark.parametrize("inherited_mode", [None, "fast48", "wide48"])
def test_d4_aot_uses_explicit_ieee64_without_changing_fp32_proxy(
    tmp_path: Path, inherited_mode: str | None
) -> None:
    prefix = tmp_path / "toolchain with spaces"
    compiler = prefix / "bin/cumetalc"
    compiler.parent.mkdir(parents=True)
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['RECORD_ARGUMENTS'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    compiler.chmod(0o755)
    # Code generation is outside this command-selection test.
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    record = tmp_path / "arguments.jsonl"
    env = {
        **os.environ,
        "CUMETAL_PREFIX": str(prefix),
        "GITHUB_WORKSPACE": str(tmp_path),
        "VIBEQC_BENCH_BUILD": str(tmp_path / "build with spaces"),
        "RECORD_ARGUMENTS": str(record),
    }
    env.pop("CUMETAL_FP64_MODE", None)
    if inherited_mode is not None:
        env["CUMETAL_FP64_MODE"] = inherited_mode
    subprocess.run(
        ["bash", "-eu", "-c", _compile_commands()],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(calls) == 2
    fp32, d4 = calls
    assert "benchmarks/cumetal_fp32_codspeed.cu" in fp32
    assert "benchmarks/cumetal_d4_codspeed.cu" in d4
    assert not any(arg.startswith("--fp64") for arg in fp32)
    assert [arg for arg in d4 if arg.startswith("--fp64")] == ["--fp64=ieee64"]
    assert d4[-2:] == [
        "-o",
        str(tmp_path / "build with spaces/vibeqc_cumetal_d4_benchmark"),
    ]

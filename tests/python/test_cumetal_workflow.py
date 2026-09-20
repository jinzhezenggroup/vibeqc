"""Exercise CuMetal toolkit version rewrites without an Apple GPU."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/cumetal-cuda.yml"
CUMETAL_APPLE_CMAKE_CONFORMANCE = "73ee848cdb88f15518f05278ce16d55edfbbbc44"


def test_cumetal_jobs_share_apple_cmake_conformance_pin() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    pins = re.findall(r"CUMETAL_COMMIT: ([0-9a-f]{40})", workflow)
    assert pins == [CUMETAL_APPLE_CMAKE_CONFORMANCE] * 2


@pytest.mark.skipif(shutil.which("sed") is None, reason="sed is not installed")
def test_cumetal_toolkit_version_rewrites_match_literal_dots() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    expressions = re.findall(r"sed -i '' '([^']+)'", workflow)
    version_rewrites = [
        expression for expression in expressions if "/12.9/" in expression
    ]
    assert len(version_rewrites) == 2, (
        "validate both runtime and benchmark toolkit setup"
    )
    source = "CUDA Version 12.2.0\nrelease 12.2, V12.2.0\n12x2 must stay unchanged\n"
    expected = "CUDA Version 12.9.0\nrelease 12.9, V12.9.0\n12x2 must stay unchanged\n"
    for expression in version_rewrites:
        result = subprocess.run(
            ["sed", expression],
            input=source,
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        assert result.stdout == expected, f"invalid toolkit rewrite: {expression}"

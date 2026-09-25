"""Keep strict D4 rejection while exposing round-trip-precision differences."""

import math
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def comparator(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = (ROOT / "benchmarks/cumetal_d4_codspeed.cu").read_text()
    begin = source.index("bool near(")
    body = source[begin : source.index("class BenchmarkServer", begin)]
    directory = tmp_path_factory.mktemp("d4-comparator")
    unit, binary = directory / "check.cpp", directory / "check"
    unit.write_text(
        "#include <algorithm>\n#include <cmath>\n#include <cstdio>\n#include <cstdlib>\n"
        + body
        + "int main(int argc, char** argv) {\n"
        "  if(argc != 4) return 99;\n"
        "  return near(std::strtod(argv[1],nullptr),std::strtod(argv[2],nullptr),\n"
        "              std::strtod(argv[3],nullptr)) ? 0 : 1;\n}\n"
    )
    subprocess.run(
        [compiler, "-std=c++20", "-O1", str(unit), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize(
    "actual,expected,tolerance,accepted",
    [
        (0.125, 0.125, 2e-10, True),
        (0.125 + 1e-11, 0.125, 2e-10, True),
        (0.125 + 1e-7, 0.125, 2e-10, False),
        (1000.0 + 1e-4, 1000.0, 2e-9, False),
        (float("nan"), 0.125, 2e-10, False),
        (float("inf"), 0.125, 2e-10, False),
        (float("-inf"), 0.125, 2e-10, False),
        (0.125, float("nan"), 2e-10, False),
    ],
)
def test_comparator_keeps_decision_and_prints_failure_values(
    comparator: Path, actual: float, expected: float, tolerance: float, accepted: bool
) -> None:
    result = subprocess.run(
        [str(comparator), repr(actual), repr(expected), repr(tolerance)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == (0 if accepted else 1), result.stderr
    assert not result.stdout
    if accepted:
        assert not result.stderr
        return
    assert result.stderr.startswith("D4 mismatch: ")
    values = dict(item.split("=", 1) for item in result.stderr.split()[2:])
    for name, value in (("actual", actual), ("expected", expected)):
        observed = float(values[name])
        assert math.isnan(observed) if math.isnan(value) else observed == value
    assert "absolute_error" in values and "limit" in values
    if math.isfinite(actual) and math.isfinite(expected):
        assert float(values["absolute_error"]) == abs(actual - expected)
        assert float(values["limit"]) == tolerance * max(1, abs(actual), abs(expected))

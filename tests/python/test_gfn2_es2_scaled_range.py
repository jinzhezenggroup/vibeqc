"""Generated ES2 must preserve scaling before otherwise overflowing products."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("case", ["gradient", "energy"])
def test_generated_es2_preserves_native_finite_range(tmp_path: Path, case: str) -> None:
    header = tmp_path / "generated_gfn2_es2_native.hpp"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools/generate_gfn2_es2_native.py"),
            "--output",
            str(header),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    calls = {
        "gradient": "evaluate_gfn2_es2_cached_gradient_weight(1e-100, 1e200, 1e200, actual)",
        "energy": "accumulate_gfn2_es2_energy(1e308, 2.0, actual)",
    }
    expected = "1e100" if case == "gradient" else "1e308"
    source = tmp_path / "probe.cpp"
    source.write_text(
        '#include <cmath>\n#include "generated_gfn2_es2_native.hpp"\n'
        "int main() { using namespace vibeqc::xtb::generated; double actual=0; "
        f"if (!{calls[case]}) return 1; "
        f"return std::isfinite(actual) && std::abs(actual/{expected}-1)<2e-15 ? 0:2; }}\n"
    )
    binary = tmp_path / "probe"
    subprocess.run(
        ["c++", "-std=c++20", "-O2", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)

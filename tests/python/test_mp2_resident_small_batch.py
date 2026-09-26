"""Full B residency must try smaller J scratch before falling back or rejecting."""

import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.method.mp2_schedule import native_header, ri_mp2_residency_plan


@pytest.mark.parametrize("nbf,virtuals,budget", [(3, 1, 48), (4, 2, 128)])
def test_full_residency_reduces_j_scratch(nbf: int, virtuals: int, budget: int) -> None:
    plan = ri_mp2_residency_plan(0, budget, nbf, 2, virtuals, 1)
    assert plan.full_resident
    assert (plan.virtual_block, plan.j_batch, plan.peak_bytes) == (virtuals, 1, budget)


def test_generated_native_full_residency_reduces_j_scratch(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    (tmp_path / "schedule.hpp").write_text(native_header())
    source = tmp_path / "check.cpp"
    source.write_text("""#include "schedule.hpp"
int main() {
  for (auto nv : {1u, 2u}) {
    const std::size_t budget = nv == 1 ? 48 : 128;
    const auto p = vibeqc::mp2::generated::ri_mp2_residency_plan(0,budget,nv+2,2,nv,1);
    if (!p.full_resident || p.j_batch != 1 || p.virtual_block != nv || p.peak_bytes != budget) return 1;
  }
}
""")
    binary = tmp_path / "check"
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, check=False, timeout=10
    )
    assert result.returncode == 0, result.stderr

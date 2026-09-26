"""Native storage admission must match the spin-specialized generated weights."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.stationary_cuda import emit_stationary_cuda
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("spin,blocks", [("unpolarized", 1), ("polarized", 2)])
def test_storage_rejects_a_different_spin_specialization(
    tmp_path: Path, spin: str, blocks: int
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native admission requires a host C++ compiler")
    # Compile the actual production allocation function, not a Python replica.
    header = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    function = re.search(r"size_t allocation\([^\n]*\) \{.*?\n\}", header, re.DOTALL)
    assert function is not None
    source = f"""#include <cstddef>
#include <stdexcept>
using std::size_t;
constexpr size_t workers=32, record_stride=26, map_stride=8;
constexpr unsigned stationary_spin_blocks={blocks};
{function.group()}
int main() {{
  if (allocation(2, 3, 8, 4, 4, {blocks}) == 0) return 1;
  try {{ allocation(2, 3, 8, 4, 4, {3 - blocks}); }}
  catch (const std::invalid_argument&) {{ return 0; }}
  return 2;
}}
"""
    path = tmp_path / "admission.cpp"
    executable = tmp_path / "admission"
    path.write_text(source)
    subprocess.run(
        [compiler, "-std=c++17", str(path), "-o", str(executable)],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


@pytest.mark.parametrize("spin,blocks", [("unpolarized", 1), ("polarized", 2)])
def test_generated_spin_contract_precedes_runtime_header(
    spin: str, blocks: int
) -> None:
    plan = StationaryGradientPlan(
        resolve_method("PBE", spin=spin), StationaryMeanField(SCF_POINT_MODEL)
    )
    source = emit_stationary_cuda("", functional=1, plan=plan)
    declaration = f"constexpr unsigned stationary_spin_blocks = {blocks};"
    assert declaration in source
    assert source.index(declaration) < source.index(
        '#include "dft/stationary_gradient_cuda.cuh"'
    )

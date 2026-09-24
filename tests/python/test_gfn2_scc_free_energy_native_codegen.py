"""Independent TensorIR/codegen gates for GFN2 SCC energy composition."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.method.gfn2_scc_free_energy_runtime import (
    build_gfn2_scc_free_energy_program,
    build_gfn2_scc_internal_energy_program,
)
from vibeqc_compiler.tensor import execute

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "src/xtb/native"


def test_scc_energy_tensorir_preserves_component_order_and_free_energy() -> None:
    components = {
        "core": 2.0,
        "es2": -0.3,
        "es3": 0.04,
        "aes2": -0.005,
        "spin": 0.0006,
        "d4_two_body": -0.00007,
        "explicit_point_charge": 0.000008,
        "electric_field": -0.0000009,
        "periodic_embedding": 0.00000001,
    }
    internal = execute(
        build_gfn2_scc_internal_energy_program(),
        {name: np.asarray(value) for name, value in components.items()},
    ).outputs["internal_energy"]
    expected = components["core"]
    for name in (
        "es2", "es3", "aes2", "spin", "d4_two_body",
        "explicit_point_charge", "electric_field", "periodic_embedding",
    ):
        expected = expected + components[name]
    assert internal == expected

    free = execute(
        build_gfn2_scc_free_energy_program(),
        {
            "electronic_temperature": np.asarray(0.02),
            "entropy": np.asarray(0.125),
            "internal_energy": np.asarray(internal),
        },
    ).outputs["free_energy"]
    assert free == pytest.approx(internal - 0.02 * 0.125, rel=0, abs=2e-16)


def _generate(output: Path) -> bytes:
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools" / "generate_gfn2_scc_free_energy_native.py"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return output.read_bytes()


def test_scc_energy_codegen_is_deterministic_and_keeps_fma_contract(
    tmp_path: Path,
) -> None:
    header = tmp_path / "generated_gfn2_scc_free_energy_native.hpp"
    first = _generate(header)
    second = _generate(header)
    assert second == first
    assert b"compose_gfn2_scc_internal_energy" in first
    assert b"compose_gfn2_scc_free_energy" in first
    assert b"std::fma" in first
    assert b"gfn2_scc_internal_energy_hash" in first
    assert b"gfn2_scc_free_energy_hash" in first


def test_generated_scc_energy_host_rounding_contract(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native SCC energy qualification requires a C++ compiler")
    header = tmp_path / "generated_gfn2_scc_free_energy_native.hpp"
    _generate(header)
    source = tmp_path / "scc_energy.cpp"
    source.write_text(
        r"""#include <limits>
#include "generated_gfn2_scc_free_energy_native.hpp"

int main() {
  using namespace vibeqc::xtb::generated;
  double internal = 0.0;
  if (!compose_gfn2_scc_internal_energy(
          1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, internal)) return 1;
  if (internal != 45.0) return 2;

  double free_energy = 0.0;
  if (!compose_gfn2_scc_free_energy(0.5, 2.0, internal, free_energy)) return 3;
  if (free_energy != 44.0) return 4;

  const double largest = std::numeric_limits<double>::max();
  if (!compose_gfn2_scc_free_energy(largest, 2.0, largest, free_energy)) return 5;
  if (free_energy != -largest) return 6;
  return 0;
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "scc_energy"
    subprocess.run(
        [
            compiler, "-std=c++20", "-O2", "-I", str(tmp_path),
            str(source), "-o", str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=30)


def test_native_scc_energy_consumers_use_generated_composition() -> None:
    cpu = (NATIVE / "src/model/gfn2/scc_driver.cpp").read_text(encoding="utf-8")
    cuda = (NATIVE / "src/backends/cuda/gfn2_scc_free_energy.cu").read_text(
        encoding="utf-8"
    )
    for source in (cpu, cuda):
        assert '#include "generated_gfn2_scc_free_energy_native.hpp"' in source
        assert "compose_gfn2_scc_internal_energy(" in source
        assert "compose_gfn2_scc_free_energy(" in source
    assert "std::fma(-data.electronic_temperature" not in cpu
    assert "fma(-batch.electronic_temperature" not in cuda
    assert (
        "for (int component = 2; component < kGfn2SccFreeEnergyInputComponents + 1;"
        not in cuda
    )

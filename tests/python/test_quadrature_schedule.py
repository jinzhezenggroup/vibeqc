"""Pure-host shape overflow/charging gates for generated molecular quadrature."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from vibeqc_compiler.xc.quadrature_cuda import emit_quadrature_cuda


def test_codegen_has_no_runtime_or_numpy_dependency(tmp_path: Path) -> None:
    """Match CMake's bare Python environment and preserve exact scalar bytes."""
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "quadrature.cuh"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tools/generate_quadrature_cuda.py"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    assert output.read_text() == emit_quadrature_cuda()
    from vibeqc_compiler.xc import grid_response, grid_response_ir

    assert grid_response.grid_response_program is grid_response_ir.grid_response_program
    assert (
        grid_response.grid_mixed_response_program
        is grid_response_ir.grid_mixed_response_program
    )


def test_emitted_layout_counts_actual_buffer_shapes_without_cuda(
    tmp_path: Path,
) -> None:
    """No device/runtime import is needed to reject impossible resource shapes."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    header = tmp_path / "quadrature.cuh"
    source = emit_quadrature_cuda()
    assert source == emit_quadrature_cuda()
    header.write_text(source)
    main = tmp_path / "main.cpp"
    main.write_text(
        r"""
#include "quadrature.cuh"
#include <iostream>
int main() {
  using namespace vibeqc::generated::quadrature;
  for (size_t atoms : {1, 3, 24, 48, 96})
    for (size_t points : {1, 4095, 4096, 4097, 2654208}) {
      const auto l = layout(atoms, points);
      const size_t tile = std::min(points, size_t{4096});
      // Enumerate owned arrays independently of generated offsets.
      const size_t doubles = 3*atoms + atoms + 2*512 + 2*256 + atoms*atoms
                           + tile*atoms + tile*atoms + 3*tile + tile;
      if (l.device_bytes != doubles*sizeof(double) + sizeof(int)) return 1;
      if (l.weights + tile != l.doubles) return 2;
    }
  unsigned rejected = 0;
  try { (void)layout(0, 1); } catch (const std::invalid_argument&) { ++rejected; }
  try { (void)layout(1, 0); } catch (const std::invalid_argument&) { ++rejected; }
  try { (void)layout(1, SIZE_MAX); } catch (const std::overflow_error&) { ++rejected; }
  try { (void)layout(UINT32_MAX, 1); } catch (const std::overflow_error&) { ++rejected; }
  try { (void)layout(size_t{UINT32_MAX}+1, 1); }
      catch (const std::invalid_argument&) { ++rejected; }
  return rejected == 5 ? 0 : 3;
}
"""
    )
    binary = tmp_path / "layout"
    subprocess.run(
        [compiler, "-std=c++20", str(main), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True)

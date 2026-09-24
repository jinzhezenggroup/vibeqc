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


def test_emitted_cuda_uses_atom_major_2d_schedule() -> None:
    """Keep hot point-center kernels free of flattened runtime div/mod decode."""
    source = emit_quadrature_cuda()
    assert source.count("for (size_t a = blockIdx.y; a < na; a += gridDim.y)") == 2
    assert source.count("point += size_t(blockDim.x) * gridDim.x)") == 2
    assert "const size_t a = i / count, point = i % count;" not in source
    assert "partition_kernel<1><<<atom_point_grid(count, na), 128" in source

    root = Path(__file__).resolve().parents[2]
    native = (root / "src/dft/cuda_quadrature.cu").read_text()
    assert "q::distances_kernel<<<q::atom_point_grid(count, l.atoms), 128" in native

    # Production PBE endpoint shapes retained in the performance record. The
    # old distance + partition kernels each decoded atom/point with / and %.
    for atoms, points, decoded in (
        (48, 1_327_104, 254_803_968),
        (96, 2_654_208, 1_019_215_872),
    ):
        assert 4 * atoms * points == decoded


def test_points_reuse_angular_factors() -> None:
    """Move angular special functions out of every molecular-point worker."""
    source = emit_quadrature_cuda()
    assert "__global__ void polar_kernel" in source
    assert "__global__ void azimuth_kernel" in source
    points = source.split("__global__ void points_kernel", 1)[1].split(
        "// Atom-major 2-D launch", 1
    )[0]
    assert "sqrt(" not in points
    assert "cos(" not in points
    assert "sin(" not in points
    assert "r * pz[0] * pp[0]" in points
    assert "wr * pz[2] * pp[2]" in points

    root = Path(__file__).resolve().parents[2]
    native = (root / "src/dft/cuda_quadrature.cu").read_text()
    polar_launch = native.index("q::polar_kernel<<<")
    azimuth_launch = native.index("q::azimuth_kernel<<<")
    point_loop = native.index("for (std::size_t begin = 0; begin < l.points")
    assert polar_launch < point_loop
    assert azimuth_launch < point_loop
    assert "std::vector<double> input(l.polar, 0.0);" in native

    # The production standard PBE grid is 54 radial x 16 polar x 32 azimuth.
    # Before this change every point evaluates sqrt + cos + sin; after it, the
    # whole grid evaluates 16 sqrt and 32 each of cos/sin exactly once.
    setup_special_functions = 16 + 2 * 32
    assert setup_special_functions == 80
    for atoms, point_count in ((48, 1_327_104), (96, 2_654_208)):
        assert atoms * 54 * 16 * 32 == point_count
        assert 3 * point_count > setup_special_functions

    # Fixed maximum cache: 3 doubles per polar/azimuth entry.
    assert 8 * (3 * 256 + 3 * 1024) == 30_720


def test_partition_reuses_inverse_center_separations() -> None:
    """Pay each center-pair division once instead of once per point visit."""
    source = emit_quadrature_cuda()
    geometry = source.split("__global__ void geometry_kernel", 1)[1].split(
        "// Angular factors", 1
    )[0]
    assert "separation > tolerance ? 1.0 / separation : 0.0" in geometry
    assert "inverse_separation[a * na + b]" in geometry

    partition = source.split("__global__ void partition_kernel", 1)[1].split(
        "__global__ void normalize_kernel", 1
    )[0]
    assert "const double inverse = inverse_separation[hi * na + lo];" in partition
    assert ") * inverse)) : 0.0;" in partition
    assert " / sep" not in partition

    root = Path(__file__).resolve().parents[2]
    native = (root / "src/dft/cuda_quadrature.cu").read_text()
    assert "data, l.atoms, spec.coincident_tolerance, data + l.geometry" in native
    assert "count, l.atoms, data + l.distances," in native

    # The production PBE shapes are all non-coincident. The old partition did
    # one division for every ordered (point, a, b!=a) visit. The new geometry
    # setup performs one division per unordered center pair for the entire grid.
    for atoms, points, old_divisions, new_divisions in (
        (48, 1_327_104, 2_993_946_624, 1_128),
        (96, 2_654_208, 24_206_376_960, 4_560),
    ):
        assert points * atoms * (atoms - 1) == old_divisions
        assert atoms * (atoms - 1) // 2 == new_divisions
        assert new_divisions < old_divisions


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
      const size_t doubles = 3*atoms + atoms + 2*512 + 2*256 + 3*256 + 3*1024
                           + atoms*atoms + tile*atoms + tile*atoms + 3*tile + tile;
      if (l.device_bytes != doubles*sizeof(double) + sizeof(int)) return 1;
      if (l.weights + tile != l.doubles) return 2;
      if (l.azimuth != l.polar + 3*256) return 3;
      if (l.geometry != l.azimuth + 3*1024) return 4;
    }
  unsigned rejected = 0;
  try { (void)layout(0, 1); } catch (const std::invalid_argument&) { ++rejected; }
  try { (void)layout(1, 0); } catch (const std::invalid_argument&) { ++rejected; }
  try { (void)layout(1, SIZE_MAX); } catch (const std::overflow_error&) { ++rejected; }
  try { (void)layout(UINT32_MAX, 1); } catch (const std::overflow_error&) { ++rejected; }
  try { (void)layout(size_t{UINT32_MAX}+1, 1); }
      catch (const std::invalid_argument&) { ++rejected; }
  return rejected == 5 ? 0 : 5;
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

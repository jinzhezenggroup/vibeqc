"""Execute generated launch admission independently of a CUDA runtime."""

from __future__ import annotations

import itertools
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from vibeqc_compiler.dft.xc_contraction_cuda import XcMatrixSchedule, _emit_tiled

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("tile", [1, 2, 4, 64, 128])
def test_unqualified_power_of_two_cannot_bypass_candidate_selection(tile: int) -> None:
    with pytest.raises(ValueError, match="supported candidate"):
        XcMatrixSchedule(tile)


@pytest.mark.parametrize("tile", [8, 16, 32])
def test_every_emitted_tile_can_publish_all_totals(tile: int) -> None:
    schedule = XcMatrixSchedule(tile)
    assert schedule.tile >= 3
    assert schedule.threads <= 1024
    assert schedule.shared_bytes <= 48 * 1024
    source = _emit_tiled(schedule)
    assert source.count("tiled_xc_admitted(n, count, spins, work_jets)") == 2
    assert "threadIdx.x < 3" in source


@pytest.fixture(scope="module")
def admission_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler, not a GPU")
    directory = tmp_path_factory.mktemp("xc-matrix-admission")
    source = "#include <cstdint>\n#include <iostream>\nusing I = std::uint64_t;\n"
    for tile in (8, 16, 32):
        emitted = _emit_tiled(XcMatrixSchedule(tile))
        start = emitted.index("inline bool tiled_xc_admitted(")
        end = emitted.index("inline void scheduled_density_product", start)
        source += emitted[start:end].replace("tiled_xc_admitted", f"admitted_{tile}")
    source += r"""
int main() {
  I tile, n, count, spins, jets;
  while (std::cin >> tile >> n >> count >> spins >> jets) {
    const bool admitted = tile == 8 ? admitted_8(n, count, spins, jets)
                        : tile == 16 ? admitted_16(n, count, spins, jets)
                        : admitted_32(n, count, spins, jets);
    std::cout << admitted << '\n';
  }
}
"""
    cpp, executable = directory / "probe.cpp", directory / "probe"
    cpp.write_text(source, encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(cpp), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("tile", [8, 16, 32])
def test_native_admission_matches_model_at_all_launch_boundaries(
    admission_probe: Path, tile: int
) -> None:
    maximum = 2**64 - 1
    shapes = list(
        itertools.product(
            (1, tile - 1, tile, 361 * tile, 362 * tile, maximum),
            (1, tile, 65535 * tile, 65535 * tile + 1, maximum),
            ((1, 1), (2, 4), (65535, 1), (65536, 1), (1, 65536), (maximum, 4)),
        )
    )
    cases = [(n, count, spins, jets) for n, count, (spins, jets) in shapes]
    # Malformed native dimensions fail closed before division or multiplication.
    cases.extend(
        [(0, tile, 1, 1), (tile, 0, 1, 1), (tile, tile, 0, 1), (tile, tile, 1, 0)]
    )
    completed = subprocess.run(
        [str(admission_probe)],
        input="".join(
            f"{tile} {n} {count} {spins} {jets}\n" for n, count, spins, jets in cases
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    actual = [bool(int(value)) for value in completed.stdout.splitlines()]
    schedule = XcMatrixSchedule(tile)
    expected = [
        schedule.admitted(n, count, spins=spins, work_jets=jets)
        if min(n, count, spins, jets) > 0
        else False
        for n, count, spins, jets in cases
    ]
    assert len(actual) == len(cases)
    assert actual == expected

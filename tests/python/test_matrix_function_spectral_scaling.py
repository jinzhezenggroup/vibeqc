"""Spectral products may be representable when the isolated coefficient is not."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_seeded_spectral_response_keeps_representable_extremes(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "weighted.cpp"
    source.write_text(CPP)
    binary = tmp_path / "weighted"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            str(source),
            str(ROOT / "src/tensor/symmetric_matrix_function.cpp"),
            str(ROOT / "src/tensor/cpu_linalg.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


CPP = r"""
#include "tensor/symmetric_matrix_function.hpp"
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
int main() {
  using namespace vibeqc::tensor;
  const std::array<double,1> q{1.0};
  const std::array<std::uint8_t,1> keep{1};
  for (const auto function : {SymmetricMatrixFunction::inverse_sqrt,
                              SymmetricMatrixFunction::pseudoinverse}) {
    const bool inverse=function==SymmetricMatrixFunction::pseudoinverse;
    for (int spectral : {-700,1000}) {
      const int seed=spectral>0?1023:-1022;
      const auto expected=std::ldexp(-1.0,seed-(inverse?2*spectral:3*(spectral/2)+1));
      try {
        const auto value=symmetric_matrix_function_vjp(
            std::array<double,1>{std::ldexp(1.0,spectral)},q,keep,
            std::array<double,1>{std::ldexp(1.0,seed)},function,0.0);
        if(value[0]!=expected) {
          std::cerr<<"representable weighted response lost: "<<value[0]<<" != "<<expected<<'\n';
          return 1;
        }
      } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
    }
  }
}
"""


@pytest.mark.parametrize("spectral", [-700, 1000])
def test_cuda_seeded_spectral_extremes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spectral: int
) -> None:
    import os

    import test_matrix_function_native_range as reference

    if os.environ.get("VIBEQC_MATRIX_FUNCTION_CUDA_TEST") != "1":
        pytest.skip("explicit allocated CUDA tier")
    seed = 1023 if spectral > 0 else -1022
    source = (
        reference._CUDA_SOURCE.replace(
            "for (int e : {500, 1023})", f"for (int e : {{{seed}}})"
        )
        .replace(
            "const int exponent = function == 0 ? 700 : 520;",
            f"const int exponent = {spectral};",
        )
        .replace(
            "const int derivative_exponent = function == 0 ? 1051 : 1040;",
            f"const int derivative_exponent = function == 0 ? {3 * (spectral // 2) + 1} : {2 * spectral};",
        )
    )
    assert source != reference._CUDA_SOURCE
    monkeypatch.setattr(reference, "_CUDA_SOURCE", source)
    reference.test_generated_cuda_matrix_function_range(tmp_path)

"""Compile the generated density body and retain its general-weight bit contract."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def density_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    source = (ROOT / "tools/generate_scf_array_native.py").read_text()
    begin = source.index("inline void density_from_orbitals(")
    end = source.index("inline void weighted_density_from_orbitals(", begin)
    # This function is a literal f-string section. IR/topology validation stays
    # covered by the existing generator suite; here execute its exact C++ body.
    body = source[begin:end].replace("{{", "{").replace("}}", "}")
    directory = tmp_path_factory.mktemp("scf-density-weight")
    cpp, executable = directory / "probe.cpp", directory / "probe"
    cpp.write_text(
        r"""
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>
"""
        + body
        + r"""
int main(int argc, char** argv) {
  if (argc != 3) return 3;
  const double weight = std::strtod(argv[1], nullptr);
  const auto occupied = static_cast<std::size_t>(std::strtoul(argv[2], nullptr, 10));
  for (std::size_t n : {2U, 4U, 9U}) {
    const std::size_t stride = 12;
    std::vector<double> coefficients(n * stride, std::numeric_limits<double>::quiet_NaN());
    const double values[] = {0.1, 0.3, 0.7, 0.9, -0.4, 0.75, -0.6};
    for (std::size_t mu = 0; mu < n; ++mu)
      for (std::size_t i = 0; i < occupied; ++i)
        coefficients[mu * stride + i] = values[(mu + 2*i) % 7];
    std::vector<double> result(n*n + 2, 123456.0);
    density_from_orbitals(result.data() + 1, coefficients.data(), n, stride, occupied, weight);
    if (result.front() != 123456.0 || result.back() != 123456.0) return 2;
    for (std::size_t mu = 0; mu < n; ++mu)
      for (std::size_t nu = 0; nu < n; ++nu) {
        double expected = 0.0;
        for (std::size_t i = 0; i < occupied; ++i)
          expected += weight * coefficients[mu * stride + i] * coefficients[nu * stride + i];
        if (std::bit_cast<std::uint64_t>(result[1 + mu*n + nu]) !=
            std::bit_cast<std::uint64_t>(expected)) return 1;
      }
  }
}
"""
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            str(cpp),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("weight", [0.0, 1.0, 2.0, 0.1, 0.3, 1.1, 1.3, 1.5, 3.0, -0.3])
@pytest.mark.parametrize("occupied", [0, 1, 2])
def test_density_retains_full_square_weight_semantics(
    density_probe: Path, weight: float, occupied: int
) -> None:
    subprocess.run(
        [str(density_probe), repr(weight), str(occupied)], check=True, timeout=10
    )

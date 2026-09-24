"""Generated native CUDA (T): unequal dimensions, reproducibility and failure gates."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_cc.triples import triples_fullsum

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RCCSDT_CUDA_TEST") != "1",
    reason="requires an allocated CUDA device and compiler",
)


@pytest.fixture(scope="module")
def executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the generated evaluator without the CC solver to vary its inputs."""
    directory = tmp_path_factory.mktemp("native-triples")
    source = directory / "triples.cu"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_rccsdt_cuda.py"),
            "--output",
            str(source),
        ],
        check=True,
    )
    driver = directory / "driver.cpp"
    driver.write_text(r"""
#include "cc/triples_cuda.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
int main(int argc, char** argv) {
  const std::size_t o = std::stoul(argv[1]), v = std::stoul(argv[2]);
  std::ifstream input(argv[3], std::ios::binary);
  const std::size_t sizes[] = {o*v*v*v,o*v*o*o,o*v*o*v,o*v,o*v,o*o*v*v,o,v};
  std::vector<std::vector<double>> arrays;
  for (auto size : sizes) {
    arrays.emplace_back(size);
    input.read(reinterpret_cast<char*>(arrays.back().data()), size*sizeof(double));
  }
  auto run = [&](std::size_t budget) {
    return vibeqc::cc::triples::evaluate_cuda(o,v,arrays[0].data(),arrays[1].data(),
      arrays[2].data(),arrays[3].data(),arrays[4].data(),arrays[5].data(),
      arrays[6].data(),arrays[7].data(),1e-10,budget,0);
  };
  const auto first = run(1<<24), second = run(first.workspace_bytes);
  if (first.energy != second.energy) return 2;
  try { run(first.workspace_bytes-1); return 3; } catch (const std::length_error&) {}
  // Finite-input validation must include unused/zero-weight entries.
  const auto saved = arrays[3][0];
  arrays[3][0] = std::numeric_limits<double>::quiet_NaN();
  try { run(1<<24); return 4; } catch (const std::invalid_argument&) {}
  arrays[3][0] = saved;
  for (auto& e : arrays[7]) e = -10.;
  try { run(1<<24); return 5; } catch (const std::invalid_argument&) {}
  std::cout << std::setprecision(17) << first.energy << " " << first.virtual_triples
            << " " << first.workspace_bytes << "\n";
}
""")
    nvcc = os.environ.get("CUDACXX") or shutil.which("nvcc")
    if not nvcc:
        pytest.fail("CUDA acceptance requested without nvcc/CUDACXX")
    output = directory / "triples"
    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O2",
            "-arch=" + os.environ.get("VIBEQC_TEST_CUDA_ARCH", "sm_120"),
            "-DVIBEQC_HAS_CUDA=1",
            f"-I{ROOT / 'src'}",
            str(source),
            str(driver),
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


@pytest.mark.parametrize("o,v", [(2, 3), (3, 2), (3, 4)])
def test_generated_cuda_triples_against_independent_fullsum(
    executable: Path, tmp_path: Path, o: int, v: int
) -> None:
    rng = np.random.default_rng(812 + o)
    arrays = [
        rng.normal(size=shape)
        for shape in [
            (o, v, v, v),
            (o, v, o, o),
            (o, v, o, v),
            (o, v),
            (o, v),
            (o, o, v, v),
        ]
    ]
    arrays[5] = (arrays[5] + arrays[5].transpose(1, 0, 3, 2)) / 2
    arrays.extend([np.linspace(-1.0, -0.5, o), np.linspace(0.5, 1.5, v)])
    inputs = tmp_path / "inputs.bin"
    np.concatenate([a.ravel() for a in arrays]).astype(np.float64).tofile(inputs)
    result = subprocess.run(
        [str(executable), str(o), str(v), str(inputs)],
        check=True,
        text=True,
        capture_output=True,
    )
    energy, count, workspace = result.stdout.split()
    assert float(energy) == pytest.approx(
        triples_fullsum(o, v, *arrays), abs=1e-10, rel=1e-12
    )
    assert int(count) == v * (v + 1) * (v + 2) // 6
    assert int(workspace) <= 1 << 24

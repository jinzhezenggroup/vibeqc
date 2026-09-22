"""Independent matrix/AD and native ragged-publication gates for spin science."""

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.method.gfn2_spin_runtime import (
    build_gfn2_spin_atom_primal,
    build_gfn2_spin_atom_vjp,
)
from vibeqc_compiler.tensor import execute

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "src/xtb/native"


@pytest.mark.parametrize("shells", (1, 2, 3))
def test_spin_energy_and_ad_match_independent_matrix_oracle(shells: int) -> None:
    rng = np.random.default_rng(671 + shells)
    for _ in range(12):
        matrix = rng.normal(size=(shells, shells))
        matrix = (matrix + matrix.T) * 0.1
        m = rng.normal(size=shells)
        feeds = {f"m_{i}": np.asarray(m[i]) for i in range(shells)}
        feeds.update(
            {
                f"w_{i}_{j}": np.asarray(matrix[i, j])
                for i in range(shells)
                for j in range(i, shells)
            }
        )
        primal = execute(build_gfn2_spin_atom_primal(shells), feeds).outputs
        adjoint = execute(
            build_gfn2_spin_atom_vjp(shells),
            {**feeds, "bar_energy": np.asarray(1.0)},
        ).outputs
        np.testing.assert_allclose(primal["energy"], 0.5 * m @ matrix @ m, atol=2e-15)
        potential = np.array([adjoint[f"bar_m_{i}"] for i in range(shells)])
        np.testing.assert_allclose(potential, matrix @ m, atol=2e-15)
        for i in range(shells):
            step = np.eye(shells)[i] * 1e-5
            finite_difference = (
                0.5 * (m + step) @ matrix @ (m + step)
                - 0.5 * (m - step) @ matrix @ (m - step)
            ) / 2e-5
            assert potential[i] == pytest.approx(finite_difference, abs=2e-10)


@pytest.fixture(scope="module")
def native_spin(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native spin qualification requires a C++ compiler")
    directory = tmp_path_factory.mktemp("gfn2-spin")
    header = directory / "generated_gfn2_spin_native.hpp"
    command = [
        sys.executable,
        "-S",
        str(ROOT / "tools/generate_gfn2_spin_native.py"),
        "--output",
        str(header),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    first = header.read_bytes()
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    assert header.read_bytes() == first
    assert b"std::fma" in first
    source = directory / "spin.cpp"
    source.write_text(r"""
#include "model/gfn2/spin.hpp"
#include "generated_gfn2_spin_native.hpp"
#include <cmath>
#include <limits>

// Exercise the actual ragged runtime, including its restricted zero branch.
extern "C" int evaluate(const double* population, double* energy, double* potential) {
  using namespace vibeqc::xtb::detail::gfn2;
  SpinPolarizationPlan plan;
  plan.batch_size=3; plan.total_atoms=3; plan.total_shells=5;
  plan.shell_population_elements=9;
  plan.atom_offsets={0,1,2,3}; plan.batch_shell_offsets={0,1,4,5};
  plan.atom_shell_offsets={0,1,4,5}; plan.shell_population_offsets={0,1,7,9};
  plan.spin_channels={1,2,2}; plan.coupling_offsets={0,1,10,11};
  plan.coupling_matrices={-0.02, -0.07,0.01,-0.02, 0.01,-0.04,0.03, -0.02,0.03,-0.06, -0.05};
  std::string error;
  return evaluate_spin_polarization_cpu(make_spin_polarization_view(plan),
      population, energy, potential, error);
}

// Contraction rounding belongs to the existing FMA contract, including
// cancellation when an unfused intermediate product would overflow.
extern "C" int fma_contract() {
  using namespace vibeqc::xtb::generated;
  double value=0;
  const double largest=std::numeric_limits<double>::max();
  if (!accumulate_gfn2_spin_potential(largest, 2.0, -largest, value) || value!=largest) return 1;
  if (!accumulate_gfn2_spin_energy(largest, 2.0, -largest, value) || value!=0.0) return 2;
  return 0;
}
""")
    library = directory / "spin.so"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-shared",
            "-I",
            str(directory),
            "-I",
            str(NATIVE),
            "-I",
            str(NATIVE / "src"),
            str(source),
            str(NATIVE / "src/model/gfn2/spin.cpp"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    native = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    native.evaluate.argtypes = [array, array, array]
    native.evaluate.restype = ctypes.c_int
    native.fma_contract.restype = ctypes.c_int
    return native


def test_native_spin_ragged_populations_and_publication(
    native_spin: ctypes.CDLL,
) -> None:
    population = np.array([0.4, 0.1, -0.2, 0.3, 0.7, -0.8, 0.9, 0.2, -0.3])
    energy = np.full(3, 777.0)
    potential = np.full(9, 777.0)
    assert native_spin.evaluate(population, energy, potential) == 0
    matrix = np.array([[-0.07, 0.01, -0.02], [0.01, -0.04, 0.03], [-0.02, 0.03, -0.06]])
    m = population[4:7]
    np.testing.assert_allclose(
        energy, [0, 0.5 * m @ matrix @ m, -0.025 * population[8] ** 2], atol=2e-16
    )
    expected = np.zeros(9)
    expected[4:7] = matrix @ m
    expected[8] = -0.05 * population[8]
    np.testing.assert_allclose(potential, expected, atol=2e-16)
    assert native_spin.fma_contract() == 0

    # CPU preflight fails before publishing any peer's output.
    population[5] = np.inf
    energy.fill(777.0)
    potential.fill(777.0)
    assert native_spin.evaluate(population, energy, potential) != 0
    assert np.all(energy == 777.0) and np.all(potential == 777.0)


def test_native_spin_consumers_retire_duplicate_fma_equations() -> None:
    for relative in ("src/model/gfn2/spin.cpp", "src/backends/cuda/gfn2_spin.cu"):
        source = (NATIVE / relative).read_text()
        assert '#include "generated_gfn2_spin_native.hpp"' in source
        assert "accumulate_gfn2_spin_potential(" in source
        assert "accumulate_gfn2_spin_energy(" in source
        assert "potential = fma(" not in source
        assert "potential = std::fma(" not in source
        assert "energy = fma(" not in source
        assert "energy = std::fma(" not in source


def test_generated_spin_cuda_matches_matrix_and_fma_oracles(tmp_path: Path) -> None:
    """Qualify nonzero spin science even though the public endpoint is restricted."""
    if os.environ.get("VIBEQC_TEST_GFN2_CUDA") != "1":
        pytest.skip("explicit GFN2 CUDA qualification is disabled")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("GFN2 CUDA qualification requires a Slurm allocation")
    compiler = shutil.which("nvcc")
    assert compiler is not None, "explicit CUDA qualification requires nvcc"
    header = tmp_path / "generated_gfn2_spin_native.hpp"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools/generate_gfn2_spin_native.py"),
            "--output",
            str(header),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    source = tmp_path / "spin.cu"
    source.write_text(r"""
#include <cuda_runtime.h>
#include "generated_gfn2_spin_native.hpp"
#include <cstdio>
#include <limits>

__global__ void evaluate(double* out) {
  using namespace vibeqc::xtb::generated;
  const double w[9]={-.07,.01,-.02, .01,-.04,.03, -.02,.03,-.06};
  const double m[3]={.7,-.8,.9};
  double energy=0;
  for (int i=0;i<3;++i) {
    double potential=0;
    for (int j=0;j<3;++j)
      if (!accumulate_gfn2_spin_potential(w[3*i+j],m[j],potential,potential)) return;
    out[i]=potential;
    if (!accumulate_gfn2_spin_energy(m[i],potential,energy,energy)) return;
  }
  out[3]=energy;
  const double largest=0x1.fffffffffffffp+1023;
  if (!accumulate_gfn2_spin_potential(largest,2.,-largest,out[4])) return;
  if (!accumulate_gfn2_spin_energy(largest,2.,-largest,out[5])) return;
  out[6]=1;
}

int main() {
  double* device=nullptr;
  if (cudaMalloc(&device,7*sizeof(double))!=cudaSuccess) return 1;
  if (cudaMemset(device,0,7*sizeof(double))!=cudaSuccess) return 2;
  evaluate<<<1,1>>>(device);
  if (cudaDeviceSynchronize()!=cudaSuccess) return 3;
  double result[7];
  if (cudaMemcpy(result,device,sizeof(result),cudaMemcpyDeviceToHost)!=cudaSuccess) return 4;
  if (cudaFree(device)!=cudaSuccess) return 5;
  for (double value:result) std::printf("%.17g\n",value);
  return 0;
}
""")
    executable = tmp_path / "spin"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-arch=sm_120",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    run = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
    values = np.fromstring(run.stdout, sep="\n")
    assert values.shape == (7,)
    matrix = np.array([[-0.07, 0.01, -0.02], [0.01, -0.04, 0.03], [-0.02, 0.03, -0.06]])
    m = np.array([0.7, -0.8, 0.9])
    np.testing.assert_allclose(values[:3], matrix @ m, rtol=0, atol=2e-16)
    assert values[3] == pytest.approx(0.5 * m @ matrix @ m, abs=2e-16)
    assert values[4] == np.finfo(float).max
    assert values[5] == 0 and values[6] == 1

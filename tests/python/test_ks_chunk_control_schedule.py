"""Host scheduling regression of the actual CUDA control body, not a GPU test.

Only CUDA launch/thread/shared-memory syntax is substituted. Delayed threads
start after thread zero reaches its first block barrier (or returns). A block
barrier drops exited threads, and each invocation owns its shared variables.
The test protects control/warm-copy ordering; it does not qualify GPU numerics.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = r"""
#include <barrier>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <thread>
#include <vector>
using std::isfinite;
struct Dim { unsigned x; };
thread_local Dim threadIdx{};
constexpr Dim blockDim{256};
std::barrier<>* block_barrier;
std::promise<void> first_fence;
bool announced = false;
void announce() {
  if (threadIdx.x == 0 && !announced) {
    announced = true;
    first_fence.set_value();
  }
}
void block_fence() {
  announce();
  block_barrier->arrive_and_wait();
}
#define __global__
#define __shared__ static
#define __syncthreads() block_fence()
"""
SUFFIX = r"""
int main(int argc, char** argv) {
  const int mode = std::stoi(argv[1]);
  const std::size_t size = 289;
  std::vector<double> density(size), proposal(size), warm(size, -99.0);
  for (std::size_t i = 0; i < size; ++i) {
    density[i] = 1.0 + i;
    proposal[i] = density[i] + 1.0;
  }
  Scalars current{};
  current.one_electron = 1.0;
  current.electrons[0] = current.electrons[1] = 1.0;
  if (mode == 1 || mode == 3) current.residual = 0.1;
  if (mode == 2) current.failure = 1;
  Control control{1.0, 1U, mode == 4 ? 0 : 1, 0, 0};
  std::uint8_t enabled = control.active, spin_enabled = control.active;
  std::barrier fence(static_cast<std::ptrdiff_t>(blockDim.x));
  block_barrier = &fence;
  auto ready = first_fence.get_future();
  const auto worker = [&](unsigned index) {
    threadIdx.x = index;
    advance_kernel(size, 1, 0.0, 1, 1, 1e-12, 1e-10, mode == 3 ? 2 : 100,
                   mode != 5, &current, &control, proposal.data(), density.data(),
                   warm.data(), &enabled, &spin_enabled);
    announce();
    fence.arrive_and_drop();
  };
  std::vector<std::thread> threads;
  threads.emplace_back(worker, 0);
  ready.wait();
  for (unsigned i = 1; i < blockDim.x; ++i) threads.emplace_back(worker, i);
  for (auto& thread : threads) thread.join();
  for (std::size_t i = 0; i < size; ++i) {
    const double expected_density = 1.0 + i + (mode == 1 ? 1.0 : 0.0);
    const double expected_warm = mode == 0 ? 1.0 + i : -99.0;
    if (density[i] != expected_density || warm[i] != expected_warm) {
      std::cerr << "incomplete publication at " << i << " density=" << density[i]
                << " warm=" << warm[i] << '\n';
      return 1;
    }
  }
  if (mode == 4 && control.iterations != 1) return 2;
  if (mode != 4 && control.iterations != 2) return 3;
  if (control.active != (mode == 1) || enabled != (mode == 1)) return 4;
  if (control.converged != (mode == 0 || mode == 5)) return 5;
  if (control.failed != (mode == 2)) return 6;
  return 0;
}
"""


@pytest.fixture(scope="module")
def control_executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++20 host compiler")
    source = (ROOT / "src/dft/cuda_ks_kernels.cu").read_text()
    start = source.index("__global__ void advance_kernel(")
    end = source.index("\n}  // namespace", start)
    header = (ROOT / "src/dft/cuda_ks_kernels.hpp").read_text()
    declarations = header[
        header.index("struct Scalars {") : header.index("void reset_control(")
    ]
    directory = tmp_path_factory.mktemp("ks-control-schedule")
    cpp, executable = directory / "test.cpp", directory / "test"
    cpp.write_text(PREFIX + declarations + source[start:end] + SUFFIX)
    subprocess.run(
        [compiler, "-std=c++20", "-O1", "-pthread", str(cpp), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return executable


@pytest.mark.parametrize(
    "mode",
    range(6),
    ids=(
        "converged-warm-copy",
        "continue-density-copy",
        "failure",
        "iteration-limit",
        "inactive",
        "converged-no-warm",
    ),
)
def test_late_threads_keep_control_publication(
    control_executable: Path, mode: int
) -> None:
    completed = subprocess.run(
        [str(control_executable), str(mode)],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

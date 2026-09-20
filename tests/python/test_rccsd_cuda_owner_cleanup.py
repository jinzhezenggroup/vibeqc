"""Compile the real CUDA owner's construction path with injected API failures."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_cuda_owner_unwinds_every_setup_failure(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/cc/cuda_solver.cu").read_text()
    helpers = source[
        source.index("std::size_t checked_mul(") : source.index(
            "__global__ void damped_advance"
        )
    ]
    # Extract the live production members/constructor/destructor, not a copied
    # model. Kernel methods are irrelevant to constructor unwind and excluded.
    owner = source[
        source.index("struct Layout {") : source.index("  template <class Output>")
    ]
    support = (ROOT / "src/cc/cuda_solver_support.cuh").read_text()
    state = support[
        support.index("struct CudaState {") : support.index(
            "struct DeviceIterationOutputs"
        )
    ]
    cpp = tmp_path / "owner.cpp"
    cpp.write_text(PREFIX + state + GENERATED + helpers + owner + "};\n" + MAIN)
    exe = tmp_path / "owner"
    subprocess.run(
        [compiler, "-std=c++20", "-I" + str(ROOT / "src"), str(cpp), "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(exe)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include "cc/solver.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
using cudaStream_t = void*;
constexpr int cudaStreamNonBlocking = 1, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToDevice = 2;
int calls = 0, fail_at = 0, streams = 0, allocations = 0, device = 7;
int step() { return ++calls == fail_at ? 999 : 0; }
int cudaGetDevice(int* p) { *p = device; return 0; }
int cudaSetDevice(int d) { device = d; return 0; }
int cudaStreamCreateWithFlags(cudaStream_t* p, int) {
  if (const int error = step()) return error;
  *p = new int(1); ++streams; return 0;
}
int cudaMalloc(void** p, std::size_t bytes) {
  if (const int error = step()) return error;
  *p = new unsigned char[bytes]; ++allocations; return 0;
}
int cudaMemcpyAsync(void* d, const void* s, std::size_t n, int, cudaStream_t) {
  if (const int error = step()) return error;
  std::memcpy(d, s, n); return 0;
}
"""

PREFIX += r"""
int cudaStreamSynchronize(cudaStream_t) { return step(); }
int cudaFree(void* p) { delete[] static_cast<unsigned char*>(p); --allocations; return 0; }
int cudaStreamDestroy(cudaStream_t p) { delete static_cast<int*>(p); --streams; return 0; }
void cuda_check(int code) { if (code) throw std::runtime_error("injected CUDA failure"); }
namespace vibeqc::cc { namespace generated {
"""
GENERATED = r"""
std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a) throw std::length_error("overflow");
  return a + b;
}
std::size_t iteration_arena_elements(std::size_t, std::size_t) { return 16; }
std::size_t replay_arena_elements(std::size_t, std::size_t) { return 16; }
}
std::size_t problem_host_bytes(const Problem&) { return 128; }
"""
MAIN = r"""
}  // namespace vibeqc::cc
int main() {
  vibeqc::cc::Problem p; p.nocc = p.nvir = 1;
  for (auto* v : {&p.foo, &p.fov, &p.fvv, &p.ovov, &p.ovvo, &p.oovv,
                 &p.ovvv, &p.ovoo, &p.oooo, &p.vvvv, &p.d1, &p.d2,
                 &p.initial_t1, &p.initial_t2}) v->push_back(1.0);
  vibeqc::cc::SolverOptions options;
  int constructor_calls = 0;
  { vibeqc::cc::Owner good(p, options, 0); constructor_calls = calls;
    const auto detached = (good.n1 + good.n2) * sizeof(double);
    if (good.diagnostic.numeric_capacity_bytes < 128 + good.layout.total + detached) {
      std::cerr << "CUDA detached result storage was not reserved\n"; return 8;
    }
  }
  if (streams || allocations || device != 7 || constructor_calls < 18) return 1;
  for (int failure = 1; failure <= constructor_calls; ++failure) {
    calls = 0; fail_at = failure;
    try { vibeqc::cc::Owner broken(p, options, 0); return 2; }
    catch (const std::runtime_error& error) {
      if (std::string(error.what()) != "injected CUDA failure") return 3;
    }
    if (streams || allocations || device != 7) {
      std::cerr << "leaked owners after setup operation " << failure << '\n';
      return 4;
    }
    calls = 0; fail_at = 0;
    { vibeqc::cc::Owner retry(p, options, 0); }
    if (streams || allocations || device != 7) return 5;
  }
  options.max_bytes = 1; calls = 0;
  try { vibeqc::cc::Owner over_budget(p, options, 0); return 6; }
  catch (const std::length_error&) {}
  if (calls || streams || allocations || device != 7) return 7;
  std::cout << "setup failures and retries checked: " << constructor_calls << '\n';
}
"""

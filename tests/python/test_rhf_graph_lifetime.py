"""Exercise the real host graph owner with a deterministic CUDA lifecycle stub."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_capture_exception_abandons_stream_and_allows_retry(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    (tmp_path / "cuda_runtime_api.h").write_text(
        """#pragma once
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
constexpr int cudaErrorInvalidResourceHandle = 400;
constexpr int cudaStreamCaptureModeThreadLocal = 1;
constexpr unsigned long long cudaGraphInstantiateFlagDeviceLaunch = 4;
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t cudaGraphDestroy(cudaGraph_t);
cudaError_t cudaSetDevice(int);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamBeginCapture(cudaStream_t, int);
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t);
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
"""
    )
    harness = tmp_path / "capture.cpp"
    harness.write_text(
        r"""#include "scf/cuda/rhf_graph.hpp"
#include <iostream>
#include <stdexcept>

namespace {
bool capturing = false;
int graphs = 0, executables = 0;
}
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t p) {
  delete static_cast<int*>(p); --executables; return cudaSuccess;
}
cudaError_t cudaGraphDestroy(cudaGraph_t p) {
  delete static_cast<int*>(p); --graphs; return cudaSuccess;
}
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaStreamBeginCapture(cudaStream_t, int) {
  if (capturing) return cudaErrorInvalidResourceHandle;
  capturing = true; return cudaSuccess;
}
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* graph) {
  if (!capturing) return cudaErrorInvalidResourceHandle;
  capturing = false; *graph = new int(1); ++graphs; return cudaSuccess;
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* out, cudaGraph_t, unsigned long long) {
  *out = new int(1); ++executables; return cudaSuccess;
}
cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaGraphLaunch(cudaGraphExec_t p, cudaStream_t) {
  return p ? cudaSuccess : cudaErrorInvalidResourceHandle;
}
int main() {
  for (bool post : {false, true}) {
    {
      vibeqc::scf::cuda_execution::RhfIterationGraphs owner;
      auto capture = [&](const std::function<vibeqc_status()>& body) {
        return post ? owner.capture_post_eigensolver(0, nullptr, body)
                    : owner.capture_iteration(0, nullptr, false, body);
      };
      try {
        capture([]() -> vibeqc_status { throw std::logic_error("original"); });
        return 2;
      } catch (const std::logic_error& error) {
        if (std::string(error.what()) != "original") return 3;
      }
      if (capturing || graphs || executables) {
        std::cerr << "capture remained active after exception\n";
        return 4;
      }
      auto rejected = capture([] { return VIBEQC_STATUS_NOT_IMPLEMENTED; });
      if (rejected.ok() || rejected.body_status != VIBEQC_STATUS_NOT_IMPLEMENTED ||
          capturing || graphs || executables) return 5;
      if (!capture([] { return VIBEQC_STATUS_SUCCESS; }).ok()) return 6;
      if (capturing || graphs != 1 || executables != 1) return 7;
      auto replay = post ? owner.launch_post_eigensolver(nullptr)
                         : owner.launch_iteration(nullptr);
      if (replay != cudaSuccess) return 8;
    }
    if (capturing || graphs || executables) return 9;
  }
  std::cout << "graph capture lifecycle passed\n";
}
"""
    )
    executable = tmp_path / "capture"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-pthread",
            "-I" + str(tmp_path),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(ROOT / "src/scf/cuda/rhf_graph.cpp"),
            str(harness),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

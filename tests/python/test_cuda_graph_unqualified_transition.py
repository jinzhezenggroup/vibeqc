"""Losing graph qualification invalidates retained bindings before ordinary work."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def transition_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    directory = tmp_path_factory.mktemp("graph-unqualified")
    (directory / "cuda_runtime.h").write_text(
        r"""
#pragma once
#include <cstddef>
using cudaStream_t = void*;
using cudaGraph_t = int*;
using cudaGraphExec_t = int*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorNotSupported=801,
  cudaErrorMemoryAllocation=4, cudaStreamCaptureModeThreadLocal=5;
inline int live_execs=0, launches=0;
inline const char* cudaGetErrorString(int) { return "fake CUDA error"; }
inline int cudaGetLastError() { return 0; }
inline int cudaMemGetInfo(size_t* free, size_t* total) {
 *total=100000; *free=100000-live_execs*4096; return 0;
}
inline int cudaStreamBeginCapture(cudaStream_t,int) { return 0; }
inline int cudaStreamEndCapture(cudaStream_t,cudaGraph_t* graph) {
 *graph=new int(1); return 0;
}
inline int cudaGraphDestroy(cudaGraph_t graph) { delete graph; return 0; }
inline int cudaGraphExecDestroy(cudaGraphExec_t graph) { delete graph; --live_execs; return 0; }
inline int cudaGraphGetNodes(cudaGraph_t,void*,size_t* count) { *count=3; return 0; }
inline int cudaGraphInstantiate(cudaGraphExec_t* out,cudaGraph_t,void*,void*,unsigned long long) {
 *out=new int(1); ++live_execs; return 0;
}
inline int cudaGraphLaunch(cudaGraphExec_t,cudaStream_t) { ++launches; return 0; }
"""
    )
    source = directory / "transition.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdlib>
#include "runtime/cuda_graph_region.cuh"
int main(int argc, char** argv) {
  assert(argc == 3);
  const bool profile = std::atoi(argv[1]) != 0;
  const bool fail = std::atoi(argv[2]) != 0;
  using namespace vibeqc::runtime;
  CudaGraphRegion graph;
  GraphBinding key{"qualified", 0, nullptr, nullptr, nullptr};
  int calls = 0;
  auto operation = [&] { ++calls; };
  graph.submit(key, true, false, operation);
  graph.submit(key, true, false, operation);
  assert(live_execs == 1 && launches == 1 && graph.metrics.captures == 1);
  bool rejected = false;
  const auto before_calls = calls;
  try {
    graph.submit({}, false, profile, [&] {
      ++calls;
      assert(live_execs == 0);
      assert(graph.metrics.node_count == 0 && graph.metrics.retained_device_bytes == 0);
      if (fail) throw std::runtime_error("ordinary failure");
    });
  } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected == fail && calls == before_calls + 1);
  assert(graph.metrics.invalidations == 1 && graph.metrics.mode == (profile ? 5 : 0));
  graph.submit({}, false, profile, operation);
  assert(graph.metrics.invalidations == 1 && live_execs == 0);
  const auto before_rebind = calls;
  graph.submit(key, true, false, operation);
  assert(graph.metrics.mode == 1 && calls == before_rebind + 1);
  assert(live_execs == 0 && launches == 1 && graph.metrics.captures == 1);
  graph.submit(key, true, false, operation);
  assert(graph.metrics.mode == 2 && live_execs == 1 && launches == 2);
  assert(graph.metrics.captures == 2);
  graph.invalidate();
  assert(live_execs == 0);
}
"""
    )
    binary = directory / "transition"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I" + str(directory),
            "-I" + str(ROOT / "src"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_unqualified_transition_releases_before_callback(
    transition_probe: Path, profile: bool, fail: bool
) -> None:
    result = subprocess.run(
        [str(transition_probe), str(int(profile)), str(int(fail))],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

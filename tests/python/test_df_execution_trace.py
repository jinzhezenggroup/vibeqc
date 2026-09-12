"""Exercise the diagnostic lifecycle with mocked CUDA, without a GPU context."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_df_trace_disabled_capture_timing_failure_and_logical_tiles(tmp_path):
    """Capture work cannot masquerade as execution; timing failures invalidate records."""
    nvcc = os.environ.get("VIBEQC_NVCC")
    compiler = shutil.which("c++")
    if not nvcc or not compiler:
        pytest.skip("set VIBEQC_NVCC and provide c++ for the mocked CUDA contract")
    root = Path(__file__).resolve().parents[2]
    include = Path(nvcc).resolve().parent.parent / "include"
    source = tmp_path / "trace.cpp"
    source.write_text(
        r"""
#include "runtime/cuda_component_trace.hpp"
#include <cstdlib>
#include <cstdint>

static unsigned calls = 0, events = 0, records = 0, syncs = 0;
static bool capturing = false, fail_elapsed = false;
static unsigned event_times[64]{};
extern "C" cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* out) {
  ++calls;
  *out = capturing ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
  return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t* out) {
  ++calls;
  *out = reinterpret_cast<cudaEvent_t>(static_cast<std::uintptr_t>(++events));
  return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t) {
  ++calls;
  event_times[reinterpret_cast<std::uintptr_t>(event)] = ++records;
  return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t) {
  ++calls;
  ++syncs;
  return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t) { ++calls; return cudaSuccess; }
extern "C" cudaError_t CUDARTAPI cudaEventElapsedTime(float* ms, cudaEvent_t first, cudaEvent_t last) {
  ++calls;
  if (fail_elapsed) return cudaErrorInvalidResourceHandle;
  *ms = event_times[reinterpret_cast<std::uintptr_t>(last)] - event_times[reinterpret_cast<std::uintptr_t>(first)];
  return cudaSuccess;
}

int main(int argc, char** argv) {
  using namespace vibeqc::runtime::cuda_trace;
  if (argc != 2) return 1;
  const auto stream = reinterpret_cast<cudaStream_t>(std::uintptr_t{1});
  unsetenv("VIBEQC_DF_TRACE");
  {
    TraceOperation disabled("disabled", stream, {1, 3, 5, true, true});
    TraceRegion no_op("ignored", stream);
    trace_tile(0, 0, 3, 0, 2, -1, true);
  }
  if (calls != 0) return 2;
  setenv("VIBEQC_DF_TRACE", argv[1], 1);
  {
    TraceOperation operation("ri_j", stream, {1, 3, 5, true, true});
    {
      TraceRegion first("first_pass", stream);
      TraceRegion child("transformed_generation", stream);
      trace_tile(0, 0, 3, 0, 2, -1, true);
    }
    {
      TraceRegion second("second_pass", stream);
      trace_tile(0, 0, 3, 0, 2, -1, true);
      trace_tile(0, 0, 3, 0, 2, -1, false);
      trace_tile(0, 0, 3, 0, 2, 0, false);
      trace_tile(0, 0, 0, 0, 0, -1, true);
      trace_counter("cache_hits", 2);
      second.finish();
      second.finish();
    }
    TraceRegion other_stream("must_not_be_recorded", nullptr);
  }
  if (events != 8 || syncs != 1) return 3;
  capturing = true;
  {
    TraceOperation capture("ri_k", stream, {1, 3, 5, true, true});
    TraceRegion generation("transformed_generation", stream);
    trace_tile(0, 0, 3, 0, 2, -1, true);
  }
  if (events != 8 || syncs != 1) return 4;
  capturing = false;
  fail_elapsed = true;
  { TraceOperation failed("timing_failure", stream, {1, 3, 5, true, true}); }
  return syncs == 2 ? 0 : 5;
}
"""
    )
    executable = tmp_path / "trace"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I" + str(root / "src"),
            "-I" + str(include),
            str(source),
            str(root / "src/runtime/cuda_component_trace.cpp"),
            "-pthread",
            "-ldl",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = tmp_path / "trace.jsonl"
    subprocess.run([str(executable), str(output)], check=True)
    executed, captured, failed = map(json.loads, output.read_text().splitlines())
    assert executed["valid"] is True
    assert executed["execution"] == "stream"
    assert [r["parent"] for r in executed["regions"]] == [-1, 0, 1, 0]
    assert all(r["gpu_ms"] > 0 for r in executed["regions"])
    assert executed["profiler_event_count"] == 8
    assert executed["counters"] == {
        "cache_hits": 2,
        "raw_tile_productions": 1,
        "raw_value_bytes": 48,
        "derivative_tile_productions": 1,
        "derivative_value_bytes": 48,
        "transformed_tile_productions": 2,
        "transformed_value_bytes": 96,
    }
    assert len(executed["tiles"]) == 3
    transformed = next(t for t in executed["tiles"] if t["transformed"])
    assert transformed["productions"] == 2
    assert transformed["derivative_coordinate"] == -1
    assert captured["execution"] == "graph_capture"
    assert captured["valid"] is True
    assert captured["profiler_event_count"] == 0
    assert all(r["gpu_ms"] is None for r in captured["regions"])
    assert captured["tiles"][0]["productions"] == 1
    assert failed["valid"] is False
    assert failed["cuda_error"] != 0

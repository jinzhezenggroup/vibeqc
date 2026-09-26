"""Shared CUDA SolverRegion executor lifecycle without a GPU."""

import shutil
import subprocess
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_solver_region_executor_bounds_and_replays(tmp_path: typing.Any) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("requires a C++17 compiler")
    (tmp_path / "cuda_runtime.h").write_text(
        r"""
#pragma once
#include <cstddef>
using cudaStream_t = void*;
using cudaGraph_t = int*;
using cudaGraphExec_t = int*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorNotSupported=801, cudaErrorMemoryAllocation=4;
constexpr int cudaStreamCaptureModeThreadLocal=5;
inline int launches=0, live_execs=0;
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
inline int cudaGraphExecDestroy(cudaGraphExec_t graph) {
 delete graph; --live_execs; return 0;
}
inline int cudaGraphGetNodes(cudaGraph_t,void*,size_t* count) { *count=3; return 0; }
inline int cudaGraphInstantiate(cudaGraphExec_t* out,cudaGraph_t,void*,void*,unsigned long long) {
 *out=new int(1); ++live_execs; return 0;
}
inline int cudaGraphLaunch(cudaGraphExec_t,cudaStream_t) { ++launches; return 0; }
"""
    )
    source = tmp_path / "test.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <stdexcept>
#include "src/runtime/solver_region_cuda.cuh"
using namespace vibeqc::runtime;
int main() {
 GraphBinding key{"region-a",0,reinterpret_cast<void*>(1),
                  reinterpret_cast<void*>(2),reinterpret_cast<void*>(3)};
 SolverRegionCudaBinding binding{key,2,SolverRegionCompletionMode::Scalar,false};
 SolverRegionCudaExecutor executor;
 int calls=0;
 auto step=[&](unsigned slot) { assert(slot<2); ++calls; };
 assert(executor.submit(binding,5,8,false,step)==2);
 assert(calls==2 && executor.metrics.submissions==1);
 assert(executor.metrics.submitted_steps==2 && executor.metrics.last_width==2);
 executor.checkpoint();
 assert(executor.metrics.checkpoints==1);
 assert(executor.submit(binding,2,1,false,step)==1);
 assert(calls==3 && executor.metrics.submitted_steps==3);
 bool threw=false;
 try { executor.submit(binding,0,1,false,step); }
 catch(const std::invalid_argument&) { threw=true; }
 assert(threw);

 executor.invalidate();
 binding.replay_enabled=true;
 assert(executor.submit(binding,2,8,false,step)==2);
 assert(calls==5 && executor.replay_metrics().mode==1);
 assert(executor.submit(binding,2,8,false,step)==2);
 assert(calls==7 && executor.replay_metrics().mode==2);
 const int before=calls;
 assert(executor.submit(binding,2,8,false,step)==2);
 assert(calls==before && launches==2 && executor.replay_metrics().mode==3);
 assert(executor.metrics.submissions==5 && executor.metrics.submitted_steps==9);

 // A different realized width is a different replay shape and must invalidate.
 assert(executor.submit(binding,2,1,false,step)==1);
 assert(calls==before+1 && executor.replay_metrics().mode==1);
 assert(executor.replay_metrics().invalidations>=2);
 return 0;
}
"""
    )
    binary = tmp_path / "test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=10
    )

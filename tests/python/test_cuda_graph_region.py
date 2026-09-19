"""Capture contracts and actual shared native lifecycle, without a GPU."""

import ctypes
import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.common.capture import CaptureContract, _GraphMetrics
from vibeqc_compiler.common.cuda_runtime import CudaArtifact
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.specialization import (
    GuardPredicate,
    SpecializationGuard,
    WorkloadSignature,
)
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    input_tensor,
)
from vibeqc_compiler.tensor.cuda_emit import emit_cuda
from vibeqc_compiler.tensor.cuda_execute import tensor_capture_contract
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

ROOT = Path(__file__).resolve().parents[2]


def contract(size=16, schedule=None):
    axis = Index("i", IndexSpace("axis", "batch", size))
    x = input_tensor("x", TensorSpec((axis,), role="input"))
    plan = plan_cuda(
        Program({"out": add(x, x)}),
        cuda_target_info("sm_120"),
        schedule=schedule or TensorSchedule(),
    )
    artifact = CudaArtifact(Path("not-loaded.so"), {"key": "a" * 64})
    return (
        plan,
        artifact,
        tensor_capture_contract(
            plan,
            artifact,
            {
                "uuid": "device-a",
                "driver": 12090,
                "runtime": 12090,
                "cublas": "12.9.1",
            },
        ),
    )


def test_contract_is_pure_deterministic_and_uses_specialization_records():
    plan, artifact, first = contract()
    assert isinstance(first, CaptureContract)
    assert first.eligible
    assert first.identity == contract()[2].identity
    assert first.artifact_key == artifact.metadata["key"]
    assert dict(first.workload.features)["plan"] == plan.identity
    assert first.workload.kind == "tensorir"
    assert "numeric_inputs" not in dataclasses.asdict(first)
    assert ctypes.sizeof(_GraphMetrics) == 88


@pytest.mark.parametrize("field", ["artifact_key", "schedule_hash", "runtime_hash"])
def test_relevant_hash_change_invalidates_contract(field):
    first = contract()[2]
    second = dataclasses.replace(first, **{field: "b" * 64})
    assert first.identity != second.identity
    with pytest.raises(ValueError, match="SHA-256"):
        dataclasses.replace(first, **{field: "unqualified"})


@pytest.mark.parametrize("feature", ["precision", "layout", "plan"])
def test_workload_shape_layout_precision_guards_invalidate(feature):
    first = contract()[2]
    facts = dict(first.workload.features)
    facts[feature] = "changed"
    second = dataclasses.replace(
        first, workload=WorkloadSignature("tensorir", tuple(facts.items()))
    )
    assert first.identity != second.identity
    assert first.identity != contract(size=17)[2].identity
    assert first.identity != contract(schedule=TensorSchedule(threads=64))[2].identity


def test_compiler_and_scientific_identity_invalidate():
    first = contract()[2]
    for field in ("scientific_hash", "compiler_hash"):
        new_compilation = dataclasses.replace(first.compilation, **{field: "c" * 64})
        assert (
            first.identity
            != dataclasses.replace(first, compilation=new_compilation).identity
        )


@pytest.mark.parametrize("field", ["uuid", "driver", "runtime", "cublas"])
def test_device_and_provider_identity_changes_invalidate(field):
    plan, artifact, first = contract()
    runtime = {
        "uuid": "device-a",
        "driver": 12090,
        "runtime": 12090,
        "cublas": "12.9.1",
    }
    runtime[field] = "changed"
    assert first.identity != tensor_capture_contract(plan, artifact, runtime).identity


def test_unbudgeted_graph_storage_and_oversized_regions_fall_back():
    plan, artifact, first = contract()
    budgeted = tensor_capture_contract(plan, artifact, {}, resource_plan=object())
    assert not budgeted.eligible
    assert "global resource plan" in budgeted.failures[0]
    facts = dict(first.workload.features)
    facts["launches"] = 4097
    oversized = dataclasses.replace(
        first, workload=WorkloadSignature("tensorir", tuple(facts.items()))
    )
    assert not oversized.eligible
    assert "launches" in oversized.failures[0]
    missing = dataclasses.replace(
        first,
        guard=SpecializationGuard(
            (GuardPredicate("workload", "unqualified", "eq", True),)
        ),
    )
    assert not missing.eligible
    effects = ["host callback"]
    unsupported = dataclasses.replace(first, unsupported_effects=effects)
    effects.clear()
    assert unsupported.failures == ("host callback",)


def test_emitted_capture_excludes_host_transfers_and_keeps_error_checks():
    source = emit_cuda(contract()[0])
    start = source.index("ctx.submit_region(")
    end = source.index("int arithmetic_error", start)
    captured = source[start:end]
    assert "cudaMemsetAsync(ctx.error" in captured
    assert "kernel_" in captured
    assert "cudaMemcpy" not in captured
    assert "cudaStreamSynchronize" not in captured
    assert "cudaMemcpyDeviceToHost" in source[end:]
    assert "if (arithmetic_error)" in source[end:]
    assert 'extern "C" int tensor_run(' in source
    assert 'extern "C" int tensor_run_graph(' in source
    assert 'extern "C" int tensor_graph_configure(' in source
    prefixed = emit_cuda(contract()[0], symbol_prefix="another_")
    assert "another_tensor_run_graph" in prefixed


@pytest.mark.parametrize("minimal_headers", [False, True])
def test_native_shared_lifecycle_recovers_capture_failure_and_never_retries_launch(
    tmp_path,
    minimal_headers,
):
    """Compile the real lifecycle against a deterministic fake CUDA API.

    No fake numerics: actual GPU numerical coverage lives in the allocated suite.
    This exposes recovery branches that must not depend on inducing GPU faults.
    """
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("requires a C++17 compiler")
    (tmp_path / "cuda_runtime.h").write_text(r"""
#pragma once
#include <cstddef>
using cudaStream_t = void*;
using cudaGraph_t = int*;
using cudaGraphExec_t = int*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorStreamCaptureUnsupported=900,
  cudaErrorStreamCaptureInvalidated=901, cudaErrorNotSupported=801,
  cudaErrorMemoryAllocation=4, cudaStreamCaptureModeThreadLocal=5;
inline int begin_error=0, end_error=0, instantiate_error=0, launch_error=0;
inline int ends=0, live_graphs=0, live_execs=0, launches=0;
inline size_t nodes=3;
inline const char* cudaGetErrorString(int) { return "fake CUDA error"; }
inline int cudaGetLastError() { return 0; }
inline int cudaMemGetInfo(size_t* free, size_t* total) {
 *total=100000; *free=100000-live_execs*4096; return 0;
}
inline int cudaStreamBeginCapture(cudaStream_t,int) { return begin_error; }
inline int cudaStreamEndCapture(cudaStream_t,cudaGraph_t* graph) {
 ++ends; *graph=end_error ? nullptr : new int(1); if(*graph) ++live_graphs; return end_error;
}
inline int cudaGraphDestroy(cudaGraph_t graph) { delete graph; --live_graphs; return 0; }
inline int cudaGraphExecDestroy(cudaGraphExec_t graph) { delete graph; --live_execs; return 0; }
inline int cudaGraphGetNodes(cudaGraph_t,void*,size_t* count) { *count=nodes; return 0; }
inline int cudaGraphInstantiate(cudaGraphExec_t* out,cudaGraph_t,void*,void*,unsigned long long) {
 *out=instantiate_error ? nullptr : new int(1); if(*out) ++live_execs; return instantiate_error;
}
inline int cudaGraphLaunch(cudaGraphExec_t,cudaStream_t) { ++launches; return launch_error; }
""")
    if minimal_headers:
        header = tmp_path / "cuda_runtime.h"
        text = (
            header.read_text()
            .replace("cudaErrorStreamCaptureUnsupported", "stub_capture_unsupported")
            .replace("cudaErrorStreamCaptureInvalidated", "stub_capture_invalidated")
        )
        header.write_text(text)
    source = tmp_path / "test.cpp"
    source.write_text(r"""
#include <cassert>
#include "src/runtime/cuda_graph_region.cuh"
using namespace vibeqc::runtime;
int main() {
 static_assert(sizeof(GraphMetrics)==88);
 GraphBinding key{"artifact-a",0,reinterpret_cast<void*>(1),reinterpret_cast<void*>(2),reinterpret_cast<void*>(3)};
 int calls=0;
 auto op=[&] { ++calls; };
 {
  CudaGraphRegion graph;
  graph.submit(key,true,false,op); assert(graph.metrics.mode==1 && calls==1);
  graph.submit(key,true,false,op); assert(graph.metrics.mode==2 && calls==2);
  graph.submit(key,true,false,op); assert(graph.metrics.mode==3 && calls==2);
  assert(graph.metrics.captures==1 && graph.metrics.replays==2);
  assert(graph.metrics.node_count==3 && graph.metrics.retained_device_bytes==4096);
  graph.submit(key,true,true,op); assert(graph.metrics.mode==5 && calls==3);
  graph.submit(key,true,false,op); assert(graph.metrics.captures==1);
  // Invalidate each owner-local binding component, as well as artifact identity.
  for (int change=0;change<5;++change) {
   auto next=key;
   if(change==0) next.qualification="artifact-b";
   if(change==1) next.device=1;
   if(change==2) next.stream=reinterpret_cast<void*>(8);
   if(change==3) next.arena=reinterpret_cast<void*>(9);
   if(change==4) next.library=reinterpret_cast<void*>(10);
   graph.submit(next,true,false,op); assert(graph.metrics.mode==1);
   graph.submit(next,true,false,op); assert(graph.metrics.mode==2);
   key=next;
  }
  assert(graph.metrics.invalidations==5 && graph.metrics.captures==6);
  graph.invalidate(); assert(live_execs==0);
  graph.submit(key,false,false,op); assert(graph.metrics.mode==0);
 }
 assert(live_execs==0 && live_graphs==0);
 for(int failure=0;failure<5;++failure) {
  CudaGraphRegion graph;
  graph.submit(key,true,false,op);
  if(failure==0) begin_error=cudaErrorNotSupported;
  if(failure==1) end_error=901;
  if(failure==2) instantiate_error=cudaErrorMemoryAllocation;
  if(failure==3) nodes=4097;
  if(failure==4) begin_error=900;
  graph.submit(key,true,false,op);
  assert(graph.metrics.mode==4 && live_execs==0 && live_graphs==0);
  graph.submit(key,true,false,op); assert(graph.metrics.capture_attempts==1);
  begin_error=end_error=instantiate_error=0; nodes=3;
  graph.invalidate();
  graph.submit(key,true,false,op); graph.submit(key,true,false,op);
  assert(graph.metrics.captures==1 && graph.metrics.capture_attempts==2);
 }
 {
  CudaGraphRegion graph; graph.submit(key,true,false,op);
  int before=ends; bool once=true;
  graph.submit(key,true,false,[&]{if(once){once=false; throw std::runtime_error("during capture");} op();});
  assert(graph.metrics.mode==4 && ends==before+1 && live_graphs==0);
 }
 {
  CudaGraphRegion graph; graph.submit(key,true,false,op); graph.submit(key,true,false,op);
  int before=calls; launch_error=99; bool threw=false;
  try { graph.submit(key,true,false,op); } catch(const std::runtime_error&) { threw=true; }
  assert(threw && calls==before); launch_error=0;
 }
 assert(live_execs==0 && live_graphs==0);
}
""")
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


def test_capture_does_not_relabel_fp32_as_qualified_fp64():
    x = input_tensor("x", TensorSpec(dtype="float32", role="input"))
    plan = plan_cuda(Program({"out": add(x, x)}), cuda_target_info("sm_120"))
    artifact = CudaArtifact(Path("not-loaded.so"), {"key": "a" * 64})
    capture = tensor_capture_contract(plan, artifact, {})
    assert dict(capture.workload.features)["precision"] == "fp32"
    assert not capture.eligible
    assert any("precision" in reason for reason in capture.failures)

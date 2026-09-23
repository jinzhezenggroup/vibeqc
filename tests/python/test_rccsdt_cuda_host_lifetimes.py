"""Borrowed host scalars must outlive delayed CUDA copies on every error path.

Compile the emitted host owner with an asynchronous CUDA shim and AddressSanitizer.
The shim delays copies until stream synchronization and inspects poisoned regions
without dereferencing them. Kernel numerics/device execution are not tested here.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_CUDA_SHIM = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sanitizer/asan_interface.h>
using cudaError_t=int;
using cudaStream_t=void*;
constexpr int cudaSuccess=0,cudaErrorMemoryAllocation=2,cudaErrorUnknown=999;
constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,cudaStreamNonBlocking=1;
static int fail_at=0, calls=0, invalid=0, streams=0, buffers=0, syncs=0, current_device=3;
static std::vector<std::function<void()>> operations;
const char* cudaGetErrorString(int) {return "injected CUDA failure";}
int cudaGetDevice(int* value) {*value=current_device;return 0;}
int cudaSetDevice(int value) {current_device=value;return 0;}
int cudaStreamCreateWithFlags(cudaStream_t* stream,int) {*stream=reinterpret_cast<void*>(1);++streams;return 0;}
int cudaStreamDestroy(cudaStream_t) {--streams;return 0;}
int cudaMalloc(void** data,std::size_t bytes) {*data=std::malloc(bytes);++buffers;return 0;}
int cudaFree(void* data) {std::free(data);--buffers;return 0;}
int cudaGetLastError() {return 0;}
int cudaMemcpyAsync(void* target,const void* source,std::size_t bytes,int kind,cudaStream_t) {
  if (++calls==fail_at) return cudaErrorUnknown;
  operations.emplace_back([=] {
    const void* host=kind==cudaMemcpyHostToDevice ? source : target;
    if (__asan_region_is_poisoned(const_cast<void*>(host),bytes)) {
      ++invalid; std::fprintf(stderr,"expired host buffer: copy=%d bytes=%zu\n",kind,bytes);
    } else {std::memcpy(target,source,bytes);}
  });
  return 0;
}
int cudaStreamSynchronize(cudaStream_t) {
  ++syncs;
  for (auto& operation:operations) operation();
  operations.clear();
  return 0;
}
void enqueue_kernel(double* e,double* m,int* err) {
  operations.emplace_back([=] {*e=-0.125;*m=1.0;*err=0;});
}
struct CudaResult { double energy{},minimum_absolute_denominator{}; std::size_t virtual_triples{},workspace_bytes{}; };
"""

_DRIVER = r"""
int main(int argc,char** argv) {
  if(argc!=2) return 99;
  fail_at=std::atoi(argv[1]);
  double input[1]={1.0};
  bool threw=false;
  try {
    const auto result=evaluate_cuda(1,1,input,input,input,input,input,input,input,input,1e-12,4096,0);
    if(result.energy!=-0.125 || result.minimum_absolute_denominator!=1.0 || result.virtual_triples!=1) return 5;
  } catch (const std::runtime_error& error) {
    if(std::string(error.what()).find("injected CUDA failure")==std::string::npos) return 6;
    threw=true;
  }
  if(threw != (fail_at!=0)) return 7;
  if(streams || buffers || !operations.empty() || current_device!=3 || syncs<1) return 8;
  return invalid ? 1 : 0;
}
"""


@pytest.fixture(scope="module")
def lifetime_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Clang's lifetime markers include exception edges for trivial stack objects.
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires Clang and its AddressSanitizer runtime")
    source = (ROOT / "tools/generate_rccsdt_cuda.py").read_text(encoding="utf-8")
    begin = source.index("void cuda_check(")
    end = source.index("__device__ inline void fail_once", begin)
    helpers = source[begin:end].replace("{{", "{").replace("}}", "}")
    begin = source.index("CudaResult evaluate_cuda(")
    end = source.index("\n}}  // namespace vibeqc::cc::triples", begin)
    owner = source[begin:end].replace("{{", "{").replace("}}", "}")
    owner, replacements = re.subn(
        r"triples_kernel<<<blocks, threads, 0, stream>>>\([\s\S]*?\);",
        "(void)blocks; enqueue_kernel(energy, minimum, error);",
        owner,
    )
    assert replacements == 1
    directory = tmp_path_factory.mktemp("triples-host-lifetimes")
    unit, executable = directory / "lifetime.cpp", directory / "lifetime"
    unit.write_text(_CUDA_SHIM + helpers + owner + _DRIVER, encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-g",
            "-fno-elide-constructors",
            "-fsanitize=address",
            "-fsanitize-address-use-after-scope",
            str(unit),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("failure_point", range(15))
def test_emitted_owner_keeps_host_buffers_alive_through_cleanup(
    lifetime_probe: Path, failure_point: int
) -> None:
    result = subprocess.run(
        [str(lifetime_probe), str(failure_point)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

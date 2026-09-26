"""Execute the emitted global-cursor worker against an independent task census."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.integral.production_emission import _streaming_fock_source
from vibeqc_compiler.integral.production_profile import resolve_production_profile


def test_component_worker_continues_after_screened_candidate(tmp_path: Path) -> None:
    """A bra-local screened tail must not retire a global pair-product worker.

    Run the actual emitted control flow as one host CTA leader. Integral math,
    atomics between concurrent CTAs and barriers are outside this scheduler
    test; the independent nested-loop census checks every retained task once.
    Numerical CUDA qualification remains separate.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    root = Path(__file__).resolve().parents[2]
    profile = resolve_production_profile(
        root / "python/vibeqc_compiler/integral/production_shell_classes.json", "sm_120"
    )
    selection = next(s for s in profile.selections if s.spec.name == "ddpp")
    source = _streaming_fock_source(selection)
    marker = "__device__ __forceinline__ void generated_ddpp_streaming_fock("
    start = source.rindex("template <bool Unrestricted>", 0, source.index(marker))
    worker = source[start : source.index("\n}\n", source.index(marker)) + 3]
    driver = tmp_path / "stream.cpp"
    driver.write_text(
        r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>
#include "scf/generated_shell_task.hpp"
#define __device__
#define __forceinline__
#define __shared__
#define __syncthreads() ((void)0)
struct { unsigned x = 0; } threadIdx;
unsigned atomicAdd(unsigned* p, unsigned v) { unsigned old = *p; *p += v; return old; }
using Topology = vibeqc::scf::detail::GeneratedShellPairStream;
using GeneratedDdppShellTask = vibeqc::scf::detail::GeneratedShellTask;
using GeneratedDdppPrimitivePairData = vibeqc::scf::detail::GeneratedPrimitivePairData;
struct GeneratedDdppVec3 { double x, y, z; };
constexpr unsigned kGeneratedDdppFockBlockThreads = 352;
using Pair = std::pair<unsigned, unsigned>;
std::vector<Pair> actual;
// This rejection is deliberately not the coarse monotonic Schwarz predicate.
// It models the finer density gate and must also leave the worker alive.
bool exact_rejection(unsigned bra, unsigned ket) { return bra == 1 && ket == 6; }
template<bool U> bool generated_ddpp_stream_survives(
    const Topology& t, unsigned bra, unsigned ket, double threshold, double* bound) {
  *bound = t.shell_pair_bounds[bra] * t.shell_pair_bounds[ket];
  return t.active[t.shell_pair_systems[bra]] && *bound >= threshold &&
         !exact_rejection(bra, ket);
}
void generated_ddpp_record_fock_precision(unsigned long long* count) { ++*count; }
void generated_ddpp_stream_populate_task(
    const Topology&, unsigned bra, unsigned ket, GeneratedDdppShellTask& task) {
  task.shell_pair[0] = bra; task.shell_pair[1] = ket;
}
template<bool U> void generated_ddpp_shell_class_fock_task(
    const GeneratedDdppShellTask* task, const GeneratedDdppPrimitivePairData*,
    const std::int64_t*, const double*, const GeneratedDdppVec3*, double,
    const double*, const double*, double*, std::size_t) {
  actual.emplace_back(task[0].shell_pair[0], task[0].shell_pair[1]);
}
"""
        + worker
        + r"""
int main() {
  const std::int32_t systems[]{0,0,0,1,1,0,0,0,1,1};
  // Each segment is correctly sorted. The product of consecutive bra/ket
  // segments is not monotonic: a screened ket is followed by a new strong bra.
  const double bounds[]{8,4,2,.1,.05,9,3,.01,100,.01};
  std::uint8_t active[]{1,1};
  std::uint32_t offsets[30]{};
  std::vector<std::uint32_t> order;
  for(unsigned cls=0; cls<10; ++cls) {
    offsets[cls*3] = order.size();
    if(cls==2) order.insert(order.end(),{5,6,7});
    if(cls==5) order.insert(order.end(),{0,1,2});
    offsets[cls*3+1] = order.size();
    if(cls==2) order.insert(order.end(),{8,9});
    if(cls==5) order.insert(order.end(),{3,4});
    offsets[cls*3+2] = order.size();
  }
  double density_maxima[20]; std::fill_n(density_maxima,20,1.0);
  Topology topology{}; topology.batch_size=2;
  topology.pair_order=order.data(); topology.pair_class_offsets=offsets;
  topology.shell_pair_bounds=bounds; topology.shell_pair_systems=systems;
  topology.system_pair_density_bounds=density_maxima; topology.active=active;
  for(unsigned workers: {1U,4U}) for(unsigned mask: {1U,3U}) {
    active[0]=mask&1U; active[1]=(mask>>1U)&1U;
    std::vector<Pair> expected;
    for(unsigned bra=0;bra<5;++bra) for(unsigned ket=5;ket<10;++ket)
      if(systems[bra]==systems[ket] && active[systems[bra]] &&
         bounds[bra]*bounds[ket]>=1.0 && !exact_rejection(bra,ket))
        expected.emplace_back(bra,ket);
    for(bool unrestricted: {false,true}) {
      actual.clear(); unsigned head=0; unsigned long long count=0;
      for(unsigned w=0;w<workers;++w) {
        if(unrestricted) generated_ddpp_streaming_fock<true>(
            &topology,nullptr,nullptr,nullptr,nullptr,1.0,nullptr,nullptr,nullptr,&head,&count);
        else generated_ddpp_streaming_fock<false>(
            &topology,nullptr,nullptr,nullptr,nullptr,1.0,nullptr,nullptr,nullptr,&head,&count);
      }
      std::sort(actual.begin(),actual.end());
      if(actual!=expected || count!=expected.size()) {
        std::cerr<<"retained "<<actual.size()<<" expected "<<expected.size()<<'\n'; return 1;
      }
    }
  }
}
"""
    )
    executable = tmp_path / "stream"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            str(root / "src"),
            str(driver),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


@pytest.mark.parametrize("name", ["ssss", "dppp"])
def test_row_streaming_binary_searches_monotonic_coarse_tail(name: str) -> None:
    """Packed/subgroup streaming should not linearly probe the Schwarz tail."""

    root = Path(__file__).resolve().parents[2]
    profile = resolve_production_profile(
        root / "python/vibeqc_compiler/integral/production_shell_classes.json", "sm_120"
    )
    selection = next(s for s in profile.selections if s.spec.name == name)
    source = _streaming_fock_source(selection)
    prefix = f"generated_{name}"
    assert f"{prefix}_stream_coarse_ket_end" in source
    assert "while (low < high)" in source
    assert "bra_bound * topology.shell_pair_bounds[ket_pair]" in source
    worker_marker = f"__device__ __forceinline__ void {prefix}_streaming_fock("
    worker_start = source.index(worker_marker)
    worker = source[worker_start:]
    assert (
        f"const std::uint32_t coarse_ket_end = {prefix}_stream_coarse_ket_end("
        in worker
    )
    assert "ket_base < coarse_ket_end" in worker

"""Thread-emulate the emitted cooperative kernel with real generated Becke math."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from generativeqc_compiler.method.stationary_cuda import _STATIONARY_SCIENTIFIC_KERNELS
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials
from test_stationary_geometry_kernel_host import PREFIX


@pytest.mark.parametrize(
    ("atoms", "aos", "layout"),
    [
        (atoms, aos, "unordered")
        for atoms, aos in (
            (12, 2),
            (33, 2),
            (96, 2),
            (128, 2),
            (12, 96),
            (12, 1024),
            (96, 768),
            (128, 1024),
        )
    ]
    + [
        (12, 96, "grouped"),
        (96, 768, "grouped"),
        (128, 1024, "grouped"),
        (12, 96, "holes"),
        (12, 96, "single"),
        (12, 96, "within_atom"),
        (12, 96, "subset"),
        (12, 1024, "grouped"),
        (12, 0, "grouped"),
    ],
)
def test_emitted_cooperative_kernel_routes_tails_and_sticky_failure(
    tmp_path: Path,
    atoms: int,
    aos: int,
    layout: str,
) -> None:
    """Execute actual generated schedules and count reduction owner-label reads."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    prefix = PREFIX[: PREFIX.index("int local_norm=")]
    prefix = prefix.replace(
        "struct { size_t x{}; } threadIdx, blockIdx, blockDim;",
        r"""
#include <atomic>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
struct Dimension { size_t x{}; };
thread_local Dimension threadIdx, blockIdx;
Dimension blockDim;
thread_local std::barrier<>* active_barrier;
#define __syncthreads() active_barrier->arrive_and_wait()
#define __shared__ static
double* geometry_pair_storage;
int atomicAnd(int* pointer, int value) {
  return std::atomic_ref(*pointer).fetch_and(value);
}
std::atomic<size_t> owner_probes{0};
size_t owner_capacity=0;
int64_t checked_owner(const int64_t* owners, size_t index) {
  if(index>=owner_capacity) std::abort();
  return owners[index];
}
int64_t read_owner(const int64_t* owners, size_t index) {
  ++owner_probes;
  return checked_owner(owners,index);
}
""",
    )
    prefix = prefix.replace(
        "const int old=*p; *p=v; return old;", "return std::atomic_ref(*p).exchange(v);"
    )
    prefix = prefix.replace(
        "*error=1; return fallback;",
        "std::atomic_ref(*error).store(1); return fallback;",
    )
    prefix = prefix.replace(
        "void ao_pullback(", "std::atomic<size_t> ao_evaluations{0};\nvoid ao_pullback("
    ).replace(
        "  for (size_t j=0;j<4;++j) out[j]=c[j]*w[j];",
        "  ++ao_evaluations;\n  for (size_t j=0;j<4;++j) out[j]=c[j]*w[j];",
    )
    begin = _STATIONARY_SCIENTIFIC_KERNELS.index(
        "__device__ bool geometry_point_setup("
    )
    end = _STATIONARY_SCIENTIFIC_KERNELS.index(
        "}  // namespace generativeqc_stationary_cuda", begin
    )
    kernels = _STATIONARY_SCIENTIFIC_KERNELS[begin:end].replace(
        "  extern __shared__ double geometry_pair_storage[];", ""
    )
    reduction_begin = kernels.index("      const bool ordered_ao_owners =")
    reduction_end = kernels.index("      if (threadIdx.x == 0)", reduction_begin)
    kernels = (
        kernels[:reduction_begin]
        + kernels[reduction_begin:reduction_end].replace(
            "ao_atoms[global_ao]", "read_owner(ao_atoms, global_ao)"
        )
        + kernels[reduction_end:]
    )
    kernels = kernels.replace(
        "ao_atoms[global_ao]", "checked_owner(ao_atoms, global_ao)"
    ).replace(
        "ao_atoms[previous_global_ao]", "checked_owner(ao_atoms, previous_global_ao)"
    )
    source = tmp_path / "kernel.cpp"
    source.write_text(
        prefix
        + emit_grid_adjoint()
        + emit_grid_partials(3)
        + "constexpr size_t stationary_becke_retained_max_atoms=32, stationary_becke_pair_tile_rows=4;\n"
        + kernels
        + r"""
int main() {
  constexpr size_t na=ATOMS,n=AOS,np=(na>32?5:17),pairs=na*(na-1)/2;
  const std::string layout="LAYOUT";
  const size_t full_aos=layout=="subset"?2*n+3:n;
  owner_capacity=full_aos;
  constexpr size_t state_count=na<=32?pairs:4*(2*na-5)/2;
  double centers[3*na];
  for(size_t a=0;a<na;++a) {
    centers[3*a]=0.7*a; centers[3*a+1]=0.4*std::sin(a); centers[3*a+2]=0.3*std::cos(a);
  }
  std::vector<generativeqc_grid_adjoint::CenterPair> geometry(pairs);
  if(!generativeqc_grid_adjoint::prepare_center_geometry(centers,na,1e-12,geometry.data(),
                                                         local_norm,local_ratio_geometry)) return 1;
  std::vector<int64_t> ao_atoms(full_aos);
  std::vector<size_t> active_ids(n);
  for(size_t ao_index=0;ao_index<full_aos;++ao_index) {
    ao_atoms[ao_index]=layout=="unordered"?(ao_index*7)%na:ao_index*na/full_aos;
    if(layout=="single") ao_atoms[ao_index]=na/2;
    if(layout=="holes") ao_atoms[ao_index]=2*(ao_index*(na/2)/full_aos);
  }
  for(size_t ao_index=0;ao_index<n;++ao_index) {
    active_ids[ao_index]=layout=="unordered"?n-1-ao_index:ao_index;
    if(layout=="subset") active_ids[ao_index]=2*ao_index+1;
    if(layout=="within_atom") active_ids[ao_index]=(ao_index/8)*8+7-ao_index%8;
  }
  for(bool cached:{false,true}) for(bool implicit:{false,true}) for(bool external:{false,true})
  for(size_t capacity:{size_t(1),size_t(7)}) for(size_t points:{size_t(0),size_t(1),np}) {
    const size_t lanes=std::min(capacity,points);
    std::vector<double> partial(capacity*9*na+2,987654),scratch(capacity*9*na+2,987654);
    std::vector<double> storage(8*state_count+2,987654);
    geometry_pair_storage=storage.data()+1;
    std::vector<double> xyz(3*points),features(10*points,1),ao(10*points*n,0.5),work(8*points*n,0.75);
    for(size_t value_index=0;value_index<ao.size();++value_index)
      ao[value_index]=0.4*std::sin(0.17*value_index);
    for(size_t value_index=0;value_index<work.size();++value_index)
      work[value_index]=0.3*std::cos(0.13*value_index);
    std::vector<double> weights(points,0.3),raw(points,0.2),seeds(6*(np+7),0.15);
    std::vector<int64_t> owners(points);
    for(size_t p=0;p<points;++p) {
      owners[p]=implicit?(p+4)/3:(p+1==points?na-1:(7*p)%na);
      xyz[3*p]=0.1+0.3*p; xyz[3*p+1]=0.7; xyz[3*p+2]=-0.8;
    }
    const auto* center_pairs=cached?geometry.data():nullptr;
    int error=0,producer_error=0;
    generativeqc::dft::GridTaskView view{points,n,full_aos,features.data(),ao.data(),xyz.data(),
                                       implicit?active_ids.data():nullptr,&producer_error};
    auto invoke=[&](bool cooperative,size_t lane,size_t rank) {
      blockIdx.x=cooperative?lane:lane/32; threadIdx.x=cooperative?rank:lane%32;
      auto kernel=cooperative?geometry_cooperative_kernel:geometry_kernel;
      kernel(view,work.data(),ao_atoms.data(),implicit?nullptr:owners.data(),4,3,centers,na,
             weights.data(),raw.data(),external?seeds.data():nullptr,np+7,2,
             lanes,partial.data()+1,scratch.data()+1,center_pairs,&error);
    };
    auto reduce=[&] {
      std::vector<double> output(9*na);
      for(size_t k=0;k<9*na;++k) {
        blockIdx.x=k/32; threadIdx.x=k%32;
        geometry_reduce(partial.data()+1,na,lanes,output.data(),&error);
      }
      return output;
    };
    blockDim.x=32;
    for(size_t lane=0;lane<lanes;++lane) invoke(false,lane,0);
    const auto expected=reduce();
    std::fill(partial.begin(),partial.end(),987654);
    auto execute=[&] {
      for(size_t lane=0;lane<lanes;++lane) {
        std::barrier barrier(32);
        std::vector<std::thread> workers;
        for(size_t rank=0;rank<32;++rank) workers.emplace_back([&,rank] {
          active_barrier=&barrier; invoke(true,lane,rank);
        });
        for(auto& worker:workers) worker.join();
      }
    };
    ao_evaluations=0;
    owner_probes=0;
    execute();
    const auto actual=reduce();
    if(error) return 2;
    if(ao_evaluations!=2*n*points) return 7;
    const bool cooperative_ao=(sizeof(StationaryPointValue)+7)/8+3*n<=8*state_count;
    bool ordered=true;
    for(size_t ao_index=1;ao_index<n;++ao_index)
      ordered=ordered && ao_atoms[implicit?active_ids[ao_index-1]:ao_index-1]
                           <=ao_atoms[implicit?active_ids[ao_index]:ao_index];
    if(!cooperative_ao && owner_probes!=0) return 9;
    if(cooperative_ao && !ordered && owner_probes!=points*na*n) return 10;
    if(cooperative_ao && ordered) {
      size_t search_depth=0;
      for(size_t remaining=n;remaining;remaining/=2) ++search_depth;
      if(owner_probes>points*(na*search_depth+n+na)) return 11;
      if(n>=96 && points && owner_probes>=points*na*n) return 12;
      if(n==0 && owner_probes!=0) return 13;
    }
    if(cached && implicit && external && capacity==7 && points==np)
      std::printf("owner-work atoms=%zu active=%zu points=%zu layout=%s cooperative=%d "
                  "ordered=%d measured=%zu dense-scan=%zu\n",na,n,points,layout.c_str(),
                  int(cooperative_ao),int(ordered),owner_probes.load(),points*na*n);
    for(size_t coordinate=0;coordinate<6*na;++coordinate)
      if(expected[coordinate]!=actual[coordinate]) return 8;
    for(size_t k=0;k<9*na;++k) if(std::abs(expected[k]-actual[k])>2e-11) return 3;
    if(partial.front()!=987654 || partial.back()!=987654 || scratch.front()!=987654 ||
       scratch.back()!=987654 || storage.front()!=987654 || storage.back()!=987654) return 4;
    if(!points) continue;
    // Invalid input late in a worker never publishes any partial output.
    for(int invalid=0;invalid<8;++invalid) {
      const auto old_xyz=xyz,old_raw=raw,old_seeds=seeds;
      const auto old_atoms=ao_atoms;
      const auto old_ids=active_ids;
      if(!n && (invalid==3 || invalid==5 || invalid==6 || invalid==7)) continue;
      if(invalid==0) std::copy(centers,centers+3,xyz.end()-3);
      if(invalid==1) raw.back()=std::numeric_limits<double>::quiet_NaN();
      if(invalid==2) producer_error=1;
      if(invalid==3) view.nao=1;
      if(invalid==4) { if(!external) continue; seeds[5*(np+7)+points+1]=std::numeric_limits<double>::infinity(); }
      if(invalid==5) ao_atoms[implicit?active_ids.back():n-1]=-1;
      if(invalid==6) { active_ids.back()=full_aos; view.ao_ids=active_ids.data(); }
      if(invalid==7) { active_ids[n-2]=full_aos; view.ao_ids=active_ids.data(); }
      error=0; execute();
      const auto failed=reduce();
      if(!error) return 5;
      for(double value:failed) if(value!=0) return 6;
      xyz=old_xyz; raw=old_raw; seeds=old_seeds; producer_error=0; view.nao=full_aos;
      ao_atoms=old_atoms;
      active_ids=old_ids;
      view.ao_ids=implicit?active_ids.data():nullptr;
      // Rebind potentially replaced vector storage before the next replay.
      view.points=xyz.data();
    }
  }
  return 0;
}
""".replace("ATOMS", str(atoms))
        .replace("AOS", str(aos))
        .replace("LAYOUT", layout)
    )
    binary = tmp_path / "kernel"
    process = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-pthread",
            "-O2",
            "-ffp-contract=off",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    print(result.stdout, end="")


def test_emitted_team_vote_completes_reads_before_the_next_failure(
    tmp_path: Path,
) -> None:
    """A slow reader of a successful vote must not see the next vote's failure."""
    from test_stationary_task_work_budget import _block

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    team = _block(_STATIONARY_SCIENTIFIC_KERNELS, "struct GeometryBlockTeam {") + ";\n"
    source = tmp_path / "vote.cpp"
    source.write_text(
        r"""
#include <atomic>
#include <barrier>
#include <chrono>
#include <thread>
#include <vector>
#include <cstddef>
#define __device__
struct Dimension { size_t x{}; };
thread_local Dimension threadIdx;
Dimension blockDim{2};
thread_local std::barrier<>* active_barrier;
thread_local size_t barrier_count=0;
void delayed_barrier() {
  active_barrier->arrive_and_wait();
  if (barrier_count++ == 0 && threadIdx.x==1)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
#define __syncthreads() delayed_barrier()
int atomicExch(int* pointer,int value) { return std::atomic_ref(*pointer).exchange(value); }
"""
        + team
        + emit_grid_adjoint()
        + emit_grid_partials(3)
        + r"""
int main() {
  using namespace generativeqc_grid_adjoint;
  constexpr size_t na=33;
  double point[3]{0.4,0.7,0.9},centers[3*na]{};
  centers[3]=1e-310;
  for(size_t atom=2;atom<na;++atom) centers[3*atom]=0.7*atom;
  std::vector<CenterPair> pairs(na*(na-1)/2);
  // Zero tolerance legally admits distinct finite centers whose inverse
  // separation overflows. The first pair state fails after a successful vote.
  if(!prepare_center_geometry(centers,na,0,pairs.data(),local_norm,local_ratio_geometry)) return 1;
  for(size_t participants:{size_t(2),size_t(32)}) for(bool cached:{false,true}) {
    blockDim.x=participants;
    std::vector<double> gradient(3*na),logs(na),products(na),bar_product(na),bar_distance(na);
    std::vector<size_t> zeros(na);
    std::vector<std::array<double,4>> distances(na);
    std::vector<PointPair> states(4*(2*na-5)/2);
    int vote=1;
    std::atomic<size_t> rejected{0};
    std::barrier barrier(static_cast<std::ptrdiff_t>(participants));
    std::vector<std::thread> workers;
    for(size_t rank=0;rank<participants;++rank) workers.emplace_back([&,rank] {
      threadIdx.x=rank; active_barrier=&barrier;
      const bool result=contract_point_tiled_cooperative(point,centers,na,0,0.3,gradient.data(),
          logs.data(),products.data(),bar_product.data(),bar_distance.data(),zeros.data(),
          distances.data(),states.data(),4,GeometryBlockTeam{&vote},local_norm,local_ratio,
          local_log,local_becke,cached?pairs.data():nullptr,local_ratio_prepared);
      if(!result) ++rejected;
    });
    for(auto& worker:workers) worker.join();
    if(rejected!=participants) return 2;
  }
  return 0;
}
"""
    )
    binary = tmp_path / "vote"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-pthread",
            "-O2",
            "-ffp-contract=off",
            str(source),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=5, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

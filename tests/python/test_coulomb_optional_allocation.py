"""Exercise the optional owner's real exception scopes with a fake CUDA runtime."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Only external runtime/geometry types and kernel launches are replaced. The
# complete production preparation and destructor bodies are compiled verbatim,
# with explicit faults immediately before their early/late allocation sites.
STUBS = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>
using cudaStream_t = void*;
enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorUnknown };
constexpr int cudaLimitStackSize=0, cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2;
int injected_stage=0, injected_kind=0, live_allocations=0, fences=0;
void fault(int stage) {
  if(stage!=injected_stage) return;
  if(injected_kind==0) throw std::bad_alloc();
  if(injected_kind==1) throw cudaErrorMemoryAllocation;
  if(injected_kind==2) throw cudaErrorUnknown;
  throw std::logic_error("not an allocation failure");
}
struct cudaDeviceProp { int major=12, minor=0, multiProcessorCount=1; };
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int) { return cudaSuccess; }
cudaError_t cudaDeviceGetLimit(std::size_t* n,int) { *n=100000; return cudaSuccess; }
cudaError_t cudaDeviceSetLimit(int,std::size_t) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { ++fences; return cudaSuccess; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
cudaError_t cudaMemcpyAsync(void* d,const void* s,std::size_t n,int,cudaStream_t) {
  std::memcpy(d,s,n); return cudaSuccess;
}
cudaError_t cudaMemsetAsync(void* p,int v,std::size_t n,cudaStream_t) {
  std::memset(p,v,n); return cudaSuccess;
}
namespace runtime {
std::size_t size_mul(std::size_t a,std::size_t b) { return a*b; }
std::size_t size_add(std::size_t a,std::size_t b) { return a+b; }
int cuda_target_info_from_properties(cudaDeviceProp) { return 0; }
template<class... T> std::size_t vector_capacities(const T&...) { return 0; }
cudaError_t resource_cuda_malloc(void** p,std::size_t n) {
  *p=std::calloc(n ? n : 1,1);
  if(!*p) return cudaErrorMemoryAllocation;
  ++live_allocations; return cudaSuccess;
}
cudaError_t resource_cuda_free(void* p) {
  std::free(p); --live_allocations; return cudaSuccess;
}
}
namespace generated {
void select_profile_for_device(int,int,int) {}
std::uint64_t enabled_fock_shell_class_mask() { return 1; }
}
namespace cuda_policy {
struct Schedule { unsigned persistent_quartet_warps_per_sm=1; std::size_t cuda_stack_limit_bytes=1; };
Schedule resolve_direct_jk_schedule_policy(int) { return {}; }
}
namespace detail {
constexpr unsigned kDirectQuartetShellClassCount=1, kDirectShellPairClassCount=1, kDirectQuartetThreads=32;
enum class GeneratedFockConsumer { Coulomb };
}
constexpr std::uint64_t kGeneratedStreamingFockShellClassMask=1, kNativeStreamingFockShellClassMask=0;
constexpr unsigned kSchwarzThreads=32;
#define METADATA(F) F(system_shell_offsets) F(system_shell_pair_offsets) F(shell_direct_ao_offsets) \
 F(shell_pair_systems) F(shell_pair_first) F(shell_pair_second) F(shell_pair_primitive_offsets) \
 F(direct_ao_shells) F(direct_ao_angular)
struct HostBatch {
  std::size_t nbf=1, direct_nbf=1;
#define V(name) std::vector<std::int64_t> name{0,1};
  METADATA(V)
#undef V
  std::vector<std::int64_t> ao_shells{0}, ao_term_counts{0}, ao_term_angular{0,0,0};
  std::vector<double> ao_to_direct_transform, ao_term_coefficients{1}, direct_ao_coefficients{1};
};
struct PrimitivePairData { double value; };
struct DeviceBatch {
  std::size_t batch_size=1, total_shell_pairs=0, nbf=1, direct_nbf=1;
#define P(name) const std::int64_t* name=nullptr;
  METADATA(P)
#undef P
  const std::int64_t *shell_atoms=nullptr, *shell_angular=nullptr, *shell_primitive_offsets=nullptr;
  const double *ao_to_direct_transform=nullptr, *direct_ao_coefficients=nullptr;
  PrimitivePairData* shell_primitive_pairs=nullptr;
};
#undef METADATA
struct GeneratedShellPairStream {
  template<class... T> GeneratedShellPairStream(T...) {}
};
struct GeneratedCoulombPlan {
  DeviceBatch batch{};
  cudaStream_t stream{};
  std::vector<void*> allocations;
  std::size_t device_bytes{}, host_preparation_bytes{};
  std::uint64_t class_mask{};
  unsigned worker_blocks{};
  double screening{};
  double *density{}, *coulomb{}, *temporary{}, *total_density{}, *zero{}, *schwarz{}, *shell_bounds{};
  std::uint8_t* active{};
  std::uint32_t* heads{};
  GeneratedShellPairStream* topology{};
  ~GeneratedCoulombPlan();
};
std::uint64_t present_direct_shell_class_mask(const HostBatch&) { return 1; }
bool make_bounded_stream_shell_pair_order(const HostBatch&, std::vector<std::uint32_t>& a,
                                        std::vector<std::uint32_t>& b) {
  a={0,1}; b={0,2}; return true;
}
template<class... T> void launch_build_shell_primitive_pair_cache_kernel(T...) {}
template<class... T> void launch_build_schwarz_and_shell_pair_bounds_packed_kernel(T...) {}
void check(cudaError_t error) { if(error!=cudaSuccess) throw error; }
std::size_t product(std::size_t a,std::size_t b) { return runtime::size_mul(a,b); }
"""

DRIVER = r"""
int main(int argc,char** argv) {
  assert(argc==3);
  HostBatch host; DeviceBatch borrowed;
  injected_stage=std::atoi(argv[1]); injected_kind=std::atoi(argv[2]);
  bool propagated=false;
  try {
    auto plan=prepare_generated_coulomb(host,borrowed,reinterpret_cast<void*>(1),0,0.0,1<<20);
    if(injected_stage==0) assert(plan && live_allocations>0);
    else assert(!plan);
  } catch(const std::bad_alloc&) { return 2; }
    catch(cudaError_t error) { assert(injected_kind==2 && error==cudaErrorUnknown); propagated=true; }
    catch(const std::logic_error&) { assert(injected_kind==3); propagated=true; }
  assert(propagated==(injected_stage!=0 && injected_kind>=2));
  assert(live_allocations==0);
  if(injected_stage>=4) assert(fences>0);
  // A rejected optional owner must not poison a fresh preparation.
  injected_stage=0;
  auto recovered=prepare_generated_coulomb(host,borrowed,reinterpret_cast<void*>(1),0,0.0,1<<20);
  assert(recovered && live_allocations>0);
  recovered.reset(); assert(live_allocations==0);
}
"""


@pytest.fixture(scope="module")
def allocation_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    begin = source.index("GeneratedCoulombPlan::~GeneratedCoulombPlan()")
    end = source.index("cudaError_t enqueue_generated_coulomb(", begin)
    preparation = source[begin:end]
    for stage, anchor in enumerate(
        (
            "  if (!make_bounded_stream_shell_pair_order(",
            "    expanded_transform.assign(",
            "  auto plan = std::make_unique<GeneratedCoulombPlan>();",
            "    auto allocate =",
            "    std::vector<double> bounds(pairs);",
        ),
        start=1,
    ):
        assert preparation.count(anchor) == 1
        preparation = preparation.replace(anchor, f"fault({stage});\n" + anchor)
    directory = tmp_path_factory.mktemp("coulomb-allocation")
    cpp, binary = directory / "probe.cpp", directory / "probe"
    cpp.write_text(STUBS + preparation + DRIVER)
    subprocess.run(
        [compiler, "-std=c++17", str(cpp), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize("stage", [1, 2, 3, 4, 5])
def test_optional_host_allocation_failure_falls_back(
    allocation_probe: Path, stage: int
) -> None:
    subprocess.run([str(allocation_probe), str(stage), "0"], check=True, timeout=10)


@pytest.mark.parametrize("kind", [1, 2, 3])
def test_late_device_oom_falls_back_but_other_errors_propagate(
    allocation_probe: Path, kind: int
) -> None:
    subprocess.run([str(allocation_probe), "5", str(kind)], check=True, timeout=10)


def test_successful_optional_preparation_is_unchanged(allocation_probe: Path) -> None:
    subprocess.run([str(allocation_probe), "0", "0"], check=True, timeout=10)

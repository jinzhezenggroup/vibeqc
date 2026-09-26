"""Compile the actual bridge transaction with explicit cache/provider stubs.

The stub CPU cache locks each operation just like the real cache. Two real host
threads exercise the gap between SCC completion and orbital snapshot copying.
No GFN2 arithmetic, basis conversion or GPU qualification is claimed here.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_PREFIX = r"""
#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <thread>
#include <optional>
#include <stdexcept>
using vibeqc_xtb_status_t=int;
constexpr int VIBEQC_XTB_STATUS_SUCCESS=0,VIBEQC_XTB_STATUS_EIGENSOLVER_FAILED=1,
 VIBEQC_XTB_STATUS_SCC_NOT_CONVERGED=2,VIBEQC_XTB_STATUS_NOT_IMPLEMENTED=3;
constexpr int VIBEQC_XTB_API_VERSION=1,VIBEQC_XTB_MEMORY_HOST=0,
 VIBEQC_XTB_MODEL_GFN2_XTB=2,VIBEQC_XTB_COMPUTE_ENERGY=1,
 VIBEQC_XTB_COMPUTE_FORCES=2,VIBEQC_XTB_COMPUTE_ATOMIC_CHARGES=4,
 VIBEQC_XTB_SCC_START_FRESH=1,VIBEQC_XTB_SCC_MIXER_MODIFIED_BROYDEN=1;
constexpr double VIBEQC_XTB_DEFAULT_ELECTRONIC_TEMPERATURE=300;
struct vibeqc_xtb_const_buffer_t {const void* data;std::size_t bytes;int location;unsigned reserved;};
struct vibeqc_xtb_buffer_t {void* data;std::size_t bytes;int location;unsigned reserved;};
template<class T> vibeqc_xtb_const_buffer_t input_buffer(std::span<const T> x) {
 return {x.empty()?nullptr:x.data(),x.size_bytes(),VIBEQC_XTB_MEMORY_HOST,0u};
}
template<class T> vibeqc_xtb_buffer_t output_buffer(std::vector<T>& x) {
 return {x.empty()?nullptr:x.data(),x.size()*sizeof(T),VIBEQC_XTB_MEMORY_HOST,0u};
}
struct vibeqc_xtb_batch_t {
 std::size_t struct_size;unsigned api_version;std::int64_t batch_size,total_atoms;
 vibeqc_xtb_const_buffer_t atom_offsets,atomic_numbers,positions,molecular_charges,unpaired_electrons,spin_channels;
};
struct vibeqc_xtb_compute_options_t {
 std::size_t struct_size;unsigned api_version;int model;unsigned flags;
 int max_scc_iterations,scc_start_mode,scc_mixer,scc_mixer_history;
 double charge_tolerance,energy_tolerance,electronic_temperature,scc_mixer_damping;
};
struct vibeqc_xtb_batch_result_t {
 std::size_t struct_size;unsigned api_version;
 vibeqc_xtb_buffer_t energies,forces,atomic_charges,scc_iterations,scc_converged,per_system_status;
};
enum class Gfn2RuntimeBackend : std::uint8_t {kCpu=1,kCuda=2};
enum class Gfn2RuntimeStatus : std::uint8_t {kSuccess=0,kInvalidArgument,kBackendUnavailable,kNotSupported,kNotImplemented,kAllocationFailed,kNotConverged,kEigensolverFailed,kInternalError};
struct Gfn2RuntimeRequest {
 std::span<const std::int32_t> atomic_numbers; std::span<const double> positions;
 int charge=0;unsigned multiplicity=1;bool compute_forces=false,compute_atomic_charges=false,compute_orbitals=false;
 std::int32_t maximum_iterations=0,mixer_history=0;double energy_tolerance=0,charge_tolerance=0;
};
struct Gfn2RuntimeOrbitals {int marker=0;};
struct Gfn2RuntimeResult {
 Gfn2RuntimeStatus status=Gfn2RuntimeStatus::kInternalError;std::string detail;double energy=0;
 std::vector<double> forces,atomic_charges;std::optional<Gfn2RuntimeOrbitals> orbitals;unsigned iterations=0;bool converged=false;
};
Gfn2RuntimeStatus map_status(int s) {return s==3?Gfn2RuntimeStatus::kNotImplemented:Gfn2RuntimeStatus::kInternalError;}
struct Gate {
 std::mutex mutex; std::condition_variable condition;
 bool race=false,first=false,second=false,overlap=false;
} gate;
namespace vibeqc::xtb::detail {
struct Gfn2CpuExecutionCache {std::mutex mutex; int marker=0;};
struct Gfn2CpuOrbitalSnapshot {int marker=0;};
int execute_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,const vibeqc_xtb_batch_t& b,
 const vibeqc_xtb_compute_options_t&,vibeqc_xtb_batch_result_t& out,std::string&) {
 const int marker=int(*static_cast<const double*>(b.molecular_charges.data));
 {
  std::lock_guard<std::mutex> lock(cache.mutex);
  cache.marker=marker;
  *static_cast<double*>(out.energies.data)=marker;
  *static_cast<std::int32_t*>(out.scc_iterations.data)=1;
  *static_cast<std::uint8_t*>(out.scc_converged.data)=1;
  *static_cast<std::int32_t*>(out.per_system_status.data)=0;
 }
 if(gate.race) {
  std::unique_lock<std::mutex> lock(gate.mutex);
  if(marker==1) {
   gate.first=true;gate.condition.notify_all();
   gate.overlap=gate.condition.wait_for(lock,std::chrono::milliseconds(1000),[]{return gate.second;});
  } else {
   gate.second=true;gate.condition.notify_all();
  }
 }
 return marker==4 ? VIBEQC_XTB_STATUS_NOT_IMPLEMENTED : VIBEQC_XTB_STATUS_SUCCESS;
}
int copy_restricted_gfn2_orbital_snapshot_cpu(Gfn2CpuExecutionCache& cache,
 Gfn2CpuOrbitalSnapshot& snapshot,std::string&) {
 std::lock_guard<std::mutex> lock(cache.mutex);
 snapshot.marker=cache.marker;
 if(cache.marker==6)throw std::runtime_error("snapshot allocation injection");
 return cache.marker==3 ? VIBEQC_XTB_STATUS_NOT_IMPLEMENTED : VIBEQC_XTB_STATUS_SUCCESS;
}
}
bool convert_cpu_orbitals(const Gfn2RuntimeRequest&,
 const vibeqc::xtb::detail::Gfn2CpuOrbitalSnapshot& snapshot,Gfn2RuntimeOrbitals& out,std::string&) {
 out.marker=snapshot.marker;return snapshot.marker!=5;
}
struct Gfn2RuntimeBridge {
 struct Impl;
 std::unique_ptr<Impl> impl_;
 Gfn2RuntimeBridge(Gfn2RuntimeBackend,int);
 ~Gfn2RuntimeBridge();
 Gfn2RuntimeBridge(Gfn2RuntimeBridge&&) noexcept;
 Gfn2RuntimeBridge& operator=(Gfn2RuntimeBridge&&) noexcept;
 Gfn2RuntimeResult execute(const Gfn2RuntimeRequest&);
};
"""

_DRIVER = r"""
int main(int argc,char** argv) {
 if(argc!=2)return 99;
 std::string mode=argv[1];
 std::vector<std::int32_t> z{2};std::vector<double> xyz{0,0,0};
 Gfn2RuntimeRequest a;a.atomic_numbers=z;a.positions=xyz;a.charge=1;a.compute_orbitals=true;
 auto b=a;b.charge=2;
 Gfn2RuntimeBridge one(Gfn2RuntimeBackend::kCpu,0),two(Gfn2RuntimeBackend::kCpu,0);
 if(mode=="shared-without-snapshot")b.compute_orbitals=false;
 if(mode=="shared-with-snapshot" || mode=="shared-without-snapshot" || mode=="distinct") {
  gate.race=true;Gfn2RuntimeResult first,second;
  std::thread t1([&]{first=one.execute(a);});
  std::thread t2([&]{
   {std::unique_lock<std::mutex> lock(gate.mutex);
    if(!gate.condition.wait_for(lock,std::chrono::seconds(5),[]{return gate.first;}))std::terminate();}
   second=(mode=="distinct"?two:one).execute(b);
  });
  t1.join();t2.join();
  if(first.status!=Gfn2RuntimeStatus::kSuccess || first.energy!=1 ||
     !first.orbitals || first.orbitals->marker!=1) {
   std::cerr<<"request 1 energy/orbital mismatch: "<<first.energy<<" / "
            <<(first.orbitals?first.orbitals->marker:-1)<<'\n';return 1;
  }
  if(second.status!=Gfn2RuntimeStatus::kSuccess || second.energy!=2)return 2;
  if(b.compute_orbitals && (!second.orbitals || second.orbitals->marker!=2))return 3;
  if(!b.compute_orbitals && second.orbitals)return 4;
  if(mode=="distinct" && !gate.overlap)return 5;
 } else {
  a.charge=mode=="snapshot-failure"?3:mode=="execution-failure"?4:mode=="conversion-failure"?5:6;
  bool failed=false;
  try {auto r=one.execute(a);failed=r.status!=Gfn2RuntimeStatus::kSuccess && !r.orbitals;}
  catch(const std::runtime_error&) {failed=a.charge==6;}
  if(!failed)return 6;
  const auto result=one.execute(b);
  if(result.status!=Gfn2RuntimeStatus::kSuccess || !result.orbitals || result.orbitals->marker!=2)return 7;
 }
 std::cout<<"PASS "<<mode<<" (execution/snapshot providers are test stubs)\n";
}
"""


@pytest.fixture(scope="module")
def snapshot_transaction(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = (ROOT / "src/methods/gfn2_runtime_bridge.cpp").read_text()
    begin = source.index("struct Gfn2RuntimeBridge::Impl {")
    end = source.index("const char* gfn2_runtime_status_name", begin)
    directory = tmp_path_factory.mktemp("gfn2-snapshot-transaction")
    unit, executable = directory / "probe.cpp", directory / "probe"
    unit.write_text(_PREFIX + source[begin:end] + _DRIVER)
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-pthread",
            "-fsanitize=undefined",
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


@pytest.mark.parametrize(
    "mode",
    [
        "shared-with-snapshot",
        "shared-without-snapshot",
        "distinct",
        "snapshot-failure",
        "execution-failure",
        "conversion-failure",
        "exception",
    ],
)
def test_orbital_snapshot_stays_with_its_completed_request(
    snapshot_transaction: Path, mode: str
) -> None:
    completed = subprocess.run(
        [str(snapshot_transaction), mode],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

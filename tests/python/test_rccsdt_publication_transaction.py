"""Execute the real post-triples publication block with fault-injected forces.

The host shim models only orchestration and diagnostics, not force algebra or
CUDA execution; molecular accuracy remains covered by the public native tests.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def publication(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = (ROOT / "src/methods/rccsdt_method.cpp").read_text(encoding="utf-8")
    start = source.index("      auto diagnostic = state.diagnostic;")
    stop = source.index("      return state.result;", start)
    body = source[start:stop] + "      return state.result;\n"
    directory = tmp_path_factory.mktemp("triples-publication")
    unit, executable = directory / "publication.cpp", directory / "publication"
    unit.write_text(
        r"""
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
struct Diagnostic {
  double minimum_absolute_denominator{9}, ccsd_t_triples_energy{};
  double response_absolute_residual{}, response_relative_residual{};
  std::uint64_t numeric_capacity_bytes{}, correlation_owned_device_bytes{};
  std::uint64_t ccsd_t_virtual_triples{}, ccsd_t_workspace_bytes{};
  std::uint64_t response_iterations{}, response_restarts{}, response_workspace_bytes{};
  std::uint64_t measured_response_workspace_peak_bytes{}, response_workspace_allocation_count{};
  std::uint64_t planned_endpoint_peak_bytes{}, force_provenance_flags{};
  char ccsd_t_equation_hash[65]{}, response_operator_hash[65]{};
};
struct Result { double energy{}; std::vector<double> forces; };
struct State {
  Diagnostic diagnostic;
  Result result;
  struct { double total_energy{10}; } solved;
  std::optional<int> reference{1};
  int problem{}, eps_o{}, eps_v{};
  std::size_t budget{1024};
};
int failure_mode{};
bool cuda_mode{};
int force_backend{};
namespace cc {
namespace triples::generated { constexpr char inventory_hash[]="triples"; }
struct Force {
  std::vector<double> forces{1,2,3};
  struct {
    std::uint64_t iterations{4}, restarts{}, workspace_bytes{64};
    std::uint64_t measured_workspace_peak_bytes{64}, workspace_allocation_count{1};
    double residual_norm{1e-12}, relative_residual{1e-13};
  } orbital_response;
  struct {
    bool cuda_actions{};
    std::uint64_t owned_device_bytes{128};
  } lambda;
  double independent_orbital_residual{1e-12};
  std::uint64_t numeric_capacity_bytes{256};
  std::uint64_t response_owned_device_bytes{192};
  std::uint64_t response_h2d_bytes{}, response_d2h_bytes{}, response_synchronizations{};
  bool cuda_response_actions{};
  std::string response_operator_hash{"force"};
};
template<class... T> Force rccsdt_force_cpu(T&&...) {
  force_backend=1;
  if (failure_mode==3) throw std::bad_alloc();
  if (failure_mode==4) throw std::length_error("force budget");
  if (failure_mode==5) throw std::runtime_error("force solve");
  return {};
}
template<class... T> Force rccsdt_force_cuda(T&&...) {
  force_backend=2;
  if (failure_mode==3) throw std::bad_alloc();
  if (failure_mode==4) throw std::length_error("force budget");
  if (failure_mode==5) throw std::runtime_error("force solve");
  Force result;
  result.lambda.cuda_actions=true;
  result.cuda_response_actions=true;
  return result;
}
}
namespace runtime { enum class ExecutionMemorySpace { Host, Device }; }
struct Execution {
  bool cuda_requested() const { return cuda_mode; }
  int device_id() const { return 0; }
  template<class... T> void observe_workspace_peak(T...) {}
  template<class... T> void observe_numeric_peak(T...) {}
};
std::size_t checked_add(std::size_t a,std::size_t b) { return a+b; }
struct Owner {
  Execution execution_;
  std::optional<Diagnostic> last_;
  int system_{};
  struct { double ccsd_denominator_threshold{1e-10}; } descriptor_;
  Result run(bool compute_forces) {
    State state;
    if (failure_mode==2) state.reference.reset();
    last_=state.diagnostic; // Retain the existing CC convergence diagnostic.
    const std::size_t retained=16, triples_virtual_count=7, triples_workspace_bytes=32;
    const double triples_energy=0.25, triples_minimum_denominator=2;
"""
        + body
        + r"""
  }
};
int main(int argc,char** argv) {
  if(argc!=3) return 99;
  failure_mode=std::atoi(argv[1]);
  cuda_mode=std::atoi(argv[2])!=0;
  Owner owner;
  try {
    const auto result=owner.run(failure_mode!=0);
    if(failure_mode>=2 || !owner.last_ || result.energy!=10.25) return 1;
    if(owner.last_->ccsd_t_virtual_triples!=7) return 2;
    if(failure_mode==1 &&
       (result.forces.size()!=3 ||
        owner.last_->force_provenance_flags!=(cuda_mode ? 15u : 7u) ||
        force_backend!=(cuda_mode ? 2 : 1)))
      return 3;
  } catch(const std::exception&) {
    if(failure_mode<2 || !owner.last_) return 4;
    if(owner.last_->ccsd_t_virtual_triples!=0 || owner.last_->ccsd_t_triples_energy!=0 ||
       owner.last_->ccsd_t_equation_hash[0]!='\0') return 5;
  }
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
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


@pytest.mark.parametrize("cuda", (False, True))
@pytest.mark.parametrize("mode", range(6))
def test_post_triples_diagnostic_is_published_only_after_success(
    publication: Path, mode: int, cuda: bool
) -> None:
    result = subprocess.run(
        [str(publication), str(mode), str(int(cuda))],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr

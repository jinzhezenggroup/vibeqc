"""Execute production source handoffs with counted host owner stand-ins.

The numerical providers and solvers are not run here; this regression checks
which storage survives each stage and the retained-input budget handoff.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
int plan_live=0, view_live=0, raw_live=0;
int source_failure=0;
std::size_t plan_bytes=64;
namespace integrals {
struct ElectronInteractionSource {
  virtual ~ElectronInteractionSource() = default;
  virtual std::size_t retained_numeric_bytes() const = 0;
};
}
namespace scf {
struct PreparedFockPlan {
  PreparedFockPlan() { ++plan_live; }
  ~PreparedFockPlan() { if (view_live) std::abort(); --plan_live; }
};
struct PreparedFockInteractionSourceView : integrals::ElectronInteractionSource {
  explicit PreparedFockInteractionSourceView(const PreparedFockPlan&) { ++view_live; }
  ~PreparedFockInteractionSourceView() { --view_live; }
  std::size_t retained_numeric_bytes() const override { return plan_bytes; }
};
}
namespace posthf {
struct RawSource : integrals::ElectronInteractionSource {
  explicit RawSource(int,const int* = nullptr) {
    if (plan_live || view_live) std::abort();
    ++raw_live;
  }
  ~RawSource() { --raw_live; }
  std::size_t retained_numeric_bytes() const override { return 16; }
};
std::size_t checked_add(std::size_t a,std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max()-a)
    throw std::overflow_error("retained-input overflow");
  return a+b;
}
}
struct Problem { std::size_t reference_retained_bytes=100, provider_peak_bytes{}; };
struct State { Problem problem; };
struct Execution { int device_id() const { return 0; } };
Problem build_problem(const integrals::ElectronInteractionSource& source,
                      int,int,bool,int,int& work,int& metrics,const int* correlation_auxiliary) {
  if (correlation_auxiliary) throw std::logic_error("conventional lifetime fixture requires no auxiliary");
  // Real providers increment work before a source read may fail. Validate only
  // this attempt's delta while retaining both attempts in endpoint diagnostics.
  const int initial_work=work, initial_metrics=metrics;
  ++work; ++metrics;
  if (plan_live && source_failure) {
    const int failure=source_failure; source_failure=0;
    if (failure==1) throw std::length_error("optional provider admission");
    throw std::bad_alloc();
  }
  if (work-initial_work!=1 || metrics-initial_metrics!=1)
    throw std::logic_error("invalid provider attempt counters");
  // The provider already charges the source exactly once during its own phase.
  return {100, source.retained_numeric_bytes()+110};
}
"""


@pytest.fixture(scope="module")
def source_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    mp2 = (ROOT / "src/methods/mp2_method.cpp").read_text()
    cc = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    mp2_setup = mp2.split("      std::unique_ptr<posthf::RawSource> raw_source;", 1)[
        1
    ].split("      const auto corr =", 1)[0]
    force_prefix = mp2.split("      if (compute_forces) {", 1)[1].split(
        "        response::GmresOptions response_options;", 1
    )[0]
    cc_handoff = cc.split("    std::unique_ptr<posthf::RawSource> raw_source;", 1)[
        1
    ].split('    allocation_stage = "CC resident solve";', 1)[0]
    program = (
        PREFIX
        + r"""
int mp2_case(bool prepared,bool compute_forces) {
  std::unique_ptr<scf::PreparedFockPlan> cpu_exact_plan_;
  if (prepared) cpu_exact_plan_=std::make_unique<scf::PreparedFockPlan>();
  auto* prepared_exact=cpu_exact_plan_.get();
  const bool density_fitted_=false;
  int system_=0;
  std::optional<int> auxiliary_;
  std::unique_ptr<posthf::RawSource> raw_source;
"""
        + mp2_setup
        + r"""
  if (!conventional_source) return 1;
  if (prepared && (raw_live || !plan_live || !view_live)) return 2;
  if (compute_forces) {
"""
        + force_prefix
        + r"""
    if (plan_live || view_live || raw_live != 1 || !raw_source) return 3;
  } else if (prepared && (!plan_live || raw_live)) return 4;
  return 0;
}
int cc_case(bool prepared,bool optional_cuda=false,int failure=0) {
  std::unique_ptr<scf::PreparedFockPlan> owner;
  if (prepared) owner=std::make_unique<scf::PreparedFockPlan>();
  auto* prepared_exact=owner.get();
  auto* cuda_source_cache=&owner;
  State state;
  int system=0, reference_value=0, solver_options=0, provider_work=0, provider_metrics=0;
  const auto* reference=&reference_value;
  const int* correlation_auxiliary=nullptr;
  const bool cuda=optional_cuda || !prepared;
  source_failure=failure;
  Execution execution;
  const auto problem_started=std::chrono::steady_clock::now();
  std::unique_ptr<posthf::RawSource> raw_source;
"""
        + cc_handoff
        + r"""
  const bool kept=prepared && !failure;
  if (state.problem.reference_retained_bytes != (kept ? 164u : 100u)) return 5;
  if (state.problem.provider_peak_bytes != (kept ? 174u : 126u)) return 6;
  if (raw_live || view_live || plan_live != int(kept)) return 7;
  if (provider_work != (failure ? 2 : 1) || provider_metrics != provider_work) return 11;
  return 0;
}
int main(int argc,char** argv) {
  if (argc != 2) return 8;
  const int mode=std::atoi(argv[1]);
  int result=0;
  if (mode < 4) result=mp2_case(mode < 2,mode%2);
  else if (mode < 6) result=cc_case(mode==4);
  else if (mode==6) {
    plan_bytes=std::numeric_limits<std::size_t>::max();
    try { (void)cc_case(true); return 9; }
    catch (const std::overflow_error&) {}
  }
  else result=cc_case(true,true,mode-7);
  if (plan_live || view_live || raw_live) return 10;
  return result;
}
"""
    )
    directory = tmp_path_factory.mktemp("posthf-source-lifetime")
    path, executable = directory / "probe.cpp", directory / "probe"
    path.write_text(program)
    compiled = subprocess.run(
        [compiler, "-std=c++20", "-O0", str(path), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


def test_posthf_source_lifetime_matches_retained_budget(source_probe: Path) -> None:
    for mode in range(10):
        process = subprocess.run(
            [str(source_probe), str(mode)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert process.returncode == 0, (mode, process.returncode, process.stderr)


def test_device_interaction_source_defaults_fail_closed(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "device_source_contract.cpp"
    executable = tmp_path / "device_source_contract"
    source.write_text(
        r"""
#include <array>
#include <cstddef>
#include <stdexcept>
#include "integrals/electron_interaction_source.hpp"

using generativeqc::integrals::DeviceInteractionTarget;
using generativeqc::integrals::ElectronInteractionOperator;
using generativeqc::integrals::ElectronInteractionSource;

struct HostOnly final : ElectronInteractionSource {
  const generativeqc::core::System& orbital() const override {
    static const generativeqc::core::System system{};
    return system;
  }
  std::size_t nbf() const override { return 1; }
  std::size_t naux() const override { return 0; }
  std::size_t retained_numeric_bytes() const override { return 0; }
  bool supports(Operator op) const noexcept override {
    return op == ElectronInteractionOperator::eri;
  }
  void read(Operator, const std::array<std::size_t, 4>&,
            const std::array<std::size_t, 4>&, double*, std::size_t) const override {}
};

int main() {
  HostOnly source;
  if (!source.supports_host_read(ElectronInteractionOperator::eri)) return 1;
  if (source.supports_device_read(ElectronInteractionOperator::eri, 0)) return 2;
  std::array<std::size_t, 4> begin{0, 0, 0, 0}, count{1, 1, 1, 1};
  try {
    source.read_device(ElectronInteractionOperator::eri, begin, count,
                       DeviceInteractionTarget{0, nullptr, nullptr, 1}, 1);
  } catch (const std::invalid_argument&) {
    return 0;
  }
  return 3;
}
"""
    )
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    process = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert process.returncode == 0, process.stderr


def test_native_provider_keeps_host_fallback_and_device_handoff() -> None:
    provider = (ROOT / "src/posthf/native_provider.cpp").read_text()
    transform_header = (ROOT / "src/posthf/cuda_transform.hpp").read_text()
    transform_cuda = (ROOT / "src/posthf/cuda_transform.cu").read_text()

    assert "source_.supports_device_read(" in provider
    assert "source_.supports_host_read(" in provider
    assert "source_.read_device(" in provider
    assert "source_.read(" in provider
    assert "posthf_cuda_batch_input_v1(" in provider
    assert "posthf_cuda_batch_add_device_v1(" in provider
    assert "posthf_cuda_batch_add_v1(" in provider

    assert "posthf_cuda_batch_input_v1" in transform_header
    assert "posthf_cuda_batch_add_device_v1" in transform_header
    assert "raw_borrowed" in transform_cuda
    assert "if (p.raw_borrowed)" in transform_cuda


def test_cc_force_hamiltonian_uses_prepared_cuda_source() -> None:
    force = (ROOT / "src/cc/rccsdt_force.cpp").read_text()
    rccsd = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    rccsdt = (ROOT / "src/methods/rccsdt_method.cpp").read_text()

    assert "raw_hamiltonian(source, reference, max_bytes, cuda_derivative" in force
    assert (
        "provider.get({all, all, all, all}, cuda, device_id, nullptr, &work)" in force
    )
    assert "if (!execution_.cuda_requested() && cpu_exact_plan_)" not in rccsd
    assert "if (!execution_.cuda_requested() && cpu_exact_plan_)" not in rccsdt
    assert "force_prepared_source.emplace(*cpu_exact_plan_)" in rccsd
    assert "force_prepared_source.emplace(*cpu_exact_plan_)" in rccsdt

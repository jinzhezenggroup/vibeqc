"""Run the production RKS closure/controller with deterministic physical providers.

The scalar provider has an RMS-small, maximum-large commutator only at the
final physical audit. All convergence, retry, ownership and history logic is
extracted unchanged from rks.cpp, including the shared iteration driver.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def closure_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    cache = shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires ccache and a host C++ compiler")
    source = (ROOT / "src/dft/rks.cpp").read_text()
    begin = source.index("  struct RksLoopEvaluation {")
    end = source.index("\n}\n\n}  // namespace", begin)
    region = source[begin:end]
    algebra = (ROOT / "src/scf/reference/mean_field.cpp").read_text()
    norm_begin = algebra.index("double residual_max_abs(")
    norm_end = algebra.index("\n\n", norm_begin)
    norm = algebra[norm_begin:norm_end]
    directory = tmp_path_factory.mktemp("cpu-ks-final-closure")
    program = (
        r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "solver/self_consistent.hpp"
struct Matrix : std::vector<double> {
 using std::vector<double>::vector;
 std::shared_ptr<int> lifetime;
};
std::weak_ptr<int> audit_fock, audit_residual;
namespace dft {
struct EnergyComponents { double value{}; };
}
namespace runtime {
template<class... T> std::size_t vector_capacities(const T&...) { return 0; }
std::size_t vector_bytes(const Matrix&) { return 0; }
std::size_t add_capacity(std::size_t a, std::size_t b) { return a+b; }
void sample_cpu_capacity(std::size_t) {}
}
struct DensityFactorIdentity { unsigned generation{}; };
struct OccupiedDensityFactor { Matrix density; unsigned generation{}; };
struct IncrementalDiagnostic {
 unsigned strict_refinement_iterations{}, final_audits{}, audit_failures{};
};
struct History {
 unsigned iteration; dft::EnergyComponents components;
 double energy_change, density_rms, residual_rms; std::array<double,2> electrons;
};
struct KsDiagnostic {
 double physical_residual{}, density_change{}; dft::EnergyComponents components;
 std::array<double,2> electrons{}; std::vector<History> history;
 IncrementalDiagnostic incremental_xc;
};
struct Result {
 unsigned iterations{}, fock_builds{}; bool converged{};
 double energy{}, energy_change{}, density_rms{}, physical_residual_rms{};
 Matrix density, ks_physical_fock;
 KsDiagnostic ks;
 std::shared_ptr<const OccupiedDensityFactor> factor;
};
struct Options {
 unsigned max_iterations; double energy_tolerance=1e-12, density_tolerance=1e-10;
 bool retain_ks_state=true;
};
unsigned observed_progress{}, strict_entries{}, full_calls{}, eigen_calls{};
struct IncrementalState {
 void observe_progress(double) { ++observed_progress; }
 void enter_strict_refinement() { ++strict_entries; }
};
struct Diis {
 unsigned clears{};
 Matrix update(const Matrix& f, const Matrix&) { return f; }
 void clear() { ++clears; }
};
struct Physical { Matrix fock; double energy; dft::EnergyComponents components; };
struct Eigen { Matrix values, vectors; };
unsigned calls{}, proposals{}, recorded_builds{}, retain_calls{};
double last_amplitude{};
bool incremental_mode{};
unsigned audit_call{};
Matrix last_density;
double residual_rms(const Matrix& r) {
 double sum=0; for (double x:r) sum+=x*x; return std::sqrt(sum/r.size());
}
double density_rms(const Matrix& a,const Matrix& b) {
 Matrix r(a); for(unsigned i=0;i<r.size();++i) r[i]-=b[i]; return residual_rms(r);
}
double dot(const Matrix& a,const Matrix&) { return a[0]; }
Matrix commutator_residual(const Matrix& f,const Matrix&,const Matrix&,std::size_t) {
 Matrix r(16); r[1]=f[1]; r[4]=-f[1];
 if(f[0]==audit_call) { r.lifetime=std::make_shared<int>(1); audit_residual=r.lifetime; }
 return r;
}
Eigen generalized_eigen(const Matrix&,const Matrix&,std::size_t) { ++eigen_calls; return {}; }
"""
        + norm
        + r"""
Result run(unsigned budget, bool incremental_xc, int scenario, unsigned& clears) {
 calls=proposals=retain_calls=0;
 observed_progress=strict_entries=full_calls=eigen_calls=0;
 incremental_mode=incremental_xc; audit_call=incremental_xc?5:4;
 const bool ordinary_primary=scenario!=9;
 Options options{budget}; Result result; auto& ks=result.ks;
 struct { double physical_residual{}; } diagnostic;
 Diis diis;
 struct { Matrix overlap; } ints;
 const std::size_t n=4;
 Matrix density(16), orthogonalizer;
 Eigen orbitals;
 std::shared_ptr<const OccupiedDensityFactor> factor =
     std::make_shared<const OccupiedDensityFactor>(OccupiedDensityFactor{density});
 DensityFactorIdentity identity;
 std::optional<IncrementalState> incremental_state;
 if(incremental_xc) incremental_state.emplace();
 const auto retained_capacity = [&](const Matrix&) { return std::size_t{}; };
 const auto evaluate_current = [&](const Matrix& current, std::size_t = 0, bool strict_full = false) {
  ++calls; full_calls+=strict_full; recorded_builds=result.fock_builds; last_density=current;
  if(calls==audit_call+1 && (!audit_fock.expired() || !audit_residual.expired()))
   throw std::runtime_error("rejected audit buffers survived restart");
  if((scenario==3 && calls==3) || (scenario==4 && calls==4))
   throw std::runtime_error("injected physical provider failure");
  if (factor->density!=current) throw std::runtime_error("physical D/factor mismatch");
  const bool audit = incremental_xc ? calls>=5 && (calls-5)%3==0 : calls>=4 && calls%4==0;
  const bool bad = audit && (scenario==2 || (scenario==1 && calls==(incremental_xc?5U:4U)));
  Matrix f(16); f[0]=calls; f[1]=bad ? 2e-10 : 5e-11;
  if(scenario==5 || (scenario==8 && calls>audit_call)) f[1]=2e-8;
  if((scenario==8 || scenario==9) && calls==audit_call) f[1]=2e-10;
  if(audit && scenario==6) f[1]=1e-10;
  if(calls==audit_call && scenario==7) f[1]=std::nextafter(1e-10,1.0);
  last_amplitude=f[1];
  if(calls==audit_call) { f.lifetime=std::make_shared<int>(1); audit_fock=f.lifetime; }
  // KS energy is supplied independently of F and must remain authoritative.
  return Physical{std::move(f), current[0], {current[0]}};
 };
 const auto next_density_from_orbitals = [&](const Matrix& current,std::size_t = 0) {
  Matrix next=current; next[0]+=1e-14; ++proposals; ++identity.generation;
  factor=std::make_shared<const OccupiedDensityFactor>(OccupiedDensityFactor{next,identity.generation});
  return next;
 };
 const auto retain_factor = [&] {
  ++retain_calls;
  if(factor->generation!=identity.generation) throw std::runtime_error("factor identity mismatch");
  result.factor=factor; clears=diis.clears;
 };
"""
        + region
        + r"""
}
int main(int argc,char** argv) {
 try {
  if(argc!=6) return 2;
  unsigned budget=std::stoul(argv[1]); bool incremental=std::stoi(argv[2]);
  int scenario=std::stoi(argv[3]); unsigned expected_iterations=std::stoul(argv[4]);
  unsigned expected_calls=std::stoul(argv[5]);
  for (double tolerance : {1e-12,1e-10,1e-7}) {
   const double gate=std::min(1e-8,tolerance);
   Matrix sparse(4096); sparse[1]=gate; sparse[64]=-gate;
   if(residual_max_abs(sparse)!=gate || residual_rms(sparse)>=tolerance)
    throw std::runtime_error("norm/boundary mismatch");
   sparse[1]=std::nextafter(gate,1.0);
   if(residual_max_abs(sparse)<=gate) throw std::runtime_error("above boundary accepted");
  }
  for(double x:{std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity()})
   if(std::isfinite(residual_max_abs(Matrix{x}))) throw std::runtime_error("nonfinite norm accepted");
  if(residual_max_abs(Matrix{})!=0 || residual_max_abs(Matrix{-0.0,0.0})!=0)
   throw std::runtime_error("zero norm changed");
  unsigned clears=0;
  Result r;
  try { r=run(budget,incremental,scenario,clears); }
  catch(const std::runtime_error& e) {
   if((scenario==3 || scenario==4) && std::string(e.what())=="injected physical provider failure" &&
      calls==expected_calls && recorded_builds==calls && retain_calls==0) return 0;
   throw;
  }
  bool success=scenario!=2 && scenario!=5 && scenario!=8 &&
      (scenario==0 || scenario==6 || scenario==9 || budget>=4);
  if(r.converged!=success) throw std::runtime_error("wrong convergence status");
  if(r.iterations!=expected_iterations || calls!=expected_calls || r.fock_builds!=calls)
   throw std::runtime_error("uncounted or unexpected iterations/builds");
  if(r.ks.history.size()!=r.iterations || r.iterations>(incremental?2:1)*budget)
   throw std::runtime_error("history/budget mismatch");
  for(unsigned i=0;i<r.ks.history.size();++i)
   if(r.ks.history[i].iteration!=i+1) throw std::runtime_error("history restarted");
  if(r.density!=last_density || r.factor->density!=r.density || r.energy!=r.density[0])
   throw std::runtime_error("returned physical D/E/factor ownership mismatch");
  if(r.ks.components.value!=r.energy) throw std::runtime_error("not true KS energy");
  if(r.ks_physical_fock.empty()==success) throw std::runtime_error("bad Fock lease");
  if(success && (r.ks_physical_fock[0]!=calls || (scenario!=9 && std::abs(r.ks_physical_fock[1])>1e-10)))
   throw std::runtime_error("accepted RMS-small maximum-large residual");
  if(std::abs(r.physical_residual_rms-std::sqrt(2.0/16)*last_amplitude)>1e-24)
   throw std::runtime_error("RMS telemetry was changed into maximum");
  if(retain_calls!=1) throw std::runtime_error("factor retained before terminal return");
  if(eigen_calls!=proposals || proposals!=r.iterations+(incremental?0:(calls-r.iterations)/2))
   throw std::runtime_error("unexpected eigensolve/projection calls");
  for(unsigned i=1;i<r.ks.history.size();++i) {
   const bool restart=(incremental && i==2) ||
       (incremental && i==4) || (!incremental && i==2 && scenario!=5);
   if(restart && r.ks.history[i].energy_change!=-1.0)
    throw std::runtime_error("restarted history has no explicit unavailable baseline");
   if(!restart && !std::isfinite(r.ks.history[i].energy_change))
    throw std::runtime_error("unexpected nonfinite measured history delta");
  }
  const unsigned audits=incremental?calls-r.iterations:(calls-r.iterations)/2;
  const unsigned failures=audits-(success?1:0);
  const unsigned expected_clears=incremental?1+failures:(audits?audits-(scenario==8?0:1):0);
  if(clears!=expected_clears || observed_progress!=(incremental?2:0) ||
     strict_entries!=(incremental?1:0) ||
     full_calls!=(incremental?r.iterations-2+audits:2*audits))
   throw std::runtime_error("DIIS reset, progress observation or strict call count changed");
  if(incremental && (r.ks.incremental_xc.strict_refinement_iterations!=r.iterations-2 ||
     r.ks.incremental_xc.final_audits!=audits || r.ks.incremental_xc.audit_failures!=failures))
   throw std::runtime_error("incremental strict audit counters changed");
  std::cout<<r.converged<<" "<<r.iterations<<" "<<calls<<" "<<clears<<"\n";
 } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
"""
    )
    path, executable = directory / "probe.cpp", directory / "probe"
    path.write_text(program)
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O0",
            "-I",
            str(ROOT / "src"),
            str(path),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    return executable


@pytest.mark.parametrize(
    "budget,incremental,scenario,iterations,calls",
    [
        (8, False, 0, 2, 4),
        (8, False, 3, 2, 3),
        (8, False, 4, 2, 4),
        (3, False, 5, 3, 3),
        (5, False, 8, 5, 7),
        (8, False, 9, 2, 4),
        (8, False, 6, 2, 4),
        (8, False, 7, 4, 8),
        (8, False, 1, 4, 8),
        (2, False, 1, 2, 4),
        (3, False, 1, 2, 4),
        (4, False, 2, 4, 8),
        (8, True, 0, 4, 5),
        (8, True, 6, 4, 5),
        (8, True, 7, 6, 8),
        (8, True, 1, 6, 8),
        (2, True, 1, 4, 5),
        (4, True, 2, 6, 8),
    ],
)
def test_rks_final_maximum_closure(
    closure_probe: Path,
    budget: int,
    incremental: bool,
    scenario: int,
    iterations: int,
    calls: int,
) -> None:
    completed = subprocess.run(
        [
            str(closure_probe),
            str(budget),
            str(int(incremental)),
            str(scenario),
            str(iterations),
            str(calls),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr

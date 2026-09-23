"""Actual native response owners reject budgets before numeric allocation."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CPP = r"""
#include "cc/lambda_response.hpp"
#include "cc/triples_response.hpp"
#include <cstdlib>
#include <new>
#include <limits>
#include <iostream>
static bool tracking=false;
static std::size_t allocations=0;
void* operator new(std::size_t n){
  if(tracking && n>=128) ++allocations;
  if(void* p=std::malloc(n?n:1))return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
int main(){
 using namespace vibeqc::cc;
 Problem p;p.nocc=2;p.nvir=3;p.reference_energy=0.;
 p.foo.assign(4,0.);p.fov.assign(6,0.);p.fvv.assign(9,0.);
 p.ovov.assign(36,0.);p.ovvo.assign(36,0.);p.oovv.assign(36,0.);
 p.ovvv.assign(54,0.);p.ovoo.assign(24,0.);p.oooo.assign(16,0.);p.vvvv.assign(81,0.);
 p.d1.assign(6,-1.);p.d2.assign(36,-2.);p.initial_t1.assign(6,0.);p.initial_t2.assign(36,0.);
 SolverResult cc;cc.status=SolveStatus::Converged;cc.correlation_energy=0.;cc.t1.assign(6,0.);cc.t2.assign(36,0.);
 LambdaOptions lo;lo.max_bytes=1;
 bool rejected=false;tracking=true;
 try{(void)solve_lambda_cpu(p,cc,lo);}catch(const std::length_error&){rejected=true;}
 tracking=false;if(!rejected || allocations){std::cerr<<"Lambda allocated before budget: "<<allocations;return 1;}
 // Corrected Lambda retains the projected energy RHS during both residual
 // evaluations. Its exact admission includes that extra live vector.
 lo.max_bytes=64ULL<<20;
 const auto corrected_capacity=lambda_cpu_numeric_capacity(p,cc,lo,true);
 lo.max_bytes=corrected_capacity-1;allocations=0;rejected=false;tracking=true;
 try{(void)solve_lambda_cpu_with_energy_source(p,cc,cc.t1,cc.t2,lo);}
 catch(const std::length_error&){rejected=true;}
 tracking=false;if(!rejected || allocations)return 5;
 lo.max_bytes=corrected_capacity;
 const auto corrected=solve_lambda_cpu_with_energy_source(p,cc,cc.t1,cc.t2,lo);
 if(corrected.diagnostic.numeric_capacity_bytes!=corrected_capacity)return 6;
 std::vector<double> eo(2,-1.),ev(3,1.);TriplesResponseOptions to;to.max_bytes=1;
 allocations=0;rejected=false;tracking=true;
 try{(void)triples_response_cpu(p,cc,eo,ev,to);}catch(const std::length_error&){rejected=true;}
 tracking=false;if(!rejected || allocations){std::cerr<<"triples allocated before budget: "<<allocations;return 2;}
 to.max_bytes=64ULL<<20;cc.t1[0]=std::numeric_limits<double>::quiet_NaN();rejected=false;
 try{(void)triples_response_cpu(p,cc,eo,ev,to);}catch(const std::invalid_argument&){rejected=true;}
 if(!rejected)return 3;
 cc.t1.assign(6,0.1);cc.t2.assign(36,1e200);p.ovvv.assign(54,1e200);rejected=false;
 try{(void)triples_response_cpu(p,cc,eo,ev,to);}catch(const std::runtime_error&){rejected=true;}
 if(!rejected)return 4;

}
"""


def test_response_budget_precedes_numeric_allocation(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tools/generate_rccsd_native.py"),
            "--cpu-header",
            str(tmp_path / "generated_rccsd_cpu.hpp"),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=90,
    )
    source = tmp_path / "admission.cpp"
    source.write_text(CPP)
    binary = tmp_path / "admission"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-DVIBEQC_HAS_CUDA=0",
            "-I" + str(root / "src"),
            "-I" + str(tmp_path),
            str(root / "src/cc/solver.cpp"),
            str(root / "src/cc/lambda_response.cpp"),
            str(root / "src/cc/triples_response.cpp"),
            str(root / "src/response/native_gmres.cpp"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)

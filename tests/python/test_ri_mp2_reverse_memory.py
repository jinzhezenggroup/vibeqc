"""Trace the actual RI reverse/metric kernels without a molecular SCF build.

Only the value provider and the BLAS dispatch are fixtures. The reverse, metric
response, spectral VJP, scalar GEMM and Jacobi bodies come from current sources.
This is a scalar-host allocation contract, not a GPU or full force qualification.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    body = source.index("{", start)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def _without_includes(path: str) -> str:
    return re.sub(
        r"^#include[^\n]*\n", "", (ROOT / path).read_text(), flags=re.MULTILINE
    )


PREFIX = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
namespace trace {
struct alignas(std::max_align_t) Header { std::size_t bytes, epoch; };
std::size_t epoch=0, live=0, peak=0, gemms=0, eigensolves=0;
bool active=false;
void start() { ++epoch; live=peak=gemms=eigensolves=0; active=true; }
}
void* operator new(std::size_t bytes) {
  if(bytes > std::numeric_limits<std::size_t>::max()-sizeof(trace::Header))
    throw std::bad_alloc();
  auto* h=static_cast<trace::Header*>(std::malloc(sizeof(trace::Header)+bytes));
  if(!h) throw std::bad_alloc();
  *h={bytes, trace::active ? trace::epoch : 0};
  if(h->epoch) { trace::live+=bytes; trace::peak=std::max(trace::peak,trace::live); }
  return h+1;
}
void operator delete(void* p) noexcept {
  if(!p) return;
  auto* h=static_cast<trace::Header*>(p)-1;
  if(h->epoch && h->epoch==trace::epoch) trace::live-=h->bytes;
  std::free(h);
}
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }
namespace vibeqc::tensor {
enum class SymmetricMatrixFunction { pseudoinverse, inverse_sqrt };
enum class CpuLinalgProvider { automatic, scalar };
enum class CpuLinalgThreadOwnership { task_parallel, provider_parallel };
struct CpuLinalgPlan {
  CpuLinalgProvider provider;
  CpuLinalgThreadOwnership thread_ownership;
  int provider_threads;
};
struct CpuSymmetricEigenResult { std::vector<double> values, vectors; };
"""

DISPATCH = r"""
CpuSymmetricEigenResult cpu_symmetric_eigen(std::vector<double> m, std::size_t n,
                                           const CpuLinalgPlan&) {
  ++trace::eigensolves;
  return scalar_symmetric_eigen(std::move(m),n);
}
void cpu_gemm(char ta,char tb,std::size_t m,std::size_t n,std::size_t k,
              const double* a,const double* b,double* c,double alpha,double beta,
              const CpuLinalgPlan&) {
  ++trace::gemms;
  scalar_gemm(ta=='T',tb=='T',m,n,k,a,b,c,alpha,beta);
}
} // namespace vibeqc::tensor
namespace vibeqc::integrals {
struct DensityFittingMetricFactor {
  std::size_t dimension{}, effective_rank{};
  double absolute_threshold{}, condition_number{};
  std::vector<double> inverse_square_root;
};
}
"""

PROVIDER = r"""
namespace vibeqc::scf {
struct PhysicalReference {
  std::size_t nbf, nocc;
  std::vector<double> coefficients;
};
}
namespace vibeqc::posthf {
std::size_t checked_mul(std::size_t a,std::size_t b) {
  if(a && b>std::numeric_limits<std::size_t>::max()/a) throw std::overflow_error("overflow");
  return a*b;
}
std::size_t checked_add(std::size_t a,std::size_t b) {
  if(b>std::numeric_limits<std::size_t>::max()-a) throw std::overflow_error("overflow");
  return a+b;
}
struct DensityFittedBlockProvider {
  const scf::PhysicalReference& ref;
  std::size_t na;
  std::vector<double> m,x,a,b;
  DensityFittedBlockProvider(const scf::PhysicalReference& r,std::size_t size)
      :ref(r),na(size),m(na*na),x(na*na),a(r.nbf*r.nbf*na),b(a.size()) {
    for(std::size_t i=0;i<na;++i) {
      m[i*na+i]=1.0+0.01*i;
      x[i*na+i]=1.0/std::sqrt(m[i*na+i]);
    }
    for(std::size_t i=0;i<a.size();++i) {
      a[i]=0.01*(1+int(i%17)); b[i]=a[i]*x[(i%na)*(na+1)];
    }
  }
  const auto& reference() const { return ref; }
  std::size_t auxiliary_count() const { return na; }
  const auto& metric() const { return m; }
  const auto& inverse_square_root() const { return x; }
  const auto& transformed_three_center() const { return a; }
  const auto& whitened_three_center() const { return b; }
  double relative_threshold() const { return 1e-10; }
  std::size_t provider_bytes() const {
    return sizeof(double)*(ref.coefficients.size()+m.size()+x.size()+a.size()+b.size());
  }
};
}
namespace vibeqc::mp2 {
std::size_t square(std::size_t n) { return posthf::checked_mul(n,n); }
std::size_t fourth_power(std::size_t n) { return square(square(n)); }
std::size_t eri_index(std::size_t n,std::size_t p,std::size_t q,std::size_t r,std::size_t s) {
  return ((p*n+q)*n+r)*n+s;
}
bool finite(std::span<const double> x) {
  return std::all_of(x.begin(),x.end(),[](double v){return std::isfinite(v);});
}
struct LagrangianWeights {
  std::size_t orbitals{},occupied{};
  std::vector<double> one_electron,overlap,two_electron;
};
struct DensityFittedLagrangianWeights {
  std::size_t orbitals{},auxiliary{};
  std::vector<double> overlap,one_electron,three_center,metric;
  std::size_t workspace_bytes{},planned_peak_bytes{};
};
"""

MAIN = r"""
} // namespace vibeqc::mp2
int main(int argc,char** argv) {
  if(argc!=3) return 2;
  using namespace vibeqc;
  const std::size_t n=std::strtoul(argv[1],nullptr,10),na=std::strtoul(argv[2],nullptr,10);
  scf::PhysicalReference ref{n,1,std::vector<double>(n*n)};
  for(std::size_t i=0;i<n;++i) ref.coefficients[i*n+i]=1.0;
  posthf::DensityFittedBlockProvider provider(ref,na);
  mp2::LagrangianWeights weights{n,1,std::vector<double>(n*n,0.2),
                                std::vector<double>(n*n,0.1),std::vector<double>(n*n*n*n)};
  for(std::size_t i=0;i<weights.two_electron.size();++i)
    weights.two_electron[i]=0.001*(int(i%11)-5);
  const std::size_t retained=provider.provider_bytes()+sizeof(double)*(2*n*n+n*n*n*n);
  std::size_t planned=0;
  for(unsigned repeat=0;repeat<2;++repeat) {
    trace::start();
    auto result=mp2::density_fitted_lagrangian_weights(
        ref,provider,weights,repeat ? planned : std::numeric_limits<std::size_t>::max());
    trace::active=false;
    planned=result.planned_peak_bytes;
    std::cout << n << ' ' << na << " measured=" << retained+trace::peak
              << " planned=" << planned << '\n';
    if(trace::eigensolves!=1 || trace::gemms!=4) return 3;
    if(retained+trace::peak>planned) return 4;
    const auto result_bytes=sizeof(double)*(2*n*n+n*n*na+na*na);
    if(planned!=retained+result_bytes+result.workspace_bytes) return 5;
  }
  trace::start();
  bool rejected=false;
  try { (void)mp2::density_fitted_lagrangian_weights(ref,provider,weights,planned-1); }
  catch(const std::length_error&) { rejected=true; }
  trace::active=false;
  if(!rejected || trace::eigensolves || trace::gemms) return 6;
}
"""


@pytest.fixture(scope="module")
def reverse_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    linalg = (ROOT / "src/tensor/cpu_linalg.cpp").read_text()
    reverse = (ROOT / "src/posthf/mp2_gradient.cpp").read_text()
    source = "\n".join(
        (
            PREFIX,
            _function(linalg, "void scalar_gemm("),
            _function(linalg, "CpuSymmetricEigenResult scalar_symmetric_eigen("),
            DISPATCH,
            _without_includes("src/tensor/symmetric_matrix_function.cpp"),
            _without_includes("src/integrals/density_fitting_metric.cpp"),
            PROVIDER,
            _function(
                reverse,
                "DensityFittedLagrangianWeights density_fitted_lagrangian_weights(",
            ),
            MAIN,
        )
    )
    directory = tmp_path_factory.mktemp("ri-reverse-memory")
    path, executable = directory / "probe.cpp", directory / "probe"
    path.write_text(source)
    process = subprocess.run(
        [compiler, "-std=c++20", "-O0", str(path), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return executable


@pytest.mark.parametrize("n,na", ((2, 1), (2, 2), (2, 4), (2, 12), (3, 64), (4, 128)))
def test_reverse_budget_covers_measured_nested_peak(
    reverse_probe: Path, n: int, na: int
) -> None:
    process = subprocess.run(
        [str(reverse_probe), str(n), str(na)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr

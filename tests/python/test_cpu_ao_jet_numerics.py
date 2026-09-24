"""Execute the production AO evaluator against an independent Leibniz oracle.

The harness supplies packed fixtures directly: it tests evaluation, not basis
construction, the native library ABI, or a complete DFT endpoint.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ao_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler")
    source = (ROOT / "src/dft/ao_grid.cpp").read_text(encoding="utf-8")
    helpers = source[source.index("namespace {") : source.index("AoBasis::AoBasis")]
    evaluate = source[source.index("void AoBasis::evaluate(") :]
    # Isolate the exact evaluator and its polynomial helper. The class fields
    # match AoBasis; the packed constructor deliberately is not under test.
    harness = (
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>
namespace molecule {
using Component = std::array<unsigned,3>;
std::vector<Component> cartesian_components(unsigned degree) {
  std::vector<Component> out;
  for (int x=degree; x>=0; --x)
    for (int y=degree-x; y>=0; --y)
      out.push_back({unsigned(x),unsigned(y),degree-unsigned(x+y)});
  return out;
}
}
namespace vibeqc::dft {
struct AoBasis {
  std::size_t natom{}, nprimitive{}, nao{};
  std::vector<double> packed;
  void evaluate(const double*,std::size_t,unsigned,std::size_t,std::size_t,
                double*,std::size_t,const std::size_t*) const;
};
"""
        + helpers
        + evaluate
        + r"""
using Component = molecule::Component;
long double axis(unsigned l,unsigned d,long double alpha,long double x) {
  // Direct Leibniz rule with explicit Gaussian derivatives through order 3;
  // independent of the production repeated polynomial differentiation.
  const long double h[] = {1, -2*alpha*x, 4*alpha*alpha*x*x-2*alpha,
                          -8*alpha*alpha*alpha*x*x*x+12*alpha*alpha*x};
  const unsigned choose[4][4]={{1,0,0,0},{1,1,0,0},{1,2,1,0},{1,3,3,1}};
  long double total=0;
  for(unsigned k=0;k<=std::min(l,d);++k) {
    long double falling=1;
    for(unsigned i=0;i<k;++i) falling*=l-i;
    total+=choose[d][k]*falling*std::pow(x,l-k)*h[d-k];
  }
  return total;
}
vibeqc::dft::AoBasis fixture(bool expanded) {
  vibeqc::dft::AoBasis b;
  b.natom=2; b.nprimitive=6;
  b.packed={0,0,0, 0.3,-0.2,0.1, 1.7,0.4, 0.51,-0.17, 0.13,0.8,
                                      1.7,0.4, 0.51,-0.17, 0.13,0.8};
  for(unsigned atom=0;atom<2;++atom) {
    for(unsigned l=0;l<=3;++l) {
      for(auto powers:molecule::cartesian_components(l)) {
        std::array<double,16> r{};
        r[0]=atom;r[1]=3*atom;r[2]=3;r[3]=1;
        for(unsigned k=0;k<3;++k) r[4+k]=powers[k];
        r[7]=1;
        b.packed.insert(b.packed.end(),r.begin(),r.end());++b.nao;
      }
    }
    if(expanded) {
      // Real d_z2 and f_z3 combinations with signed component weights.
      for(unsigned z=0;z<=1;++z) {
        std::array<double,16> r{};
        r[0]=atom;r[1]=3*atom;r[2]=3;r[3]=3;
        r[4]=2;r[6]=z;r[7]=-0.5;
        r[9]=2;r[10]=z;r[11]=-0.5;
        r[14]=2+z;r[15]=1;
        b.packed.insert(b.packed.end(),r.begin(),r.end());++b.nao;
      }
    }
  }
  return b;
}
int main(int argc,char** argv) {
  assert(argc==4);
  const unsigned order=std::atoi(argv[1]);
  const bool expanded=std::atoi(argv[2]);
  const unsigned mode=std::atoi(argv[3]);
  auto b=fixture(expanded);
  std::vector<double> points={0,0,0, .1,-.2,.3, 1.1,.7,-.9, -2.1,.5,1.7};
  std::vector<std::size_t> ids;
  if(mode==1) ids={0,3,8,b.nao-1};
  else for(std::size_t i=0;i<b.nao;++i) ids.push_back(i);
  const auto* selected=mode==1 ? ids.data() : nullptr;
  if(mode==2) points.assign(3,1e150);
  if(mode==3) points.clear();
  if(mode==4) points[0]=std::numeric_limits<double>::quiet_NaN();
  if(mode==5) { ids={1,1}; selected=ids.data(); }
  const std::size_t npoint=points.size()/3, jets=(order+1)*(order+2)*(order+3)/6;
  std::vector<double> guarded(jets*npoint*ids.size()+2,12345.0);
  bool rejected=false;
  try {
    b.evaluate(points.data(),npoint,order,0,ids.size(),guarded.data()+1,
               guarded.size()-2,selected);
  } catch(const std::invalid_argument&) {rejected=true;}
  assert(rejected==(mode>=4));
  assert(guarded.front()==12345.0 && guarded.back()==12345.0);
  if(rejected || mode==3) return 0;
  if(mode==2) {
    assert(std::all_of(guarded.begin()+1,guarded.end()-1,[](double x){return x==0;}));
    return 0;
  }
  std::size_t jet=0;
  for(unsigned degree=0;degree<=order;++degree)
    for(Component d:molecule::cartesian_components(degree)) {
      for(std::size_t point=0;point<npoint;++point)
        for(std::size_t ao=0;ao<ids.size();++ao) {
          const auto* r=b.packed.data()+3*b.natom+2*b.nprimitive+16*ids[ao];
          long double xyz[3],r2=0;
          for(unsigned k=0;k<3;++k) {
            xyz[k]=static_cast<long double>(points[3*point+k])-b.packed[3*std::size_t(r[0])+k];
            r2+=xyz[k]*xyz[k];
          }
          long double expected=0;
          for(std::size_t p=r[1];p<r[1]+r[2];++p) {
            const auto* primitive=b.packed.data()+3*b.natom+2*p;
            const long double alpha=primitive[0];
            for(unsigned t=0;t<unsigned(r[3]);++t) {
              long double value=primitive[1]*std::exp(-alpha*r2)*r[7+4*t];
              for(unsigned k=0;k<3;++k) value*=axis(unsigned(r[4+4*t+k]),d[k],alpha,xyz[k]);
              expected+=value;
            }
          }
          const auto actual=guarded[1+(jet*npoint+point)*ids.size()+ao];
          assert(std::isfinite(actual));
          assert(std::abs(actual-expected)<=1e-12L+5e-13L*std::abs(expected));
        }
      ++jet;
    }
  return 0;
}
"""
    )
    directory = tmp_path_factory.mktemp("ao-jets")
    cpp, executable = directory / "probe.cpp", directory / "probe"
    cpp.write_text(harness, encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O3",
            "-ffp-contract=off",
            str(cpp),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("order", range(4))
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("mode", [0, 1])
def test_ao_jets_match_independent_leibniz(
    ao_probe: Path, order: int, expanded: bool, mode: int
) -> None:
    subprocess.run(
        [str(ao_probe), str(order), str(int(expanded)), str(mode)],
        check=True,
        timeout=10,
    )


@pytest.mark.parametrize("mode", [2, 3, 4, 5])
def test_underflow_empty_and_invalid_inputs(ao_probe: Path, mode: int) -> None:
    subprocess.run([str(ao_probe), "3", "1", str(mode)], check=True, timeout=10)

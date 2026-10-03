"""Host numerical probes for the shared all-center full/LR primitive algebra.

A displaced-value Hermite contraction is a different recurrence from the emitted
Wick gradient. Native molecular tests additionally use independent CPU ERIs and
exercise shell canonicalization, normalization, spin weights and repeated atoms.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
    emit_direct_cartesian_contraction_headers,
)
from generativeqc_compiler.integral.direct_pair_gradient_cuda import (
    emit_direct_high_order_pair_gradient_header,
)
from generativeqc_compiler.integral.direct_pair_support_cuda import (
    emit_direct_pair_support_headers,
)
from generativeqc_compiler.integral.direct_recurrence_cuda import (
    emit_direct_recurrence_headers,
)

ROOT = Path(__file__).resolve().parents[2]


def test_full_and_lr_all_center_gradients(tmp_path: Path) -> None:
    """Check every center/axis, all pair partitions of orders 4--6 and LR limits."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host gradient probes require c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    headers = {
        **emit_direct_pair_support_headers(),
        **emit_direct_recurrence_headers(),
        **emit_direct_cartesian_contraction_headers(),
        "gradient.cuh": emit_direct_high_order_pair_gradient_header(),
    }
    for name, source in headers.items():
        (tmp_path / name).write_text(source)
    (tmp_path / "cuda_runtime.h").write_text(
        "#pragma once\n#include <cmath>\n#include <algorithm>\n"
        "#define __device__\n#define __host__\n#define __forceinline__ inline\n"
        "#define __noinline__\n"
        "using std::min; using std::max;\n"
        "inline unsigned __popc(unsigned x) { return __builtin_popcount(x); }\n"
    )
    source = tmp_path / "probe.cpp"
    source.write_text(PROBE)
    executable = tmp_path / "probe"
    build = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            str(source),
            "-o",
            str(executable),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "all-center finite differences PASS" in result.stdout
    print(result.stdout)


PROBE = r"""
#include "gradient.cuh"
#include "generated_direct_shell_class.cuh"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
using namespace generativeqc::scf::cuda_execution;
using generativeqc::integrals::CoulombRange;
unsigned checks = 0;
double maximum_error = 0.0;

template<unsigned A, unsigned B, unsigned C, unsigned D>
void check_partition() {
  constexpr std::array<unsigned,4> order{A,B,C,D};
  for (unsigned orientation = 0; orientation < 9; ++orientation) {
    Angular angular[4]{};
    for (unsigned center = 0; center < 4; ++center)
      for (unsigned quantum = 0; quantum < order[center]; ++quantum)
        add_angular_axis(angular[center], (center + quantum + orientation * (quantum+1))%3, 1);
    for (double scale : {0.4, 1.0, 3.0}) {
      std::array<Vec3<double>,4> positions{{{0.1,-0.2,-0.8},{0.3,0.1,0.7},
                                          {-0.5,0.6,0.2},{0.8,-0.4,0.3}}};
      const double exponent[4]{0.8*scale,0.6*scale,0.7*scale,0.9*scale};
      for (double omega : {0.0,1e-8,0.3,2.0,1e4}) {
        double gradient[4][3];
        primitive_eri_order456_gradient<A+B,C+D,true>(
            exponent[0],positions[0],angular[0],exponent[1],positions[1],angular[1],
            exponent[2],positions[2],angular[2],exponent[3],positions[3],angular[3],
            gradient,omega);
        for (unsigned center = 0; center < 4; ++center) {
          for (unsigned axis = 0; axis < 3; ++axis) {
            auto value = [&](double displacement) {
              auto moved = positions;
              double* coordinate = axis==0 ? &moved[center].x :
                                   axis==1 ? &moved[center].y : &moved[center].z;
              *coordinate += displacement;
              return primitive_eri_cartesian_shell_pairs<A,B,C,D,double>(
                  exponent[0],moved[0],angular[0],exponent[1],moved[1],angular[1],
                  exponent[2],moved[2],angular[2],exponent[3],moved[3],angular[3],
                  CoulombRange::Long,omega);
            };
            constexpr double h=2e-4;
            const double expected=(value(-2*h)-8*value(-h)+8*value(h)-value(2*h))/(12*h);
            const double error=std::abs(expected-gradient[center][axis]);
            if (!std::isfinite(gradient[center][axis]) || error > 2e-8*(1+std::abs(expected))) {
              std::fprintf(stderr,"partition %u%u%u%u omega %.8g coordinate %u/%u: %.17g != %.17g\n",
                           A,B,C,D,omega,center,axis,gradient[center][axis],expected);
              std::exit(1);
            }
            maximum_error=std::max(maximum_error,error);++checks;
          }
        }
        for (unsigned axis=0;axis<3;++axis) {
          const double sum=gradient[0][axis]+gradient[1][axis]+gradient[2][axis]+gradient[3][axis];
          if (std::abs(sum)>1e-12) std::exit(2);
        }
      }
      // The default specialization must preserve the full-range primitive;
      // use a separate Hermite value recurrence for this finite-difference gate.
      double full[4][3];
      primitive_eri_order456_gradient<A+B,C+D>(
          exponent[0],positions[0],angular[0],exponent[1],positions[1],angular[1],
          exponent[2],positions[2],angular[2],exponent[3],positions[3],angular[3],full);
      for (unsigned center=0;center<4;++center) for (unsigned axis=0;axis<3;++axis) {
        auto value=[&](double displacement) {
          auto moved=positions;
          double* coordinate=axis==0?&moved[center].x:axis==1?&moved[center].y:&moved[center].z;
          *coordinate+=displacement;
          return primitive_eri_cartesian_shell_pairs<A,B,C,D,double>(
              exponent[0],moved[0],angular[0],exponent[1],moved[1],angular[1],
              exponent[2],moved[2],angular[2],exponent[3],moved[3],angular[3]);
        };
        constexpr double h=2e-4;
        const double expected=(value(-2*h)-8*value(-h)+8*value(h)-value(2*h))/(12*h);
        if (!std::isfinite(full[center][axis]) || std::abs(full[center][axis]-expected)>2e-8*(1+std::abs(expected)))
          std::exit(3);
        ++checks;
      }
      for (double invalid : {-0.3,std::numeric_limits<double>::quiet_NaN()}) {
        primitive_eri_order456_gradient<A+B,C+D,true>(
            exponent[0],positions[0],angular[0],exponent[1],positions[1],angular[1],
            exponent[2],positions[2],angular[2],exponent[3],positions[3],angular[3],full,invalid);
        for (auto& row:full) for(double v:row) if(std::isfinite(v)) std::exit(4);
      }
    }
  }
}
int main() {
  check_partition<2,2,0,0>();check_partition<2,1,1,0>();check_partition<1,1,1,1>();
  check_partition<3,2,0,0>();check_partition<2,2,1,0>();check_partition<2,1,1,1>();
  check_partition<3,3,0,0>();check_partition<3,2,1,0>();check_partition<2,2,1,1>();
  check_partition<2,1,2,1>();
  std::printf("%u all-center finite differences PASS; max LR absolute error %.5g\n",checks,maximum_error);
}
"""

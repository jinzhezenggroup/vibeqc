"""Execute emitted Fock geometry and the actual Boys table without CUDA runtime."""

import ctypes
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.integral import (
    FUSED_SHELL_SPEC_BY_NAME,
    KernelConsumer,
    build_fused_shell_plan,
    cuda_target_info,
    emit_shell_class_fused_cuda,
)

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, marker: str, *, structure: bool = False) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start : end + int(structure)]


@pytest.mark.parametrize("name", ("ssss", "psss", "dppp", "dddd"))
def test_value_geometry_retains_all_live_state_and_prunes_only_dead_state(
    tmp_path: Path, name: str
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        target=cuda_target_info("sm_120"),
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    tag = "Generated" + name.capitalize()
    stem = "generated_" + name
    maximum = plan.kernel.integral.value_coulomb_order
    force_maximum = spec.maximum_force_coulomb_order
    declarations = "\n".join(
        _definition(source, "struct " + tag + suffix, structure=True)
        for suffix in ("Vec3", "PrimitivePairData", "PrimitiveGeometry")
    )
    functions = "\n".join(
        _definition(source, "__device__ __forceinline__ " + return_type + stem + suffix)
        for return_type, suffix in (
            ("double ", "_axis("),
            ("void ", "_make_primitive_geometry("),
            ("void ", "_make_fock_primitive_geometry("),
        )
    )
    boys = (ROOT / "src/scf/cuda/boys_table.cuh").read_text()
    boys = re.sub(r"^#(?:include|pragma).*\n", "", boys, flags=re.MULTILINE)
    prefix = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#define __device__
#define __forceinline__ inline
struct MixedPrecisionFloat { double value; };
constexpr unsigned kMaximumCoulombOrder=32;
constexpr double kPi=3.141592653589793238462643383279502884;
template<class T> T scalar(double x) { return T(x); }
inline double scalar_value(double x) { return x; }
inline double qexp(double x) { return std::exp(x); }
inline double qsqrt(double x) { return std::sqrt(x); }
inline double qerf(double x) { return std::erf(x); }
"""
    body = r"""
extern "C" unsigned run(const double* input, double* output, unsigned mode) {
  using G=TAGPrimitiveGeometry; using V=TAGVec3; using P=TAGPrimitivePairData;
  V points[4];
  for(unsigned i=0;i<4;++i) points[i]={input[4+3*i],input[5+3*i],input[6+3*i]};
  const auto pair=[](double a,double b,V A,V B,double coefficient) {
    const double p=a+b, mu=a*b/p;
    const double distance=(A.x-B.x)*(A.x-B.x)+(A.y-B.y)*(A.y-B.y)+(A.z-B.z)*(A.z-B.z);
    return P{p,mu,{(a*A.x+b*B.x)/p,(a*A.y+b*B.y)/p,(a*A.z+b*B.z)/p},
             coefficient*std::exp(-mu*distance),a/p,b/p};
  };
  const P first=pair(input[0],input[1],points[0],points[1],input[16]);
  const P second=pair(input[2],input[3],points[2],points[3],input[17]);
  G value{}, full{};
  const double nan=std::numeric_limits<double>::quiet_NaN();
  std::fill_n(value.product_scales,3,nan);
  for(auto& axis:value.decay_gradients) std::fill_n(axis,3,nan);
  std::fill_n(value.boys,FORCE_MAX+1,nan);
  for(auto& axis:value.coordinate_powers) std::fill_n(axis,FORCE_MAX+1,nan);
  std::fill_n(value.negative_two_rho_powers,FORCE_MAX+1,nan);
  STEM_make_primitive_geometry(first,second,mode&1,mode&2,points[0],points[1],points[2],points[3],full);
  STEM_make_fock_primitive_geometry(first,second,mode&1,mode&2,points[0],points[1],points[2],points[3],value);
  unsigned offset=0;
  for(const G* g:{&value,&full}) {
    output[offset++]=g->inverse_two_p;output[offset++]=g->inverse_two_q;output[offset++]=g->rho;
    output[offset++]=g->prefactor;output[offset++]=g->primitive_coefficient;
    for(const auto& center:g->pair_shifts) for(double x:center) output[offset++]=x;
    for(double x:g->difference) output[offset++]=x;
    for(unsigned k=0;k<=VALUE_MAX;++k) output[offset++]=g->boys[k];
    for(const auto& axis:g->coordinate_powers) for(unsigned k=0;k<=VALUE_MAX;++k) output[offset++]=axis[k];
    for(unsigned k=0;k<=VALUE_MAX;++k) output[offset++]=g->negative_two_rho_powers[k];
  }
  for(double x:value.product_scales) if(!std::isnan(x)) return 1;
  for(const auto& axis:value.decay_gradients) for(double x:axis) if(!std::isnan(x)) return 2;
  for(unsigned k=VALUE_MAX+1;k<=FORCE_MAX;++k) {
    if(!std::isnan(value.boys[k]) || !std::isnan(value.negative_two_rho_powers[k])) return 3;
    for(const auto& axis:value.coordinate_powers) if(!std::isnan(axis[k])) return 4;
  }
  return 0;
}
"""
    for old, new in (
        ("TAG", tag),
        ("STEM", stem),
        ("FORCE_MAX", str(force_maximum)),
        ("VALUE_MAX", str(maximum)),
    ):
        body = body.replace(old, new)
    cpp, library = tmp_path / "geometry.cpp", tmp_path / "geometry.so"
    cpp.write_text(
        prefix
        + boys
        + "\nusing vibeqc::scf::cuda_execution::boys_values;\n"
        + declarations
        + functions
        + body
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(cpp),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    native = ctypes.CDLL(str(library)).run
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    native.argtypes = [pointer, pointer, ctypes.c_uint]
    native.restype = ctypes.c_uint
    width = 20 + 5 * (maximum + 1)
    nodes, weights = np.polynomial.legendre.leggauss(256)
    quadrature_nodes = (nodes + 1.0) / 2.0
    quadrature_weights = weights / 2.0
    rng = np.random.default_rng(806)
    for scale in (0.0, 0.01, 0.5, 2.0, 5.0):
        for _ in range(8):
            data = np.concatenate(
                (
                    np.exp(rng.uniform(-2, 2, 4)),
                    rng.normal(size=12) * scale,
                    rng.uniform(-1, 1, 2),
                )
            )
            output = np.empty((2, width), dtype=np.float64)
            for mode in range(4):
                assert native(data, output, mode) == 0
                assert np.isfinite(output).all()
                np.testing.assert_allclose(output[0], output[1], atol=2e-14, rtol=3e-13)
                argument = output[0, 2] * np.sum(output[0, 17:20] ** 2)
                # Independent quadrature of integral_0^1 t^(2m) exp(-T*t*t) dt.
                independent = np.array(
                    [
                        np.dot(
                            quadrature_weights,
                            quadrature_nodes ** (2 * order)
                            * np.exp(-argument * quadrature_nodes**2),
                        )
                        for order in range(maximum + 1)
                    ]
                )
                np.testing.assert_allclose(
                    output[0, 20 : 21 + maximum],
                    independent,
                    atol=2e-14,
                    rtol=3e-11,
                )

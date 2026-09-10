"""Link the exact shared DF headers from independent consumer translation units."""

import ctypes
import shutil
import subprocess

import pytest
from vibeqc_compiler.integral.df_cuda import emit_df_values_cuda
from vibeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda
from vibeqc_compiler.integral.df_policy import emit_df_policy_cuda


def test_shared_df_headers_have_translation_unit_safe_linkage(tmp_path):
    """Catch duplicate functions/tables without requiring a CUDA device or SDK.

    Only CUDA attributes are replaced for host compilation; scientific source
    and policy headers are the actual emitted bytes. Numerical family tests
    separately validate those expressions against libcint and finite differences.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    (tmp_path / "cuda_runtime.h").write_text("")
    for name, source in (
        ("df_values.cuh", emit_df_values_cuda()),
        ("generated_df_derivatives.cuh", emit_df_derivatives_cuda()),
        ("values.cuh", emit_df_policy_cuda()),
        ("derivatives.cuh", emit_df_policy_cuda(derivatives=True)),
    ):
        (tmp_path / name).write_text(source)
    source = r"""
#define __device__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#include "values.cuh"
#include "derivatives.cuh"
extern "C" double ENTRY(double exponent) {
  namespace policy = vibeqc::scf::generated_df_policy;
  double e[2]{exponent, 0.8}, value = 0;
  policy::Value::Vec3 v[2]{{0,0,0},{0.3,0.2,0.1}};
  policy::Value::Angular a[2]{{0,0,0},{0,0,0}};
  policy::Value::accumulate<2>(value,e,v,a,1);
  policy::Derivative::Vec3 r[2]{{0,0,0},{0.3,0.2,0.1}};
  policy::Derivative::Angular b[2]{{0,0,0},{0,0,0}};
  policy::Derivative::Accumulator gradient{};
  policy::Derivative::accumulate<2>(gradient,e,r,b,1);
  return value + gradient.gradient[0][0];
}
"""
    paths = [tmp_path / f"{name}.cpp" for name in ("first", "second")]
    for path in paths:
        path.write_text(source.replace("ENTRY", path.stem))
    output = tmp_path / "consumers.so"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            *map(str, paths),
            "-I",
            str(tmp_path),
            "-o",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=240,
    )
    assert compiled.returncode == 0, compiled.stderr
    library = ctypes.CDLL(str(output))
    for name in ("first", "second"):
        function = getattr(library, name)
        function.argtypes, function.restype = [ctypes.c_double], ctypes.c_double
    assert library.first(0.7) == library.second(0.7)

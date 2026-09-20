"""Small device-free guards on composition, packaging and admission."""

import os
import subprocess
import sys
import typing

import pytest


def test_cuda_source_generation_is_device_and_runtime_independent() -> None:
    script = """
import sys
class Block:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'vibeqc','pyscf','cupy','torch'}:
            raise AssertionError('source generation imported runtime/oracle: '+name)
sys.meta_path.insert(0,Block())
from vibeqc_compiler.integral.first_derivative_native import emit_first_derivative_cuda, emit_first_derivative_cpu
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.stationary_cuda import emit_stationary_cuda
from vibeqc_compiler.method.stationary_gradient import SCF_POINT_MODEL, StationaryGradientPlan, StationaryMeanField
requests=(('overlap',('','')),('kinetic',('','')),('nuclear_attraction',('','')),
          ('four_center_eri',('','','','')),('nuclear',()))
import ctypes, subprocess

def forbidden(*args, **kwargs):
    raise AssertionError('generation must not compile, probe CUDA or load a native library')
subprocess.Popen=forbidden
ctypes.CDLL=forbidden
primitive=emit_first_derivative_cuda(requests)
assert primitive == emit_first_derivative_cuda(requests)
cpu=emit_first_derivative_cpu(requests)
assert '__device__' not in cpu
assert 'vibeqc_first_derivative_cpu' in cpu
for functional in (0,1,2):
    method=resolve_method(('LDA_XC_PW','PBE','R2SCAN')[functional],spin='unpolarized')
    plan=StationaryGradientPlan(method,StationaryMeanField(SCF_POINT_MODEL))
    s=emit_stationary_cuda(primitive,functional=functional,plan=plan)
    assert 'vibeqc_first_derivative_cpu' not in s
    assert '__device__ bool first_derivative' in s
    assert 'stationary_gradient_cuda.cuh' in s
    assert 'ao_pullback' in s
    assert 'local_becke' in s
    assert 'namespace vibeqc_grid_adjoint {' in s
    assert 'grid_response_adjoint.hpp' not in s
    assert f'stationary-plan: {plan.identity}' in s
    assert 'stationary-weight-program-one_electron:' in s
    assert 'stationary-weight-program-overlap_pulay:' in s
    assert 'stationary-weight-program-coulomb:' in s
    assert '__device__ inline bool stationary_source_weight' in s
    include = s.index('#include "dft/stationary_gradient_cuda.cuh"')
    for scientific in ('__global__ void primitive_kernel', '__global__ void geometry_kernel'):
        assert scientific in s
        assert s.index(scientific) > include
    assert f'stationary_functional = {functional}' in s
    assert s == emit_stationary_cuda(primitive,functional=functional,plan=plan)
template=open('src/dft/stationary_gradient_cuda.cuh').read()
assert '__global__ void primitive_kernel' in template
assert 'for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count' not in template
r2scan=emit_stationary_cuda(primitive,functional=2,plan=plan)
assert 'stationary_coefficients = 5' in r2scan
assert 'tau[0]' in r2scan and 'kinetic[0]' in r2scan
for functional in (0,1):
    plan=StationaryGradientPlan(resolve_method(('LDA_XC_PW','PBE')[functional],spin='unpolarized'),StationaryMeanField(SCF_POINT_MODEL))
    assert emit_stationary_cuda(primitive,pbe=bool(functional),plan=plan) == emit_stationary_cuda(primitive,functional=functional,plan=plan)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": ".:python"},
        check=True,
        timeout=30,
    )


@pytest.mark.parametrize("method_name", ["LDA_XC_PW", "PBE", "R2SCAN"])
@pytest.mark.parametrize(
    ("spin", "spin_blocks"), [("unpolarized", 1), ("polarized", 2)]
)
def test_generated_stationary_weight_lowering_tracks_plan(
    method_name: str, spin: str, spin_blocks: int
) -> None:
    from vibeqc_compiler.method import resolve_method
    from vibeqc_compiler.method.stationary_cuda import emit_stationary_weight_cuda
    from vibeqc_compiler.method.stationary_gradient import (
        SCF_POINT_MODEL,
        StationaryGradientPlan,
        StationaryMeanField,
    )

    plan = StationaryGradientPlan(
        resolve_method(method_name, spin=spin), StationaryMeanField(SCF_POINT_MODEL)
    )
    source = emit_stationary_weight_cuda(plan)
    assert f"stationary-plan: {plan.identity}" in source
    for name in ("one_electron", "overlap_pulay", "coulomb"):
        identity = plan.integral_block(name, terms=1).weights.logical_hash
        assert f"stationary-weight-program-{name}: {identity}" in source
    assert ("density[1 * n * n" in source) == (spin_blocks == 2)
    assert ("weighted_density[1 * n * n" in source) == (spin_blocks == 2)


@pytest.mark.parametrize("name", ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"])
@pytest.mark.parametrize("flags", ["--use_fast_math", "--fmad=true", "--ftz=true"])
def test_strict_stationary_cuda_rejects_environment_overrides(
    monkeypatch: typing.Any, tmp_path: typing.Any, name: typing.Any, flags: typing.Any
) -> None:
    from pathlib import Path

    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.cuda_target import cuda_target_info
    from vibeqc_compiler.method import resolve_method, stationary_cuda
    from vibeqc_compiler.method.stationary_gradient import (
        SCF_POINT_MODEL,
        StationaryGradientPlan,
        StationaryMeanField,
    )

    compiler = CudaCompilerAdapter(
        Path("must-not-execute-nvcc"), cuda_target_info("sm_120")
    )
    plan = StationaryGradientPlan(
        resolve_method("LDA_XC_PW", spin="unpolarized"),
        StationaryMeanField(SCF_POINT_MODEL),
    )
    monkeypatch.setenv(name, flags)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail(
            "strict arithmetic override reached source generation or compilation"
        )

    monkeypatch.setattr(stationary_cuda, "emit_stationary_cuda", forbidden)
    monkeypatch.setattr(stationary_cuda, "compile_runtime", forbidden)
    cache = tmp_path / "uncreated"
    with pytest.raises(ValueError, match="strict.*NVCC.*override"):
        stationary_cuda.compile_stationary_cuda(
            "",
            pbe=False,
            plan=plan,
            iterations=3,
            compiler=compiler,
            cache=cache,
        )
    assert not cache.exists()


def test_native_gradient_grid_helpers_do_not_duplicate_the_ao_translation_unit() -> (
    None
):
    from vibeqc_compiler.dft.ao_cuda import emit_grid_policy
    from vibeqc_compiler.xc.geometry_cuda import emit_native_geometry_cuda

    ao = emit_grid_policy()
    gradient = emit_native_geometry_cuda()
    assert "namespace vibeqc_grid_policy {" in ao
    assert "namespace vibeqc_grid_policy {" not in gradient
    assert "vibeqc_grid_policy::" not in gradient
    assert "namespace vibeqc_xc_gradient_grid_policy {" in gradient
    assert "vibeqc_xc_gradient_grid_policy::axis_jet" in gradient
    assert "namespace vibeqc_grid_adjoint {" in gradient
    assert "grid_response_adjoint.hpp" not in gradient

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
from vibeqc_compiler.method.stationary_cuda import emit_stationary_cuda, emit_stationary_wrapper_cuda
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
    wrapper=emit_stationary_wrapper_cuda(functional=functional,plan=plan)
    assert 'primitive_0(' not in wrapper
    assert 'extern __device__ bool first_derivative' in wrapper
    s=emit_stationary_cuda(primitive,functional=functional,plan=plan)
    assert s.startswith(primitive)
    assert 'extern __device__ bool first_derivative' not in s[len(primitive):]
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
    assert '__global__ void source_reduce' in s
    include = s.index('#include "dft/stationary_gradient_cuda.cuh"')
    for scientific in ('__global__ void task_kernel', '__global__ void geometry_kernel'):
        assert scientific in s
        assert s.index(scientific) > include
    assert f'stationary_functional = {functional}' in s
    assert 'stationary_records' not in s
    assert s == emit_stationary_cuda(primitive,functional=functional,plan=plan)
template=open('src/dft/stationary_gradient_cuda.cuh').read()
assert '__global__ void primitive_kernel' not in template
assert '__global__ void task_kernel' in template
assert 'stationary_tasks' in template
assert 'stationary_topology' in template
assert 'stationary_records' not in template
assert 'for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count' not in template
assert 'for (size_t linear = 0; linear < primitive_work; ++linear)' not in template
r2scan=emit_stationary_cuda(primitive,functional=2,plan=plan)
assert 'stationary_coefficients = 5' in r2scan
assert 'tau[0]' in r2scan and 'kinetic[0]' in r2scan
for functional in (0,1):
    plan=StationaryGradientPlan(resolve_method(('LDA_XC_PW','PBE')[functional],spin='unpolarized'),StationaryMeanField(SCF_POINT_MODEL))
    assert emit_stationary_cuda(primitive,pbe=bool(functional),plan=plan) == emit_stationary_cuda(primitive,functional=functional,plan=plan)
driver=open('python/vibeqc/_stationary_cuda.py').read()
assert 'stationary_records' not in driver
assert 'for ids in product(*ranges)' not in driver
assert '"stationary_tasks"' in driver
assert 'np.lexsort' in driver
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": os.pathsep.join((".", "python"))},
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
        assert f"stationary-weight-specialization-{name}:" in source
        assert f"stationary-weight-lowered-{name}:" in source
        assert f"stationary-weight-optimizer-{name}:" in source
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

    monkeypatch.setattr(stationary_cuda, "emit_stationary_wrapper_cuda", forbidden)
    monkeypatch.setattr(stationary_cuda, "compile_cuda_object", forbidden)
    monkeypatch.setattr(stationary_cuda, "link_cuda_objects", forbidden)
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


def test_stationary_aot_inventory_is_fixed_full_sp_domain() -> None:
    from itertools import product

    from vibeqc_compiler.integral.first_derivative_native import (
        emit_first_derivative_cuda,
    )
    from vibeqc_compiler.method.stationary_cuda import (
        QUALIFIED_SP_COMPONENTS,
        emit_stationary_aot_cuda,
        qualified_sp_requests,
        stationary_aot_plan_identity,
        stationary_aot_source_identity,
    )

    requests = qualified_sp_requests()
    primitive_source = emit_first_derivative_cuda(requests)
    assert len(requests) == 3 * 4**2 + 4**4 + 1
    assert len(requests) == len(set(requests)) == 305
    assert ("overlap", ("", "")) in requests
    assert ("four_center_eri", ("z", "x", "", "y")) in requests
    assert requests[-1] == ("nuclear", ())
    assert QUALIFIED_SP_COMPONENTS == ("", "x", "y", "z")
    assert {
        components for operator, components in requests if operator == "four_center_eri"
    } == set(product(QUALIFIED_SP_COMPONENTS, repeat=4))
    for functional in (0, 1, 2):
        identities = set()
        for spin, blocks in (("unpolarized", 1), ("polarized", 2)):
            source = emit_stationary_aot_cuda(
                functional, primitive_source=primitive_source, spin=spin
            )
            identities.add(
                stationary_aot_source_identity(
                    functional, primitive_source=primitive_source, spin=spin
                )
            )
            assert stationary_aot_plan_identity(functional, spin=spin)
            assert f"stationary_functional = {functional}" in source
            assert f"stationary_spin_blocks = {blocks}" in source
            assert "__global__ void source_reduce" in source
        assert len(identities) == 2
    with pytest.raises(ValueError, match="partition_iterations=3"):
        emit_stationary_aot_cuda(
            0, primitive_source=primitive_source, spin="unpolarized", iterations=2
        )


def test_stationary_aot_loader_checks_plan_target_and_binary_identity(
    tmp_path: typing.Any,
) -> None:
    import json

    from vibeqc_compiler.common.provenance import file_hash
    from vibeqc_compiler.method import resolve_method
    from vibeqc_compiler.method.stationary_cuda import (
        load_stationary_aot_artifact,
        stationary_aot_contract_identity,
        stationary_aot_plan_identity,
    )
    from vibeqc_compiler.method.stationary_gradient import (
        SCF_POINT_MODEL,
        StationaryGradientPlan,
        StationaryMeanField,
    )

    spin = "unpolarized"
    plan = StationaryGradientPlan(
        resolve_method("PBE", spin=spin), StationaryMeanField(SCF_POINT_MODEL)
    )
    library = tmp_path / "libvibeqc_stationary_pbe_rks.so"
    library.write_bytes(b"aot-binary")
    manifest = tmp_path / "vibeqc_stationary_pbe_rks.json"
    payload = {
        "schema": "vibeqc.stationary-cuda-aot.v2",
        "functional": 1,
        "spin": spin,
        "plan_identity": stationary_aot_plan_identity(1, spin=spin),
        "partition_iterations": 3,
        "architectures": ["sm_90", "sm_120"],
        "compile_architectures": ["90-real", "120"],
        "code_objects": [
            {"architecture": "sm_90", "kind": "cubin"},
            {"architecture": "sm_120", "kind": "cubin"},
            {"architecture": "sm_120", "kind": "ptx"},
        ],
        "source_identity": "build-recorded-source",
        "contract_identity": stationary_aot_contract_identity(1, spin=spin),
        "source_sha256": "unused-by-loader",
        "binary_sha256": file_hash(library),
        "binary_bytes": library.stat().st_size,
        "compile_contract": {"fp64": True, "fmad": False},
    }
    manifest.write_text(json.dumps(payload))

    artifact = load_stationary_aot_artifact(
        tmp_path,
        functional=1,
        spin=spin,
        plan=plan,
        architecture="sm_120",
    )
    assert artifact.library == library
    assert artifact.metadata["identity"]["plan"] == plan.identity
    assert artifact.metadata["identity"]["target"] == {
        "architecture": "sm_120",
        "code_kinds": ["cubin", "ptx"],
    }
    assert artifact.metadata["driver_ptx_jit_possible"] is True
    assert artifact.metadata["driver_ptx_jit_required"] is False

    with pytest.raises(NotImplementedError, match="sm_80"):
        load_stationary_aot_artifact(
            tmp_path,
            functional=1,
            spin=spin,
            plan=plan,
            architecture="sm_80",
        )
    polarized = StationaryGradientPlan(
        resolve_method("PBE", spin="polarized"), StationaryMeanField(SCF_POINT_MODEL)
    )
    with pytest.raises(ValueError, match="plan identity"):
        load_stationary_aot_artifact(
            tmp_path,
            functional=1,
            spin=spin,
            plan=polarized,
            architecture="sm_120",
        )
    bad = {**payload, "contract_identity": "0" * 64}
    manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="contract_identity"):
        load_stationary_aot_artifact(
            tmp_path,
            functional=1,
            spin=spin,
            plan=plan,
            architecture="sm_120",
        )
    manifest.write_text(json.dumps(payload))
    library.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="binary integrity"):
        load_stationary_aot_artifact(
            tmp_path,
            functional=1,
            spin=spin,
            plan=plan,
            architecture="sm_120",
        )


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, ()),
        ({"VIBEQC_STATIONARY_CUDA_SPLIT_COMPILE_THREADS": "1"}, ()),
        (
            {"VIBEQC_STATIONARY_CUDA_SPLIT_COMPILE_THREADS": "8"},
            ("--split-compile=8",),
        ),
    ],
)
def test_stationary_split_compile_options_are_explicit(
    environment: dict[str, str], expected: tuple[str, ...]
) -> None:
    from vibeqc_compiler.method.stationary_cuda import _split_compile_options

    assert _split_compile_options(environment) == expected


@pytest.mark.parametrize("value", ["0", "33", "many"])
def test_stationary_split_compile_options_fail_closed(value: str) -> None:
    from vibeqc_compiler.method.stationary_cuda import _split_compile_options

    with pytest.raises(
        ValueError, match="VIBEQC_STATIONARY_CUDA_SPLIT_COMPILE_THREADS"
    ):
        _split_compile_options({"VIBEQC_STATIONARY_CUDA_SPLIT_COMPILE_THREADS": value})

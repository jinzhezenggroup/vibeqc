"""Small device-free guards on composition, packaging and admission."""

import os
import subprocess
import sys

import pytest


def test_cuda_source_generation_is_device_and_runtime_independent():
    script = """
import sys
class Block:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'vibeqc','pyscf','cupy','torch'}:
            raise AssertionError('source generation imported runtime/oracle: '+name)
sys.meta_path.insert(0,Block())
from vibeqc_compiler.integral.first_derivative_native import emit_first_derivative_cuda, emit_first_derivative_cpu
from vibeqc_compiler.method.stationary_cuda import emit_stationary_cuda
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
for pbe in (False,True):
    s=emit_stationary_cuda(primitive,pbe=pbe)
    assert 'vibeqc_first_derivative_cpu' not in s
    assert '__device__ bool first_derivative' in s
    assert 'stationary_gradient_cuda.cuh' in s
    assert 'ao_pullback' in s
    assert 'local_becke' in s
    assert s == emit_stationary_cuda(primitive,pbe=pbe)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": ".:python"},
        check=True,
        timeout=30,
    )


@pytest.mark.parametrize("name", ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"])
@pytest.mark.parametrize("flags", ["--use_fast_math", "--fmad=true", "--ftz=true"])
def test_strict_stationary_cuda_rejects_environment_overrides(
    monkeypatch, tmp_path, name, flags
):
    from pathlib import Path

    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.cuda_target import cuda_target_info
    from vibeqc_compiler.method import stationary_cuda

    compiler = CudaCompilerAdapter(
        Path("must-not-execute-nvcc"), cuda_target_info("sm_120")
    )
    monkeypatch.setenv(name, flags)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "strict arithmetic override reached source generation or compilation"
        )

    monkeypatch.setattr(stationary_cuda, "emit_stationary_cuda", forbidden)
    monkeypatch.setattr(stationary_cuda, "compile_runtime", forbidden)
    cache = tmp_path / "uncreated"
    with pytest.raises(ValueError, match="strict.*NVCC.*override"):
        stationary_cuda.compile_stationary_cuda(
            "", pbe=False, iterations=3, compiler=compiler, cache=cache
        )
    assert not cache.exists()

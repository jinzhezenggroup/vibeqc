"""Object-cache identity resolves the same host command accepted by NVCC."""

import shutil
from pathlib import Path

import pytest
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.native_runtime import _cuda_host_identity
from vibeqc_compiler.common.provenance import file_hash


@pytest.mark.parametrize("spelling", ("name", "absolute"))
def test_host_compiler_command_is_resolved_before_hashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spelling: str
) -> None:
    executable = shutil.which("gcc")
    if executable is None:
        pytest.skip("GCC is required for this host identity test")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NVCC_CCBIN", "gcc" if spelling == "name" else executable)
    adapter = CudaCompilerAdapter(Path("/unused/nvcc"), cuda_target_info("sm_120"))
    selected, identity = _cuda_host_identity(adapter)
    assert selected == Path(executable).resolve()
    assert identity["host_compiler"] == file_hash(Path(executable).resolve())
    assert identity["host_version"]

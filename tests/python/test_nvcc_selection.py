"""Honor the compiler configuration advertised by the public CUDA force API."""

import pytest
from vibeqc_compiler.common.provenance import find_nvcc


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path.resolve()


@pytest.mark.parametrize("command_name", [False, True])
def test_cudacxx_selects_the_advertised_compiler(monkeypatch, tmp_path, command_name):
    compiler = executable(tmp_path / "bin" / "nvcc-requested")
    monkeypatch.delenv("VIBEQC_NVCC", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("PATH", str(compiler.parent) if command_name else "")
    monkeypatch.setenv("CUDACXX", compiler.name if command_name else str(compiler))
    assert find_nvcc() == compiler


def test_project_override_precedes_cudacxx_and_toolkit_root(monkeypatch, tmp_path):
    override = executable(tmp_path / "project-nvcc")
    cudacxx = executable(tmp_path / "cmake-nvcc")
    toolkit = executable(tmp_path / "toolkit/bin/nvcc")
    monkeypatch.setenv("VIBEQC_NVCC", str(override))
    monkeypatch.setenv("CUDACXX", str(cudacxx))
    monkeypatch.setenv("CUDA_PATH", str(toolkit.parent.parent))
    monkeypatch.setenv("PATH", "")
    assert find_nvcc() == override
    monkeypatch.delenv("VIBEQC_NVCC")
    assert find_nvcc() == cudacxx
    monkeypatch.delenv("CUDACXX")
    assert find_nvcc() == toolkit

"""Exercise the real CUDA qualification setup without loading native owners."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import pytest

SOURCE = Path(__file__).with_name("test_cc_complete_gradient_cuda.py")


@pytest.fixture
def compiler_setup(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[tuple]]:
    """Execute the actual setup prefix, stopping before molecular/GPU work."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "test_complete_ccsdt_cuda_response_gradient_matches_pinned_pyscf"
    )
    end = next(
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "expected"
            for target in node.targets
        )
    )
    function.body = function.body[:end] + [
        ast.Return(value=ast.Name(id="compiler", ctx=ast.Load()))
    ]
    calls: list[tuple] = []

    def find_nvcc() -> str:
        calls.append(("nvcc",))
        return "allocated-nvcc"

    def target_info(architecture: str) -> str:
        calls.append(("target", architecture))
        if architecture not in {"sm_80", "sm_89", "sm_90", "sm_120"}:
            raise ValueError("unsupported target")
        return architecture

    def compiler(nvcc: str, target: str) -> tuple[str, str]:
        calls.append(("compiler", nvcc, target))
        return nvcc, target

    namespace = {
        "os": os,
        "pytest": pytest,
        "Path": Path,
        "find_nvcc": find_nvcc,
        "cuda_target_info": target_info,
        "CudaCompilerAdapter": compiler,
    }
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    monkeypatch.setenv("SLURM_JOB_ID", "host-control-flow-test")
    return namespace[function.name], calls


@pytest.mark.parametrize("architecture", [None, "", "   "])
def test_target_is_required_before_compiler_lookup(
    compiler_setup: tuple[Any, list[tuple]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str | None,
) -> None:
    setup, calls = compiler_setup
    if architecture is None:
        monkeypatch.delenv("VIBEQC_TENSOR_ARCH", raising=False)
    else:
        monkeypatch.setenv("VIBEQC_TENSOR_ARCH", architecture)
    with pytest.raises(pytest.fail.Exception, match="VIBEQC_TENSOR_ARCH"):
        setup(tmp_path)
    assert calls == []


@pytest.mark.parametrize(
    "architecture", ["sm_80", "sm_89", "sm_90", "sm_120", " sm_90 "]
)
def test_explicit_target_is_preserved(
    compiler_setup: tuple[Any, list[tuple]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
) -> None:
    setup, calls = compiler_setup
    monkeypatch.setenv("VIBEQC_TENSOR_ARCH", architecture)
    assert setup(tmp_path) == ("allocated-nvcc", architecture.strip())
    assert calls[-1] == ("compiler", "allocated-nvcc", architecture.strip())


def test_invalid_target_has_no_default_fallback(
    compiler_setup: tuple[Any, list[tuple]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    setup, calls = compiler_setup
    monkeypatch.setenv("VIBEQC_TENSOR_ARCH", "not-a-target")
    with pytest.raises(ValueError, match="unsupported target"):
        setup(tmp_path)
    assert not any(call[0] == "compiler" for call in calls)


def test_slurm_requirement_is_preserved(
    compiler_setup: tuple[Any, list[tuple]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    setup, calls = compiler_setup
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("VIBEQC_TENSOR_ARCH", "sm_90")
    with pytest.raises(AssertionError, match="requires Slurm"):
        setup(tmp_path)
    assert calls == []

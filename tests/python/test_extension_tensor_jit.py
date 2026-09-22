"""Explicit advanced-user TensorIR JIT contract for #558."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from vibeqc.extensions import tensor
import vibeqc_compiler.common.cpp_adapter as cpp_adapter
import vibeqc_compiler.tensor.cpu as tensor_cpu


def _program() -> tensor.Program:
    space = tensor.IndexSpace("ao", "ao", 2)
    index = tensor.Index("i", space)
    spec = tensor.TensorSpec((index,), role="input")
    node = tensor.input_tensor("x", spec)
    return tensor.Program({"value": node})


def test_explicit_cpu_jit_wraps_native_artifact_without_leaking_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    class FakeCompilerAdapter:
        def __init__(self, cxx: Path, compile_timeout: float = 300.0) -> None:
            calls["compiler"] = cxx
            calls["compile_timeout"] = compile_timeout

    class FakeNativeTensorProgram:
        def __init__(
            self,
            program: tensor.Program,
            *,
            compiler: object,
            cache: Path,
            max_bytes: int,
            max_work: int,
            max_nodes: int,
        ) -> None:
            calls["native_compiler"] = compiler
            calls["cache"] = cache
            calls["budgets"] = (max_bytes, max_work, max_nodes)
            self.identity = "native-source-identity"
            self.resources = {
                "required_bytes": 128,
                "scalar_work": 16,
            }
            self.artifact = SimpleNamespace(
                library=cache / "runtime.so",
                metadata={"key": "verified-artifact", "compile_seconds": 0.125},
            )
            self._program = program

        def execute(self, feeds: object) -> object:
            return {"feeds": feeds, "logical_hash": self._program.logical_hash}

    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", FakeCompilerAdapter)
    monkeypatch.setattr(tensor_cpu, "NativeTensorProgram", FakeNativeTensorProgram)

    program = _program()
    compiled = tensor.compile(
        program,
        compiler="test-c++",
        cache=tmp_path,
        compile_timeout=12.5,
        max_bytes=4096,
        max_work=8192,
        max_nodes=32,
    )

    assert isinstance(compiled, tensor.CompiledTensorProgram)
    assert compiled.target == "cpu"
    assert compiled.mode == "jit"
    assert compiled.logical_hash == program.logical_hash
    assert compiled.identity == "native-source-identity"
    assert calls["compiler"] == Path("test-c++")
    assert calls["compile_timeout"] == 12.5
    assert calls["cache"] == tmp_path
    assert calls["budgets"] == (4096, 8192, 32)

    report = compiled.inspect()
    assert report["extension_api_version"] == tensor.API_VERSION
    assert report["kind"] == "compiled-tensor"
    assert report["artifact"]["metadata"]["key"] == "verified-artifact"
    assert report["resources"] == {"required_bytes": 128, "scalar_work": 16}
    assert compiled.execute({"x": [1.0, 2.0]})["logical_hash"] == program.logical_hash

    # Public provenance is detached: callers cannot mutate the compiler owner.
    report["artifact"]["metadata"]["key"] = "tampered"
    report["resources"]["required_bytes"] = 0
    assert compiled.artifact["metadata"]["key"] == "verified-artifact"
    assert compiled.resources["required_bytes"] == 128


def test_tensor_jit_rejects_unsupported_requests_before_toolchain_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activations = 0

    class ForbiddenCompilerAdapter:
        def __init__(self, *_: object, **__: object) -> None:
            nonlocal activations
            activations += 1
            raise AssertionError("toolchain must not activate for rejected requests")

    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", ForbiddenCompilerAdapter)
    program = _program()

    with pytest.raises(ValueError, match="mode='jit'"):
        tensor.compile(program, mode="aot")
    with pytest.raises(ValueError, match="target='cpu'"):
        tensor.compile(program, target="cuda")
    with pytest.raises(TypeError, match="tensor compilation requires Program"):
        tensor.compile(object())  # type: ignore[arg-type]

    assert activations == 0


def test_tensor_jit_resolves_toolchain_and_cache_only_on_explicit_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    class FakeCompilerAdapter:
        def __init__(self, cxx: Path, compile_timeout: float = 300.0) -> None:
            calls["compiler"] = cxx
            calls["compile_timeout"] = compile_timeout

    class FakeNativeTensorProgram:
        def __init__(self, program: tensor.Program, **kwargs: object) -> None:
            calls.update(kwargs)
            self.identity = "identity"
            self.resources = {}
            cache = Path(kwargs["cache"])
            self.artifact = SimpleNamespace(library=cache / "runtime.so", metadata={})
            self.program = program

        def execute(self, feeds: object) -> object:
            return feeds

    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", FakeCompilerAdapter)
    monkeypatch.setattr(tensor_cpu, "NativeTensorProgram", FakeNativeTensorProgram)
    monkeypatch.setenv("CXX", "configured-c++")
    monkeypatch.setenv("VIBEQC_TENSOR_CACHE", str(tmp_path))

    tensor.compile(_program())

    assert calls["compiler"] == Path("configured-c++")
    assert calls["cache"] == tmp_path / "extensions"


@pytest.mark.parametrize("timeout", [True, "30", float("inf"), 0.0, -1.0])
def test_tensor_jit_validates_timeout_before_toolchain_activation(timeout: object) -> None:
    program = _program()
    error = TypeError if isinstance(timeout, (bool, str)) else ValueError
    with pytest.raises(error, match="compile_timeout"):
        tensor.compile(program, compile_timeout=timeout)  # type: ignore[arg-type]

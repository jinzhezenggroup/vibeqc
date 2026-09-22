"""Public TensorIR compile-capability contract for #558."""

from __future__ import annotations

import shutil
import subprocess

import pytest
from vibeqc.extensions import tensor
from vibeqc_compiler.common import cpp_adapter


def _program() -> tensor.Program:
    space = tensor.IndexSpace("ao", "ao", 2)
    index = tensor.Index("i", space)
    node = tensor.input_tensor("x", tensor.TensorSpec((index,), role="input"))
    return tensor.Program({"value": node})


def test_compile_capabilities_reports_lowering_without_toolchain_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("capability inspection must not activate a toolchain")

    class ForbiddenCompilerAdapter:
        def __init__(self, *_: object, **__: object) -> None:
            forbidden()

    monkeypatch.setattr(shutil, "which", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(cpp_adapter, "CppCompilerAdapter", ForbiddenCompilerAdapter)

    program = _program()
    report = tensor.compile_capabilities(
        program,
        max_bytes=4096,
        max_work=4096,
        max_nodes=32,
    )

    assert report["extension_api_version"] == tensor.API_VERSION
    assert report["kind"] == "tensor-compile-capabilities"
    assert report["logical_hash"] == program.logical_hash
    assert report["target"] == "cpu"
    assert report["mode"] == "jit"
    assert report["represented"] is True
    assert report["compilable"] is True
    assert report["lowering_validated"] is True
    assert report["validated"] is False
    assert report["production_promoted"] is False
    assert report["toolchain_checked"] is False
    assert report["reason"] is None
    assert report["resources"]["required_bytes"] > 0
    assert report["resources"]["scalar_work"] > 0


@pytest.mark.parametrize(
    ("target", "mode", "message"),
    [
        ("cpu", "aot", "mode='jit'"),
        ("cuda", "jit", "target='cpu'"),
    ],
)
def test_compile_capabilities_reports_unsupported_target_or_mode(
    target: str, mode: str, message: str
) -> None:
    report = tensor.compile_capabilities(_program(), target=target, mode=mode)

    assert report["represented"] is True
    assert report["compilable"] is False
    assert report["lowering_validated"] is False
    assert report["validated"] is False
    assert report["production_promoted"] is False
    assert report["toolchain_checked"] is False
    assert report["resources"] is None
    assert message in report["reason"]


def test_compile_capabilities_reports_unsupported_ir_without_promotion() -> None:
    program = _program()
    unsupported = tensor.Program({"value": tensor.exp(program.outputs["value"])})

    report = tensor.compile_capabilities(unsupported)

    assert report["represented"] is True
    assert report["compilable"] is False
    assert report["lowering_validated"] is False
    assert report["validated"] is False
    assert report["production_promoted"] is False
    assert report["resources"] is None
    assert "unsupported CPU primitive" in report["reason"]


def test_compile_capabilities_applies_requested_resource_bounds() -> None:
    report = tensor.compile_capabilities(_program(), max_bytes=0)

    assert report["represented"] is True
    assert report["compilable"] is False
    assert report["resources"] is None
    assert "budget" in report["reason"]


def test_compile_capabilities_rejects_non_program() -> None:
    with pytest.raises(TypeError, match="capability query requires Program"):
        tensor.compile_capabilities(object())  # type: ignore[arg-type]

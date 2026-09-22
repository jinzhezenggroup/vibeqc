"""Shared semilocal CPU/CUDA code-generation ownership regressions."""

from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest
from vibeqc_compiler.xc.semilocal_codegen import (
    emit_polarized_semilocal,
    polarized_feature_count,
)
from vibeqc_compiler.xc.spec import functional

ROOT = Path(__file__).resolve().parents[2]


def _identity(source: str, name: str) -> str:
    match = re.search(rf'{name} = "([0-9a-f]+)"', source)
    assert match is not None
    return match.group(1)


def test_r2scan_cpu_cuda_wrappers_share_expression_identity() -> None:
    cpu = runpy.run_path(str(ROOT / "tools/generate_xc_cpu.py"))[
        "emit_r2scan_polarized"
    ]()
    cuda = runpy.run_path(str(ROOT / "tools/generate_xc_r2scan_cuda.py"))[
        "emit_r2scan_device"
    ]()

    assert _identity(cpu, "kR2scanPolarizedExpressionIdentity") == _identity(
        cuda, "kR2scanDeviceExpressionIdentity"
    )
    assert "inline R2scanPolarizedValue r2scan_polarized(" in cpu
    assert "__device__ inline R2scanDeviceValue r2scan_device(" in cuda


@pytest.mark.parametrize(
    ("name", "features"),
    (("PBE", 5), ("SCAN", 7), ("R2SCAN", 7)),
)
def test_shared_semilocal_lowerer_derives_native_feature_width(
    name: str, features: int
) -> None:
    spec = functional(name, spin="polarized")
    assert polarized_feature_count(spec) == features

    source = emit_polarized_semilocal(
        spec,
        value_type="TestValue",
        function_name="test_point",
        identity_constant="kTestIdentity",
        production=name == "R2SCAN",
        function_qualifier="__device__ inline",
    )
    assert f"double feature_derivative[{features}];" in source
    assert "__device__ inline TestValue test_point(" in source


def test_generator_tools_do_not_reown_semilocal_differentiation() -> None:
    cpu = (ROOT / "tools/generate_xc_cpu.py").read_text()
    cuda = (ROOT / "tools/generate_xc_r2scan_cuda.py").read_text()

    assert "def build_roots(" not in cpu
    assert "from vibeqc_compiler.xc.semilocal_codegen import" in cpu
    assert "ScalarCEmitter" not in cuda
    assert "build_roots" not in cuda
    assert "tools.generate_xc_cpu" not in cuda
    assert "emit_polarized_semilocal" in cuda


def test_unpolarized_spec_is_not_a_native_polarized_abi() -> None:
    with pytest.raises(ValueError, match="polarized FunctionalSpec"):
        polarized_feature_count(functional("PBE", spin="unpolarized"))

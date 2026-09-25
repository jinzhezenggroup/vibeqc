"""Shared semilocal CPU/CUDA code-generation ownership regressions."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from vibeqc_compiler.xc.semilocal_codegen import (
    emit_polarized_semilocal,
    polarized_feature_count,
)
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS, UnsupportedXC, functional

ROOT = Path(__file__).resolve().parents[2]


def _identity(source: str, name: str) -> str:
    match = re.search(rf'{name} = "([0-9a-f]+)"', source)
    assert match is not None
    return match.group(1)


def test_r2scan_cpu_cuda_wrappers_share_expression_identity(tmp_path: Path) -> None:
    cpu_path = tmp_path / "xc_cpu.hpp"
    cuda_path = tmp_path / "r2scan_device.cuh"
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "tools/generate_xc_cpu.py"),
            "--output",
            str(cpu_path),
        ],
        check=True,
        cwd=tmp_path,
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "tools/generate_xc_r2scan_cuda.py"),
            "--output",
            str(cuda_path),
        ],
        check=True,
        cwd=tmp_path,
    )
    cpu, cuda = cpu_path.read_text(), cuda_path.read_text()

    assert _identity(cpu, "kR2scanPolarizedExpressionIdentity") == _identity(
        cuda, "kR2scanDeviceExpressionIdentity"
    )
    assert "inline R2scanPolarizedValue r2scan_polarized(" in cpu
    assert "__device__ inline R2scanDeviceValue r2scan_device(" in cuda


def test_wb97mv_cpu_cuda_wrappers_share_expression_identity(tmp_path: Path) -> None:
    cpu_path = tmp_path / "xc_cpu.hpp"
    cuda_path = tmp_path / "wb97mv_device.cuh"
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "tools/generate_xc_cpu.py"),
            "--output",
            str(cpu_path),
        ],
        check=True,
        cwd=tmp_path,
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "tools/generate_xc_wb97mv_cuda.py"),
            "--output",
            str(cuda_path),
        ],
        check=True,
        cwd=tmp_path,
    )
    cpu, cuda = cpu_path.read_text(), cuda_path.read_text()

    assert _identity(cpu, "kWb97mvSemilocalExpressionIdentity") == _identity(
        cuda, "kWb97mvDeviceExpressionIdentity"
    )
    assert "inline Wb97mvPolarizedValue wb97mv_polarized(" in cpu
    assert "__device__ inline Wb97mvDeviceValue wb97mv_device(" in cuda
    assert "kWb97mvDeviceDensityThreshold" in cuda
    assert "kWb97mvDeviceSigmaThreshold" in cuda
    assert "kWb97mvDeviceTauThreshold" in cuda


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


@pytest.mark.parametrize(
    ("name", "features"),
    (("LDA_C_VWN_4", 2), ("GGA_X_PBE_SOL", 5)),
)
def test_bulk_imports_reach_shared_native_pointwise_lowerer_without_admission(
    name: str, features: int
) -> None:
    assert name in AUTO_BULK_COMPONENTS
    spec = functional(name, spin="polarized")
    assert polarized_feature_count(spec) == features

    cpu = emit_polarized_semilocal(
        spec,
        value_type="BulkValue",
        function_name="bulk_point",
        identity_constant="kBulkIdentity",
        pointwise_bulk=True,
    )
    cuda = emit_polarized_semilocal(
        spec,
        value_type="BulkValue",
        function_name="bulk_point",
        identity_constant="kBulkIdentity",
        function_qualifier="__device__ inline",
        pointwise_bulk=True,
    )
    assert _identity(cpu, "kBulkIdentity") == _identity(cuda, "kBulkIdentity")
    assert f"double feature_derivative[{features}];" in cpu
    assert "inline BulkValue bulk_point(" in cpu
    assert "__device__ inline BulkValue bulk_point(" in cuda

    with pytest.raises(UnsupportedXC, match="not production-domain admitted"):
        emit_polarized_semilocal(
            spec,
            value_type="BulkValue",
            function_name="bulk_point",
            identity_constant="kBulkIdentity",
        )


def test_bulk_tau_mgga_stays_explicitly_blocked_until_feature_projection_lands() -> (
    None
):
    spec = functional("MGGA_X_R2SCAN01", spin="polarized")
    assert "MGGA_X_R2SCAN01" in AUTO_BULK_COMPONENTS
    with pytest.raises(UnsupportedXC, match="tau/laplacian projection"):
        emit_polarized_semilocal(
            spec,
            value_type="BulkMggaValue",
            function_name="bulk_mgga_point",
            identity_constant="kBulkMggaIdentity",
            pointwise_bulk=True,
        )


def test_generator_tools_do_not_reown_semilocal_differentiation() -> None:
    cpu = (ROOT / "tools/generate_xc_cpu.py").read_text()
    cuda = (ROOT / "tools/generate_xc_r2scan_cuda.py").read_text()
    wb97mv_cuda = (ROOT / "tools/generate_xc_wb97mv_cuda.py").read_text()

    assert "def build_roots(" not in cpu
    assert "from vibeqc_compiler.xc.semilocal_codegen import" in cpu
    assert "ScalarCEmitter" not in cuda
    assert "build_roots" not in cuda
    assert "tools.generate_xc_cpu" not in cuda
    assert "emit_r2scan_program" in cuda
    assert "ScalarCEmitter" not in wb97mv_cuda
    assert "build_roots" not in wb97mv_cuda
    assert "tools.generate_xc_cpu" not in wb97mv_cuda
    assert "emit_polarized_semilocal" in wb97mv_cuda


def test_unpolarized_spec_is_not_a_native_polarized_abi() -> None:
    with pytest.raises(ValueError, match="polarized FunctionalSpec"):
        polarized_feature_count(functional("PBE", spin="unpolarized"))

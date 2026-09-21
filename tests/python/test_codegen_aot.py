"""AOT code-generation determinism, cache identity, and CLI contract tests."""

from __future__ import annotations

import subprocess
import sys
import typing
from dataclasses import replace

import pytest

if typing.TYPE_CHECKING:
    from pathlib import Path
from vibeqc_compiler.integral import (
    DDPS_SPEC,
    DPDS_SPEC,
    DPPP_SPEC,
    FDDD_SPEC,
    PSSS_SPEC,
    SSSS_SPEC,
    ContractionSpec,
    NvrtcCacheSpec,
    OperatorFamily,
    OperatorSpec,
    TranslationInvariant,
    build_dppp_component_kernel,
    build_dppp_contraction_kernel,
    build_integral_ir,
    build_psss_kernel,
    cuda_target_info,
    emit_shell_class_fused_cuda,
    nvrtc_cache_key,
)
from vibeqc_compiler.integral.shell_class import (
    emit_dppp_component_cuda,
    emit_dppp_contraction_cuda,
    emit_psss_cuda,
)

TEST_CUDA_TARGET = cuda_target_info("sm_120")


def test_cuda_emission_is_deterministic_and_runtime_ad_free() -> None:
    first = emit_psss_cuda(build_psss_kernel("z"))
    second = emit_psss_cuda(build_psss_kernel("z"))
    assert first == second
    assert "boys_values<2>" in first
    assert "generated_psss_z_gradient" in first
    assert "Dual3" not in first


def test_dppp_cuda_emission_is_deterministic_and_runtime_ad_free() -> None:
    kernel = build_dppp_component_kernel("xy", tuple("xyz"))
    first = emit_dppp_component_cuda(kernel)
    second = emit_dppp_component_cuda(build_dppp_component_kernel("xy", tuple("xyz")))
    assert first == second
    assert "boys_values<6>" in first
    assert "generated_dppp_xy_xyz_gradient" in first
    assert "Dual3" not in first

    factored = emit_dppp_contraction_cuda(
        build_dppp_contraction_kernel("xy", tuple("xyz"))
    )
    assert "GeneratedDpppGeometry" in factored
    assert "generated_dppp_xy_xyz_factored_gradient" in factored
    assert "boys_values" not in factored
    assert "Dual3" not in factored


def test_dppp_contraction_cuda_honors_explicit_nonfinal_recovery() -> None:
    """Pack factored decay rows by IR order when center B is recovered."""

    operator = OperatorSpec(
        family=OperatorFamily.FOUR_CENTER_ERI,
        centers=(0, 1, 2, 3),
        invariants=(TranslationInvariant(dependent_center=1),),
    )
    force = ContractionSpec(
        consumer="direct_force",
        density="rhf|uhf",
        output="atomic_force",
    )
    integral = build_integral_ir(
        DPPP_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    kernel = build_dppp_contraction_kernel(
        "xy",
        tuple("xyz"),
        integral=integral,
    )
    assert kernel.integral is integral
    source = emit_dppp_contraction_cuda(kernel)

    # The ket fourth-center derivative needs the complement of the stored
    # third-center product scale.  Decay rows are dense A/C/D slots, not
    # physical center indices A/B/C/D, so row 1 must remain present while
    # row 3 is not referenced by this three-independent-center geometry ABI.
    assert "1.0 - geometry.product_scales[2]" in source
    assert "geometry.decay_gradients[1]" in source
    assert "geometry.decay_gradients[2]" in source
    assert "geometry.decay_gradients[3]" not in source


def test_nvrtc_cache_key_covers_binary_compatibility_inputs() -> None:
    specification = NvrtcCacheSpec(
        generator_abi="1",
        shell_class=(2, 1, 2, 0),
        derivative_centers=(0, 1, 2),
        precision_policy="fp64",
        screening_policy="density-v1",
        source_digest="abc123",
        compute_capability="sm_120",
        nvrtc_version="12.9",
        driver_version="580.95.05",
    )
    key = nvrtc_cache_key(specification)
    assert len(key) == 64
    assert key == nvrtc_cache_key(specification)
    assert key != nvrtc_cache_key(
        replace(specification, precision_policy="mixed-fp32-control")
    )


def test_codegen_cli_writes_deterministic_aot_candidate(tmp_path: Path) -> None:
    output = tmp_path / "generated" / "psss_x.cuh"
    command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        "psss",
        "--axis",
        "x",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, check=True)
    assert output.read_text(encoding="utf-8") == first
    assert "generated_psss_x_gradient" in first


def test_codegen_cli_writes_dppp_component_candidate(tmp_path: Path) -> None:
    output = tmp_path / "generated" / "dppp_xy_xyz.cuh"
    command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        "dppp",
        "--d-component",
        "xy",
        "--p-components",
        "xyz",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, check=True)
    assert output.read_text(encoding="utf-8") == first
    assert "generated_dppp_xy_xyz_gradient" in first

    factored_output = tmp_path / "generated" / "dppp_xy_xyz_factored.cuh"
    factored_command = [
        *command[:-2],
        "--lowering",
        "factored",
        "--output",
        str(factored_output),
    ]
    subprocess.run(factored_command, check=True)
    factored = factored_output.read_text(encoding="utf-8")
    subprocess.run(factored_command, check=True)
    assert factored_output.read_text(encoding="utf-8") == factored
    assert "generated_dppp_xy_xyz_factored_gradient" in factored

    fused_output = tmp_path / "generated" / "dppp_fused.cuh"
    fused_command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        "dppp",
        "--lowering",
        "fused",
        "--architecture",
        TEST_CUDA_TARGET.architecture,
        "--output",
        str(fused_output),
    ]
    subprocess.run(fused_command, check=True)
    fused = fused_output.read_text(encoding="utf-8")
    subprocess.run(fused_command, check=True)
    assert fused_output.read_text(encoding="utf-8") == fused
    assert "generated_dppp_shell_class_force_rhf_kernel" in fused


@pytest.mark.parametrize(
    "spec", (SSSS_SPEC, PSSS_SPEC, DPDS_SPEC, DDPS_SPEC, FDDD_SPEC)
)
def test_codegen_cli_writes_generated_fused_candidate(
    tmp_path: Path, spec: typing.Any
) -> None:
    output = tmp_path / "generated" / f"{spec.name}_fused.cuh"
    command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        spec.name,
        "--lowering",
        "fused",
        "--architecture",
        TEST_CUDA_TARGET.architecture,
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, check=True)
    assert output.read_text(encoding="utf-8") == first
    assert first == emit_shell_class_fused_cuda(spec, target=TEST_CUDA_TARGET)
    assert f"generated_{spec.name}_shell_class_force_rhf_kernel" in first

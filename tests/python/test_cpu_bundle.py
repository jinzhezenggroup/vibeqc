"""Portable multi-ISA CPU bundle manifests and fail-closed dispatch."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cpu_dispatch import (
    CpuRuntimeFeatures,
    cpu_binary_target_supported,
    detect_cpu_features,
    select_cpu_target,
)
from vibeqc_compiler.common.cpu_target import (
    AVX2_FMA_TARGET,
    AVX512F_FMA_TARGET,
    CPU_TARGETS,
    GENERIC_CPU_TARGET,
)
from vibeqc_compiler.integral.cpu_bundle import (
    FirstDerivativeCpuDispatchEvaluator,
    compile_first_derivative_cpu_bundle,
    load_first_derivative_cpu_bundle,
)
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir


def test_cpu_target_selection_prefers_widest_supported_and_forces_safely(monkeypatch):
    monkeypatch.delenv("VIBEQC_CPU_TARGET", raising=False)
    none = CpuRuntimeFeatures("x86_64", ())
    avx2 = CpuRuntimeFeatures("amd64", ("fma", "avx2"))
    avx512 = CpuRuntimeFeatures("x86_64", ("avx512f", "fma", "avx2"))
    assert select_cpu_target(CPU_TARGETS, none).selected_target == "generic"
    assert select_cpu_target(CPU_TARGETS, avx2).selected_target == AVX2_FMA_TARGET.name
    assert (
        select_cpu_target(CPU_TARGETS, avx512).selected_target
        == AVX512F_FMA_TARGET.name
    )
    assert (
        select_cpu_target(
            CPU_TARGETS,
            avx512,
            forced_target="generic",
        ).selected_target
        == "generic"
    )
    with pytest.raises(ValueError, match="unsupported"):
        select_cpu_target(
            CPU_TARGETS,
            avx2,
            forced_target=AVX512F_FMA_TARGET.name,
        )
    with pytest.raises(ValueError, match="not in this bundle"):
        select_cpu_target(CPU_TARGETS, avx512, forced_target="x86_64-amx")

    monkeypatch.setenv("VIBEQC_CPU_TARGET", "generic")
    decision = select_cpu_target(CPU_TARGETS, avx512)
    assert decision.selected_target == "generic"
    assert decision.forced_target == "generic"


@pytest.fixture(scope="module")
def cpu_bundle(tmp_path_factory):
    executable = shutil.which("c++")
    if executable is None:
        pytest.skip("CPU C++ compiler required")
    return compile_first_derivative_cpu_bundle(
        build_weighted_eri_ir((0, 0, 0, 0)),
        CppCompilerAdapter(Path(executable)),
        tmp_path_factory.mktemp("cpu-bundle"),
        component_indices=(0,),
    )


def test_bundle_materializes_portable_manifest_and_target_specific_cache(
    cpu_bundle, tmp_path
):
    cpu_bundle.validate()
    assert tuple(target.name for target in cpu_bundle.targets) == (
        GENERIC_CPU_TARGET.name,
        AVX2_FMA_TARGET.name,
        AVX512F_FMA_TARGET.name,
    )
    manifest = json.loads((cpu_bundle.directory / "manifest.json").read_text())
    assert manifest["bundle_identity"] == cpu_bundle.program_identity
    assert len(manifest["candidates"]) == 3
    keys = set()
    for row in manifest["candidates"]:
        path = cpu_bundle.directory / row["library"]
        assert path.is_file()
        assert path.parent == cpu_bundle.directory / "libraries"
        flags = row["native_metadata"]["identity"]["flags"]
        assert "-march=native" not in flags
        assert (
            row["binary_target"]
            == row["native_metadata"]["identity"]["target"]["architecture"]
        )
        assert row["binary_target"] != "portable"
        keys.add(row["artifact_key"])
    assert len(keys) == 3

    relocated_directory = tmp_path / "installed-package-data" / "cpu-kernel"
    shutil.copytree(cpu_bundle.directory, relocated_directory)
    loaded = load_first_derivative_cpu_bundle(relocated_directory)
    assert loaded.program_identity == cpu_bundle.program_identity
    assert tuple(target.name for target in loaded.targets) == tuple(
        target.name for target in cpu_bundle.targets
    )
    assert all(
        candidate.native.library.is_relative_to(relocated_directory)
        for candidate in loaded.candidates
    )


def test_dispatch_loads_only_selected_compatible_candidate(cpu_bundle, monkeypatch):
    monkeypatch.delenv("VIBEQC_CPU_TARGET", raising=False)
    primitives = (
        (
            (0.71, 0.8),
            (0.49, -0.2),
            (0.31, 0.5),
            (0.19, 0.1),
            (0.11, -0.07),
        ),
        ((0.83, 1.0),),
        ((1.07, 1.0),),
        ((0.93, 1.0),),
    )
    centers = np.array(
        [
            [0.13, -0.31, 0.24],
            [-0.43, 0.27, 0.51],
            [0.68, -0.14, -0.22],
            [-0.21, 0.48, -0.63],
        ]
    )

    generic = FirstDerivativeCpuDispatchEvaluator(
        cpu_bundle,
        runtime=CpuRuntimeFeatures("x86_64", ()),
        record_capacity=7,
    )
    assert generic.selected_target == "generic"
    expected = generic.contract(primitives, centers)

    runtime = detect_cpu_features()
    automatic = FirstDerivativeCpuDispatchEvaluator(
        cpu_bundle,
        runtime=runtime,
        record_capacity=7,
    )
    actual = automatic.contract(primitives, centers)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)

    diagnostics = automatic.diagnostics()
    assert diagnostics["selected_target"] == automatic.selected_target
    assert diagnostics["bundle_identity"] == cpu_bundle.program_identity
    assert diagnostics["runtime"]["architecture"] == runtime.architecture
    assert set(diagnostics["candidate_artifact_keys"]) == {
        GENERIC_CPU_TARGET.name,
        AVX2_FMA_TARGET.name,
        AVX512F_FMA_TARGET.name,
    }


def test_bundle_rejects_cross_architecture_before_loading(cpu_bundle, monkeypatch):
    import vibeqc_compiler.integral.cpu_bundle as module

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("incompatible CPU binary must be rejected before dlopen")

    monkeypatch.setattr(module, "FirstDerivativeCpuLaneEvaluator", forbidden_loader)
    with pytest.raises(ValueError, match="runtime architecture/ABI"):
        module.FirstDerivativeCpuDispatchEvaluator(
            cpu_bundle,
            runtime=CpuRuntimeFeatures("aarch64", ()),
        )


def test_binary_target_gate_checks_runtime_abi(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    runtime = CpuRuntimeFeatures("x86_64", ())
    assert cpu_binary_target_supported("x86_64-linux-gnu", runtime)
    assert not cpu_binary_target_supported("x86_64-apple-darwin23", runtime)
    assert not cpu_binary_target_supported("aarch64-linux-gnu", runtime)


def test_bundle_rejects_forced_unsupported_isa_before_loading(cpu_bundle):
    with pytest.raises(ValueError, match="unsupported"):
        FirstDerivativeCpuDispatchEvaluator(
            cpu_bundle,
            runtime=CpuRuntimeFeatures("x86_64", ("avx2", "fma")),
            forced_target=AVX512F_FMA_TARGET.name,
        )

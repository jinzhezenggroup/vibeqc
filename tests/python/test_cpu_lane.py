"""CPU scalar/AVX lane lowering from the shared four-center ERI DAG."""

import platform
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Primitive, Shell
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cpu_target import (
    AVX2_FMA_TARGET,
    AVX512F_FMA_TARGET,
    GENERIC_CPU_TARGET,
)
from vibeqc_compiler.integral.cpu_lane import emit_first_components_cpu_lanes
from vibeqc_compiler.integral.cpu_lane_execute import (
    FirstDerivativeCpuLaneShellEvaluator,
    compile_first_derivative_cpu_lane,
    compile_first_derivative_cpu_lane_shell,
)
from vibeqc_compiler.integral.cpu_schedule import CpuScheduleIR, default_cpu_schedule
from vibeqc_compiler.integral.weight_pullback import (
    normalized_cartesian_components,
    normalized_radial_primitives,
)
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir

from tools.vibeqc_posthf.sources import NativeSource


def _x86():
    return platform.machine().lower() in ("x86_64", "amd64")


def _has_flag(flag):
    try:
        text = Path("/proc/cpuinfo").read_text().lower()
    except OSError:
        return False
    return flag.lower() in text


@pytest.fixture(scope="module")
def compiler():
    executable = shutil.which("c++")
    if executable is None:
        pytest.skip("CPU C++ compiler required")
    return CppCompilerAdapter(Path(executable))


def test_cpu_lane_schedule_and_sources_are_target_explicit():
    ir = build_weighted_eri_ir((1, 0, 0, 0))
    scalar = default_cpu_schedule(GENERIC_CPU_TARGET)
    avx2 = default_cpu_schedule(AVX2_FMA_TARGET)
    avx512 = default_cpu_schedule(AVX512F_FMA_TARGET)
    assert (scalar.vector_lanes, avx2.vector_lanes, avx512.vector_lanes) == (1, 4, 8)
    with pytest.raises(ValueError, match="vector width"):
        CpuScheduleIR(vector_lanes=4).validate_for(GENERIC_CPU_TARGET)

    sources = tuple(
        emit_first_components_cpu_lanes(
            ir,
            (0,),
            target,
            schedule,
        )
        for target, schedule in (
            (GENERIC_CPU_TARGET, scalar),
            (AVX2_FMA_TARGET, avx2),
            (AVX512F_FMA_TARGET, avx512),
        )
    )
    assert "_mm256_" not in sources[0] and "_mm512_" not in sources[0]
    assert "_mm256_" in sources[1] and "_mm512_" not in sources[1]
    assert "_mm512_" in sources[2] and "_mm256_" not in sources[2]
    assert all("psss" not in source.lower() for source in sources)
    assert all("ffast-math" not in source for source in sources)


@pytest.mark.skipif(not _x86(), reason="initial SIMD targets are x86_64")
def test_scalar_avx2_avx512_candidates_compile(compiler, tmp_path):
    ir = build_weighted_eri_ir((1, 0, 0, 0))
    artifacts = []
    for target in (GENERIC_CPU_TARGET, AVX2_FMA_TARGET, AVX512F_FMA_TARGET):
        artifact = compile_first_derivative_cpu_lane(
            ir,
            compiler,
            tmp_path,
            component_indices=(0,),
            target=target,
            schedule=default_cpu_schedule(target),
        )
        artifact.validate()
        flags = artifact.native.metadata["identity"]["flags"]
        assert all(option in flags for option in target.compiler_options)
        artifacts.append(artifact)
    assert len({item.program_identity for item in artifacts}) == 3


@pytest.mark.skipif(not _has_flag("avx2"), reason="AVX2 execution unavailable")
def test_avx2_full_shell_tail_matches_independent_native_oracle(compiler, tmp_path):
    ir = build_weighted_eri_ir((1, 0, 0, 0))
    targets = [GENERIC_CPU_TARGET, AVX2_FMA_TARGET]
    if _has_flag("avx512f") and _has_flag("fma"):
        targets.append(AVX512F_FMA_TARGET)
    artifacts = [
        compile_first_derivative_cpu_lane_shell(
            ir,
            compiler,
            tmp_path / target.name,
            target=target,
            schedule=default_cpu_schedule(target),
            tile_size=3,
        )
        for target in targets
    ]
    coordinates = np.array(
        [
            [0.13, -0.31, 0.24],
            [-0.43, 0.27, 0.51],
            [0.68, -0.14, -0.22],
            [-0.21, 0.48, -0.63],
        ]
    )
    primitive_specs = (
        (
            (1.7, 0.30),
            (0.91, -0.24),
            (0.53, 0.61),
            (0.29, 0.18),
            (0.12, -0.08),
        ),
        ((0.80, 1.0),),
        ((1.10, 1.0),),
        ((0.90, 1.0),),
    )
    primitives = tuple(
        normalized_radial_primitives(l, specs)
        for l, specs in zip(ir.signature.angular, primitive_specs, strict=True)
    )
    actual = [
        FirstDerivativeCpuLaneShellEvaluator(
            artifact,
            record_capacity=7,
            budget_bytes=8 << 20,
        ).contract(primitives, coordinates)
        for artifact in artifacts
    ]
    for candidate in actual[1:]:
        np.testing.assert_allclose(candidate, actual[0], atol=2e-12, rtol=2e-12)

    factors = np.array(
        [
            scale
            for _, scale in normalized_cartesian_components(
                ir.signature.angular,
                np.ones(ir.signature.component_shape),
            )
        ]
    )
    normalized = actual[-1] * factors[:, None]
    shells = tuple(
        Shell(
            atom,
            l,
            tuple(Primitive(exponent, coefficient) for exponent, coefficient in specs),
        )
        for atom, (l, specs) in enumerate(
            zip(ir.signature.angular, primitive_specs, strict=True)
        )
    )
    with NativeSource([(1, xyz) for xyz in coordinates], basis=shells) as source:
        sizes = tuple(source.shell_sizes)
        offsets = tuple(int(x) for x in np.cumsum((0, *sizes[:-1]), dtype=np.int64))
        value = source._read("four_center_eri", offsets, sizes).reshape(-1)
        slices = tuple(
            slice(offset, offset + size)
            for offset, size in zip(offsets, sizes, strict=True)
        )
        derivative = (
            source.integral_derivatives()["eri"][(slice(None), *slices)]
            .reshape(12, -1)
            .T
        )
    np.testing.assert_allclose(normalized[:, 0], value, atol=3e-11, rtol=2e-10)
    np.testing.assert_allclose(
        normalized[:, 1:],
        derivative,
        atol=5e-11,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        normalized[:, 1:].reshape(-1, 4, 3).sum(axis=1),
        0.0,
        atol=3e-11,
    )


@pytest.mark.parametrize(
    "option",
    [
        "-ffast-math",
        "-Ofast",
        "-ffinite-math-only",
        "-ffp-contract=fast",
        "-march=native",
        "@override.rsp",
    ],
)
def test_cpu_target_rejects_unqualified_compiler_options(option):
    with pytest.raises(ValueError, match="compiler options"):
        replace(GENERIC_CPU_TARGET, compiler_options=(option,))


@pytest.mark.parametrize("lanes", [True, 1.0, 4.0, 8.0])
def test_cpu_target_and_schedule_require_integer_lanes(lanes):
    with pytest.raises(ValueError, match="lanes"):
        replace(GENERIC_CPU_TARGET, vector_lanes=lanes)
    with pytest.raises(ValueError, match="lanes"):
        CpuScheduleIR(vector_lanes=lanes)


def test_simd_target_requires_matching_isa_flags():
    with pytest.raises(ValueError, match="compiler options"):
        replace(AVX2_FMA_TARGET, compiler_options=())
    with pytest.raises(ValueError, match="compiler options"):
        replace(AVX2_FMA_TARGET, compiler_options=AVX512F_FMA_TARGET.compiler_options)


def test_strict_lane_execution_rejects_nonfinite_primitive(compiler, tmp_path):
    from vibeqc_compiler.integral.cpu_lane_execute import (
        FirstDerivativeCpuLaneEvaluator,
    )

    target = GENERIC_CPU_TARGET
    artifact = compile_first_derivative_cpu_lane(
        build_weighted_eri_ir((0, 0, 0, 0)),
        compiler,
        tmp_path,
        component_indices=(0,),
        target=target,
        schedule=default_cpu_schedule(target),
    )
    with pytest.raises(ValueError, match="status 1"):
        FirstDerivativeCpuLaneEvaluator(artifact).contract(
            (((float("nan"), 1.0),), ((1.0, 1.0),), ((1.0, 1.0),), ((1.0, 1.0),)),
            np.zeros((4, 3)),
        )

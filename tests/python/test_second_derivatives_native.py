"""Compiled CPU/CUDA coordinate tiles, bounded lifecycle and independent gates."""

import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.integral.second_derivatives import (
    build_eri_second_ir,
    build_one_electron_second_ir,
    build_second_derivative_kernel,
)
from vibeqc_compiler.integral.second_derivatives_execute import (
    PreparedSecondDerivative,
    SecondPrimitive,
    compile_second_derivative,
    pack_second_primitive,
)

from tools.vibeqc_validation.second_derivatives import (
    evaluate_second_primitive,
    libcint_eri_hessian,
    libcint_one_electron_hessian,
)

POSITIONS = (
    (0.13, -0.31, 0.24),
    (-0.43, 0.27, 0.51),
    (0.68, -0.14, -0.22),
    (-0.21, 0.48, -0.63),
)
EXPONENTS = (0.6, 0.8, 1.1, 0.9)


@pytest.fixture(scope="module", params=("cpu", "cuda"))
def compiler(request, tmp_path_factory):
    """Compile with the common adapter; every actual CUDA call requires Slurm."""
    cuda = request.param == "cuda"
    if cuda and os.environ.get("VIBEQC_TEST_SECOND_CUDA") != "1":
        pytest.skip("set VIBEQC_TEST_SECOND_CUDA=1 inside a Slurm GPU job")
    if cuda and not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("second derivative CUDA validation requires Slurm")
    executable = shutil.which("nvcc" if cuda else "c++")
    if executable is None:
        pytest.skip("native compiler unavailable")
    adapter = (
        CudaCompilerAdapter(Path(executable), cuda_target_info("sm_120"))
        if cuda
        else CppCompilerAdapter(Path(executable))
    )
    return adapter, tmp_path_factory.mktemp(f"second-runtime-{request.param}")


def fixture(family, output):
    """Asymmetric primitive fixtures with distinct packed signed weights."""
    ir = (
        build_eri_second_ir((2, 1, 0, 1), output=output)
        if family == "eri"
        else build_one_electron_second_ir(family, (1, 2), charge=2.3, output=output)
    )
    count = len(ir.operator.centers)
    indices = (4,) if output == "raw_hessian" else (0, 4)
    outputs = (
        tuple(range(3 * count))
        if output == "weighted_hvp"
        else (0, 1, 3, (3 * count) ** 2 - 1)
    )
    primitive = SecondPrimitive(
        EXPONENTS[: len(ir.signature.shells)],
        POSITIONS[:count],
        None if output == "raw_hessian" else (0.73, -0.31),
        tuple(tuple(row) for row in np.random.default_rng(178).normal(size=(count, 3)))
        if output == "weighted_hvp"
        else None,
    )
    return ir, indices, outputs, primitive


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction", "eri"])
@pytest.mark.parametrize("output", ["raw_hessian", "weighted_hessian", "weighted_hvp"])
def test_native_tiles_match_independent_analytic_hessian(compiler, family, output):
    pytest.importorskip("pyscf")
    ir, indices, outputs, primitive = fixture(family, output)
    artifact = compile_second_derivative(
        ir, *compiler, component_indices=indices, output_indices=outputs
    )
    reference = (
        libcint_eri_hessian(
            ir.signature.angular, primitive.exponents, primitive.centers
        )
        if family == "eri"
        else libcint_one_electron_hessian(
            family, ir.signature.angular, primitive.exponents, primitive.centers, 2.3
        )
    )
    dimension = 3 * len(primitive.centers)
    reference = reference.reshape(dimension, dimension, -1)
    matrix = (
        reference[:, :, indices[0]]
        if primitive.weights is None
        else sum(
            reference[:, :, i] * w
            for i, w in zip(indices, primitive.weights, strict=True)
        )
    )
    expected = (
        (matrix @ np.asarray(primitive.direction).ravel())
        if output == "weighted_hvp"
        else matrix.ravel()
    )[list(outputs)]
    with PreparedSecondDerivative(artifact, record_capacity=2, tile_capacity=2) as plan:
        records = [
            replace(primitive, scale=s, output_tile=i % 2)
            for i, s in enumerate((1.0, -0.7, 0.3, 0.9, -0.2))
        ]
        result = plan.contract(iter(records), tile_count=2, profile=True)
        np.testing.assert_allclose(
            result.values,
            np.array([1.1, 0.2])[:, None] * expected,
            atol=6e-11,
            rtol=3e-11,
        )
        assert result.diagnostics["records"] == 5 and result.diagnostics["chunks"] == 3
        assert result.diagnostics["electronic_response"] == "excluded"
        np.testing.assert_array_equal(plan.contract((), tile_count=2).values, 0)
        np.testing.assert_array_equal(
            plan.contract(records, tile_count=2).values, result.values
        )
        assert not result.values.flags.writeable


def test_failed_late_chunk_native_errors_empty_replay_and_budget(compiler):
    ir, indices, outputs, primitive = fixture("eri", "weighted_hvp")
    artifact = compile_second_derivative(
        ir, *compiler, component_indices=indices, output_indices=outputs
    )
    with pytest.raises(ValueError):
        PreparedSecondDerivative(
            artifact, budget=ResourceBudget(host_bytes=1, device_bytes=1)
        )
    with pytest.raises(ValueError, match="identity"):
        PreparedSecondDerivative(replace(artifact, output_indices=(0,)))
    with PreparedSecondDerivative(artifact, record_capacity=1, tile_capacity=2) as plan:
        saved = plan.contract((primitive,)).values
        for changed in (
            replace(primitive, weights=(float("nan"), 0)),
            replace(primitive, exponents=(1e308,) * 4),
            replace(primitive, direction=None),
            replace(primitive, output_tile=2),
        ):
            with pytest.raises((ValueError, FloatingPointError)):
                plan.contract((primitive, changed), tile_count=2)
            np.testing.assert_allclose(
                plan.contract((primitive,)).values, saved, atol=1e-13
            )
        output = np.full((2, len(outputs)), 173.0)
        plan._records[0] = np.frombuffer(
            pack_second_primitive(artifact, primitive), dtype=np.uint8
        )
        with pytest.raises(ValueError, match="stride"):
            plan._call(
                "vibeqc_second_run_v1",
                plan._handle,
                plan._records.ctypes.data,
                1,
                artifact.record_format.size - 8,
                2,
                output.ctypes.data,
                0,
            )
        np.testing.assert_array_equal(output, 173)
        # A valid shape with invalid geometry must fail transactionally inside
        # the native owner, independently of Python's record input checks.
        malformed = replace(primitive, exponents=(1e308,) * 4)
        plan._records[0] = np.frombuffer(
            pack_second_primitive(artifact, malformed), dtype=np.uint8
        )
        with pytest.raises(FloatingPointError):
            plan._call(
                "vibeqc_second_run_v1",
                plan._handle,
                plan._records.ctypes.data,
                1,
                artifact.record_format.size,
                2,
                output.ctypes.data,
                0,
            )
        np.testing.assert_array_equal(output, 173)
        plan._call(
            "vibeqc_second_run_v1",
            plan._handle,
            None,
            0,
            artifact.record_format.size,
            2,
            output.ctypes.data,
            0,
        )
        np.testing.assert_array_equal(output, 0)
        with pytest.raises(AttributeError):
            plan.tile_capacity = 999
    with pytest.raises(RuntimeError, match="closed"):
        plan.contract((primitive,))
    plan.close()


def test_explicit_fifteenth_moment_native_hvp(compiler):
    ir = build_eri_second_ir((3, 3, 3, 3))
    kernel = build_second_derivative_kernel(ir, (0,), output_indices=(0,))
    artifact = compile_second_derivative(
        ir, *compiler, component_indices=(0,), output_indices=(0,)
    )
    direction = np.random.default_rng(178).normal(size=(4, 3))
    primitive = SecondPrimitive(EXPONENTS, POSITIONS, (0.7,), direction)
    weights = np.zeros(ir.signature.component_count)
    weights[0] = 0.7
    expected = evaluate_second_primitive(
        kernel, EXPONENTS, POSITIONS, weights, direction
    )
    with PreparedSecondDerivative(artifact, record_capacity=1) as plan:
        np.testing.assert_allclose(
            plan.contract((primitive,)).values[0], expected, atol=2e-11, rtol=2e-11
        )


def test_native_svec_preserves_offdiagonal_inner_product_factors(compiler):
    pytest.importorskip("pyscf")
    from vibeqc_compiler.integral.second_order_layout import HessianLayout

    ir = build_eri_second_ir((1, 0, 0, 0), output="weighted_hessian", packing="svec")
    # Two negative signs and a nonunit factor exercise generated consumer
    # scaling independently of the record's fixed primitive normalization.
    consumer = ir.contractions[0]
    ir = replace(
        ir,
        contractions=(
            replace(
                consumer,
                output_sign=-1,
                weights=replace(consumer.weights, sign=-1, prefactor=0.37),
            ),
        ),
    )
    indices, selected = (0, 2), (0, 1, 4, 12, 13, 77)
    artifact = compile_second_derivative(
        ir, *compiler, component_indices=indices, output_indices=selected
    )
    raw = libcint_eri_hessian(ir.signature.angular, EXPONENTS, POSITIONS).reshape(
        12, 12, -1
    )
    matrix = 0.37 * (0.73 * raw[:, :, 0] - 0.31 * raw[:, :, 2])
    expected = HessianLayout(ir.operator.centers, "svec").encode(matrix)[list(selected)]
    primitive = SecondPrimitive(EXPONENTS, POSITIONS, (0.73, -0.31))
    with PreparedSecondDerivative(artifact, record_capacity=1) as plan:
        np.testing.assert_allclose(
            plan.contract((primitive,)).values[0], expected, atol=5e-11, rtol=2e-11
        )


def test_native_ffff_hvp_against_independent_libcint(compiler):
    pytest.importorskip("pyscf")
    ir = build_eri_second_ir((3, 3, 3, 3))
    artifact = compile_second_derivative(
        ir, *compiler, component_indices=(0,), output_indices=(0,)
    )
    direction = np.random.default_rng(178).normal(size=(4, 3))
    # This bounded test oracle owns one f/f/f/f raw block (about 12 MB). The
    # production HVP retains one coordinate output and one component weight.
    matrix = libcint_eri_hessian(ir.signature.angular, EXPONENTS, POSITIONS).reshape(
        12, 12, -1
    )[:, :, 0]
    expected = 0.7 * (matrix @ direction.ravel())[0]
    primitive = SecondPrimitive(EXPONENTS, POSITIONS, (0.7,), direction)
    with PreparedSecondDerivative(artifact, record_capacity=1) as plan:
        assert plan.contract((primitive,)).values[0, 0] == pytest.approx(
            expected, abs=5e-11, rel=2e-11
        )

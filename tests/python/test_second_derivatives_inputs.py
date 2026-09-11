"""Public contraction normalization and the physical-atom HVP chain rule."""

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
from vibeqc_compiler.integral.blocks import TensorLayout, WeightTile
from vibeqc_compiler.integral.second_derivatives import (
    build_eri_second_ir,
    build_one_electron_second_ir,
)
from vibeqc_compiler.integral.second_derivatives_execute import (
    PreparedSecondDerivative,
    compile_second_derivative,
)
from vibeqc_compiler.integral.second_derivatives_inputs import (
    prepare_second_shell_stream,
)
from vibeqc_compiler.integral.second_order_layout import SecondAtomMap
from vibeqc_compiler.integral.shell_signature import BasisConvention

from tools.validate_weighted_eri import make_fixture
from tools.vibeqc_validation.f_shell_numerics import _normalized_primitives
from tools.vibeqc_validation.one_electron_values import make_one_electron_fixture
from tools.vibeqc_validation.second_derivatives import (
    contracted_public_first_gradient,
    normalized_cartesian_rotation,
)


@pytest.fixture(scope="module", params=("cpu", "cuda"))
def compiler(request, tmp_path_factory):
    pytest.importorskip("pyscf")
    cuda = request.param == "cuda"
    if cuda and os.environ.get("VIBEQC_TEST_SECOND_CUDA") != "1":
        pytest.skip("set VIBEQC_TEST_SECOND_CUDA=1 inside a Slurm GPU job")
    if cuda and not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("public second derivative CUDA validation requires Slurm")
    executable = shutil.which("nvcc" if cuda else "c++")
    if executable is None:
        pytest.skip("native compiler unavailable")
    adapter = (
        CudaCompilerAdapter(Path(executable), cuda_target_info("sm_120"))
        if cuda
        else CppCompilerAdapter(Path(executable))
    )
    return adapter, tmp_path_factory.mktemp(f"second-public-{request.param}")


def public_fixture(family, spherical):
    """Reuse first-integral normalization fixtures with independent public weights."""
    if family == "eri":
        fixture = make_fixture("dpsp", "spherical" if spherical else "cartesian")
        ir = build_eri_second_ir((2, 1, 0, 1))
        inputs, centers, primitives = (
            fixture["inputs"],
            fixture["centers"],
            fixture["primitives"],
        )
        projections = fixture["projections"]
    else:
        fixture = make_one_electron_fixture((1, 2), lengths=(2, 2))
        ir = build_one_electron_second_ir(
            family, (1, 2), charge=2.3, output="weighted_hvp"
        )
        inputs = {
            **fixture.inputs,
            "basis_representation": "spherical" if spherical else "cartesian",
        }
        centers = np.array(
            inputs["coordinates"]
            + ([inputs["operator_center"]] if family == "nuclear_attraction" else [])
        )
        primitives = tuple(_normalized_primitives(s) for s in inputs["shells"])
        projections = fixture.projections if spherical else None
    signature = ir.signature
    if spherical:
        signature = replace(
            signature,
            legacy_class=None,
            shells=tuple(
                replace(s, convention=BasisConvention.REAL_SPHERICAL)
                for s in signature.shells
            ),
        )
    weights = np.random.default_rng(178).normal(size=signature.component_shape)
    return ir, inputs, centers, primitives, signature, projections, weights


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction", "eri"])
@pytest.mark.parametrize("spherical", [False, True])
def test_contracted_public_hvp_matches_three_step_independent_gradient_fd(
    compiler, family, spherical
):
    ir, inputs, centers, primitives, signature, projections, weights = public_fixture(
        family, spherical
    )
    artifact = compile_second_derivative(ir, *compiler)
    center_atoms = (7, 7, 9, 11)[: len(centers)]
    mapping = SecondAtomMap(ir.requested_derivative_centers, center_atoms)
    centers[1] = centers[0]
    direction = mapping.expand_direction(
        np.random.default_rng(17).normal(size=(len(mapping.atom_indices), 3))
    )
    layout = TensorLayout(signature.tensor_indices, signature.component_shape)
    tile = WeightTile(layout, weights.ravel())
    stream = prepare_second_shell_stream(
        artifact,
        primitives,
        centers,
        tile,
        public_signature=signature,
        projections=projections,
        direction=direction,
    )
    with PreparedSecondDerivative(artifact, record_capacity=3) as plan:
        result = plan.contract(stream, profile=True)
    actual = mapping.scatter_hvp(result.values.reshape(len(centers), 3))
    assert stream.record_count == result.diagnostics["records"]
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        finite = (
            contracted_public_first_gradient(
                family, inputs, centers + step * direction, weights
            )
            - contracted_public_first_gradient(
                family, inputs, centers - step * direction, weights
            )
        ) / (2 * step)
        errors.append(np.max(np.abs(actual - mapping.scatter_hvp(finite))))
    scale = max(1, np.max(np.abs(actual)))
    assert errors[-1] < 2e-7 * scale, errors
    assert errors[-1] <= max(5e-11 * scale, errors[0] * 0.03), errors
    with pytest.raises(ValueError, match="numeric budget"):
        prepare_second_shell_stream(
            artifact,
            primitives,
            centers,
            tile,
            public_signature=signature,
            projections=projections,
            direction=direction,
            budget_bytes=0,
        )


def test_public_coverage_and_composed_budget_fail_before_execution(compiler):
    ir, _, centers, primitives, signature, projections, weights = public_fixture(
        "nuclear_attraction", True
    )
    partial = compile_second_derivative(ir, *compiler, component_indices=(0,))
    direction = np.ones_like(centers)
    tile = WeightTile(
        TensorLayout(signature.tensor_indices, signature.component_shape),
        weights.ravel(),
    )
    from vibeqc_compiler.integral.shell_signature import CenterBinding

    for malformed in (
        replace(
            signature,
            shells=(
                replace(signature.shells[0], role="auxiliary"),
                signature.shells[1],
            ),
        ),
        replace(
            signature, center_bindings=signature.center_bindings + (CenterBinding(3),)
        ),
    ):
        with pytest.raises(ValueError, match="roles and center bindings"):
            prepare_second_shell_stream(
                partial,
                primitives,
                centers,
                tile,
                public_signature=malformed,
                projections=projections,
                direction=direction,
            )
    with pytest.raises(ValueError, match="compiled Cartesian subset"):
        prepare_second_shell_stream(
            partial,
            primitives,
            centers,
            tile,
            public_signature=signature,
            projections=projections,
            direction=direction,
        )
    full = compile_second_derivative(ir, *compiler)
    stream = prepare_second_shell_stream(
        full,
        primitives,
        centers,
        tile,
        public_signature=signature,
        projections=projections,
        direction=direction,
    )
    # The native owner fits alone; composing the public adapter's storage must
    # still reject before consuming its stream or allocating result buffers.
    with (
        PreparedSecondDerivative(
            full,
            record_capacity=1,
            budget=ResourceBudget(host_bytes=2000, device_bytes=2000),
        ) as plan,
        pytest.raises(ValueError),
    ):
        plan.contract(stream)


def test_atom_chain_rule_uses_both_hessian_indices_and_noncontiguous_labels():
    mapping = SecondAtomMap((0, 1, 2, 3), (8, 8, 1, 5))
    random = np.random.default_rng(178)
    matrix = random.normal(size=(12, 12))
    matrix += matrix.T
    direction = random.normal(size=(3, 3))
    shell = mapping.expand_direction(direction)
    expected = mapping.scatter_hvp((matrix @ shell.ravel()).reshape(4, 3))
    np.testing.assert_allclose(
        expected.ravel(),
        mapping.scatter_hessian(matrix) @ direction.ravel(),
        atol=2e-14,
    )
    assert mapping.atom_indices == (1, 5, 8)


@pytest.mark.parametrize("family", ["nuclear_attraction", "eri"])
def test_public_hvp_arbitrary_rotation_and_bra_shell_permutation(compiler, family):
    ir, _, centers, primitives, signature, _, weights = public_fixture(family, False)
    direction = np.random.default_rng(18).normal(size=centers.shape)
    artifact = compile_second_derivative(ir, *compiler)

    def run(artifact, primitives, centers, weights, direction):
        layout = TensorLayout(artifact.integral.signature.tensor_indices, weights.shape)
        stream = prepare_second_shell_stream(
            artifact,
            primitives,
            centers,
            WeightTile(layout, weights.ravel()),
            direction=direction,
        )
        with PreparedSecondDerivative(artifact, record_capacity=3) as plan:
            return plan.contract(stream).values.reshape(len(centers), 3)

    actual = run(artifact, primitives, centers, weights, direction)
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation = (
        np.cos(0.37) * np.eye(3)
        + (1 - np.cos(0.37)) * np.outer(axis, axis)
        + np.sin(0.37) * skew
    )
    rotated_weights = weights.copy()
    for slot, angular in enumerate(signature.angular):
        inverse_transpose = np.linalg.inv(
            normalized_cartesian_rotation(angular, rotation)
        ).T
        rotated_weights = np.moveaxis(
            np.tensordot(inverse_transpose, rotated_weights, axes=(1, slot)), 0, slot
        )
    rotated = run(
        artifact,
        primitives,
        centers @ rotation.T + [0.2, -0.7, 0.3],
        rotated_weights,
        direction @ rotation.T,
    )
    np.testing.assert_allclose(rotated, actual @ rotation.T, atol=2e-11, rtol=2e-11)
    order = (1, 0, *range(2, len(signature.shells)))
    center_order = (1, 0, *range(2, len(centers)))
    angular = tuple(signature.angular[i] for i in order)
    reverse_ir = (
        build_eri_second_ir(angular)
        if family == "eri"
        else build_one_electron_second_ir(
            family, angular, charge=2.3, output="weighted_hvp"
        )
    )
    reverse = compile_second_derivative(reverse_ir, *compiler)
    permuted = run(
        reverse,
        tuple(primitives[i] for i in order),
        centers[list(center_order)],
        weights.transpose(order),
        direction[list(center_order)],
    )
    np.testing.assert_allclose(
        permuted, actual[list(center_order)], atol=2e-11, rtol=2e-11
    )

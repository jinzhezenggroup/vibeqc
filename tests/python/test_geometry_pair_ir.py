"""GeometryIR / PairIR qualification on the CPU reference and CUDA emitter."""

import typing

import numpy as np
import pytest
from vibeqc_compiler.geometry import (
    GeometryIR,
    PairCutoff,
    PairTopology,
    inverse_power_program,
    lower_geometry,
    pair_to_atom,
    pair_to_system,
)
from vibeqc_compiler.tensor import Program, TensorSpec, execute, input_tensor


def _geometry() -> typing.Any:
    return GeometryIR((1, 6, 8), parameter_identity="elements-v1")


def _coordinates() -> typing.Any:
    return np.array(
        [
            [0.1, -0.2, 0.3],
            [1.4, 0.5, -0.1],
            [-0.7, 1.2, 0.9],
        ],
        dtype=np.float64,
    )


def _energy(pair_program: typing.Any, coordinates: typing.Any) -> typing.Any:
    return (
        execute(
            pair_program.program,
            {pair_program.geometry.coordinate_name: coordinates},
        )
        .outputs["energy"]
        .item()
    )


def test_pair_topology_is_canonical_explicit_and_deterministic() -> None:
    cutoff = PairCutoff(8.0, 6.0)
    topology = PairTopology.complete(4, cutoff=cutoff)
    assert topology.pairs == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )
    assert topology.owners == (0, 0, 0, 1, 1, 2)
    assert topology.to_payload()["cutoff"] == {
        "radius": (8.0).hex(),
        "switch_start": (6.0).hex(),
    }
    assert topology.identity == PairTopology.complete(4, cutoff=cutoff).identity
    for pairs in (
        ((1, 0),),
        ((0, 1), (0, 1)),
        ((1, 2), (0, 1)),
        ((0, 4),),
    ):
        with pytest.raises(ValueError):
            PairTopology(4, pairs)


def test_geometry_lowering_has_explicit_pair_displacements_and_distances() -> None:
    geometry = _geometry()
    topology = PairTopology.complete(3)
    context = lower_geometry(geometry, topology)
    result = execute(
        Program(
            {
                "displacement": context.displacement,
                "squared_distance": context.squared_distance,
                "distance": context.distance,
            }
        ),
        {"coordinates": _coordinates()},
    ).outputs
    expected = np.array(
        [
            _coordinates()[1] - _coordinates()[0],
            _coordinates()[2] - _coordinates()[0],
            _coordinates()[2] - _coordinates()[1],
        ]
    )
    np.testing.assert_allclose(result["displacement"], expected, rtol=0, atol=0)
    np.testing.assert_allclose(
        result["squared_distance"], np.einsum("pc,pc->p", expected, expected)
    )
    np.testing.assert_allclose(
        result["distance"], np.linalg.norm(expected, axis=1), rtol=1e-14, atol=1e-14
    )


def test_pair_reductions_are_shared_tensorir_not_method_kernels() -> None:
    context = lower_geometry(_geometry(), PairTopology.complete(3))
    values = input_tensor(
        "q",
        TensorSpec((context.pair_index,), role="input"),
    )
    program = Program(
        {
            "system": pair_to_system(values, context),
            "incident": pair_to_atom(values, context),
            "owned": pair_to_atom(values, context, owners_only=True),
        }
    )
    assert all(node.op != "einsum" for node in program.live_nodes)
    assert sum(node.op == "scatter_add" for node in program.live_nodes) == 3
    result = execute(program, {"q": np.array([1.0, 2.0, 3.0])}).outputs
    np.testing.assert_array_equal(result["system"], 6.0)
    np.testing.assert_array_equal(result["incident"], np.array([3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(result["owned"], np.array([3.0, 3.0, 0.0]))


def test_inverse_power_energy_is_translation_and_permutation_invariant() -> None:
    geometry = _geometry()
    topology = PairTopology.complete(3)
    program = inverse_power_program(geometry, topology, 1, exponent=-1)
    coordinates = _coordinates()
    reference = _energy(program, coordinates)
    shift = np.array([3.25, -7.0, 1.5])
    assert _energy(program, coordinates + shift) == pytest.approx(reference, abs=2e-15)

    order = np.array([2, 0, 1])
    permuted_geometry = GeometryIR(
        tuple(geometry.elements[i] for i in order),
        parameter_identity=geometry.parameter_identity,
    )
    permuted = inverse_power_program(
        permuted_geometry, PairTopology.complete(3), 1, exponent=-1
    )
    assert _energy(permuted, coordinates[order]) == pytest.approx(
        reference, rel=2e-15, abs=2e-15
    )


def test_generated_coordinate_jvp_matches_directional_finite_difference() -> None:
    program = inverse_power_program(
        _geometry(), PairTopology.complete(3), (1, 2, 3), exponent=-1
    )
    coordinates = _coordinates()
    direction = np.array(
        [[0.3, -0.2, 0.1], [-0.4, 0.5, 0.2], [0.1, -0.1, -0.3]],
        dtype=np.float64,
    )
    tangent = (
        execute(
            program.coordinate_jvp().program,
            {"coordinates": coordinates, "d_coordinates": direction},
        )
        .outputs["d_energy"]
        .item()
    )
    step = 2e-6
    finite_difference = (
        _energy(program, coordinates + step * direction)
        - _energy(program, coordinates - step * direction)
    ) / (2 * step)
    assert tangent == pytest.approx(finite_difference, rel=3e-8, abs=3e-9)

    translation = np.ones_like(coordinates)
    translated = (
        execute(
            program.coordinate_jvp().program,
            {"coordinates": coordinates, "d_coordinates": translation},
        )
        .outputs["d_energy"]
        .item()
    )
    assert translated == pytest.approx(0.0, abs=3e-14)


def test_generated_coordinate_vjp_matches_finite_difference_and_newton_sum() -> None:
    program = inverse_power_program(
        _geometry(),
        PairTopology.complete(3),
        ("3/2", "-2/3", "5/4"),
        exponent=-2,
        parameter_identity="qualification-v1",
    )
    coordinates = _coordinates()
    reverse = program.coordinate_vjp().program
    gradient = execute(
        reverse,
        {"coordinates": coordinates, "bar_energy": np.array(1.0)},
    ).outputs["bar_coordinates"]
    step = 2e-6
    finite_difference = np.empty_like(coordinates)
    for atom in range(coordinates.shape[0]):
        for axis in range(3):
            plus, minus = coordinates.copy(), coordinates.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            finite_difference[atom, axis] = (
                _energy(program, plus) - _energy(program, minus)
            ) / (2 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=3e-8, atol=3e-9)
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, rtol=0, atol=3e-14)


def test_compiler_identity_invalidates_cutoff_topology_and_parameter_contracts() -> (
    None
):
    geometry = _geometry()
    topology = PairTopology.complete(3, cutoff=PairCutoff(8.0, 6.0))
    baseline = inverse_power_program(
        geometry, topology, 1, parameter_identity="parameter-table-a"
    )
    cutoff_changed = inverse_power_program(
        geometry,
        PairTopology.complete(3, cutoff=PairCutoff(9.0, 6.0)),
        1,
        parameter_identity="parameter-table-a",
    )
    parameter_changed = inverse_power_program(
        geometry, topology, 1, parameter_identity="parameter-table-b"
    )
    topology_changed = inverse_power_program(
        geometry,
        PairTopology(3, ((0, 1), (1, 2)), cutoff=topology.cutoff),
        1,
        parameter_identity="parameter-table-a",
    )

    assert cutoff_changed.program.logical_hash == baseline.program.logical_hash
    assert parameter_changed.program.logical_hash == baseline.program.logical_hash
    assert topology_changed.program.logical_hash != baseline.program.logical_hash
    assert (
        len(
            {
                baseline.identity,
                cutoff_changed.identity,
                parameter_changed.identity,
                topology_changed.identity,
            }
        )
        == 4
    )
    baseline.validate_execution_identity(baseline.identity)
    with pytest.raises(ValueError, match="stale pair compiler execution state"):
        baseline.validate_execution_identity(cutoff_changed.identity)


def test_primal_and_generated_reverse_lower_through_existing_cuda_tensorir() -> None:
    from vibeqc_compiler.common.cuda_target import CUDA_TARGETS
    from vibeqc_compiler.tensor.cuda_emit import emit_cuda
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    pair_program = inverse_power_program(
        _geometry(),
        PairTopology.complete(3),
        (1, 2, 3),
        exponent=-1,
        parameter_identity="cuda-qualification-v1",
    )
    target = CUDA_TARGETS["sm_80"]
    primal_plan = plan_cuda(pair_program.program, target)
    reverse_plan = plan_cuda(pair_program.coordinate_vjp().program, target)
    primal_source = emit_cuda(primal_plan)
    reverse_source = emit_cuda(reverse_plan)
    assert "tensor_create" in primal_source and "tensor_run" in primal_source
    assert "tensor_create" in reverse_source and "tensor_run" in reverse_source
    assert primal_plan.program.logical_hash == pair_program.program.logical_hash
    assert reverse_plan.program.provenance["mode"] == "vjp"

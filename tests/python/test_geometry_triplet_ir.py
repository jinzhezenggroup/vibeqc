"""Generic TripletIR/AngleIR qualification for issue #853 Stage A."""

import typing

import numpy as np
import pytest
from vibeqc_compiler.geometry.ir import GeometryIR
from vibeqc_compiler.geometry.triplet import (
    TRIPLET_CONVENTION,
    TRIPLET_OWNERSHIP,
    TripletTopology,
    cosine_angle_program,
    lower_triplet_geometry,
    triplet_to_system,
)
from vibeqc_compiler.tensor import Program, TensorSpec, execute, input_tensor


def _geometry() -> GeometryIR:
    return GeometryIR((9, 6, 1, 8), parameter_identity="triplet-elements-v1")


def _coordinates() -> np.ndarray:
    return np.array(
        [
            [1.2, -0.3, 0.5],
            [0.1, 0.2, -0.4],
            [-0.8, 1.4, 0.7],
            [0.6, -1.1, 1.3],
        ],
        dtype=np.float64,
    )


def _energy(program: typing.Any, coordinates: np.ndarray) -> float:
    return float(
        execute(
            program.program,
            {program.geometry.coordinate_name: coordinates},
        ).outputs["energy"]
    )


def test_triplet_topology_is_role_ordered_unique_and_deterministic() -> None:
    topology = TripletTopology(4, ((0, 1, 2), (3, 1, 2)))
    assert topology.triplets == ((0, 1, 2), (3, 1, 2))
    assert topology.owners == (1, 1)
    assert topology.to_payload()["convention"] == TRIPLET_CONVENTION
    assert topology.to_payload()["ownership"] == TRIPLET_OWNERSHIP
    assert topology.identity == TripletTopology(4, topology.triplets).identity

    for triplets in (
        ((0, 1, 1),),
        ((0, 1, 4),),
        ((0, 1, 2), (0, 1, 2)),
        ((3, 1, 2), (0, 1, 2)),
    ):
        with pytest.raises(ValueError):
            TripletTopology(4, triplets)


def test_role_order_is_scientific_semantics_not_unordered_canonicalization() -> None:
    forward = TripletTopology(4, ((0, 1, 2),))
    reversed_roles = TripletTopology(4, ((2, 1, 0),))
    assert forward.identity != reversed_roles.identity
    assert forward.owners == reversed_roles.owners == (1,)


def test_triplet_lowering_exposes_vectors_distances_dot_and_cosine() -> None:
    topology = TripletTopology(4, ((0, 1, 2), (3, 1, 2)))
    context = lower_triplet_geometry(_geometry(), topology)
    result = execute(
        Program(
            {
                "first_vector": context.first_vector,
                "third_vector": context.third_vector,
                "first_distance": context.first_distance,
                "third_distance": context.third_distance,
                "dot": context.dot,
                "cosine": context.cosine,
            }
        ),
        {"coordinates": _coordinates()},
    ).outputs

    coordinates = _coordinates()
    first = np.array([coordinates[0] - coordinates[1], coordinates[3] - coordinates[1]])
    third = np.array([coordinates[2] - coordinates[1], coordinates[2] - coordinates[1]])
    dot = np.einsum("tc,tc->t", first, third)
    expected_cosine = dot / (
        np.linalg.norm(first, axis=1) * np.linalg.norm(third, axis=1)
    )
    np.testing.assert_allclose(result["first_vector"], first, rtol=0, atol=0)
    np.testing.assert_allclose(result["third_vector"], third, rtol=0, atol=0)
    np.testing.assert_allclose(result["first_distance"], np.linalg.norm(first, axis=1))
    np.testing.assert_allclose(result["third_distance"], np.linalg.norm(third, axis=1))
    np.testing.assert_allclose(result["dot"], dot)
    np.testing.assert_allclose(
        result["cosine"], expected_cosine, rtol=2e-15, atol=2e-15
    )


def test_triplet_system_reduction_reuses_shared_tensorir() -> None:
    context = lower_triplet_geometry(
        _geometry(), TripletTopology(4, ((0, 1, 2), (3, 1, 2)))
    )
    values = input_tensor(
        "q", TensorSpec((context.triplet_index,), dtype="float64", role="input")
    )
    program = Program({"system": triplet_to_system(values, context)})
    assert all(node.op != "einsum" for node in program.live_nodes)
    assert execute(program, {"q": np.array([1.25, -0.5])}).outputs[
        "system"
    ] == pytest.approx(0.75)


def test_cosine_angle_energy_is_translation_and_rotation_invariant() -> None:
    topology = TripletTopology(4, ((0, 1, 2), (3, 1, 2)))
    program = cosine_angle_program(_geometry(), topology, ("3/2", "-2/3"))
    coordinates = _coordinates()
    reference = _energy(program, coordinates)
    shift = np.array([4.0, -3.0, 2.5])
    assert _energy(program, coordinates + shift) == pytest.approx(reference, abs=2e-15)

    angle = 0.61
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert _energy(program, coordinates @ rotation.T) == pytest.approx(
        reference, abs=3e-15
    )


def test_generated_coordinate_vjp_matches_finite_difference_and_translation() -> None:
    program = cosine_angle_program(
        _geometry(),
        TripletTopology(4, ((0, 1, 2), (3, 1, 2))),
        ("3/2", "-2/3"),
        parameter_identity="triplet-angle-qualification-v1",
    )
    coordinates = _coordinates()
    gradient = execute(
        program.coordinate_vjp().program,
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
    np.testing.assert_allclose(gradient, finite_difference, rtol=4e-8, atol=4e-9)
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, rtol=0, atol=2e-14)


def test_triplet_identity_invalidates_topology_parameter_and_role_changes() -> None:
    geometry = _geometry()
    baseline = cosine_angle_program(
        geometry,
        TripletTopology(4, ((0, 1, 2),)),
        1,
        parameter_identity="angle-table-a",
    )
    topology_changed = cosine_angle_program(
        geometry,
        TripletTopology(4, ((0, 1, 3),)),
        1,
        parameter_identity="angle-table-a",
    )
    roles_changed = cosine_angle_program(
        geometry,
        TripletTopology(4, ((2, 1, 0),)),
        1,
        parameter_identity="angle-table-a",
    )
    parameter_changed = cosine_angle_program(
        geometry,
        TripletTopology(4, ((0, 1, 2),)),
        1,
        parameter_identity="angle-table-b",
    )
    assert (
        len(
            {
                baseline.identity,
                topology_changed.identity,
                roles_changed.identity,
                parameter_changed.identity,
            }
        )
        == 4
    )
    baseline.validate_execution_identity(baseline.identity)
    with pytest.raises(ValueError, match="stale triplet compiler execution state"):
        baseline.validate_execution_identity(parameter_changed.identity)


def test_primal_and_generated_reverse_lower_through_shared_cuda_tensorir() -> None:
    from vibeqc_compiler.common.cuda_target import CUDA_TARGETS
    from vibeqc_compiler.tensor.cuda_emit import emit_cuda
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    triplet_program = cosine_angle_program(
        _geometry(),
        TripletTopology(4, ((0, 1, 2), (3, 1, 2))),
        (1, 2),
        parameter_identity="triplet-cuda-qualification-v1",
    )
    target = CUDA_TARGETS["sm_80"]
    primal_plan = plan_cuda(triplet_program.program, target)
    reverse_plan = plan_cuda(triplet_program.coordinate_vjp().program, target)
    primal_source = emit_cuda(primal_plan)
    reverse_source = emit_cuda(reverse_plan)
    assert "tensor_create" in primal_source and "tensor_run" in primal_source
    assert "tensor_create" in reverse_source and "tensor_run" in reverse_source
    assert primal_plan.program.logical_hash == triplet_program.program.logical_hash
    assert reverse_plan.program.provenance["mode"] == "vjp"

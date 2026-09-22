"""Pinned GFN1 halogen TripletIR and generated-VJP qualification (#853)."""

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc_compiler.geometry import (
    GFN1_HALOGEN_CUTOFF_BOHR,
    GFN1_HALOGEN_PARAMETER_IDENTITY,
    GFN1_HALOGEN_VERSION,
    build_gfn1_halogen_topology,
    gfn1_element_parameters,
    gfn1_geometry,
)
from vibeqc_compiler.geometry.triplet import TripletTopology
from vibeqc_compiler.method import (
    build_gfn1_halogen_program,
    compile_gfn1_halogen,
    resolve_xtb_method,
)
from vibeqc_compiler.tensor import execute

TBLITE_CASES = (
    (
        "Br2-NH3",
        (35, 35, 7, 1, 1, 1),
        np.array(
            [
                [0.0, 0.0, 3.114952513],
                [0.0, 0.0, -1.256718806],
                [0.0, 0.0, -6.302011301],
                [0.0, 1.787127097, -6.9747084],
                [-1.547696925, -0.893562604, -6.9747084],
                [1.547696925, -0.893562604, -6.9747084],
            ],
            dtype=np.float64,
        ),
        2.4763110097465683e-3,
        2,
    ),
    (
        "Br2-OCH2",
        (35, 35, 8, 6, 1, 1),
        np.array(
            [
                [-1.785333747, -3.126082999, 0.0],
                [0.0, 0.816042264, 0.0],
                [2.658286999, 5.297075806, 0.0],
                [4.885971586, 4.861161373, 0.0],
                [5.615509753, 2.908222159, 0.0],
                [6.289076126, 6.399636435, 0.0],
            ],
            dtype=np.float64,
        ),
        -6.7587305781592112e-4,
        2,
    ),
    (
        "FI-NCH",
        (9, 53, 7, 6, 1),
        np.array(
            [
                [0.0, 0.0, 4.376378627],
                [0.0, 0.0, 0.699818447],
                [0.0, 0.0, -4.241811239],
                [0.0, 0.0, -6.395206917],
                [0.0, 0.0, -8.413872692],
            ],
            dtype=np.float64,
        ),
        1.1857937381795408e-2,
        1,
    ),
)


def _gfn1_method() -> object:
    return resolve_xtb_method(
        "GFN1-xTB", requested_products=("energy", "nuclear-gradient")
    )


def _tblite_oracle(elements: tuple[int, ...], coordinates: np.ndarray) -> float:
    """Independent scalar transcription of pinned tblite fa8a4416 halogen.f90."""

    energy = 0.0
    donors = {17, 35, 53, 85}
    acceptors = {7, 8, 15, 16}
    for donor, donor_element in enumerate(elements):
        if donor_element not in donors:
            continue
        neighbor = None
        neighbor_distance = np.inf
        for candidate in range(len(elements)):
            distance = float(
                np.linalg.norm(coordinates[candidate] - coordinates[donor])
            )
            if distance > 0.0 and distance < neighbor_distance:
                neighbor = candidate
                neighbor_distance = distance
        for acceptor, acceptor_element in enumerate(elements):
            if acceptor_element not in acceptors:
                continue
            donor_acceptor = coordinates[acceptor] - coordinates[donor]
            distance = float(np.linalg.norm(donor_acceptor))
            if distance > GFN1_HALOGEN_CUTOFF_BOHR:
                continue
            if distance == 0.0 or neighbor is None:
                raise ValueError("pinned tblite halogen expression is undefined")
            if neighbor == acceptor:
                continue
            donor_neighbor = coordinates[neighbor] - coordinates[donor]
            cosine = float(
                np.dot(donor_neighbor, donor_acceptor)
                / (np.linalg.norm(donor_neighbor) * distance)
            )
            angular = (0.5 - 0.5 * cosine) ** 6
            donor_parameter = gfn1_element_parameters(donor_element)
            acceptor_parameter = gfn1_element_parameters(acceptor_element)
            radius = 1.3 * (
                donor_parameter.atomic_radius_bohr
                + acceptor_parameter.atomic_radius_bohr
            )
            ratio_six = (radius / distance) ** 6
            radial = (ratio_six * ratio_six - 0.44 * ratio_six) / (
                1.0 + ratio_six * ratio_six
            )
            energy += angular * donor_parameter.xbond * radial
    return energy


def _gradient(program: object, coordinates: np.ndarray) -> np.ndarray:
    return execute(
        program.coordinate_vjp().program,
        {"coordinates": coordinates, "bar_energy": np.array(1.0)},
    ).outputs["bar_coordinates"]


@pytest.mark.parametrize(
    ("_name", "elements", "coordinates", "expected", "triplet_count"),
    TBLITE_CASES,
)
def test_gfn1_halogen_matches_pinned_tblite_energy_goldens(
    _name: str,
    elements: tuple[int, ...],
    coordinates: np.ndarray,
    expected: float,
    triplet_count: int,
) -> None:
    compiled = compile_gfn1_halogen(_gfn1_method(), elements, coordinates)
    compiled.validate_coordinates(coordinates)
    actual = (
        execute(compiled.program, {"coordinates": coordinates}).outputs["energy"].item()
    )
    assert len(compiled.topology.triplets) == triplet_count
    assert actual == pytest.approx(expected, rel=0, abs=3e-14)
    assert actual == pytest.approx(_tblite_oracle(elements, coordinates), abs=3e-16)
    assert compiled.parameter_identity == GFN1_HALOGEN_PARAMETER_IDENTITY


def test_generated_vjp_matches_independent_oracle_at_multiple_steps() -> None:
    _name, elements, coordinates, _expected, _count = TBLITE_CASES[1]
    method = resolve_xtb_method(
        "GFN1-xTB", requested_products=("energy", "nuclear-gradient")
    )
    compiled = compile_gfn1_halogen(method, elements, coordinates)
    gradient = _gradient(compiled, coordinates)
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, rtol=0, atol=2e-15)

    for step, tolerance in ((4e-6, 3e-10), (2e-6, 2e-10), (1e-6, 4e-10)):
        finite_difference = np.empty_like(coordinates)
        for atom in range(len(elements)):
            for axis in range(3):
                plus, minus = coordinates.copy(), coordinates.copy()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                finite_difference[atom, axis] = (
                    _tblite_oracle(elements, plus) - _tblite_oracle(elements, minus)
                ) / (2 * step)
        np.testing.assert_allclose(
            gradient, finite_difference, rtol=2e-7, atol=tolerance
        )


def test_topology_preserves_roles_ties_repeats_and_zero_degeneracy() -> None:
    geometry = gfn1_geometry((6, 1, 35, 7, 8))
    coordinates = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [4.0, 0.2, 0.0],
            [5.0, -0.3, 0.1],
        ]
    )
    topology = build_gfn1_halogen_topology(geometry, coordinates)
    assert topology.triplets == ((0, 2, 3), (0, 2, 4))

    repeated_pair = TripletTopology(5, ((0, 2, 3), (1, 2, 3)))
    with pytest.raises(ValueError, match="repeats a donor/acceptor pair"):
        build_gfn1_halogen_program(_gfn1_method(), geometry, repeated_pair)

    invalid_roles = TripletTopology(5, ((0, 1, 3),))
    with pytest.raises(ValueError, match="center must be a halogen donor"):
        build_gfn1_halogen_program(_gfn1_method(), geometry, invalid_roles)

    degenerate_coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    degenerate = compile_gfn1_halogen(_gfn1_method(), (35, 8), degenerate_coordinates)
    assert degenerate.topology.triplets == ()
    assert (
        execute(degenerate.program, {"coordinates": degenerate_coordinates})
        .outputs["energy"]
        .item()
        == 0.0
    )


def test_topology_cutoff_and_undefined_coincidence_boundaries() -> None:
    geometry = gfn1_geometry((6, 35, 8))
    exact = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [GFN1_HALOGEN_CUTOFF_BOHR, 0.0, 0.0]]
    )
    assert build_gfn1_halogen_topology(geometry, exact).triplets == ((0, 1, 2),)

    above = exact.copy()
    above[2, 0] = np.nextafter(GFN1_HALOGEN_CUTOFF_BOHR, np.inf)
    assert build_gfn1_halogen_topology(geometry, above).triplets == ()

    coincident = exact.copy()
    coincident[2] = coincident[1]
    with pytest.raises(ValueError, match="coincidence is undefined"):
        build_gfn1_halogen_topology(geometry, coincident)


def test_changed_cutoff_and_nearest_neighbor_invalidate_topology() -> None:
    elements = (6, 1, 35, 8)
    coordinates = np.array(
        [[-1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.2, 0.0]]
    )
    compiled = compile_gfn1_halogen(_gfn1_method(), elements, coordinates)
    compiled.validate_coordinates(coordinates)

    neighbor_changed = coordinates.copy()
    neighbor_changed[1, 0] = 0.5
    with pytest.raises(ValueError, match="stale GFN1 halogen topology"):
        compiled.validate_coordinates(neighbor_changed)

    cutoff_changed = coordinates.copy()
    cutoff_changed[3] = (GFN1_HALOGEN_CUTOFF_BOHR + 0.1, 0.0, 0.0)
    with pytest.raises(ValueError, match="stale GFN1 halogen topology"):
        compiled.validate_coordinates(cutoff_changed)


def test_rotation_translation_and_permutation_covariance() -> None:
    _name, elements, coordinates, _expected, _count = TBLITE_CASES[1]
    compiled = compile_gfn1_halogen(_gfn1_method(), elements, coordinates)
    energy = (
        execute(compiled.program, {"coordinates": coordinates}).outputs["energy"].item()
    )
    gradient = _gradient(compiled, coordinates)

    angle = 0.47
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = coordinates @ rotation.T + np.array([4.0, -3.0, 2.5])
    rotated = compile_gfn1_halogen(_gfn1_method(), elements, transformed)
    rotated_energy = (
        execute(rotated.program, {"coordinates": transformed}).outputs["energy"].item()
    )
    assert rotated_energy == pytest.approx(energy, rel=0, abs=3e-18)
    np.testing.assert_allclose(
        _gradient(rotated, transformed), gradient @ rotation.T, rtol=2e-14, atol=3e-17
    )

    permutation = np.array([3, 1, 5, 0, 4, 2])
    permuted_elements = tuple(elements[index] for index in permutation)
    permuted_coordinates = coordinates[permutation]
    permuted = compile_gfn1_halogen(
        _gfn1_method(), permuted_elements, permuted_coordinates
    )
    permuted_energy = (
        execute(permuted.program, {"coordinates": permuted_coordinates})
        .outputs["energy"]
        .item()
    )
    restored_gradient = np.empty_like(coordinates)
    restored_gradient[permutation] = _gradient(permuted, permuted_coordinates)
    assert permuted_energy == pytest.approx(energy, rel=0, abs=3e-18)
    np.testing.assert_allclose(restored_gradient, gradient, rtol=2e-14, atol=3e-17)


def test_method_topology_and_parameter_identities_are_bound() -> None:
    _name, elements, coordinates, _expected, _count = TBLITE_CASES[0]
    energy_method = resolve_xtb_method("GFN1-xTB", requested_products=("energy",))
    gradient_method = resolve_xtb_method(
        "GFN1-xTB", requested_products=("energy", "nuclear-gradient")
    )
    energy_program = compile_gfn1_halogen(energy_method, elements, coordinates)
    gradient_program = compile_gfn1_halogen(gradient_method, elements, coordinates)
    assert energy_program.identity != gradient_program.identity
    assert energy_program.program.logical_hash == gradient_program.program.logical_hash
    assert energy_program.geometry_program.triplet_program.parameter_identity == (
        GFN1_HALOGEN_PARAMETER_IDENTITY
    )
    assert energy_program.geometry_program.version == GFN1_HALOGEN_VERSION
    with pytest.raises(ValueError, match="stale GFN1 halogen MethodIR"):
        energy_program.validate_execution_identity(gradient_program.identity)

    changed = coordinates.copy()
    changed[0, 2] = 30.0
    changed_program = compile_gfn1_halogen(energy_method, elements, changed)
    assert changed_program.topology.identity != energy_program.topology.identity
    assert changed_program.identity != energy_program.identity

    with pytest.raises(ValueError, match="canonical GFN1 XtbMethodIR"):
        compile_gfn1_halogen(resolve_xtb_method("GFN2-xTB"), elements, coordinates)

    with pytest.raises(ValueError, match="primitive identity mismatch"):
        replace(energy_program, primitive_identity="0" * 64)


@pytest.mark.parametrize("unreviewed", ["unreviewed-parameters", "0" * 64])
def test_method_binding_rejects_agreeing_noncanonical_geometry_parameters(
    unreviewed: str,
) -> None:
    """Matching nested strings cannot relabel an unqualified parameter set."""
    _name, elements, coordinates, _expected, _count = TBLITE_CASES[0]
    compiled = compile_gfn1_halogen(_gfn1_method(), elements, coordinates)
    geometry = compiled.geometry_program
    assert compiled.parameter_identity == geometry.parameter_identity
    with pytest.raises(ValueError, match="canonical GFN1 halogen parameter"):
        other = replace(
            geometry,
            parameter_identity=unreviewed,
            triplet_program=replace(
                geometry.triplet_program, parameter_identity=unreviewed
            ),
        )
        replace(compiled, geometry_program=other)


def test_method_binding_rejects_structural_impostors_and_unrequested_vjp() -> None:
    _name, elements, coordinates, _expected, _count = TBLITE_CASES[2]
    energy_method = resolve_xtb_method("GFN1-xTB", requested_products=("energy",))
    geometry = gfn1_geometry(elements)
    topology = build_gfn1_halogen_topology(geometry, coordinates)
    fake = SimpleNamespace(
        kind=energy_method.kind,
        model_flavor=energy_method.model_flavor,
        parameter_set=energy_method.parameter_set,
        primitives=energy_method.primitives,
        requested_products=energy_method.requested_products,
        identity=energy_method.identity,
    )
    with pytest.raises(TypeError, match="XtbMethodIR"):
        build_gfn1_halogen_program(fake, geometry, topology)

    energy_only = build_gfn1_halogen_program(energy_method, geometry, topology)
    with pytest.raises(ValueError, match="nuclear-gradient compiler product"):
        energy_only.coordinate_jvp()
    with pytest.raises(ValueError, match="nuclear-gradient compiler product"):
        energy_only.coordinate_vjp()

    # The method-neutral geometry graph still exposes AD for compiler qualification.
    assert energy_only.geometry_program.coordinate_vjp().program is not None


def test_primal_and_generated_vjp_lower_through_shared_cuda_tensorir() -> None:
    from vibeqc_compiler.common.cuda_target import CUDA_TARGETS
    from vibeqc_compiler.tensor.cuda_emit import emit_cuda
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    _name, elements, coordinates, _expected, _count = TBLITE_CASES[2]
    compiled = compile_gfn1_halogen(_gfn1_method(), elements, coordinates)
    target = CUDA_TARGETS["sm_80"]
    for program in (compiled.program, compiled.coordinate_vjp().program):
        source = emit_cuda(plan_cuda(program, target))
        assert "tensor_create" in source and "tensor_run" in source


@pytest.mark.skipif(
    os.environ.get("VIBEQC_GFN1_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
def test_gfn1_halogen_energy_and_generated_vjp_execute_on_cuda(
    tmp_path: Path,
) -> None:
    from vibeqc.profiles import find_nvcc
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.cuda_target import cuda_target_info
    from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    nvcc = find_nvcc()
    if nvcc is None:
        pytest.fail("VIBEQC_GFN1_CUDA_TEST requires a CUDA compiler")
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_90"))
    )
    _name, elements, coordinates, expected, _count = TBLITE_CASES[1]
    compiled = compile_gfn1_halogen(_gfn1_method(), elements, coordinates)
    jobs = (
        (compiled.program, {"coordinates": coordinates}),
        (
            compiled.coordinate_vjp().program,
            {"coordinates": coordinates, "bar_energy": np.array(1.0)},
        ),
    )
    references = tuple(execute(program, feeds).outputs for program, feeds in jobs)
    actual = []
    for index, (program, feeds) in enumerate(jobs):
        plan = plan_cuda(program, compiler.target)
        cache = tmp_path / f"gfn1-halogen-{index}"
        cache.mkdir()
        with PreparedCuda(plan, compile_cuda(plan, compiler, cache)) as prepared:
            actual.append(prepared.execute(feeds).outputs)
    assert actual[0]["energy"].item() == pytest.approx(expected, rel=0, abs=3e-14)
    np.testing.assert_allclose(
        actual[0]["energy"], references[0]["energy"], rtol=0, atol=3e-15
    )
    np.testing.assert_allclose(
        actual[1]["bar_coordinates"],
        references[1]["bar_coordinates"],
        rtol=2e-12,
        atol=2e-14,
    )

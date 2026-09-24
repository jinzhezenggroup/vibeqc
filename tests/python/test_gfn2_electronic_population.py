"""Population, mixed-spin, and CUDA qualification for GFN2 TensorIR (#505)."""

import os
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.method import (
    Gfn2ElectronicTopology,
    build_gfn2_electronic_program,
    build_gfn2_mixed_electronic_program,
    build_gfn2_population_program,
)
from vibeqc_compiler.tensor import dot_test, execute


def _h2_topology() -> Gfn2ElectronicTopology:
    return Gfn2ElectronicTopology(
        system_atom_offsets=(0, 2),
        system_shell_offsets=(0, 2),
        system_orbital_offsets=(0, 2),
        orbital_to_shell=(0, 1),
        orbital_to_atom=(0, 1),
    )


def _two_h2_topology() -> Gfn2ElectronicTopology:
    return Gfn2ElectronicTopology(
        system_atom_offsets=(0, 2, 4),
        system_shell_offsets=(0, 2, 4),
        system_orbital_offsets=(0, 2, 4),
        orbital_to_shell=(0, 1, 2, 3),
        orbital_to_atom=(0, 1, 2, 3),
    )


def test_population_matches_pinned_vibeqc_xtb_two_ao_fixture() -> None:
    """Freeze tests/mulliken_test.cpp::test_two_ao_population_fixture."""

    compiled = build_gfn2_population_program(
        "GFN2-xTB",
        _h2_topology(),
        spin_channels=(1,),
        reference_shell_occupations=(1.0, 1.0),
    )
    overlap = np.array([1.0, 0.2, 0.3, 1.0])
    density = np.array([0.8, 0.1, 0.4, 0.6])
    dipole = np.vstack(
        [np.array([1.0, 2.0, 3.0, 4.0]) * scale for scale in (1.0, 2.0, 3.0)]
    )
    quadrupole = np.vstack(
        [
            np.array([0.5, 1.5, 2.5, 3.5]) * scale
            for scale in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        ]
    )
    h0 = np.array([0.1, -0.2, 0.3, 0.4])
    out = execute(
        compiled.program,
        {
            "density": density,
            "h0": h0,
            "overlap": overlap,
            "dipole_integrals": dipole,
            "quadrupole_integrals": quadrupole,
        },
    ).outputs
    np.testing.assert_allclose(out["shell_charge"], [0.08, 0.38], rtol=0, atol=2e-15)
    np.testing.assert_allclose(out["atomic_charge"], [0.08, 0.38], rtol=0, atol=2e-15)
    np.testing.assert_array_equal(out["shell_magnetization"], [0.0, 0.0])
    np.testing.assert_array_equal(out["atomic_magnetization"], [0.0, 0.0])
    np.testing.assert_allclose(
        out["atomic_dipole"],
        [[-2.0, -4.0, -6.0], [-2.6, -5.2, -7.8]],
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        out["atomic_quadrupole"],
        [
            [-1.4 * scale for scale in (1, 2, 3, 4, 5, 6)],
            [-2.25 * scale for scale in (1, 2, 3, 4, 5, 6)],
        ],
        rtol=0,
        atol=3e-15,
    )
    np.testing.assert_array_equal(out["atomic_dipole_magnetization"], np.zeros((2, 3)))
    np.testing.assert_array_equal(
        out["atomic_quadrupole_magnetization"], np.zeros((2, 6))
    )
    np.testing.assert_allclose(out["core_energy"], [0.42], rtol=0, atol=2e-15)


def test_population_supports_heterogeneous_restricted_unrestricted_batch() -> None:
    topology = _two_h2_topology()
    compiled = build_gfn2_population_program(
        "GFN2-xTB",
        topology,
        spin_channels=(1, 2),
        reference_shell_occupations=(1.0, 1.0, 1.0, 1.0),
    )
    # One restricted H2 matrix followed by alpha/beta matrices for another H2.
    density = np.array(
        [
            0.2,
            0.0,
            0.0,
            0.4,
            0.3,
            0.0,
            0.0,
            0.5,
            0.1,
            0.0,
            0.0,
            0.2,
        ]
    )
    overlap = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    dipole = np.zeros((3, 8))
    quadrupole = np.zeros((6, 8))
    for component in range(3):
        dipole[component, (0, 3, 4, 7)] = component + 1.0
    for component in range(6):
        quadrupole[component, (0, 3, 4, 7)] = 0.5 * (component + 1.0)
    h0 = np.array([0.2, 0.0, 0.0, 0.4, -0.3, 0.0, 0.0, 0.6])
    out = execute(
        compiled.program,
        {
            "density": density,
            "h0": h0,
            "overlap": overlap,
            "dipole_integrals": dipole,
            "quadrupole_integrals": quadrupole,
        },
    ).outputs
    np.testing.assert_allclose(out["shell_charge"], [0.8, 0.6, 0.6, 0.3], atol=2e-15)
    np.testing.assert_allclose(
        out["shell_magnetization"], [0.0, 0.0, -0.2, -0.3], atol=2e-15
    )
    np.testing.assert_allclose(out["atomic_charge"], out["shell_charge"], atol=0)
    np.testing.assert_allclose(
        out["atomic_magnetization"], out["shell_magnetization"], atol=0
    )
    for component in range(3):
        scale = component + 1.0
        np.testing.assert_allclose(
            out["atomic_dipole"][:, component],
            [-0.2 * scale, -0.4 * scale, -0.4 * scale, -0.7 * scale],
            atol=2e-15,
        )
        np.testing.assert_allclose(
            out["atomic_dipole_magnetization"][:, component],
            [0.0, 0.0, -0.2 * scale, -0.3 * scale],
            atol=2e-15,
        )
    for component in range(6):
        scale = 0.5 * (component + 1.0)
        np.testing.assert_allclose(
            out["atomic_quadrupole"][:, component],
            [-0.2 * scale, -0.4 * scale, -0.4 * scale, -0.7 * scale],
            atol=2e-15,
        )
        np.testing.assert_allclose(
            out["atomic_quadrupole_magnetization"][:, component],
            [0.0, 0.0, -0.2 * scale, -0.3 * scale],
            atol=2e-15,
        )
    np.testing.assert_allclose(out["core_energy"], [0.2, 0.30], atol=2e-15)


def _mixed_feeds(topology: Gfn2ElectronicTopology) -> dict[str, np.ndarray]:
    matrices = topology.matrix_count
    return {
        "h0": np.linspace(-0.2, 0.3, matrices),
        "overlap": np.linspace(0.4, 0.8, matrices),
        "dipole_integrals": np.linspace(-0.1, 0.2, 3 * matrices).reshape(3, matrices),
        "quadrupole_integrals": np.linspace(-0.07, 0.11, 6 * matrices).reshape(
            6, matrices
        ),
        "shell_scalar_potential": np.array([0.13, -0.09, 0.04, 0.17]),
        "atomic_dipole_potential": np.linspace(
            -0.08, 0.12, topology.atom_count * 3
        ).reshape(topology.atom_count, 3),
        "atomic_quadrupole_potential": np.linspace(
            -0.05, 0.09, topology.atom_count * 6
        ).reshape(topology.atom_count, 6),
        "shell_magnetization_potential": np.array([0.07, -0.03, 0.11, -0.06]),
        "atomic_dipole_magnetization_potential": np.linspace(
            0.06, -0.04, topology.atom_count * 3
        ).reshape(topology.atom_count, 3),
        "atomic_quadrupole_magnetization_potential": np.linspace(
            0.03, -0.02, topology.atom_count * 6
        ).reshape(topology.atom_count, 6),
    }


def test_mixed_spin_hamiltonian_matches_existing_channel_graphs() -> None:
    topology = _two_h2_topology()
    feeds = _mixed_feeds(topology)
    mixed = build_gfn2_mixed_electronic_program(
        "GFN2-xTB", topology, spin_channels=(1, 2)
    )
    restricted = build_gfn2_electronic_program("GFN2-xTB", topology)
    unrestricted = build_gfn2_electronic_program(
        "GFN2-xTB", topology, reference="unrestricted"
    )
    actual = execute(mixed.program, feeds).outputs["hamiltonian"]
    restricted_h = execute(restricted.program, feeds).outputs["hamiltonian"]
    unrestricted_h = execute(unrestricted.program, feeds).outputs
    np.testing.assert_allclose(actual[:4], restricted_h[:4], rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        actual[4:8], unrestricted_h["hamiltonian_alpha"][4:8], rtol=0, atol=2e-15
    )
    np.testing.assert_allclose(
        actual[8:12], unrestricted_h["hamiltonian_beta"][4:8], rtol=0, atol=2e-15
    )
    assert mixed.identity != restricted.identity
    assert mixed.spin_layout.channels == (1, 2)


def test_mixed_spin_generated_sdq_adjoint_passes_dot_test() -> None:
    topology = _two_h2_topology()
    compiled = build_gfn2_mixed_electronic_program(
        "GFN2-xTB", topology, spin_channels=(1, 2)
    )
    feeds = _mixed_feeds(topology)
    matrices = topology.matrix_count
    tangents = {
        "overlap": np.linspace(-0.02, 0.03, matrices),
        "dipole_integrals": np.linspace(-0.01, 0.025, 3 * matrices).reshape(
            3, matrices
        ),
        "quadrupole_integrals": np.linspace(0.02, -0.015, 6 * matrices).reshape(
            6, matrices
        ),
    }
    cotangents = {"hamiltonian": np.linspace(-0.3, 0.5, 12)}
    assert dot_test(compiled.program, feeds, tangents, cotangents).passed
    generated = compiled.integral_vjp().program
    assert set(generated.outputs) == {
        "bar_overlap",
        "bar_dipole_integrals",
        "bar_quadrupole_integrals",
    }


def test_new_population_and_mixed_graphs_lower_through_cuda() -> None:
    from vibeqc_compiler.common.cuda_target import CUDA_TARGETS
    from vibeqc_compiler.tensor.cuda_emit import emit_cuda
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    topology = _two_h2_topology()
    population = build_gfn2_population_program(
        "GFN2-xTB",
        topology,
        spin_channels=(1, 2),
        reference_shell_occupations=(1.0, 1.0, 1.0, 1.0),
    )
    mixed = build_gfn2_mixed_electronic_program(
        "GFN2-xTB", topology, spin_channels=(1, 2)
    )
    for program in (population.program, mixed.program, mixed.integral_vjp().program):
        source = emit_cuda(plan_cuda(program, CUDA_TARGETS["sm_80"]))
        assert "tensor_create" in source
        assert "tensor_run" in source


@pytest.mark.skipif(
    os.environ.get("VIBEQC_GFN2_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
def test_new_population_and_mixed_graphs_execute_on_cuda(tmp_path: Path) -> None:
    from vibeqc.profiles import find_nvcc
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.cuda_target import cuda_target_info
    from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    nvcc = find_nvcc()
    if nvcc is None:
        pytest.fail("VIBEQC_GFN2_CUDA_TEST requires a CUDA compiler")
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )
    topology = _two_h2_topology()
    population = build_gfn2_population_program(
        "GFN2-xTB",
        topology,
        spin_channels=(1, 2),
        reference_shell_occupations=(1.0, 1.0, 1.0, 1.0),
    )
    population_feeds = {
        "density": np.linspace(0.02, 0.24, 12),
        "h0": np.linspace(-0.2, 0.3, 8),
        "overlap": np.linspace(0.4, 0.8, 8),
        "dipole_integrals": np.linspace(-0.1, 0.2, 24).reshape(3, 8),
        "quadrupole_integrals": np.linspace(-0.07, 0.11, 48).reshape(6, 8),
    }
    mixed = build_gfn2_mixed_electronic_program(
        "GFN2-xTB", topology, spin_channels=(1, 2)
    )
    mixed_feeds = _mixed_feeds(topology)
    programs = ((population.program, population_feeds), (mixed.program, mixed_feeds))
    for index, (program, feeds) in enumerate(programs):
        plan = plan_cuda(program, compiler.target)
        expected = execute(program, feeds).outputs
        cache = tmp_path / f"gfn2-505-{index}"
        cache.mkdir()
        with PreparedCuda(plan, compile_cuda(plan, compiler, cache)) as prepared:
            actual = prepared.execute(feeds).outputs
        assert actual.keys() == expected.keys()
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=3e-14)

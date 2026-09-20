"""Qualification of the fixed-state GFN2 electronic TensorIR slice (#505)."""

import numpy as np
import pytest
from vibeqc_compiler.method import (
    Gfn2ElectronicTopology,
    build_gfn2_electronic_program,
)
from vibeqc_compiler.tensor import dot_test, execute, vjp

# Frozen from xTBloom tests/cuda_hamiltonian_test.cu::make_case(2) and
# evaluate_cpu at revision 2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3.
XTBLOOM_EXPECTED = np.array(
    [
        -0.31025400000000003,
        -0.062346000000000026,
        0.09920149999999994,
        -0.05505300000000003,
        0.17932699999999996,
        -0.11390100000000002,
        0.0567435,
        -0.09380450000000001,
        -0.010296000000000027,
        -0.2777985,
        -0.037576000000000026,
        -0.228678,
        -0.03948550000000001,
        0.18022299999999997,
        -0.12467550000000001,
        0.1550525,
        -0.01819649999999999,
        0.15294699999999994,
        -0.12467800000000002,
        0.14692599999999997,
        -0.16708750000000003,
        0.05805399999999998,
        -0.2181105,
        0.02673349999999997,
        -0.20221350000000002,
        0.010326999999999989,
        0.16851449999999996,
        -0.06308750000000003,
        0.19404,
        -0.03294200000000003,
        0.07858149999999994,
        -0.18383500000000003,
        0.10787349999999997,
        -0.17890100000000003,
        0.011222999999999983,
        -0.21494600000000003,
        0.07105799999999995,
        -0.238973,
        -0.06808400000000003,
        0.16643249999999993,
        -0.04006100000000003,
        0.09574349999999998,
        -0.1896755,
        0.09388949999999999,
        -0.19441850000000002,
        0.03591599999999995,
        -0.202672,
        -0.04723850000000002,
        -0.23076549999999998,
        0.04919550000000001,
        0.1940525,
        -0.03826650000000001,
        0.12816499999999997,
        -0.10656750000000002,
        0.056761499999999965,
        -0.185816,
        -0.03138450000000004,
        -0.244296,
        0.12480350000000003,
        -0.1632135,
        0.04287349999999997,
        -0.209061,
        -0.022765500000000008,
        0.07261549999999994,
        -0.139563,
    ],
    dtype=np.float64,
)


def _topology() -> Gfn2ElectronicTopology:
    return Gfn2ElectronicTopology(
        system_atom_offsets=(0, 1, 3),
        system_shell_offsets=(0, 1, 4),
        system_orbital_offsets=(0, 1, 9),
        orbital_to_shell=(0, 1, 1, 2, 2, 2, 3, 3, 3),
        orbital_to_atom=(0, 1, 1, 1, 1, 1, 2, 2, 2),
    )


def _restricted_feeds() -> dict[str, np.ndarray]:
    topology = _topology()
    matrices = topology.matrix_count
    shell_scalar = np.array([-0.23, 0.33, -0.15, 0.41])
    dipole_potential = np.empty((topology.atom_count, 3))
    quadrupole_potential = np.empty((topology.atom_count, 6))
    for atom in range(topology.atom_count):
        for component in range(3):
            dipole_potential[atom, component] = (
                0.035 * (1 + (atom * 11 + component * 5) % 17) - 0.16
            )
        for component in range(6):
            quadrupole_potential[atom, component] = (
                0.019 * (1 + (atom * 13 + component * 7) % 23) - 0.21
            )

    h0 = np.empty(matrices)
    overlap = np.empty(matrices)
    dipole = np.empty((3, matrices))
    quadrupole = np.empty((6, matrices))
    for matrix in range(matrices):
        h0[matrix] = 0.013 * (1 + (matrix * 17) % 37) - 0.26
        overlap[matrix] = 0.009 * (1 + (matrix * 19) % 41) - 0.14
        for component in range(3):
            dipole[component, matrix] = (
                0.006 * (1 + (matrix * (component + 3) + component * 29) % 43) - 0.12
            )
        for component in range(6):
            quadrupole[component, matrix] = (
                0.003 * (1 + (matrix * (component + 5) + component * 31) % 47) - 0.07
            )
    return {
        "h0": h0,
        "overlap": overlap,
        "dipole_integrals": dipole,
        "quadrupole_integrals": quadrupole,
        "shell_scalar_potential": shell_scalar,
        "atomic_dipole_potential": dipole_potential,
        "atomic_quadrupole_potential": quadrupole_potential,
    }


def test_ragged_fixed_state_hamiltonian_matches_pinned_xtbloom_reference() -> None:
    compiled = build_gfn2_electronic_program(
        "GFN2-xTB",
        _topology(),
    )
    actual = execute(compiled.program, _restricted_feeds()).outputs["hamiltonian"]
    np.testing.assert_allclose(actual, XTBLOOM_EXPECTED, rtol=0, atol=8e-16)
    assert compiled.topology.matrix_offsets == (0, 1, 65)
    assert compiled.topology.canonical_forward[1] == 1
    assert compiled.topology.canonical_forward[8] == 8
    assert compiled.topology.canonical_forward[9] == 2


def test_generated_sdq_adjoint_matches_reference_vjp_and_dot_product() -> None:
    compiled = build_gfn2_electronic_program(
        "GFN2-xTB",
        _topology(),
    )
    feeds = _restricted_feeds()
    matrices = compiled.topology.matrix_count
    tangents = {
        "overlap": np.linspace(-0.03, 0.05, matrices),
        "dipole_integrals": np.linspace(
            -0.02,
            0.04,
            3 * matrices,
        ).reshape(3, matrices),
        "quadrupole_integrals": np.linspace(
            0.03,
            -0.01,
            6 * matrices,
        ).reshape(6, matrices),
    }
    cotangent = {
        "hamiltonian": np.linspace(-0.4, 0.7, matrices),
    }
    assert dot_test(
        compiled.program,
        feeds,
        tangents,
        cotangent,
    ).passed

    reference = vjp(
        compiled.program,
        feeds,
        cotangent,
    ).input_cotangents
    generated = compiled.integral_vjp().program
    actual = execute(
        generated,
        {
            **feeds,
            "bar_hamiltonian": cotangent["hamiltonian"],
        },
    ).outputs
    for name in (
        "overlap",
        "dipole_integrals",
        "quadrupole_integrals",
    ):
        np.testing.assert_allclose(
            actual[f"bar_{name}"],
            reference[name],
            rtol=0,
            atol=2e-14,
        )


def test_unrestricted_spin_semantics_are_explicit_and_reduce_to_charge_average() -> (
    None
):
    topology = _topology()
    restricted = build_gfn2_electronic_program(
        "GFN2-xTB",
        topology,
        reference="restricted",
    )
    unrestricted = build_gfn2_electronic_program(
        "GFN2-xTB",
        topology,
        reference="unrestricted",
    )
    feeds = _restricted_feeds()
    restricted_h = execute(
        restricted.program,
        feeds,
    ).outputs["hamiltonian"]

    shell_magnetization = np.array([-0.06, 0.02, -0.04, 0.01])
    dipole_magnetization = np.linspace(
        -0.025,
        0.035,
        topology.atom_count * 3,
    ).reshape(topology.atom_count, 3)
    quadrupole_magnetization = np.linspace(
        0.018,
        -0.022,
        topology.atom_count * 6,
    ).reshape(topology.atom_count, 6)
    spin_feeds = {
        **feeds,
        "shell_magnetization_potential": shell_magnetization,
        "atomic_dipole_magnetization_potential": dipole_magnetization,
        "atomic_quadrupole_magnetization_potential": quadrupole_magnetization,
    }
    spin = execute(unrestricted.program, spin_feeds).outputs
    np.testing.assert_allclose(
        0.5 * (spin["hamiltonian_alpha"] + spin["hamiltonian_beta"]),
        restricted_h,
        rtol=0,
        atol=2e-15,
    )
    assert np.max(np.abs(spin["hamiltonian_alpha"] - spin["hamiltonian_beta"])) > 1e-4
    assert restricted.identity != unrestricted.identity


def test_topology_rejects_cross_system_ownership() -> None:
    with pytest.raises(ValueError, match="crosses a system boundary"):
        Gfn2ElectronicTopology(
            system_atom_offsets=(0, 1, 2),
            system_shell_offsets=(0, 1, 2),
            system_orbital_offsets=(0, 1, 2),
            orbital_to_shell=(1, 1),
            orbital_to_atom=(0, 1),
        )


def test_primal_and_generated_sdq_adjoint_lower_through_cuda_tensorir() -> None:
    from vibeqc_compiler.common.cuda_target import CUDA_TARGETS
    from vibeqc_compiler.tensor.cuda_emit import emit_cuda
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    compiled = build_gfn2_electronic_program(
        "GFN2-xTB",
        _topology(),
    )
    programs = (
        compiled.program,
        compiled.integral_vjp().program,
    )
    for program in programs:
        source = emit_cuda(
            plan_cuda(
                program,
                CUDA_TARGETS["sm_80"],
            )
        )
        assert "tensor_create" in source
        assert "tensor_run" in source

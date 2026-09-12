"""Independent libcint ECP matrices and full valence Hamiltonian force gates."""

import ctypes
import json
import os
from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Atom, BasisProvenance, BasisSet, BasisShell, Calculator, ElementBasis
from vibeqc.ecp import ecp_integrals, resolve_ecp
from vibeqc.profiles import canonical_hash


def fixture(*, representation="spherical", d_shell=False, spin=0):
    gto = pytest.importorskip("pyscf.gto")
    # Parameters are read only by this independent test, never by runtime.
    basis = {"Na": gto.basis.load("lanl2dz", "Na"), "H": gto.basis.load("sto-3g", "H")}
    if d_shell:
        basis["Na"] += [[2, [0.35, 1.0]]]
    ecp = {"Na": gto.basis.load_ecp("lanl2dz", "Na")}
    atoms = [("Na", (0.13, -0.21, 0.17)), ("H", (0.43, 0.19, 3.2))]
    mol = gto.M(
        atom=atoms,
        basis=basis,
        ecp=ecp,
        unit="Bohr",
        spin=spin,
        charge=spin,
        cart=representation == "cartesian",
        verbose=0,
    )
    elements = []
    for symbol, z in (("Na", 11), ("H", 1)):
        shells = []
        for shell in basis[symbol]:
            l, rows = shell[0], shell[1:]
            shells.append(
                BasisShell(
                    l,
                    tuple(str(r[0]) for r in rows),
                    tuple(
                        tuple(str(r[c]) for r in rows) for c in range(1, len(rows[0]))
                    ),
                )
            )
        potentials = None
        if symbol in ecp:
            potentials = []
            local_channel = max(ch[0] for ch in ecp[symbol][1]) + 1
            for channel, powers in ecp[symbol][1]:
                terms = [(n, a, c) for n, rows in enumerate(powers) for a, c in rows]
                potentials.append(
                    {
                        "ecp_type": "scalar_ecp",
                        "angular_momentum": [
                            local_channel if channel == -1 else channel
                        ],
                        "r_exponents": [n for n, _, _ in terms],
                        "gaussian_exponents": [str(a) for _, a, _ in terms],
                        "coefficients": [[str(c) for _, _, c in terms]],
                    }
                )
        elements.append(
            ElementBasis(
                z,
                tuple(shells),
                ecp_core_electrons=10 if symbol == "Na" else 0,
                ecp_data=json.dumps(potentials) if potentials else None,
            )
        )
    record = BasisSet(
        "LANL2DZ-Na/STO-3G-H test oracle",
        tuple(elements),
        BasisProvenance(
            "PySCF test-only installed basis library",
            "2.14.0",
            "upstream external test data; not redistributed",
            canonical_hash({"basis": basis, "ecp": ecp}),
        ),
        representation=representation,
    )
    return atoms, record, mol


def reference(mol):
    norms = np.sqrt(mol.intor("int1e_ovlp").diagonal())
    return mol.intor("ECPscalar") / norms[:, None] / norms[None, :]


@pytest.mark.parametrize("representation", ["spherical", "cartesian"])
@pytest.mark.parametrize("d_shell", [False, True])
def test_raw_matrices_libcint_and_quadrature(representation, d_shell):
    atoms, basis, mol = fixture(representation=representation, d_shell=d_shell)
    actual = ecp_integrals(atoms, basis)
    refined = ecp_integrals(atoms, basis, radial_points=224, polar_points=44)
    np.testing.assert_allclose(actual.matrix, reference(mol), atol=2e-9, rtol=1e-9)
    np.testing.assert_allclose(actual.local, refined.local, atol=2e-9, rtol=1e-9)
    np.testing.assert_allclose(
        actual.nonlocal_, refined.nonlocal_, atol=2e-9, rtol=1e-9
    )
    np.testing.assert_allclose(
        actual.local_derivative, refined.local_derivative, atol=2e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        actual.nonlocal_derivative, refined.nonlocal_derivative, atol=2e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        (actual.local_derivative + actual.nonlocal_derivative).sum(axis=0),
        0,
        atol=2e-13,
    )


@pytest.mark.parametrize("step", [2e-4, 7e-5])
def test_all_atom_derivatives_against_independent_libcint(step):
    atoms, basis, mol = fixture(d_shell=True)
    actual = ecp_integrals(atoms, basis)
    analytical = actual.local_derivative + actual.nonlocal_derivative
    xyz = mol.atom_coords()
    for a in range(2):
        for axis in range(3):
            delta = np.zeros_like(xyz)
            delta[a, axis] = step
            plus = mol.copy().set_geom_(xyz + delta, unit="Bohr")
            minus = mol.copy().set_geom_(xyz - delta, unit="Bohr")
            expected = (reference(plus) - reference(minus)) / (2 * step)
            np.testing.assert_allclose(
                analytical[a, axis], expected, atol=2e-7, rtol=2e-6
            )
    weights = np.random.default_rng(171).normal(size=actual.matrix.shape)
    np.testing.assert_allclose(
        actual.contract(weights), np.einsum("axij,ij->ax", analytical, weights)
    )


@pytest.mark.parametrize("spin", [0, 1])
def test_complete_hf_energy_force_and_core_bookkeeping(spin):
    scf = pytest.importorskip("pyscf.scf")
    atoms, basis, mol = fixture(spin=spin)
    method = "uhf" if spin else "rhf"
    oracle = (scf.UHF(mol) if spin else scf.RHF(mol)).run(conv_tol=1e-12)
    expected_force = -oracle.nuc_grad_method().kernel()
    actual = Calculator(method=method, basis=basis, device="cpu").singlepoint(
        atoms, charge=spin, multiplicity=spin + 1
    )
    assert actual.converged
    np.testing.assert_allclose(actual.energy, oracle.e_tot, atol=2e-8, rtol=0)
    np.testing.assert_allclose(actual.forces, expected_force, atol=2e-6, rtol=0)
    assert sum(resolve_ecp(basis, tuple(Atom.from_value(a) for a in atoms))[0]) == 10


def test_parameters_invalidate_identity_and_malformed_channels_fail():
    atoms, basis, _ = fixture()
    potentials = json.loads(basis.by_element[11].ecp_data)
    potentials[0]["coefficients"][0][0] = str(
        float(potentials[0]["coefficients"][0][0]) + 0.1
    )
    changed = replace(
        basis,
        elements=tuple(
            replace(e, ecp_data=json.dumps(potentials)) if e.atomic_number == 11 else e
            for e in basis.elements
        ),
    )
    first = Calculator(basis=basis)
    second = Calculator(basis=changed)
    assert (
        first.basis_metadata(atoms)["orbital"]["mathematical_identity"]
        != second.basis_metadata(atoms)["orbital"]["mathematical_identity"]
    )
    assert (
        first.estimate_resources([atoms]).identity
        != second.estimate_resources([atoms]).identity
    )
    potentials.append(potentials[0])
    elements = tuple(
        replace(e, ecp_data=json.dumps(potentials)) if e.atomic_number == 11 else e
        for e in basis.elements
    )
    bad = replace(basis, elements=elements)
    with pytest.raises(ValueError, match="duplicate"):
        resolve_ecp(bad, tuple(Atom.from_value(a) for a in atoms))


def detached_native(xyz, *, device="cpu"):
    """An ECP atom with no Gaussian shell tests its independent center motion."""
    from vibeqc import _native

    calculator = Calculator(device=device)
    library = calculator._library
    context = ctypes.c_void_p()
    _native.check(
        library,
        library.vibeqc_context_create(
            ctypes.byref(calculator._context_descriptor()), ctypes.byref(context)
        ),
    )
    system = ctypes.c_void_p()
    atoms = (_native.AtomDescriptor * 3)(
        *(_native.AtomDescriptor(z, *r) for z, r in zip((11, 1, 2), xyz))
    )
    shells = (_native.ShellDescriptor * 2)(
        _native.ShellDescriptor(1, 0, 0, 1), _native.ShellDescriptor(2, 1, 1, 1)
    )
    primitives = (_native.PrimitiveDescriptor * 2)(
        _native.PrimitiveDescriptor(0.7, 1.0), _native.PrimitiveDescriptor(1.1, 1.0)
    )
    descriptor = _native.SystemDescriptor(
        ctypes.sizeof(_native.SystemDescriptor),
        _native.ABI_VERSION,
        atoms,
        3,
        shells,
        2,
        primitives,
        2,
        0,
        1,
        _native.BASIS_CARTESIAN,
    )
    cores = (ctypes.c_int32 * 3)(10, 0, 0)
    terms = (_native.EcpTermDescriptor * 3)(
        _native.EcpTermDescriptor(0, -1, 2, 0.8, -2.0),
        _native.EcpTermDescriptor(0, 0, 2, 0.5, 3.0),
        _native.EcpTermDescriptor(0, 1, 2, 0.4, -1.0),
    )
    try:
        _native.check(
            library,
            library.vibeqc_system_create_ecp(
                context, ctypes.byref(descriptor), cores, terms, 3, ctypes.byref(system)
            ),
        )
        result = np.empty((2, 10, 4, 4))
        _native.check(
            library,
            library.vibeqc_system_ecp_integrals(
                context,
                system,
                160,
                32,
                1,
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                result.size,
            ),
        )
        return result
    finally:
        if system:
            library.vibeqc_system_destroy(system)
        library.vibeqc_context_destroy(context)


def detached_reference(xyz):
    gto = pytest.importorskip("pyscf.gto")
    channels = [
        [channel, [[], [], [[exponent, coefficient]]]]
        for channel, exponent, coefficient in (
            (-1, 0.8, -2.0),
            (0, 0.5, 3.0),
            (1, 0.4, -1.0),
        )
    ]
    mol = gto.M(
        atom=list(zip(("Na", "H", "He"), xyz)),
        unit="Bohr",
        verbose=0,
        basis={"H": [[0, [0.7, 1.0]]], "He": [[1, [1.1, 1.0]]]},
        ecp={"Na": [10, channels]},
        cart=True,
    )
    original = mol._ecpbas.copy()
    mol._ecpbas = original[original[:, gto.ANG_OF] == -1]
    local = mol.intor("ECPscalar")
    mol._ecpbas = original[original[:, gto.ANG_OF] != -1]
    nonlocal_ = mol.intor("ECPscalar")
    return np.array([local, nonlocal_])


def test_ecp_and_basis_centers_move_independently():
    xyz = np.array([[0.13, -0.27, 0.32], [0.43, 0.38, 1.12], [-0.31, 0.12, -0.62]])
    actual = detached_native(xyz)
    np.testing.assert_allclose(
        actual[:, 0], detached_reference(xyz), atol=2e-10, rtol=1e-10
    )
    for step in (1e-4, 4e-5):
        for a in range(3):
            for d in range(3):
                delta = np.zeros_like(xyz)
                delta[a, d] = step
                reference = (
                    detached_reference(xyz + delta) - detached_reference(xyz - delta)
                ) / (2 * step)
                np.testing.assert_allclose(
                    actual[:, 1 + a * 3 + d], reference, atol=1e-8, rtol=2e-7
                )
    np.testing.assert_allclose(
        actual[:, 1:].reshape(2, 3, 3, 4, 4).sum(axis=1), 0, atol=2e-13
    )


def test_isolated_valence_atom_and_wrong_core_count():
    scf = pytest.importorskip("pyscf.scf")
    atoms, basis, mol = fixture()
    isolated = mol.copy()
    isolated.spin = 1
    isolated.set_geom_([atoms[0]], unit="Bohr")
    target = scf.UHF(isolated).run(conv_tol=1e-12)
    result = Calculator(method="uhf", basis=basis).singlepoint(
        [atoms[0]], multiplicity=2
    )
    assert result.converged
    np.testing.assert_allclose(result.energy, target.e_tot, atol=2e-8, rtol=0)
    np.testing.assert_allclose(result.forces, 0, atol=2e-10)
    bad = replace(
        basis,
        elements=tuple(
            replace(e, ecp_core_electrons=9) if e.atomic_number == 11 else e
            for e in basis.elements
        ),
    )
    with pytest.raises(RuntimeError, match="invalid argument"):
        Calculator(basis=bad).singlepoint(atoms)


def test_unresolved_quadrature_rejects_full_method():
    atoms, basis, _ = fixture()
    potentials = json.loads(basis.by_element[11].ecp_data)
    for p in potentials:
        p["coefficients"] = [[str(float(c) * 1e8) for c in p["coefficients"][0]]]
    amplified = replace(
        basis,
        elements=tuple(
            replace(e, ecp_data=json.dumps(potentials)) if e.atomic_number == 11 else e
            for e in basis.elements
        ),
    )
    with pytest.raises(RuntimeError, match="quadrature convergence"):
        Calculator(basis=amplified).singlepoint(atoms)


@pytest.mark.skipif(
    os.getenv("VIBEQC_ECP_CUDA_TEST") != "1", reason="requires explicit real CUDA run"
)
@pytest.mark.parametrize("spin", [0, 1])
def test_real_cuda_matrices_complete_hf_and_replay(spin):
    atoms, basis, _ = fixture(spin=spin)
    cpu = ecp_integrals(atoms, basis, charge=spin, multiplicity=spin + 1)
    gpu = ecp_integrals(atoms, basis, charge=spin, multiplicity=spin + 1, device="cuda")
    np.testing.assert_allclose(gpu.local, cpu.local, atol=2e-11, rtol=1e-11)
    np.testing.assert_allclose(gpu.nonlocal_, cpu.nonlocal_, atol=2e-11, rtol=1e-11)
    np.testing.assert_allclose(
        gpu.local_derivative, cpu.local_derivative, atol=2e-10, rtol=1e-10
    )
    np.testing.assert_allclose(
        gpu.nonlocal_derivative, cpu.nonlocal_derivative, atol=2e-10, rtol=1e-10
    )
    calculator = Calculator(method="uhf" if spin else "rhf", basis=basis, device="cuda")
    reference_result = Calculator(
        method="uhf" if spin else "rhf", basis=basis
    ).singlepoint(atoms, charge=spin, multiplicity=spin + 1)
    first = calculator.singlepoint(atoms, charge=spin, multiplicity=spin + 1)
    assert first.executed_backend == "cuda" and first.converged
    np.testing.assert_allclose(first.energy, reference_result.energy, atol=2e-8, rtol=0)
    np.testing.assert_allclose(first.forces, reference_result.forces, atol=2e-6, rtol=0)
    with calculator.prepare_batch(
        [atoms], charges=[spin], multiplicities=[spin + 1]
    ) as batch:
        for _ in range(2):
            result = batch.execute(strict=True)
            np.testing.assert_allclose(
                result.energies[0], first.energy, atol=2e-8, rtol=0
            )
        moved = [
            (z, (x + 0.03 * (i + 1), y - 0.02 * i, zz + 0.015))
            for i, (z, (x, y, zz)) in enumerate(atoms)
        ]
        replay = batch.execute(coordinates=[[r for _, r in moved]], strict=True)
        fresh = calculator.singlepoint(moved, charge=spin, multiplicity=spin + 1)
        np.testing.assert_allclose(replay.energies[0], fresh.energy, atol=2e-8, rtol=0)
        np.testing.assert_allclose(
            replay.items[0].forces, fresh.forces, atol=2e-6, rtol=0
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ecp_resource_budget_and_parameter_invalidation(device):
    if device == "cuda" and os.getenv("VIBEQC_ECP_CUDA_TEST") != "1":
        pytest.skip("requires explicit real CUDA run")
    from vibeqc import ResourceBudget

    atoms, basis, _ = fixture()
    calculator = Calculator(basis=basis, device=device)
    plan = calculator.estimate_resources([atoms]).require_feasible()
    budget = ResourceBudget(
        host_bytes=plan.peak_bytes["host"], device_bytes=plan.peak_bytes.get("device")
    )
    bounded = Calculator(basis=basis, device=device, resource_budget=budget)
    with bounded.prepare_batch([atoms]) as batch:
        result = batch.execute(strict=True)
        assert result.items[0].converged
        if device == "cuda":
            ledger = batch.resource_diagnostics["observation"]["device_ledger"]
            assert ledger["rejected_allocations"] == 0
            assert 0 < ledger["peak_bytes"] <= ledger["limit_bytes"]
        potentials = json.loads(basis.by_element[11].ecp_data)
        potentials[0]["coefficients"][0][0] = str(
            float(potentials[0]["coefficients"][0][0]) + 0.1
        )
        bounded._basis = replace(
            basis,
            elements=tuple(
                replace(e, ecp_data=json.dumps(potentials))
                if e.atomic_number == 11
                else e
                for e in basis.elements
            ),
        )
        with pytest.raises(RuntimeError, match="identity changed"):
            batch.execute()
    with pytest.raises(NotImplementedError, match="density-fitting"):
        Calculator(basis=basis, device=device, density_fitting="auto").singlepoint(
            atoms
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("r_exponents", [5], "radial powers"),
        ("angular_momentum", [4], "local channel"),
        ("unknown", True, "unknown ECP"),
    ],
)
def test_unsupported_ecp_formats_reject_before_execution(field, value, match):
    atoms, basis, _ = fixture()
    potentials = [
        {
            "ecp_type": "scalar_ecp",
            "angular_momentum": [0],
            "r_exponents": [2],
            "gaussian_exponents": ["1"],
            "coefficients": [["-1"]],
            field: value,
        }
    ]
    bad = replace(
        basis,
        elements=tuple(
            replace(e, ecp_data=json.dumps(potentials)) if e.atomic_number == 11 else e
            for e in basis.elements
        ),
    )
    with pytest.raises(NotImplementedError, match=match):
        resolve_ecp(bad, tuple(Atom.from_value(a) for a in atoms))


@pytest.mark.skipif(
    os.getenv("VIBEQC_ECP_CUDA_TEST") != "1", reason="requires explicit real CUDA run"
)
@pytest.mark.parametrize("representation", ["spherical", "cartesian"])
def test_cuda_d_shell_and_independent_potential_center(representation):
    atoms, basis, _ = fixture(representation=representation, d_shell=True)
    cpu = ecp_integrals(atoms, basis)
    gpu = ecp_integrals(atoms, basis, device="cuda")
    for field in ("local", "nonlocal_", "local_derivative", "nonlocal_derivative"):
        np.testing.assert_allclose(
            getattr(gpu, field), getattr(cpu, field), atol=2e-10, rtol=1e-10
        )
    xyz = np.array([[0.13, -0.27, 0.32], [0.43, 0.38, 1.12], [-0.31, 0.12, -0.62]])
    np.testing.assert_allclose(
        detached_native(xyz, device="cuda"),
        detached_native(xyz),
        atol=2e-10,
        rtol=1e-10,
    )

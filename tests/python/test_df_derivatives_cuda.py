"""Fail-closed Slurm tests for external DF response weights and bounded tiles."""

import copy
import os

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell

from tools.vibeqc_validation.df_gradient import (
    execute_df_gradient,
    reference_df_matrices,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_DF_DERIVATIVE_CUDA_TEST") != "1",
    reason="explicit Slurm DF derivative tier",
)


def fixture_inputs(representation, shared):
    """A third atom owns only auxiliary functions unless shared centers are requested."""
    common = {
        "atomic_numbers": [2, 1, 1],
        "coordinates": [[0.1, -0.2, -0.7], [0.3, 0.1, 0.8], [-0.5, 0.4, 0.2]],
        "charge": 0,
        "multiplicity": 1,
        "basis_representation": representation,
    }

    def shell(atom, angular):
        return {
            "atom_index": atom,
            "angular_momentum": angular,
            "primitives": [[0.6 + 0.2 * angular, 0.8], [1.7 + 0.1 * angular, -0.1]],
        }

    orbital = {**common, "shells": [shell(0, 0), shell(0, 1), shell(1, 0), shell(1, 2)]}
    auxiliary = {
        **copy.deepcopy(common),
        "shells": [
            shell(0, 0),
            shell(0 if shared else 2, 0),
            shell(0 if shared else 2, 2),
            shell(1 if shared else 2, 3),
        ],
    }
    return orbital, auxiliary


def calculator(inputs):
    return Calculator(
        device="cuda",
        basis_representation=inputs["basis_representation"],
        basis=[
            Shell(
                s["atom_index"],
                s["angular_momentum"],
                tuple(Primitive(*p) for p in s["primitives"]),
            )
            for s in inputs["shells"]
        ],
    )


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("shared", [False, True])
def test_arbitrary_raw_fused_and_two_budgets_with_auxiliary_motion(
    representation, shared
):
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    oi, xi = fixture_inputs(representation, shared)
    o, x = calculator(oi), calculator(xi)
    atoms = [(z, tuple(r)) for z, r in zip(("He", "H", "H"), oi["coordinates"])]
    a, m, da, dm = reference_df_matrices(oi, xi)
    rng = np.random.default_rng(143)
    wa, wm = rng.normal(size=a.shape), rng.normal(size=m.shape)
    expected = np.einsum("axijp,ijp->ax", da, wa) + np.einsum("axpq,pq->ax", dm, wm)
    if not shared:
        assert np.max(np.abs(expected[-1])) > 1e-7
    records = []
    for budget in (12288, 65536):
        for schedule in (0, 1):
            actual, resources = execute_df_gradient(
                o, x, atoms, wa, wm, schedule=schedule, maximum_bytes=budget
            )
            np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-11)
            np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-10)
            assert (
                resources["host_bytes"] <= budget
                and resources["device_bytes"] <= budget
            )
            assert resources["device_to_host_bytes"] == 3 * len(atoms) * 8
            assert resources["stream_synchronizations"] == 1
            records.append(resources)
    assert records[0]["weight_tile_elements"] < wa.size
    assert records[-1]["weight_tile_elements"] == wa.size
    first, _ = execute_df_gradient(
        o, x, atoms, wa, wm, schedule=1, maximum_tile_elements=13
    )
    second, _ = execute_df_gradient(
        o, x, atoms, wa, wm, schedule=1, maximum_tile_elements=13
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, expected, atol=2e-9, rtol=2e-11)
    with pytest.raises(RuntimeError, match="budget|maximum_bytes"):
        execute_df_gradient(o, x, atoms, wa, wm, maximum_bytes=32)
    for step in (2e-4, 5e-5):
        for atom in range(3):
            energies = []
            for sign in (1, -1):
                op, xp = copy.deepcopy(oi), copy.deepcopy(xi)
                op["coordinates"][atom][2] += sign * step
                xp["coordinates"][atom][2] += sign * step
                av, mv, _, _ = reference_df_matrices(op, xp)
                energies.append(np.sum(av * wa) + np.sum(mv * wm))
            assert expected[atom, 2] == pytest.approx(
                (energies[0] - energies[1]) / (2 * step), abs=5e-6, rel=5e-6
            )


@pytest.mark.parametrize("tile_elements", [1, 7, 13])
def test_cooperative_sparse_weights_and_ragged_primitive_partitions(tile_elements):
    """Zero-weight groups and partial warps must preserve the shuffle mask.

    Unequal odd contraction lengths leave different primitive remainders in
    each subgroup. The oracle contracts libcint derivatives with the original
    sparse weights, independently of CUDA lane ownership and reduction order.
    """
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    oi, xi = fixture_inputs("spherical", False)
    oi["shells"][0]["primitives"].append([0.31, 0.12])
    xi["shells"][0]["primitives"].extend([[0.27, 0.13], [0.41, -0.08], [2.2, 0.07]])
    a, m, da, dm = reference_df_matrices(oi, xi)
    wa, wm = np.zeros_like(a), np.zeros_like(m)
    wa.flat[::5] = np.linspace(-0.2, 0.3, wa.flat[::5].size)
    wm.flat[::3] = np.linspace(0.1, -0.4, wm.flat[::3].size)
    expected = np.einsum("axijp,ijp->ax", da, wa) + np.einsum("axpq,pq->ax", dm, wm)
    atoms = [(z, tuple(r)) for z, r in zip(("He", "H", "H"), oi["coordinates"])]
    actual, _ = execute_df_gradient(
        calculator(oi),
        calculator(xi),
        atoms,
        wa,
        wm,
        schedule=0,
        maximum_tile_elements=tile_elements,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-11)


@pytest.mark.parametrize(
    "orbital_rep,auxiliary_rep",
    [("cartesian", "spherical"), ("spherical", "cartesian")],
)
def test_mixed_representations_and_permuted_auxiliary_shells(
    orbital_rep, auxiliary_rep
):
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    oi, xi = fixture_inputs(orbital_rep, False)
    xi["basis_representation"] = auxiliary_rep
    xi["shells"] = [xi["shells"][i] for i in (3, 0, 2, 1)]
    o, x = calculator(oi), calculator(xi)
    atoms = [(z, tuple(r)) for z, r in zip(("He", "H", "H"), oi["coordinates"])]
    a, m, da, dm = reference_df_matrices(oi, xi)
    rng = np.random.default_rng(1431)
    wa, wm = rng.normal(size=a.shape), rng.normal(size=m.shape)
    expected = np.einsum("axijp,ijp->ax", da, wa) + np.einsum("axpq,pq->ax", dm, wm)
    actual, _ = execute_df_gradient(
        o, x, atoms, wa, wm, maximum_bytes=16384, maximum_tile_elements=19
    )
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-11)


@pytest.mark.parametrize("method,charge,multiplicity", [("rhf", 0, 1), ("uhf", 1, 2)])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("budget", [0, 1 << 20, 4 << 20])
def test_complete_hf_replay_two_budgets_and_force_components(
    monkeypatch, method, charge, multiplicity, representation, budget
):
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.2, 0.7))]
    base = np.array([r for _, r in atoms])
    moved = base.copy()
    moved[1] += [0.01, -0.02, 0.03]
    calc = Calculator(
        device="cuda",
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=budget,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", "generated")
    oracle = Calculator(
        device="cpu",
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        density_fitting="cpu",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    expected = [
        oracle.singlepoint(
            [("H", tuple(r)) for r in positions],
            charge=charge,
            multiplicity=multiplicity,
        )
        for positions in (base, base, moved, base)
    ]
    monkeypatch.setenv("VIBEQC_DF_DERIVATIVE_MAPPING", "thread")
    with calc.prepare_batch(
        [atoms], charges=[charge], multiplicities=[multiplicity]
    ) as batch:
        for positions, reference in zip((base, base, moved, base), expected):
            actual = batch.execute([positions], strict=True).items[0]
            assert actual.executed_backend == "cuda"
            assert actual.energy == pytest.approx(reference.energy, abs=3e-10)
            np.testing.assert_allclose(
                actual.forces, reference.forces, atol=3e-9, rtol=0
            )
            np.testing.assert_allclose(actual.forces.sum(axis=0), 0, atol=3e-9)
        # Mapping changes must invalidate the response cache. Retired selector
        # values have no dispatch/provenance effect and cannot resurrect an oracle.
        for mapping in ("serial", "thread"):
            monkeypatch.setenv(
                "VIBEQC_DF_DERIVATIVES",
                "reference" if mapping == "serial" else "generated",
            )
            monkeypatch.setenv("VIBEQC_DF_DERIVATIVE_MAPPING", mapping)
            actual = batch.execute([base], strict=True).items[0]
            np.testing.assert_allclose(
                actual.forces, expected[0].forces, atol=3e-9, rtol=0
            )
            policy = batch._warm_metadata[0]["controls"]["runtime_policy"]
            assert "VIBEQC_DF_DERIVATIVES" not in policy
            assert policy["VIBEQC_DF_DERIVATIVE_MAPPING"] == mapping


@pytest.mark.parametrize("method,charge,multiplicity", [("rhf", 0, 1), ("uhf", 1, 2)])
def test_generated_df_hf_matches_pyscf_and_two_energy_difference_steps(
    monkeypatch, method, charge, multiplicity
):
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    # This tier is explicitly enabled in an allocated GPU job. Missing oracle
    # dependencies must fail the requested numerical gate instead of skipping it.
    from pyscf import gto, scf

    atoms = [("H", (0, 0, -0.7)), ("H", (0.1, 0.2, 0.7))]
    mol = gto.M(
        atom=atoms,
        basis="def2-svp",
        unit="Bohr",
        cart=True,
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )
    reference = (scf.RHF if method == "rhf" else scf.UHF)(mol).density_fit(
        auxbasis="def2-svp"
    )
    reference.conv_tol, reference.conv_tol_grad = 1e-13, 1e-10
    reference.kernel()
    assert reference.converged
    expected = -reference.nuc_grad_method().kernel()
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", "generated")
    calc = Calculator(
        device="cuda",
        method=method,
        basis="def2-svp",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=1 << 20,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    result = calc.singlepoint(atoms, charge=charge, multiplicity=multiplicity)
    assert result.energy == pytest.approx(reference.e_tot, abs=3e-10)
    np.testing.assert_allclose(result.forces, expected, atol=3e-9, rtol=0)
    for step in (2e-4, 5e-5):
        energies = []
        for sign in (1, -1):
            displaced = copy.deepcopy(atoms)
            r = np.array(displaced[1][1], dtype=float)
            r[2] += sign * step
            displaced[1] = ("H", tuple(r))
            energies.append(
                calc.singlepoint(
                    displaced, charge=charge, multiplicity=multiplicity
                ).energy
            )
        assert result.forces[1, 2] == pytest.approx(
            -(energies[0] - energies[1]) / (2 * step), abs=2e-6
        )


def test_metric_only_weight_response_moves_both_auxiliary_centers():
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    oi, xi = fixture_inputs("spherical", False)
    o, x = calculator(oi), calculator(xi)
    atoms = [(z, tuple(r)) for z, r in zip(("He", "H", "H"), oi["coordinates"])]
    a, m, _, dm = reference_df_matrices(oi, xi)
    wm = np.random.default_rng(1432).normal(size=m.shape)
    expected = np.einsum("axpq,pq->ax", dm, wm)
    assert np.max(np.abs(expected[2])) > 1e-5
    actual, _ = execute_df_gradient(
        o, x, atoms, np.zeros_like(a), wm, maximum_tile_elements=11
    )
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-11)
    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-10)


@pytest.mark.parametrize("null_a,null_m", [(True, False), (False, True), (True, True)])
def test_null_weight_channels_are_documented_zero_operators(
    monkeypatch, null_a, null_m
):
    """The C ABI retains full shape counts when a null pointer denotes zero."""
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    oi, xi = fixture_inputs("spherical", False)
    orbital, auxiliary = calculator(oi), calculator(xi)
    atoms = [(z, tuple(r)) for z, r in zip(("He", "H", "H"), oi["coordinates"])]
    n = sum(2 * shell["angular_momentum"] + 1 for shell in oi["shells"])
    a = sum(2 * shell["angular_momentum"] + 1 for shell in xi["shells"])
    random = np.random.default_rng(1433)
    wa, wm = random.normal(size=(n, n, a)), random.normal(size=(a, a))
    expected, _ = execute_df_gradient(
        orbital,
        auxiliary,
        atoms,
        np.zeros_like(wa) if null_a else wa,
        np.zeros_like(wm) if null_m else wm,
    )
    native = orbital._library.vibeqc_system_df_gradient_cuda

    def nullable(*arguments):
        native.argtypes, native.restype = nullable.argtypes, nullable.restype
        arguments = list(arguments)
        assert arguments[4] == wa.size and arguments[6] == wm.size
        if null_a:
            arguments[3] = None
        if null_m:
            arguments[5] = None
        return native(*arguments)

    monkeypatch.setattr(orbital._library, "vibeqc_system_df_gradient_cuda", nullable)
    actual, resources = execute_df_gradient(orbital, auxiliary, atoms, wa, wm)
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-11)
    if null_a and null_m:
        np.testing.assert_array_equal(actual, np.zeros_like(actual))
        assert resources["tiles"] == 0


@pytest.mark.parametrize("method,charge,multiplicity", [("rhf", 0, 1), ("uhf", 1, 2)])
def test_auxiliary_only_atom_hf_energy_derivative(
    monkeypatch, method, charge, multiplicity
):
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    oi, xi = fixture_inputs("cartesian", False)
    atoms = [(z, tuple(r)) for z, r in zip(("He", "H", "H"), oi["coordinates"])]

    def shells(inputs):
        return [
            Shell(
                s["atom_index"],
                s["angular_momentum"],
                tuple(Primitive(*p) for p in s["primitives"]),
            )
            for s in inputs["shells"]
        ]

    calc = Calculator(
        device="cuda",
        method=method,
        basis=shells(oi),
        auxiliary_basis=shells(xi),
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=4 << 20,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    oracle = Calculator(
        device="cpu",
        method=method,
        basis=shells(oi),
        auxiliary_basis=shells(xi),
        density_fitting="cpu",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    expected = oracle.singlepoint(atoms, charge=charge, multiplicity=multiplicity)
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", "generated")
    actual = calc.singlepoint(atoms, charge=charge, multiplicity=multiplicity)
    np.testing.assert_allclose(actual.forces, expected.forces, atol=3e-9, rtol=0)
    for step in (2e-4, 5e-5):
        energies = []
        for sign in (1, -1):
            displaced = copy.deepcopy(atoms)
            r = np.array(displaced[2][1], dtype=float)
            r[2] += sign * step
            displaced[2] = ("H", tuple(r))
            energies.append(
                calc.singlepoint(
                    displaced, charge=charge, multiplicity=multiplicity
                ).energy
            )
        assert actual.forces[2, 2] == pytest.approx(
            -(energies[0] - energies[1]) / (2 * step), abs=3e-6
        )


@pytest.mark.parametrize(
    "method,representation", [("rhf", "spherical"), ("uhf", "cartesian")]
)
def test_df_generated_sdf_bucket_preserves_all_geometry_phases(
    monkeypatch, method, representation
):
    from pyscf import gto, scf
    from test_one_electron_values_cuda import run_case, sdf_case_inputs

    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    kwargs = {
        "method": method,
        "representation": representation,
        "fitted": True,
        "count": 3,
    }
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", "generated")
    actual = run_case(monkeypatch, mapping="thread", **kwargs)
    basis, systems, charge, multiplicity = sdf_case_inputs(method, kwargs["count"])
    # Libcint/PySCF provides an independent complete-force oracle, including
    # the moving auxiliary basis. Reuse only identical reference geometries;
    # the CUDA calculation above still executes all four prepared phases.
    expected = {}
    for moved, result in zip((False, False, True, False), actual, strict=True):
        for item, right in enumerate(result.items):
            if (moved, item) not in expected:
                atoms = [
                    (f"{z}{i}", np.array(r)) for i, (z, r) in enumerate(systems[item])
                ]
                if moved:
                    atoms[1][1][0] += 0.013 * (item + 1)
                reference_basis = {label: [] for label, _ in atoms}
                for shell in basis:
                    reference_basis[atoms[shell.atom_index][0]].append(
                        [shell.angular_momentum]
                        + [[p.exponent, p.coefficient] for p in shell.primitives]
                    )
                molecule = gto.M(
                    atom=atoms,
                    basis=reference_basis,
                    unit="Bohr",
                    cart=representation == "cartesian",
                    charge=charge,
                    spin=multiplicity - 1,
                    verbose=0,
                )
                reference = (scf.RHF if method == "rhf" else scf.UHF)(
                    molecule
                ).density_fit(auxbasis=reference_basis)
                reference.conv_tol, reference.conv_tol_grad = 1e-13, 1e-10
                reference.kernel()
                assert reference.converged
                gradient = reference.nuc_grad_method()
                gradient.auxbasis_response = True
                expected[moved, item] = reference.e_tot, -gradient.kernel()
            energy, forces = expected[moved, item]
            assert right.executed_backend == "cuda"
            np.testing.assert_allclose(right.energy, energy, atol=3e-10, rtol=0)
            np.testing.assert_allclose(right.forces, forces, atol=3e-9, rtol=0)


def test_df_generated_failed_item_preserves_successful_neighbor(monkeypatch):
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", "generated")
    atoms = [("H", (0, 0, -0.7)), ("H", (0.1, 0, 0.7))]
    other = [("He", (0, 0, -0.7)), ("H", (0.1, 0, 0.7))]
    calc = Calculator(device="cuda", density_fitting="cuda", max_iterations=3)
    with calc.prepare_batch([atoms, other], charges=[0, 1]) as batch:
        result = batch.execute()
        assert result.items[0].succeeded and not result.items[1].succeeded
        expected = Calculator(device="cuda", density_fitting="cuda").singlepoint(atoms)
        np.testing.assert_allclose(
            result.items[0].forces, expected.forces, atol=3e-9, rtol=0
        )


def test_df_rank_crossing_is_reported_without_oracle_retry(monkeypatch):
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    inputs = {
        "atomic_numbers": [1, 1],
        "coordinates": [[0, 0, -0.7], [0, 0, 0.7]],
        "charge": 0,
        "multiplicity": 1,
        "basis_representation": "cartesian",
        "shells": [
            {"atom_index": i, "angular_momentum": 0, "primitives": [[0.8, 1.0]]}
            for i in range(2)
        ],
    }
    _, metric, _, _ = reference_df_matrices(inputs, inputs)
    eigenvalues = np.linalg.eigvalsh(metric)
    threshold = float(eigenvalues[0] / eigenvalues[-1])
    basis = [Shell(i, 0, (Primitive(0.8, 1.0),)) for i in range(2)]
    calc = Calculator(
        device="cuda",
        basis=basis,
        auxiliary_basis=basis,
        density_fitting="cuda",
        density_fitting_relative_threshold=threshold,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with pytest.raises(RuntimeError, match="rank crossing|subspaces are unresolved"):
        calc.singlepoint([("H", tuple(r)) for r in inputs["coordinates"]])

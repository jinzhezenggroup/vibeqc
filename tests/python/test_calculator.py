import numpy as np
import pytest

from vibeqc import Calculator, Primitive, Shell, method_capabilities


def test_h2_energy_and_force_invariance():
    calculator = Calculator(method="rhf", basis="sto-3g", device="cpu")
    result = calculator.singlepoint(
        [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    )
    assert abs(result.energy - (-1.11671432506255)) < 2.0e-9
    assert np.max(np.abs(result.forces.sum(axis=0))) < 2.0e-10
    assert result.executed_backend == "cpu_reference"


def test_wb97m_v_is_reserved_not_implemented():
    try:
        Calculator(method="wb97m-v")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("wB97M-V must report that it is not implemented")


def test_method_capabilities_report_families_and_properties():
    rhf = method_capabilities("rhf")
    assert rhf.family == "hartree_fock"
    assert rhf.available
    assert rhf.supports_batch
    assert rhf.supported_properties == frozenset(("energy", "forces"))

    ccsd_t = method_capabilities("ccsd(t)")
    assert ccsd_t.family == "coupled_cluster"
    assert not ccsd_t.available
    assert not ccsd_t.supports_batch


def test_helium_sto3g_reference():
    """Match PySCF when it consumes the exact bundled BSE coefficients.

    PySCF's built-in STO-3G table rounds the helium exponents and contraction
    coefficients to fewer digits than our pinned Basis Set Exchange pack.  Its
    built-in energy therefore differs by about 9e-10 Eh; the value below comes
    from PySCF with the exact data stored in ``basis_pack.json``.
    """

    result = Calculator().singlepoint([("He", (0.0, 0.0, 0.0))])
    assert abs(result.energy - (-2.8077839566141964)) < 2.0e-12
    assert np.max(np.abs(result.forces)) == 0.0


def test_bundled_def2_tzvp_h2_cpu_reference():
    """Keep generated named-basis data available without optional packages."""

    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    result = Calculator(
        basis="def2-tzvp",
        device="cpu",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    ).singlepoint(atoms)
    assert result.energy == pytest.approx(-1.1325356608719206, abs=2.0e-11)
    assert result.forces[0, 2] == pytest.approx(0.00427329, abs=2.0e-8)
    assert result.forces[1, 2] == pytest.approx(-0.00427329, abs=2.0e-8)


def test_real_spherical_d_energy_and_force_match_pyscf():
    """Validate the Python/ABI path for normalized real spherical AOs."""

    atoms = [("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    result = Calculator(
        basis=basis,
        basis_representation="spherical",
        device="cpu",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    ).singlepoint(atoms, charge=1)

    assert result.energy == pytest.approx(-2.3341870407859284, abs=3.0e-12)
    assert result.forces[0, 2] == pytest.approx(0.3502792384052442, abs=8.0e-12)
    assert result.forces[1, 2] == pytest.approx(-0.3502792384052438, abs=8.0e-12)


def test_cuda_real_spherical_d_energy_and_force_match_pyscf():
    """Expose the validated sparse d transform through the public CUDA ABI."""

    atoms = [("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    try:
        result = Calculator(
            basis=basis,
            basis_representation="spherical",
            device="cuda",
            energy_tolerance=1.0e-12,
            density_tolerance=1.0e-10,
        ).singlepoint(atoms, charge=1)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    assert result.executed_backend == "cuda"
    assert result.energy == pytest.approx(-2.3341870407859284, abs=3.0e-9)
    assert result.forces[0, 2] == pytest.approx(0.3502792384052442, abs=3.0e-8)
    assert result.forces[1, 2] == pytest.approx(-0.3502792384052438, abs=3.0e-8)


def test_cuda_spherical_def2_svp_water_matches_pyscf():
    """Validate pure def2-SVP through CUDA's spherical direct-J/K path."""

    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, -1.43233673, 1.10715266)),
        ("H", (0.0, 1.43233673, 1.10715266)),
    ]
    try:
        result = Calculator(
            basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            energy_tolerance=1.0e-12,
            density_tolerance=1.0e-10,
        ).singlepoint(atoms)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    expected_forces = np.array(
        [
            [0.0, 0.0, 0.018058935411787047],
            [0.0, 0.011439597724475004, -0.009029467705892413],
            [0.0, -0.011439597724475448, -0.009029467705893301],
        ]
    )
    assert result.executed_backend == "cuda"
    assert result.energy == pytest.approx(-75.96097281179767, abs=3.0e-9)
    assert np.allclose(result.forces, expected_forces, atol=3.0e-8)


def test_cuda_def2_tzvp_water_uses_graph_native_eigensolver():
    """Validate the capture-safe batched provider on a realistic AO matrix."""

    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, -1.43233673, 1.10715266)),
        ("H", (0.0, 1.43233673, 1.10715266)),
    ]
    try:
        result = Calculator(
            basis="def2-tzvp",
            device="cuda",
            energy_tolerance=1.0e-12,
            density_tolerance=1.0e-10,
        ).singlepoint(atoms)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    expected_forces = np.array(
        [
            [0.0, 0.0, 0.024987059842175086],
            [0.0, 0.012545806964672668, -0.012493529921093316],
            [0.0, -0.012545806964674444, -0.012493529921094648],
        ]
    )
    assert result.executed_backend == "cuda"
    assert result.energy == pytest.approx(-76.0594049849294, abs=3.0e-9)
    assert np.allclose(result.forces, expected_forces, atol=3.0e-8)


def test_cuda_spherical_uhf_doublet_matches_pyscf():
    """Exercise public CUDA UHF with a multi-term real-spherical d shell."""

    atoms = [("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    try:
        result = Calculator(
            method="uhf",
            basis=basis,
            basis_representation="spherical",
            device="cuda",
            energy_tolerance=1.0e-12,
            density_tolerance=1.0e-10,
        ).singlepoint(atoms, charge=2, multiplicity=2)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    assert result.executed_backend == "cuda"
    assert result.energy == pytest.approx(-1.0477980686214317, abs=3.0e-9)
    assert result.forces[0, 2] == pytest.approx(-0.34919100072002696, abs=3.0e-8)
    assert result.forces[1, 2] == pytest.approx(0.34919100072002696, abs=3.0e-8)


def test_cuda_spherical_named_basis_uhf_breaks_excited_state_symmetry():
    """Keep linear OH out of the high-energy sigma-hole UHF fixed point."""

    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.8323918340046244)),
    ]
    try:
        result = Calculator(
            method="uhf",
            basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            energy_tolerance=1.0e-12,
            density_tolerance=1.0e-10,
        ).singlepoint(atoms, multiplicity=2)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    expected_forces = np.array(
        [
            [0.0, 0.0, 0.014500852953795551],
            [0.0, 0.0, -0.014500852953795551],
        ]
    )
    assert result.executed_backend == "cuda"
    assert result.energy == pytest.approx(-75.32510951561675, abs=3.0e-9)
    assert np.allclose(result.forces, expected_forces, atol=3.0e-8)


def test_h3_plus_exercises_diis_and_force_invariants():
    coordinates = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms = [("H", coordinate) for coordinate in coordinates]
    result = Calculator().singlepoint(atoms, charge=1)
    assert result.converged
    assert result.iterations > 2
    assert np.max(np.abs(result.forces.sum(axis=0))) < 2.0e-10
    total_torque = np.cross(coordinates, result.forces).sum(axis=0)
    assert np.max(np.abs(total_torque)) < 2.0e-10


def test_single_system_cuda_executes_full_scientific_path_when_available():
    atoms = [("H", (-1.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.0)),
             ("H", (1.0, 0.0, 0.0))]
    reference = Calculator(device="cpu").singlepoint(atoms, charge=1)
    try:
        candidate = Calculator(device="cuda").singlepoint(atoms, charge=1)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")
    assert candidate.executed_backend == "cuda"
    assert candidate.iterations == reference.iterations
    assert abs(candidate.energy - reference.energy) < 2.0e-12
    assert np.allclose(candidate.forces, reference.forces, atol=2.0e-10)


def test_cartesian_p_shell_energy_force_and_cuda_agreement():
    """Exercise s/p one-electron, ERI, Pulay, and two-electron derivatives."""

    basis = (
        Shell(0, 0, (Primitive(1.2, 1.0),)),
        Shell(0, 1, (Primitive(0.7, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
        Shell(1, 1, (Primitive(0.7, 1.0),)),
    )
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    reference = Calculator(
        basis=basis,
        device="cpu",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    ).singlepoint(atoms)

    # Independent PySCF 2.11/libcint Cartesian RHF oracle.
    assert reference.energy == pytest.approx(-0.2897023252480543, abs=3.0e-12)
    assert reference.forces[0, 2] == pytest.approx(0.34807084478701356,
                                                  abs=3.0e-11)
    assert np.max(np.abs(reference.forces.sum(axis=0))) < 2.0e-11

    try:
        candidate = Calculator(
            basis=basis,
            device="cuda",
            energy_tolerance=1.0e-12,
            density_tolerance=1.0e-10,
        ).singlepoint(atoms)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")
    assert candidate.executed_backend == "cuda"
    assert candidate.iterations == reference.iterations
    assert candidate.energy == pytest.approx(reference.energy, abs=3.0e-12)
    assert np.allclose(candidate.forces, reference.forces, atol=3.0e-11)


def test_bundled_def2_svp_water_matches_pyscf_cartesian_reference():
    """Validate named H-Ar basis ingestion on a realistic direct-J/K case."""

    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, -1.43233673, 1.10715266)),
        ("H", (0.0, 1.43233673, 1.10715266)),
    ]
    try:
        result = Calculator(
            basis="def2-svp",
            device="cuda",
            energy_tolerance=1.0e-11,
            density_tolerance=1.0e-9,
        ).singlepoint(atoms)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    # Independent PySCF/libcint reference with cart=True and coordinates in
    # Bohr. This has 25 Cartesian AOs and therefore exercises tiled direct J/K.
    assert result.energy == pytest.approx(-75.96220846602468, abs=8.0e-11)
    assert result.forces[0, 2] == pytest.approx(
        0.0178004578, abs=8.0e-10
    )
    assert result.forces[1, 1] == pytest.approx(
        0.0113730678, abs=8.0e-10
    )
    assert result.forces[1, 2] == pytest.approx(
        -0.00890022892, abs=8.0e-10
    )
    assert np.max(np.abs(result.forces.sum(axis=0))) < 2.0e-9


def test_cartesian_d_f_cuda_matches_pyscf_libcint_reference():
    """Gate the full device path through mixed s/d/f values and derivatives."""

    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(0, 3, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    atoms = [("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    try:
        result = Calculator(
            basis=basis,
            device="cuda",
            energy_tolerance=1.0e-12,
            density_tolerance=1.0e-10,
        ).singlepoint(atoms, charge=1)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    # Independent PySCF/libcint Cartesian reference. This 18-AO system makes
    # off-center d/f functions participate in Fock construction and exercises
    # their nuclear, one-electron, Pulay, and ERI derivative paths.
    assert result.executed_backend == "cuda"
    assert result.energy == pytest.approx(-2.644619635687887, abs=8.0e-12)
    assert result.forces[0, 2] == pytest.approx(-0.025113094749739884,
                                               abs=3.0e-11)
    assert result.forces[1, 2] == pytest.approx(0.02511309474973966,
                                               abs=3.0e-11)
    assert np.max(np.abs(result.forces.sum(axis=0))) < 3.0e-11


def test_screened_direct_jk_force_matches_energy_finite_difference(
    monkeypatch: pytest.MonkeyPatch,
):
    """Keep the direct-J/K screening decision variational and force-consistent."""

    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(0, 3, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    calculator = Calculator(
        basis=basis,
        device="cuda",
        energy_tolerance=1.0e-10,
        density_tolerance=1.0e-8,
        # Deliberately loose enough to exercise screening. Production defaults
        # remain 1e-12. Four shells create ten shell pairs, so this also selects
        # the grouped direct-J/K consumer and guards the mathematical
        # consistency of that approximate path.
        screening_tolerance=1.0e-2,
    )

    def coordinates(distance: float) -> np.ndarray:
        return np.array(
            [[0.0, 0.0, -0.5 * distance],
             [0.0, 0.0, 0.5 * distance]],
            dtype=np.float64,
        )

    atoms = [("He", coordinates(1.4)[0]), ("H", coordinates(1.4)[1])]
    try:
        with calculator.prepare_batch([atoms], charges=[1]) as batch:
            # Seed the resident density before comparing force-screening modes;
            # otherwise the cold-to-warm SCF refinement obscures their tiny
            # numerical difference.
            batch.execute([coordinates(1.4)], strict=True)
            center = batch.execute([coordinates(1.4)], strict=True).items[0]
            # The force-only density-product gate is evaluated on every warm
            # execution rather than cached in the immutable direct-J/K plan.
            # Compare both paths on one prepared batch so the A/B switch and
            # the conservative loose-threshold cap remain covered together.
            monkeypatch.setenv("VIBEQC_FORCE_DENSITY_PRODUCT_SCREENING", "0")
            unscreened = batch.execute(
                [coordinates(1.4)], strict=True
            ).items[0]
            monkeypatch.delenv("VIBEQC_FORCE_DENSITY_PRODUCT_SCREENING")
            plus = batch.execute([coordinates(1.4001)], strict=True).items[0]
            minus = batch.execute([coordinates(1.3999)], strict=True).items[0]
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    derivative = (plus.energy - minus.energy) / 2.0e-4
    # Both atoms move by +/-dR/2, so translational invariance makes the force
    # on atom 1 equal to -dE/dR.
    assert center.forces is not None
    assert unscreened.forces is not None
    assert center.energy == pytest.approx(unscreened.energy, abs=1.0e-12)
    assert np.max(np.abs(center.forces - unscreened.forces)) < 2.0e-10
    assert center.forces[1, 2] == pytest.approx(-derivative, abs=2.0e-6)
    assert np.max(np.abs(center.forces.sum(axis=0))) < 3.0e-10


def test_larger_direct_jk_matches_cpu_oracle():
    """Validate a larger high-accuracy direct J/K workload on the GPU."""

    atoms = [
        ("He", (0.0, 0.0, -2.0)),
        ("He", (0.0, 0.0, 0.0)),
        ("He", (0.0, 0.0, 2.0)),
    ]
    basis = tuple(
        shell
        for atom_index in range(3)
        for shell in (
            Shell(atom_index, 0, (Primitive(1.5, 1.0),)),
            Shell(atom_index, 2, (Primitive(0.8, 1.0),)),
        )
    )
    calculator = Calculator(
        basis=basis,
        device="cuda",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )
    try:
        # Six shells produce 21 canonical shell pairs and 21 Cartesian AOs,
        # providing a medium direct-J/K regression above the persistent-ERI
        # threshold while production-accuracy screening remains enabled.
        with calculator.prepare_batch([atoms], warm_start=True) as prepared:
            first = prepared.execute(strict=True).items[0]
            repeated = prepared.execute(strict=True).items[0]
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    # These values come from the independent CPU integral/SCF/gradient oracle,
    # not from the CUDA implementation under test.
    assert first.energy == pytest.approx(-7.43079582681513, abs=5.0e-11)
    assert first.forces[0, 2] == pytest.approx(-0.28152878739456416,
                                               abs=5.0e-10)
    assert first.forces[1, 2] == pytest.approx(0.0, abs=5.0e-10)
    assert first.forces[2, 2] == pytest.approx(0.28152878739456361,
                                               abs=5.0e-10)
    # The replay consumes the converged density as a warm start, so it follows
    # a shorter SCF path; agreement should be numerical rather than bitwise.
    assert repeated.energy == pytest.approx(first.energy, abs=2.0e-13)
    assert np.allclose(repeated.forces, first.forces, atol=2.0e-12)


def test_cuda_uhf_h2_plus_matches_pyscf_and_cpu_oracles():
    """Validate the public all-GPU UHF energy and analytic-force path."""

    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    cpu = Calculator(
        method="uhf",
        device="cpu",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    ).singlepoint(atoms, charge=1, multiplicity=2)
    try:
        gpu = Calculator(
            method="uhf",
            device="cuda",
            energy_tolerance=1.0e-12,
            density_tolerance=1.0e-10,
        ).singlepoint(atoms, charge=1, multiplicity=2)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    assert gpu.executed_backend == "cuda"
    # PySCF oracle uses the exact bundled BSE STO-3G coefficients rather than
    # the historically rounded coefficients in the native smoke test.
    assert gpu.energy == pytest.approx(-0.53851134832246783, abs=2.0e-10)
    assert gpu.forces[0, 2] == pytest.approx(-0.19038408686848268,
                                             abs=3.0e-9)
    assert gpu.forces[1, 2] == pytest.approx(0.19038408686848290,
                                             abs=3.0e-9)
    assert gpu.energy == pytest.approx(cpu.energy, abs=2.0e-10)
    assert np.allclose(gpu.forces, cpu.forces, atol=3.0e-9)


def test_cuda_uhf_closed_shell_limit_matches_rhf():
    """Require alpha=beta UHF to reduce exactly to closed-shell HF."""

    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    try:
        rhf = Calculator(method="rhf", device="cuda").singlepoint(atoms)
        uhf = Calculator(method="uhf", device="cuda").singlepoint(
            atoms, multiplicity=1
        )
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    assert uhf.energy == pytest.approx(rhf.energy, abs=3.0e-11)
    assert np.allclose(uhf.forces, rhf.forces, atol=3.0e-9)


def test_cuda_uhf_direct_quartet_closed_shell_matches_rhf():
    """Exercise the spin-resolved symmetry-reduced s/d quartet path."""

    atoms = [
        ("He", (0.0, 0.0, -2.0)),
        ("He", (0.0, 0.0, 0.0)),
        ("He", (0.0, 0.0, 2.0)),
    ]
    basis = tuple(
        shell
        for atom_index in range(3)
        for shell in (
            Shell(atom_index, 0, (Primitive(1.5, 1.0),)),
            Shell(atom_index, 2, (Primitive(0.8, 1.0),)),
        )
    )
    options = dict(
        basis=basis,
        device="cuda",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )
    try:
        rhf = Calculator(method="rhf", **options).singlepoint(atoms)
        uhf = Calculator(method="uhf", **options).singlepoint(
            atoms, multiplicity=1
        )
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    assert uhf.energy == pytest.approx(rhf.energy, abs=8.0e-11)
    assert np.allclose(uhf.forces, rhf.forces, atol=8.0e-9)


def test_cuda_uhf_direct_jk_matches_pyscf_and_force_finite_difference():
    """Exercise spin-resolved screened direct J/K above the ERI-cache limit."""

    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(0, 3, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    calculator = Calculator(
        method="uhf",
        basis=basis,
        device="cuda",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )

    def coordinates(distance: float) -> np.ndarray:
        return np.array(
            [[0.0, 0.0, -0.5 * distance],
             [0.0, 0.0, 0.5 * distance]],
            dtype=np.float64,
        )

    atoms = [("He", coordinates(1.4)[0]), ("H", coordinates(1.4)[1])]
    try:
        with calculator.prepare_batch(
            [atoms], multiplicities=[2], warm_start=True
        ) as prepared:
            center = prepared.execute([coordinates(1.4)], strict=True).items[0]
            plus = prepared.execute([coordinates(1.4001)], strict=True).items[0]
            minus = prepared.execute([coordinates(1.3999)], strict=True).items[0]
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    assert center.executed_backend == "cuda"
    assert center.energy == pytest.approx(-2.5806263483643965, abs=8.0e-11)
    assert center.forces[0, 2] == pytest.approx(-0.19311929542563755,
                                                abs=5.0e-9)
    assert center.forces[1, 2] == pytest.approx(0.19311929542563733,
                                                abs=5.0e-9)
    derivative = (plus.energy - minus.energy) / 2.0e-4
    assert center.forces[1, 2] == pytest.approx(-derivative, abs=3.0e-6)
    assert np.max(np.abs(center.forces.sum(axis=0))) < 5.0e-9


def test_screened_cuda_uhf_direct_force_matches_energy_finite_difference():
    """Keep approximate spin-resolved J/K and its analytic force consistent."""

    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(0, 3, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    calculator = Calculator(
        method="uhf",
        basis=basis,
        device="cuda",
        energy_tolerance=1.0e-10,
        density_tolerance=1.0e-8,
        screening_tolerance=1.0e-2,
    )

    def coordinates(distance: float) -> np.ndarray:
        return np.array(
            [[0.0, 0.0, -0.5 * distance],
             [0.0, 0.0, 0.5 * distance]],
            dtype=np.float64,
        )

    atoms = [("He", coordinates(1.4)[0]), ("H", coordinates(1.4)[1])]
    try:
        with calculator.prepare_batch(
            [atoms], multiplicities=[2], warm_start=True
        ) as prepared:
            center = prepared.execute([coordinates(1.4)], strict=True).items[0]
            plus = prepared.execute([coordinates(1.4001)], strict=True).items[0]
            minus = prepared.execute([coordinates(1.3999)], strict=True).items[0]
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    derivative = (plus.energy - minus.energy) / 2.0e-4
    assert center.forces[1, 2] == pytest.approx(-derivative, abs=3.0e-6)
    assert np.max(np.abs(center.forces.sum(axis=0))) < 5.0e-9

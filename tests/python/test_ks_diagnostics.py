"""Physical KS histories, independent components and additive ABI state gates."""

import ctypes
import json
import os
import typing
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from vibeqc import Atom, Calculator, KsDiagnostic, KsTransportDiagnostic, _native
from vibeqc_compiler.dft.grid import MolecularGrid

H3 = [("H", (0, 0, 0)), ("H", (0.15, 0.13, 1.5)), ("H", (0.6, 0.26, 3.0))]


@pytest.fixture(params=("cpu", "cuda"))
def device(request: typing.Any) -> typing.Any:
    if request.param == "cuda" and os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    return request.param


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"))
def test_physical_components_and_history_match_independent_state(
    method: typing.Any, device: typing.Any
) -> None:
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    pyscf.lib.num_threads(1)
    unrestricted = method.endswith("uks")
    charge, multiplicity = (0, 2) if unrestricted else (1, 1)
    atoms = tuple(Atom.from_value(a) for a in H3)
    calculator = Calculator(
        method=method,
        device=device,
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    labels = [f"H{i}" for i in range(3)]
    basis = {label: [] for label in labels}
    for shell in calculator._shells_for_atoms(atoms):
        basis[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[(p.exponent, p.coefficient) for p in shell.primitives],
            ]
        )
    mol = gto.M(
        atom=[(label, atom.position) for label, atom in zip(labels, atoms)],
        basis=basis,
        charge=charge,
        spin=multiplicity - 1,
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    grid = MolecularGrid(atoms, charge=charge, multiplicity=multiplicity).explicit()
    reference = dft.UKS(mol) if unrestricted else dft.RKS(mol)
    reference.xc = "PBE" if method.startswith("pbe") else "LDA_X,LDA_C_PW"
    reference.grids.coords = np.array(grid.points)
    reference.grids.weights = np.array(grid.weights)
    reference.small_rho_cutoff = 0
    reference.conv_tol, reference.conv_tol_grad = 1e-13, 1e-10
    reference.max_cycle = 150
    reference.kernel()
    assert reference.converged
    density = reference.make_rdm1()
    total = density.sum(axis=0) if unrestricted else density
    potential = reference.get_veff(dm=density)
    fock, overlap = reference.get_fock(dm=density), reference.get_ovlp()
    residual = fock @ density @ overlap - overlap @ density @ fock
    assert np.max(np.sqrt(np.mean(residual**2, axis=(-2, -1)))) < 1e-9
    independent = (
        mol.energy_nuc(),
        np.einsum("ij,ji", total, reference.get_hcore()),
        0.5 * np.einsum("ij,ji", total, reference.get_j(dm=total)),
        potential.exc,
    )
    native = calculator.singlepoint(atoms, charge=charge, multiplicity=multiplicity)
    diagnostic = native.ks_diagnostic
    assert native.converged and isinstance(diagnostic, KsDiagnostic)
    assert native.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    assert diagnostic.occupations == ((2, 1) if unrestricted else (1, 1))
    assert diagnostic.electrons == pytest.approx(diagnostic.occupations, abs=1e-10)
    assert diagnostic.grid_points == len(grid.weights)
    assert diagnostic.tile_points == calculator.ks_options.tile_points
    assert diagnostic.ao_order == int(method.startswith("pbe"))
    assert diagnostic.scf_domain == calculator.ks_options.scf_domain
    components = diagnostic.components
    assert (
        components.nuclear,
        components.one_electron,
        components.hartree,
        components.xc,
    ) == pytest.approx(independent, abs=1e-8)
    assert components.total == native.energy
    assert native.energy == pytest.approx(reference.e_tot, abs=1e-8)
    assert diagnostic.physical_residual_max < 1e-9
    assert diagnostic.density_change_max < 1e-10
    assert diagnostic.physical_residual_max >= native.physical_residual_rms * (
        1 - 1e-12
    )
    assert len(diagnostic.history) == native.iterations
    assert [row.iteration for row in diagnostic.history] == list(
        range(1, native.iterations + 1)
    )
    assert diagnostic.history[0].energy_change is None
    assert not diagnostic.initial_density_used
    # CPU RKS and UKS both close a converged iterate with the physical Fock.
    expected_rebuilds = 0 if device == "cuda" else 2
    assert diagnostic.fock_builds == native.iterations + expected_rebuilds
    for previous, current in zip(diagnostic.history, diagnostic.history[1:]):
        assert current.energy_change == pytest.approx(
            abs(current.components.total - previous.components.total), abs=1e-14
        )
        assert current.electrons == pytest.approx(diagnostic.occupations, abs=1e-10)
    json.dumps(diagnostic.to_payload(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        diagnostic.grid_points = 0


def test_batch_snapshot_history_abi_invalidation_and_old_library(
    monkeypatch: typing.Any, device: typing.Any
) -> None:
    calculator = Calculator(
        method="pbe-uks",
        device=device,
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calculator.prepare_batch(
        [H3, [("H", (0, 0, 0))]], multiplicities=[2, 2]
    ) as batch:
        query = batch._library.vibeqc_batch_get_ks_diagnostic
        summary = _native.KsDiagnosticDescriptor(
            ctypes.sizeof(_native.KsDiagnosticDescriptor), _native.ABI_VERSION
        )
        original = bytes(summary)
        assert (
            query(batch._batch, 0, ctypes.byref(summary), None, 0)
            == _native.STATUS_NOT_IMPLEMENTED
        )
        assert bytes(summary) == original
        assert query(batch._batch, 2, None, None, 0) == _native.STATUS_INVALID_ARGUMENT
        cold = batch.execute(strict=True)
        saved = cold.items[0].ks_diagnostic.to_payload()
        replay = batch.execute(strict=True)
        assert replay.items[0].ks_diagnostic.initial_density_used
        assert replay.items[0].ks_diagnostic.history[0].energy_change is None
        assert cold.items[0].ks_diagnostic.to_payload() == saved
        assert (
            query(batch._batch, 0, ctypes.byref(summary), None, 0)
            == _native.STATUS_SUCCESS
        )
        rows = (_native.KsIterationDescriptor * (summary.history_count + 1))()
        for row in rows:
            row.struct_size, row.abi_version = ctypes.sizeof(row), _native.ABI_VERSION
        # Validate the whole output before touching either summary or rows.
        for capacity, bad_row in (
            (summary.history_count - 1, None),
            (summary.history_count, summary.history_count - 1),
        ):
            if bad_row is not None:
                rows[bad_row].abi_version += 1
            before = bytes(summary), bytes(rows)
            status = query(batch._batch, 0, ctypes.byref(summary), rows, capacity)
            assert status == (
                _native.STATUS_INVALID_ARGUMENT
                if bad_row is None
                else _native.STATUS_ABI_MISMATCH
            )
            assert (bytes(summary), bytes(rows)) == before
            if bad_row is not None:
                rows[bad_row].abi_version = _native.ABI_VERSION
        canary = bytes(rows[-1])
        assert (
            query(batch._batch, 0, ctypes.byref(summary), rows, len(rows))
            == _native.STATUS_SUCCESS
        )
        assert bytes(rows[-1]) == canary
        failed = batch.execute([np.full((3, 3), np.nan), None])
        assert failed.failure_indices == (0,)
        assert failed.items[0].ks_diagnostic is None
        assert failed.items[1].ks_diagnostic is not None
        assert query(batch._batch, 0, None, None, 0) == _native.STATUS_NOT_IMPLEMENTED
        assert cold.items[0].ks_diagnostic.to_payload() == saved
        batch.execute(strict=True)
        outputs = (_native.BatchItemResultDescriptor * 1)()
        assert (
            batch._library.vibeqc_batch_execute(batch._batch, None, 0, outputs, 1)
            == _native.STATUS_INVALID_ARGUMENT
        )
        assert query(batch._batch, 1, None, None, 0) == _native.STATUS_NOT_IMPLEMENTED
        monkeypatch.setattr(batch._library, "vibeqc_batch_get_ks_diagnostic", None)
        assert batch.execute(strict=True).items[0].ks_diagnostic is None


def test_valid_iteration_limit_keeps_its_actual_history(
    device: typing.Any,
) -> None:
    calculator = Calculator(method="pbe-uks", device=device, max_iterations=1)
    with calculator.prepare_batch([H3], multiplicities=[2]) as batch:
        result = batch.execute().items[0]
        assert result.status == _native.STATUS_NOT_CONVERGED
        diagnostic = result.ks_diagnostic
        assert len(diagnostic.history) == 1
        assert diagnostic.components.total == result.energy
        assert diagnostic.history[0].energy_change is None
        assert diagnostic.fock_builds == 1
        assert not diagnostic.initial_density_used
        json.dumps(diagnostic.to_payload(), allow_nan=False)


def test_hf_has_no_ks_snapshot() -> None:
    calculator = Calculator(method="rhf", device="cpu")
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    assert calculator.singlepoint(atoms, properties=("energy",)).ks_diagnostic is None
    with calculator.prepare_batch([atoms]) as batch:
        assert batch.execute(strict=True).items[0].ks_diagnostic is None
        assert (
            batch._library.vibeqc_batch_get_ks_diagnostic(
                batch._batch, 0, None, None, 0
            )
            == _native.STATUS_NOT_IMPLEMENTED
        )


def test_cpu_and_old_libraries_report_no_cuda_ks_transport(
    monkeypatch: typing.Any,
) -> None:
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calculator = Calculator(method="lda-rks", device="cpu")
    assert calculator.singlepoint(atoms).ks_transport_diagnostic is None
    with calculator.prepare_batch([atoms]) as batch:
        assert batch.ks_transport_diagnostics == (None,)
        query = batch._library.vibeqc_batch_get_ks_transport_diagnostic
        value = _native.KsTransportDiagnosticDescriptor(
            ctypes.sizeof(_native.KsTransportDiagnosticDescriptor),
            _native.ABI_VERSION,
        )
        value.setup_h2d_bytes = 19
        assert query(batch._batch, 0, None) == _native.STATUS_NOT_IMPLEMENTED
        assert (
            query(batch._batch, 0, ctypes.byref(value))
            == _native.STATUS_NOT_IMPLEMENTED
        )
        assert value.setup_h2d_bytes == 19
        assert query(batch._batch, 1, None) == _native.STATUS_INVALID_ARGUMENT
        monkeypatch.setattr(
            batch._library, "vibeqc_batch_get_ks_transport_diagnostic", None
        )
        assert batch.ks_transport_diagnostics == (None,)


def test_cuda_ks_transport_covers_setup_replay_and_geometry_rebuild(
    device: typing.Any,
) -> None:
    if device != "cuda":
        pytest.skip("transport ledger is CUDA-only")
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    changed = np.asarray([[0, 0, -0.7], [0, 0, 0.735]])
    calculator = Calculator(
        method="pbe-rks",
        device="cuda",
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calculator.prepare_batch([atoms]) as batch:
        setup = batch.ks_transport_diagnostics[0]
        assert isinstance(setup, KsTransportDiagnostic)
        assert setup.setup_h2d_bytes > 0
        assert setup.density_h2d_bytes == setup.scalar_d2h_bytes == 0
        assert setup.matrix_d2h_bytes == setup.iterations == 0

        cold_result = batch.execute(strict=True).items[0]
        cold = batch.ks_transport_diagnostics[0]
        assert cold.setup_h2d_bytes == setup.setup_h2d_bytes
        assert cold.density_h2d_bytes > 0
        assert cold.scalar_d2h_bytes > 0
        assert cold.matrix_d2h_bytes == 0
        assert cold.iterations == cold_result.iterations
        assert cold.synchronizations > setup.synchronizations

        warm_result = batch.execute(strict=True).items[0]
        warm = batch.ks_transport_diagnostics[0]
        assert warm.setup_h2d_bytes == cold.setup_h2d_bytes
        assert warm.density_h2d_bytes == cold.density_h2d_bytes
        assert warm.matrix_d2h_bytes == 0
        assert warm.iterations == cold.iterations + warm_result.iterations
        assert warm.scalar_d2h_bytes > cold.scalar_d2h_bytes

        changed_result = batch.execute([changed], strict=True).items[0]
        rebuilt = batch.ks_transport_diagnostics[0]
        assert changed_result.warm_start_used
        assert rebuilt.setup_h2d_bytes > warm.setup_h2d_bytes
        assert rebuilt.density_h2d_bytes > warm.density_h2d_bytes
        assert rebuilt.matrix_d2h_bytes > warm.matrix_d2h_bytes
        assert rebuilt.iterations == warm.iterations + changed_result.iterations
        assert rebuilt.synchronizations > warm.synchronizations
        json.dumps(rebuilt.to_payload(), allow_nan=False)
        with pytest.raises(FrozenInstanceError):
            rebuilt.iterations = 0


def test_cold_retry_replaces_the_failed_warm_attempt_history(
    device: typing.Any,
) -> None:
    """A normalized virtual determinant forces a retry within a two-step limit."""
    from vibeqc import cross_overlap

    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calculator = Calculator(method="pbe-rks", device=device, max_iterations=2)
    overlap = cross_overlap(calculator, calculator, atoms)
    with calculator.prepare_batch([atoms]) as batch:
        cold = batch.execute(strict=True).items[0]
        state = _native.HfWarmState(
            ctypes.sizeof(_native.HfWarmState), _native.ABI_VERSION
        )
        getter = batch._library.vibeqc_batch_get_hf_warm_state
        _native.check(batch._library, getter(batch._batch, 0, ctypes.byref(state)))
        density, coordinates = (
            np.empty(state.density_count),
            np.empty(state.coordinate_count),
        )
        state.density = density.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        state.coordinates = coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _native.check(batch._library, getter(batch._batch, 0, ctypes.byref(state)))
        # The complement is another normalized/idempotent determinant. Cached
        # energy metadata must not replace rebuilding its actual physical F/D.
        density[:] = (2 * np.linalg.inv(overlap) - density.reshape(2, 2)).ravel()
        states = (_native.HfWarmState * 1)(state)
        _native.check(
            batch._library,
            batch._library.vibeqc_batch_restore_hf_warm_states(batch._batch, states, 1),
        )
        retried = batch.execute(strict=True).items[0]
        assert retried.warm_start_used and retried.warm_start_fallback
        assert retried.energy == pytest.approx(cold.energy, abs=1e-9)
        diagnostic = retried.ks_diagnostic
        assert not diagnostic.initial_density_used
        assert len(diagnostic.history) == retried.iterations == 2
        assert diagnostic.history[0].energy_change is None
        assert diagnostic.components.total == retried.energy

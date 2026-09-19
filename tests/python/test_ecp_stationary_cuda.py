"""Opt-in real-device ECP stationary gradients; independent CPU/PySCF oracles."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_dft_complete_cuda import no_cpu_derivatives
from test_ecp import fixture
from test_ecp_stationary_cpu import GRID, reference
from vibeqc import Calculator, KsOptions
from vibeqc._dft_gradient import StationaryDerivativeContract, StationaryKsState
from vibeqc._stationary_cuda import complete_rks_cuda_gradient_diagnostic
from vibeqc_compiler.dft import NativeAO

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_ECP_CUDA_TEST") != "1",
    reason="explicit real-device ECP gate",
)


@pytest.fixture(scope="module")
def compiler():
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.cuda_target import cuda_target_info

    # The caller selects an allocated GPU and its actual target explicitly.
    return CudaCompilerAdapter(
        Path(os.environ["CUDACXX"]),
        cuda_target_info(os.environ["VIBEQC_ECP_CUDA_TARGET"]),
        compile_timeout=600,
    )


def diagnostic(state, basis, compiler, **kwargs):
    with no_cpu_derivatives():
        return complete_rks_cuda_gradient_diagnostic(
            state,
            basis,
            compiler=compiler,
            cache=os.environ.get(
                "VIBEQC_STATIONARY_CACHE", ".cache/ecp-stationary-cuda"
            ),
            tile_points=137,
            primitive_tile=29,
            integral_terms=17,
            **kwargs,
        )


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_ecp_complete_cuda_gradient_analytic_fd_and_live_owner(
    method, record_property, compiler
):
    spin = int(method.endswith("uks"))
    atoms, record, mol = fixture(spin=spin, representation="cartesian")
    calc = Calculator(
        basis=record,
        method=method,
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with (
        calc.prepare_batch([atoms], charges=[spin], multiplicities=[spin + 1]) as batch,
        NativeAO(
            atoms,
            basis=record,
            charge=spin,
            multiplicity=spin + 1,
        ) as basis,
    ):
        energy = batch.execute(strict=True, properties=("energy",)).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        assert state._source.metadata[0] == 5
        assert state._source.ecp_cores == (10, 0)
        assert state._source.hamiltonian == "scalar-semilocal-ecp"
        assert state.occupations.sum() == 2 - spin
        result = diagnostic(state, basis, compiler)
        assert result.execution.startswith("cuda-nine-source/")
        assert len(result.components) == 9
        assert (
            result.work["ecp_derivative_export_bytes"]
            == 48 * basis.natom * basis.nao**2
        )
        assert (
            result.work["additional_device_peak_bound"]
            <= result.work["additional_device_budget"]
        )
        from vibeqc.ecp import ecp_integrals

        oracle = ecp_integrals(
            atoms,
            record,
            charge=spin,
            multiplicity=spin + 1,
            radial_points=224,
            polar_points=44,
        )
        raw = state._source.ecp_derivatives()
        np.testing.assert_allclose(raw[0], oracle.local_derivative, atol=2e-9, rtol=0)
        np.testing.assert_allclose(
            raw[1], oracle.nonlocal_derivative, atol=2e-9, rtol=0
        )
        expected_energy, expected = reference(mol, state, method)
        assert abs(energy - expected_energy) < 2e-8
        np.testing.assert_allclose(result.gradient, expected, atol=1e-7, rtol=0)
        np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=1e-9, rtol=0)
        ecp = result.components["ecp_local"] + result.components["ecp_nonlocal"]
        assert np.max(np.abs(ecp)) > 1e-3
        assert np.max(np.abs(result.gradient - ecp - expected)) > 1e-3
        # Effective ionic charges are +1/+1, not bare Na(+11)/H(+1).
        delta = mol.atom_coord(0) - mol.atom_coord(1)
        np.testing.assert_allclose(
            result.components["nuclear"][0],
            -delta / np.linalg.norm(delta) ** 3,
            atol=1e-13,
        )
        record_property("work", dict(result.work))
        record_property(
            "analytic_max_error", float(np.max(np.abs(result.gradient - expected)))
        )
        direction = np.array([[0.2, -0.13, 0.07], [-0.11, 0.08, 0.19]])
        projection = float(np.sum(result.gradient * direction))
        estimates = []
        for step in (1e-3, 3e-4, 1e-4):
            energies = []
            for sign in (1, -1):
                moved = [
                    (a, np.asarray(r) + sign * step * d)
                    for (a, r), d in zip(atoms, direction)
                ]
                energies.append(
                    calc.singlepoint(
                        moved,
                        charge=spin,
                        multiplicity=spin + 1,
                        properties=("energy",),
                    ).energy
                )
            estimates.append((energies[0] - energies[1]) / (2 * step))
        np.testing.assert_allclose(estimates[-2:], projection, atol=2e-7, rtol=0)
        record_property("fd_errors", [abs(x - projection) for x in estimates])
        with pytest.raises(ValueError, match="identity"):
            forged = replace(
                state, identity=replace(state.identity, model_identity="all-electron")
            )
            StationaryDerivativeContract(forged.identity).validate(forged)
        # A fresh ECP owner must never authorize the old derivative proof.
        batch.execute(strict=True, properties=("energy",))
        with pytest.raises(ValueError, match="stale"):
            state._source.ecp_derivatives()
        current = StationaryKsState.from_native(batch, basis)
        assert current.identity.solve_epoch > state.identity.solve_epoch
        assert current._source.ecp_terms == state._source.ecp_terms
        replay = diagnostic(current, basis, compiler)
        np.testing.assert_allclose(replay.gradient, result.gradient, atol=1e-9, rtol=0)
        with pytest.raises(ValueError, match="forces"):
            calc.singlepoint(
                atoms,
                charge=spin,
                multiplicity=spin + 1,
                properties=("energy", "forces"),
            )


def test_cuda_same_core_count_different_ecp_is_bound_to_actual_energy_owner():
    import json

    atoms, record, _ = fixture(representation="cartesian")
    changed = []
    for element in record.elements:
        if element.ecp_core_electrons:
            potentials = json.loads(element.ecp_data)
            potentials[0]["coefficients"][0][0] = str(
                float(potentials[0]["coefficients"][0][0]) * 1.01
            )
            element = replace(element, ecp_data=json.dumps(potentials))
        changed.append(element)
    other_record = replace(record, elements=tuple(changed))
    options = {
        "method": "lda-rks",
        "device": "cuda",
        "ks_options": KsOptions(grid=GRID),
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    with (
        Calculator(basis=record, **options).prepare_batch([atoms]) as batch,
        Calculator(basis=other_record, **options).prepare_batch([atoms]) as other,
        NativeAO(atoms, basis=record) as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        other.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
        changed_state = StationaryKsState.from_native(other, basis)
        assert state.identity.basis_identity == changed_state.identity.basis_identity
        assert state._source.ecp_cores == changed_state._source.ecp_cores
        assert state._source.ecp_terms != changed_state._source.ecp_terms
        assert (
            np.max(
                np.abs(
                    state._source.ecp_derivatives()
                    - changed_state._source.ecp_derivatives()
                )
            )
            > 1e-6
        )
        with pytest.raises(ValueError, match="identity"):
            StationaryDerivativeContract(state.identity).validate(
                replace(state, _source=changed_state._source)
            )
    with pytest.raises(RuntimeError, match="closed"):
        state._source.ecp_derivatives()


def test_cuda_ecp_admission_failure_recovery_and_legacy_guard(compiler, monkeypatch):
    from vibeqc._ks_snapshot import NativeKsSnapshot

    atoms, record, _ = fixture(representation="cartesian")
    calc = Calculator(
        basis=record,
        method="pbe-rks",
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calc.prepare_batch([atoms]) as batch, NativeAO(atoms, basis=record) as basis:
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
        original = NativeKsSnapshot.ecp_derivatives

        def forbidden(*args):
            pytest.fail("budget rejection occurred after ECP execution")

        with monkeypatch.context() as patch:
            patch.setattr(NativeKsSnapshot, "ecp_derivatives", forbidden)
            for kwargs in (
                {"max_device_bytes": 1},
                {"max_host_bytes": 1},
                {"max_primitive_records": 1},
            ):
                with pytest.raises(ValueError, match="budget"):
                    diagnostic(state, basis, compiler, **kwargs)

        def nonfinite(source):
            result = np.array(original(source))
            result[0, 0, 0, 0, 0] = np.nan
            return result

        with monkeypatch.context() as patch:
            patch.setattr(NativeKsSnapshot, "ecp_derivatives", nonfinite)
            with pytest.raises((ValueError, RuntimeError), match="finite"):
                diagnostic(state, basis, compiler)
        # Fresh transaction after a late TensorIR failure must recover.
        result = diagnostic(state, basis, compiler)
        assert np.isfinite(result.gradient).all()
        # Old CUDA wire has no ECP records: preserve its fail-closed guard.
        source = state._source
        metadata, cores, hamiltonian = (
            source.metadata,
            source.ecp_cores,
            source.hamiltonian,
        )
        try:
            object.__setattr__(source, "metadata", (3, *metadata[1:]))
            object.__setattr__(source, "ecp_cores", (0, 0))
            object.__setattr__(source, "hamiltonian", "unbound")
            with pytest.raises(NotImplementedError, match="bound ECP"):
                diagnostic(state, basis, compiler)
        finally:
            object.__setattr__(source, "metadata", metadata)
            object.__setattr__(source, "ecp_cores", cores)
            object.__setattr__(source, "hamiltonian", hamiltonian)


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_all_electron_cuda_v3_regression(method, compiler):
    from test_dft_complete_cpu import (
        ATOMS,
        independent_gradient,
        independent_uks_gradient,
    )

    spin = int(method.endswith("uks"))
    calc = Calculator(
        method=method,
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
    )
    with (
        calc.prepare_batch([ATOMS], charges=[spin], multiplicities=[spin + 1]) as batch,
        NativeAO(ATOMS, charge=spin, multiplicity=spin + 1) as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
        assert state._source.metadata[0] == 3
        result = diagnostic(state, basis, compiler)
        oracle = (independent_uks_gradient if spin else independent_gradient)(
            basis, state, method
        )
        np.testing.assert_allclose(result.gradient, oracle[1], atol=1e-7, rtol=0)
        assert result.execution.startswith("cuda-seven-source/")

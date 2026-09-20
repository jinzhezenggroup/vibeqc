"""Native CUDA CPKS with the same independent endpoint gates as CPU KS.

The shared oracle assertions use libcint/Libxc and independently reconverged
perturbations. They do not substitute a CPU VibeQC response as the reference.
"""

import os
import typing
from dataclasses import replace

import numpy as np
import pytest
from test_ecp import fixture as _ecp_fixture
from test_response_native_rks import (
    ATOMS,
    GRID,
    H2,
)
from test_response_native_rks import (
    test_action_finite_rotations_transpose_and_independent_fxc as _check_rks_action,
)
from test_response_native_rks import (
    test_complete_solve_and_reconverged_density_response as _check_rks_solve,
)
from test_response_native_uks import (
    LIH,
)
from test_response_native_uks import (
    test_solve_reconverged_spin_densities_and_recycling as _check_uks_solve,
)
from test_response_native_uks import (
    test_spin_action_finite_rotations_and_transpose as _check_uks_action,
)
from vibeqc import Calculator, KsOptions
from vibeqc._ks_snapshot import NativeKsSnapshot
from vibeqc_compiler.dft import ExplicitGrid, NativeAO
from vibeqc_compiler.xc import functional

from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import (
    NativeJKBackend,
    NativeRKSResponse,
    NativeUKSResponse,
    ResponseUnsupported,
    solve,
    solve_many,
)
from tools.vibeqc_response.native_ks import _NativeKSXCKernel

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESPONSE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated real GPU",
)


def _no_cpu(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    raise AssertionError("CPU AO/XC/J fallback or SCF rerun used by CUDA CPKS")


def _calculator(method: str) -> Calculator:
    return Calculator(
        method=method,
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )


@pytest.fixture(params=("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"), scope="module")
def native(request: typing.Any) -> typing.Iterator[typing.Any]:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    spin = request.param.endswith("uks")
    atoms = LIH if spin else ATOMS
    cls = NativeUKSResponse if spin else NativeRKSResponse
    with (
        _calculator(request.param).prepare_batch(
            [atoms],
            charges=[int(spin)],
            multiplicities=[1 + int(spin)],
        ) as batch,
        NativeAO(atoms, charge=int(spin), multiplicity=1 + int(spin)) as basis,
    ):
        result = batch.execute(strict=True)
        with cls.from_native(batch, basis, tile_points=257) as response:
            assert response.problem.reference.reference_energy == result.items[0].energy
            assert response.state._source.hamiltonian == "all-electron"
            yield response, basis


def _block_fallbacks(
    response: typing.Any, basis: typing.Any, monkeypatch: typing.Any
) -> None:
    monkeypatch.setattr(basis, "evaluate", _no_cpu)
    monkeypatch.setattr(NativeSource, "tile", _no_cpu)
    monkeypatch.setattr(NativeSource, "requests", _no_cpu)
    monkeypatch.setattr(NativeJKBackend, "coulomb_exchange", _no_cpu)
    monkeypatch.setattr(_NativeKSXCKernel, "_response_tile", _no_cpu)
    monkeypatch.setattr(NativeKsSnapshot, "evaluate_rks_response_points", _no_cpu)
    monkeypatch.setattr(NativeKsSnapshot, "evaluate_uks_response_points", _no_cpu)
    monkeypatch.setattr(response.state._source._batch, "execute", _no_cpu)


def test_cuda_action_finite_rotations_and_independent_kernel(
    native: typing.Any, monkeypatch: typing.Any
) -> None:
    response, basis = native
    _block_fallbacks(response, basis, monkeypatch)
    check = (
        _check_uks_action
        if response.problem.reference.algorithm == "UKS"
        else _check_rks_action
    )
    before = response.diagnostics["xc"]
    check(native)
    after = response.diagnostics["xc"]
    count = after["enqueues"] - before["enqueues"]
    assert count > 0
    matrix_bytes = after["spins"] * basis.nao**2 * 8
    assert (
        after["action_h2d_bytes"] - before["action_h2d_bytes"] == count * matrix_bytes
    )
    assert after["d2h_bytes"] - before["d2h_bytes"] == count * (matrix_bytes + 28)
    assert after["synchronizations"] - before["synchronizations"] == count * 2
    assert after["setup_h2d_bytes"] == before["setup_h2d_bytes"]
    # Preparing the response reads the actual native source a second time;
    # neither that export nor its fences may disappear from diagnostics.
    for name in (
        "preparation_export_d2h_bytes",
        "preparation_export_reads",
        "preparation_export_synchronizations",
    ):
        assert before[name] > 0
        assert after[name] == before[name]
    assert response.diagnostics["owned_device_bytes"] <= 128 << 20
    assert response.backend._plan.spec.exchange.present is False


def test_cuda_complete_solves_reconverged_density_and_multirhs(
    native: typing.Any, monkeypatch: typing.Any
) -> None:
    response, basis = native
    _block_fallbacks(response, basis, monkeypatch)
    check = (
        _check_uks_solve
        if response.problem.reference.algorithm == "UKS"
        else _check_rks_solve
    )
    check(native)


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"))
def test_cuda_resource_lifetime_and_identity_failures(
    method: str, monkeypatch: typing.Any
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    spin = method.endswith("uks")
    cls = NativeUKSResponse if spin else NativeRKSResponse
    with (
        _calculator(method).prepare_batch(
            [H2],
            charges=[int(spin)],
            multiplicities=[1 + int(spin)],
        ) as batch,
        NativeAO(H2, charge=int(spin), multiplicity=1 + int(spin)) as basis,
    ):
        batch.execute(strict=True)
        # A legacy library's CUDA wire layout alone does not prove that no
        # ECP/core-adjusted Hamiltonian was used. Require the live native proof.
        with monkeypatch.context() as legacy:
            legacy.setattr(batch._library, "vibeqc_ks_snapshot_hamiltonian_v1", None)
            with pytest.raises(ResponseUnsupported, match="all-electron"):
                cls.from_native(batch, basis)
        with pytest.raises(MemoryError):
            cls.from_native(batch, basis, device_budget_bytes=1)
        with cls.from_native(batch, basis, tile_points=127) as response:
            expected = response.apply(np.ones(response.dimension))
            if spin:
                assert response.problem.layout.block_dimension("beta") == 0
            bad = replace(response.problem.reference, geometry_hash="changed")
            with pytest.raises(ValueError, match="geometry_hash"):
                response.backend.validate_reference(bad)
            altered_grid = ExplicitGrid(
                response.state.grid.points,
                response.state.grid.weights * 1.001,
                response.state.grid.owners,
                {"bad": True},
            )
            with pytest.raises(ValueError, match="grid"):
                cls.from_native(batch, basis, altered_grid)
            wrong = functional(
                "PBE" if method.startswith("lda") else "LDA_XC_PW",
                spin="polarized" if spin else "unpolarized",
            )
            with pytest.raises(ValueError, match="functional"):
                cls.from_native(batch, basis, functional=wrong)
            # Native validation rejects an asymmetric direction before result
            # publication, then the same arena is reset for a valid replay.
            owner = response.xc_kernel._cuda
            direction = np.zeros(owner.shape)
            direction[0, 0, 1] = 1.0
            with pytest.raises(RuntimeError, match="response"):
                owner.apply(direction)
            np.testing.assert_allclose(
                response.apply(np.ones(response.dimension)), expected, atol=2e-12
            )
            if spin:
                direction.fill(0)
                direction[1, 0, 0] = 1.0
                with pytest.raises(RuntimeError, match="response"):
                    owner.apply(direction)
                np.testing.assert_allclose(
                    response.apply(np.ones(response.dimension)), expected, atol=2e-12
                )
            response.xc_kernel.close()
            with pytest.raises(RuntimeError, match="closed"):
                solve(response, np.zeros(response.dimension))
        with cls.from_native(batch, basis) as response:
            batch.execute(strict=True)
            for strategy in ("sequential", "blocked", "recycled"):
                with pytest.raises(ValueError, match="stale"):
                    solve_many(
                        response, np.zeros((response.dimension, 2)), strategy=strategy
                    )
        with cls.from_native(batch, basis) as response:
            failed = batch.execute(coordinates=[[0.0]], strict=False)
            assert not failed.items[0].succeeded
            with pytest.raises(ValueError, match="stale"):
                solve(response, np.zeros(response.dimension))
        batch.execute(strict=True)
        with cls.from_native(batch, basis) as response:
            batch.close()
            with pytest.raises((ValueError, RuntimeError)):
                solve(response, np.zeros(response.dimension))


@pytest.mark.parametrize("restricted", (True, False))
def test_cuda_meta_gga_is_not_promoted_to_cpks(restricted: bool) -> None:
    """Actual r2SCAN convergence does not qualify a tau response kernel."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    method = "r2scan-rks" if restricted else "r2scan-uks"
    adapter = NativeRKSResponse if restricted else NativeUKSResponse
    with _calculator(method).prepare_batch([H2]) as batch, NativeAO(H2) as basis:
        # Only the converged state is needed; CUDA tau nuclear gradients have
        # their own capability boundary and do not participate in this gate.
        batch.execute(strict=True, properties=("energy",))
        with pytest.raises(ValueError, match="LDA/PBE"):
            adapter.from_native(batch, basis)


@pytest.mark.parametrize("spin", (0, 1))
def test_cuda_ecp_is_not_promoted_to_all_electron_response(spin: int) -> None:
    """A real ECP KS state must not inherit all-electron CPKS support."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms, record, _ = _ecp_fixture(spin=spin, representation="cartesian")
    adapter = NativeUKSResponse if spin else NativeRKSResponse
    calculator = Calculator(
        basis=record,
        method="lda-uks" if spin else "lda-rks",
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with (
        calculator.prepare_batch(
            [atoms], charges=[spin], multiplicities=[spin + 1]
        ) as batch,
        NativeAO(atoms, basis=record, charge=spin, multiplicity=spin + 1) as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        with pytest.raises((ValueError, ResponseUnsupported), match="all-electron"):
            adapter.from_native(batch, basis)

"""Public DF-KS energy endpoints against independently converged PySCF states."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from vibeqc import Atom, Calculator, KsOptions
from vibeqc_compiler.dft.grid import GridSpec, MolecularGrid


@pytest.fixture(params=("cpu", "cuda"))
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "cuda" and os.environ.get("VIBEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an allocated native CUDA library/device")
    return request.param


def calculator(device: str, method: str = "pbe-rks", **kwargs: Any) -> Calculator:
    """Use tight physical convergence and an explicitly selected auxiliary basis."""
    if method.startswith(("pbe0", "b3lyp")):
        kwargs.setdefault("ks_options", KsOptions(grid=GridSpec()))
    return Calculator(
        method=method,
        basis="sto-3g",
        device=device,
        density_fitting="auto",
        auxiliary_basis="def2-svp",
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        **kwargs,
    )


WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]


def pyscf_energy(
    calc: Calculator, raw_atoms: Any, multiplicity: int, functional: str
) -> float:
    """Copy orbital/auxiliary primitives and quadrature; only the solvers differ."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    pyscf.lib.num_threads(1)
    atoms = tuple(Atom.from_value(a) for a in raw_atoms)
    labels = [f"{symbol}{i}" for i, (symbol, _) in enumerate(raw_atoms)]

    def basis_dict(selected: Any) -> dict:
        result = {label: [] for label in labels}
        for shell in calc._shells_for_atoms(atoms, selected):
            result[labels[shell.atom_index]].append(
                [
                    shell.angular_momentum,
                    *[(p.exponent, p.coefficient) for p in shell.primitives],
                ]
            )
        return result

    mol = gto.M(
        atom=[(label, atom.position) for label, atom in zip(labels, atoms)],
        basis=basis_dict(calc._basis),
        unit="Bohr",
        cart=True,
        spin=multiplicity - 1,
        verbose=0,
    )
    mf = (dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)).density_fit(
        auxbasis=basis_dict(calc._auxiliary_basis)
    )
    grid = MolecularGrid(
        atoms, spec=calc.ks_options.grid, multiplicity=multiplicity
    ).explicit()
    mf.xc = functional
    mf.grids.coords = np.asarray(grid.points)
    mf.grids.weights = np.asarray(grid.weights)
    mf.small_rho_cutoff = 0
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf.e_tot


@pytest.mark.parametrize(
    "method,functional,atoms,multiplicity",
    [
        ("lda-rks", "LDA_X,LDA_C_PW", WATER, 1),
        ("pbe-rks", "PBE", WATER, 1),
        ("r2scan-rks", "R2SCAN", WATER, 1),
        ("pbe-uks", "PBE", [("Li", (0.0, 0.0, 0.0))], 2),
    ],
)
def test_df_semilocal_matches_independent_reference(
    device: str, method: str, functional: str, atoms: Any, multiplicity: int
) -> None:
    calc = calculator(device, method)
    result = calc.singlepoint(atoms, multiplicity=multiplicity, properties=("energy",))
    assert result.converged and result.forces is None
    assert result.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    assert result.physical_residual_rms < 1e-9
    assert result.energy == pytest.approx(
        pyscf_energy(calc, atoms, multiplicity, functional), abs=1e-8
    )


@pytest.mark.parametrize(
    "method,functional", [("pbe0-rks", "PBE0"), ("b3lyp-rks", "B3LYP")]
)
def test_df_cpu_global_hybrid_matches_independent_reference(
    method: str, functional: str
) -> None:
    calc = calculator("cpu", method)
    result = calc.singlepoint(WATER, properties=("energy",))
    assert result.energy == pytest.approx(
        pyscf_energy(calc, WATER, 1, functional), abs=1e-8
    )


def test_df_batch_warm_replay_rebinds_auxiliary_centers(device: str) -> None:
    calc = calculator(device)
    moved = [
        (symbol, (x, y, z + (0.04 if i == 1 else 0.0)))
        for i, (symbol, (x, y, z)) in enumerate(WATER)
    ]
    expected = calc.singlepoint(moved, properties=("energy",)).energy
    with calc.prepare_batch([WATER, WATER], warm_start=True) as batch:
        cold = batch.execute(strict=True)
        # Private/native snapshots also reject the conventional force path.
        from vibeqc._ks_snapshot import NativeKsSnapshot

        with pytest.raises(NotImplementedError):
            NativeKsSnapshot(batch, 0)
        warm = batch.execute(strict=True)
        updated = batch.execute(
            coordinates=[None, [xyz for _, xyz in moved]], strict=True
        )
        assert warm.energies == pytest.approx(cold.energies, abs=1e-9)
        assert all(item.warm_start_used for item in warm.items)
        assert updated.items[1].energy == pytest.approx(expected, abs=1e-9)
        if device == "cuda":
            metrics = batch.last_density_fitting_metric_diagnostics()
            assert [item.system_index for item in metrics] == [0, 1]
            assert [item.bucket_id for item in metrics] == [0, 1]
            assert all(
                item.effective_rank > 0 and item.device_resident_bytes > 0
                for item in metrics
            )
            transport = batch.ks_transport_diagnostics
            assert all(item.matrix_d2h_bytes == 0 for item in transport)


def test_df_rejects_unqualified_force_precision_and_resource_consumers(
    device: str,
) -> None:
    calc = calculator(device)
    with pytest.raises(ValueError, match="does not support.*forces"):
        calc.singlepoint(WATER, properties=("energy", "forces"))
    with pytest.raises(NotImplementedError, match="resource plans"):
        calc.estimate_resources([WATER])
    if device == "cuda":
        with pytest.raises(NotImplementedError, match="fp64"):
            calculator(device, precision="auto")
        with pytest.raises(ValueError, match="backend must match"):
            Calculator(method="pbe-rks", device=device, density_fitting="cpu")


def test_df_cuda_provider_budget_is_not_silently_enlarged(device: str) -> None:
    if device != "cuda":
        pytest.skip("DF device allocation budget")
    with pytest.raises(
        (MemoryError, RuntimeError), match="budget|memory|fit|workspace"
    ):
        calculator(device, density_fitting_memory_budget_bytes=1).singlepoint(WATER)


def test_df_default_auxiliary_ragged_batch(device: str) -> None:
    calc = Calculator(
        method="pbe-rks",
        basis="sto-3g",
        device=device,
        density_fitting="auto",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    systems = [
        [("He", (0.0, 0.0, 0.0))],
        [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    ]
    expected = [pyscf_energy(calc, atoms, 1, "PBE") for atoms in systems]
    with calc.prepare_batch(systems) as batch:
        result = batch.execute(strict=True)
        assert result.energies == pytest.approx(expected, abs=1e-8)

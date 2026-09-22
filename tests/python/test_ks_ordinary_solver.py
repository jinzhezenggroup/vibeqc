"""Allocated energy endpoints and final-state handoff above the tiny KS solver."""

import os
import typing

import numpy as np
import pytest
from vibeqc import Atom, Calculator, ResourceBudget
from vibeqc._dft_gradient import StationaryKsState
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.grid import MolecularGrid

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_DFT_CUDA_TEST") != "1", reason="allocated CUDA KS gate"
)
WATER = [("O", (0, 0, 0)), ("H", (0, -1.43, 1.11)), ("H", (0, 1.43, 1.11))]


def reference_energy(
    calculator: Calculator, atoms: typing.Any, charge: int, multiplicity: int
) -> float:
    """Independent PySCF oracle with exact primitive input and native quadrature."""
    from pyscf import dft, gto

    owned = tuple(Atom.from_value(atom) for atom in atoms)
    labels = [f"{atom[0]}{i}" for i, atom in enumerate(atoms)]
    basis = {label: [] for label in labels}
    for shell in calculator._shells_for_atoms(owned):
        basis[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[(p.exponent, p.coefficient) for p in shell.primitives],
            ]
        )
    mol = gto.M(
        atom=[
            (label, atom.position) for label, atom in zip(labels, owned, strict=True)
        ],
        unit="Bohr",
        basis=basis,
        cart=False,
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )
    grid = MolecularGrid(
        owned, spec=calculator.ks_options.grid, charge=charge, multiplicity=multiplicity
    ).explicit()
    mf = dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)
    mf.xc = "PBE"
    mf.grids.coords, mf.grids.weights = np.array(grid.points), np.array(grid.weights)
    mf.small_rho_cutoff = 0
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 150
    mf.kernel()
    assert mf.converged
    return mf.e_tot


@pytest.mark.parametrize("unrestricted", (False, True))
def test_large_ks_solver_energy_replay_geometry_and_final_state(
    unrestricted: bool,
) -> None:
    """Exercise both spin slots, explicit energy selection and a charged owner."""
    charge, multiplicity = (1, 2) if unrestricted else (0, 1)
    kwargs = {
        "method": "pbe-uks" if unrestricted else "pbe-rks",
        "basis": "def2-svp",
        "basis_representation": "spherical",
        "device": "cuda",
        "max_iterations": 150,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    resolver = Calculator(**kwargs)
    plan = resolver.estimate_resources(
        [WATER], charges=[charge], multiplicities=[multiplicity]
    ).require_feasible()
    calculator = Calculator(
        **kwargs,
        resource_budget=ResourceBudget(
            host_bytes=plan.peak_bytes["host"], device_bytes=plan.peak_bytes["device"]
        ),
    )
    expected = reference_energy(calculator, WATER, charge, multiplicity)
    with calculator.prepare_batch(
        [WATER], charges=[charge], multiplicities=[multiplicity]
    ) as batch:
        for replay in (False, True):
            result = batch.execute(properties=("energy",), strict=True).items[0]
            assert result.warm_start_used is replay
            assert result.energy == pytest.approx(expected, abs=1e-8, rel=0)
            assert result.physical_residual_rms < 1e-9
        observed = batch.resource_diagnostics["observation"]["device_ledger"]
        # Shape-only cuSOLVER capacity is conservative; queried allocations may
        # be smaller. Warm execution must allocate no new provider workspace.
        assert 0 < observed["live_bytes"] <= plan.resident_bytes["device"]
        assert observed["allocations"] == 0 and observed["rejected_allocations"] == 0
        with NativeAO(
            WATER,
            basis="def2-svp",
            representation="spherical",
            charge=charge,
            multiplicity=multiplicity,
        ) as basis:
            state = StationaryKsState.from_native(batch, basis)
            assert state.density.shape == ((2 if unrestricted else 1), 24, 24)
            assert state.physical_residual < 1e-9
            state._source.close()
        moved = [
            (element, tuple(np.array(xyz) + (0, 0, 0.03) if i == 2 else xyz))
            for i, (element, xyz) in enumerate(WATER)
        ]
        result = batch.execute(
            [np.array([xyz for _, xyz in moved])], properties=("energy",), strict=True
        ).items[0]
        assert result.energy == pytest.approx(
            reference_energy(calculator, moved, charge, multiplicity), abs=1e-8, rel=0
        )

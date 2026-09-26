"""Opt-in matched-grid Libxc/PySCF acceptance for public CUDA split hybrids.

Run from a clean checkout on a Slurm-assigned GPU with
VIBEQC_SPLIT_HYBRID_CUDA_TEST=1.
The reference consumes exported quadrature and density, never production XC
code, and independently rebuilds J, K, semilocal XC, and the full SCF state.
"""

import hashlib
import json
import os
import subprocess
import typing
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_SPLIT_HYBRID_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)

H2 = (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7)))
H3 = (
    ("H", (0.0, 0.0, -1.4)),
    ("H", (0.15, 0.1, 0.0)),
    ("H", (0.0, 0.0, 1.4)),
)
H3_MOVED = (
    ("H", (0.0, 0.0, -1.4)),
    ("H", (0.17, 0.12, 0.02)),
    ("H", (0.0, 0.0, 1.43)),
)


def independent_reference(
    basis: typing.Any, state: typing.Any, method: str, polarized: bool
) -> tuple[dict[str, float], float, float]:
    """Evaluate the same physical density and independently reconverge the grid."""
    from pyscf import dft, gto, lib
    from pyscf.data.elements import ELEMENTS

    lib.num_threads(1)
    labels = [
        f"{ELEMENTS[atom.atomic_number]}{index}"
        for index, atom in enumerate(basis.atoms)
    ]
    shells = {label: [] for label in labels}
    for shell in basis.shells:
        shells[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[(p.exponent, p.coefficient) for p in shell.primitives],
            ]
        )
    molecule = gto.M(
        atom=[
            (label, atom.position)
            for label, atom in zip(labels, basis.atoms, strict=True)
        ],
        basis=shells,
        unit="Bohr",
        cart=True,
        charge=basis.charge,
        spin=basis.multiplicity - 1,
        verbose=0,
    )
    reference = dft.UKS(molecule) if polarized else dft.RKS(molecule)
    reference.xc = method
    reference.grids.coords = np.array(state.grid.points)
    reference.grids.weights = np.array(state.grid.weights)
    reference.small_rho_cutoff = 0
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-9
    reference.max_cycle = 180
    density = tuple(state.density) if polarized else state.density[0]
    total_density = sum(density) if polarized else density
    spin_k = reference.get_k(dm=density)
    exact_coefficient = {"M06-2X": 0.54, "MN15": 0.44}[method]
    exact_k = (
        -0.5
        * exact_coefficient
        * sum(
            np.einsum("ij,ji->", spin, term)
            for spin, term in zip(density, spin_k, strict=True)
        )
        if polarized
        else -0.25 * exact_coefficient * np.einsum("ij,ji->", density, spin_k)
    )
    numerical_integrator = reference._numint
    nr = numerical_integrator.nr_uks if polarized else numerical_integrator.nr_rks
    _, semilocal_xc, semilocal_potential = nr(
        molecule, reference.grids, method, density
    )
    veff = reference.get_veff(dm=density)
    independent = {
        "nuclear": float(molecule.energy_nuc()),
        "one_electron": float(
            np.einsum("ij,ji->", total_density, reference.get_hcore())
        ),
        "hartree": float(
            0.5 * np.einsum("ij,ji->", total_density, reference.get_j(dm=total_density))
        ),
        "semilocal_xc": float(semilocal_xc),
        "exact_k": float(exact_k),
    }
    np.testing.assert_allclose(
        veff.exc, independent["semilocal_xc"] + exact_k, atol=2e-9, rtol=0
    )
    hartree_potential = reference.get_j(dm=total_density)
    exact_k_potential = (
        -exact_coefficient * spin_k if polarized else -0.5 * exact_coefficient * spin_k
    )
    native_fock = state.fock if polarized else state.fock[0]
    native_exchange = (
        native_fock - reference.get_hcore() - hartree_potential - semilocal_potential
    )
    np.testing.assert_allclose(native_exchange, exact_k_potential, atol=2e-7, rtol=0)
    fock = reference.get_hcore() + veff
    np.testing.assert_allclose(native_fock, fock, atol=2e-7, rtol=0)
    reference.kernel()
    assert reference.converged
    density_error = float(
        np.max(np.abs(np.asarray(reference.make_rdm1()) - np.asarray(density)))
    )
    independent["exact_k_matrix_max_error"] = float(
        np.max(np.abs(native_exchange - exact_k_potential))
    )
    return independent, float(reference.e_tot), density_error


@pytest.mark.parametrize("method", ("M06-2X", "MN15"))
@pytest.mark.parametrize("polarized", (False, True))
def test_split_hybrid_complete_endpoint(method: str, polarized: bool) -> None:
    """Enforce decomposition, SCF, and recovery with both work spins occupied.

    The independent 113-bit point gate owns exact-empty spin: binary64 Libxc
    cancellation there cannot independently arbitrate the XC potential.
    """
    import pyscf
    from pyscf.dft import libxc
    from vibeqc import Calculator, KsOptions
    from vibeqc._dft_gradient import StationaryKsState
    from vibeqc_compiler.dft import NativeAO
    from vibeqc_compiler.dft.grid import GridSpec

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require a Slurm allocation"
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ), "split-hybrid endpoint acceptance requires a clean tested head"
    assert pyscf.__version__ == "2.14.0"
    assert libxc.libxc_version() == "7.0.0"
    selector = f"{method.lower()}-{'uks' if polarized else 'rks'}"
    systems = (H3, H3_MOVED) if polarized else (H2, (("He", (0.0, 0.0, 0.0)),))
    charges = (0, 0)
    multiplicities = (2, 2) if polarized else (1, 1)
    calculator = Calculator(
        method=selector,
        device="cuda",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=180,
    )
    begin = perf_counter()
    with calculator.prepare_batch(
        systems, charges=charges, multiplicities=multiplicities, warm_start=True
    ) as prepared:
        setup_seconds = perf_counter() - begin
        begin = perf_counter()
        cold = prepared.execute(properties=("energy",), strict=True)
        cold_seconds = perf_counter() - begin
        evidence = []
        for index, atoms in enumerate(systems):
            item = cold.items[index]
            with NativeAO(
                atoms, charge=charges[index], multiplicity=multiplicities[index]
            ) as basis:
                state = StationaryKsState.from_native(prepared, basis, index=index)
                assert state._source.metadata[0] == 8
                independent, converged_energy, density_error = independent_reference(
                    basis, state, method, polarized
                )
            diagnostic = item.ks_diagnostic
            assert item.executed_backend == "cuda" and item.converged
            assert diagnostic.scf_domain == calculator.ks_options.scf_domain
            assert diagnostic.fock_builds >= item.iterations > 0
            assert diagnostic.grid_points == len(state.grid.points)
            assert item.physical_residual_rms < 1e-9
            assert density_error < 1e-6
            assert abs(item.energy - converged_energy) < 2e-8
            components = diagnostic.components
            for name in ("nuclear", "one_electron", "hartree"):
                assert abs(getattr(components, name) - independent[name]) < 2e-8
            assert (
                abs(
                    components.xc - independent["semilocal_xc"] - independent["exact_k"]
                )
                < 2e-8
            )
            assert abs(components.total - item.energy) < 2e-9
            evidence.append(
                {
                    "energy": item.energy,
                    "reference_energy": converged_energy,
                    "density_error": density_error,
                    "independent_components": independent,
                    "native_components": components.__dict__,
                    "fock_builds": diagnostic.fock_builds,
                    "grid_points": diagnostic.grid_points,
                    "semantic_direct_jk_builds": diagnostic.fock_builds,
                    "semantic_xc_point_evaluations": diagnostic.fock_builds
                    * diagnostic.grid_points,
                }
            )
        begin = perf_counter()
        replay = prepared.execute(properties=("energy",), strict=True)
        replay_seconds = perf_counter() - begin
        for old, new in zip(cold.items, replay.items, strict=True):
            assert new.warm_start_used and new.executed_backend == "cuda"
            assert abs(new.energy - old.energy) < 2e-9
        with pytest.raises((ValueError, TypeError)):
            prepared.execute(
                coordinates=(((0.0,),),), properties=("energy",), strict=True
            )
        recovered = prepared.execute(properties=("energy",), strict=True)
        for old, new in zip(cold.items, recovered.items, strict=True):
            assert new.converged and abs(new.energy - old.energy) < 2e-9
        begin = perf_counter()
        single = calculator.singlepoint(
            systems[0],
            charge=charges[0],
            multiplicity=multiplicities[0],
            properties=("energy",),
        )
        single_seconds = perf_counter() - begin
        assert single.converged and single.executed_backend == "cuda"
        assert abs(single.energy - cold.items[0].energy) < 2e-9
        native_library = Path(os.environ["VIBEQC_LIBRARY"])
        payload = {
            "method": selector,
            "slurm_job": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
            "libxc_version": libxc.libxc_version(),
            "pyscf_version": pyscf.__version__,
            "native_library_sha256": hashlib.sha256(
                native_library.read_bytes()
            ).hexdigest(),
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "nvcc_version": subprocess.check_output(
                ["nvcc", "--version"], text=True
            ).strip(),
            "setup_seconds": setup_seconds,
            "cold_seconds": cold_seconds,
            "replay_seconds": replay_seconds,
            "single_seconds": single_seconds,
            "replay_fock_builds": [
                item.ks_diagnostic.fock_builds for item in replay.items
            ],
            "cases": evidence,
        }
        directory = os.environ.get("VIBEQC_SPLIT_HYBRID_EVIDENCE")
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / f"{selector}.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )

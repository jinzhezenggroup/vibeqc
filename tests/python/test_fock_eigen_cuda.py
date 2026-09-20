"""Independent fitted Fock consumers borrow the device provider by solve stage."""

import os
import typing

import numpy as np
import pytest
from vibeqc.fock import FockBuildSpec, FockPlan
from vibeqc_compiler.dft import NativeAO

from benchmarks.df_component_ledger import aggregate_host, read_host_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)
CONTROLS = {"energy_tolerance": 1e-12, "density_tolerance": 1e-10}


def calls(
    components: typing.Any, provider: typing.Any, reason: typing.Any
) -> typing.Any:
    """Count actual leaves, so a missing or unused callback cannot pass."""
    return components[provider].get(reason, {}).get("calls", 0)


@pytest.mark.parametrize("spin", ("restricted", "unrestricted"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize(
    "j,k",
    (
        ("density_fitted", "density_fitted"),
        ("exact", "density_fitted"),
        ("density_fitted", "exact"),
    ),
)
def test_independent_fitted_scf_providers_and_complete_forces(
    spin: typing.Any,
    representation: typing.Any,
    j: typing.Any,
    k: typing.Any,
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
) -> None:
    """Compare device setup/iteration/final solves to the independent CPU oracle.

    Oxygen's d functions distinguish the AO representations. A separate CUDA
    owner checks the explicit setup/final reference controls, while its SCF
    iterations still use the device. Immutable sources support energy/force
    replay without manufacturing a compact-solver snapshot or epoch.
    """
    from pyscf import gto

    assert os.environ.get("SLURM_JOB_ID")
    separate = spin == "unrestricted"
    # Bent H2O+ avoids OH's degenerate Pi occupation: equally valid density
    # orientations must not be mistaken for a device-provider discrepancy.
    atoms = [("O", (0, 0, 0)), ("H", (0, 0, 1.8)), ("H", (1.7, 0, -0.6))]
    controls = dict(CONTROLS)
    if separate and j != k:
        # The open-shell frontier perturbation has a solver-phase-dependent
        # sign. Its small out-of-plane remnant decays slowly for mixed J/K;
        # converge both independent owners more tightly before comparing D
        # elementwise, retaining the same 1e-8 physical comparison threshold.
        controls["density_tolerance"] = 1e-12
    spec = FockBuildSpec.hf(spin, coulomb=j, exchange=k)
    mol = gto.M(
        atom=atoms,
        basis="def2-svp",
        unit="Bohr",
        spin=int(separate),
        charge=int(separate),
        cart=representation == "cartesian",
        verbose=0,
    )
    overlap = mol.intor("int1e_ovlp")
    # Native Cartesian AOs are individually normalized; PySCF's Cartesian
    # d components retain angular normalization factors. Transform S into the
    # native convention before electron, idempotency and commutator checks.
    norms = np.sqrt(np.diag(overlap))
    overlap = overlap / np.outer(norms, norms)
    with (
        NativeAO(
            atoms,
            basis="def2-svp",
            representation=representation,
            multiplicity=2 if separate else 1,
            charge=int(separate),
        ) as basis,
        FockPlan(basis, spec, device="cpu") as oracle,
        FockPlan(basis, spec, device="cuda", device_budget_bytes=16 << 20) as device,
        FockPlan(
            basis, spec, device="cuda", device_budget_bytes=16 << 20
        ) as diagnostic,
    ):
        seeds = {}
        for step, forces in enumerate((False, True, False)):
            outputs = []
            for name, owner in (
                ("cpu", oracle),
                ("device", device),
                ("diagnostic", diagnostic),
            ):
                reference = name == "diagnostic"
                monkeypatch.setenv(
                    "VIBEQC_DF_REFERENCE_SETUP_EIGEN", str(int(reference))
                )
                monkeypatch.setenv(
                    "VIBEQC_DF_REFERENCE_FINAL_EIGEN", str(int(reference))
                )
                path = tmp_path / f"{step}-{name}.jsonl"
                monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(path))
                try:
                    result = owner.solve(
                        initial_density=seeds.get(name),
                        compute_forces=forces,
                        **controls,
                    )
                finally:
                    monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
                    monkeypatch.delenv("VIBEQC_DF_REFERENCE_SETUP_EIGEN")
                    monkeypatch.delenv("VIBEQC_DF_REFERENCE_FINAL_EIGEN")
                components = aggregate_host(read_host_trace(path))
                spins = 2 if separate else 1
                expected = {
                    "overlap": int(step == 0 or name == "cpu"),
                    "core_guess": int(step == 0),
                    "iteration": spins * result.iterations,
                    "seed_validation": (1 + spins) * int(step != 0),
                    "final_fock": spins,
                }
                for reason, count in expected.items():
                    host = name == "cpu" or (reference and reason != "iteration")
                    assert calls(components, "eigensolves_by_reason", reason) == (
                        count if host else 0
                    )
                    assert calls(
                        components, "device_eigensolves_by_reason", reason
                    ) == (0 if host else count)
                if name == "device":
                    assert not components["eigensolves_by_reason"]
                assert result.initial_density_used == (step != 0)
                physical = owner.evaluate(result.density)
                assert physical.energy == pytest.approx(result.energy, abs=1e-9)
                densities = result.density if separate else [result.density]
                focks = physical.fock
                focks = focks if separate else [focks]
                for density, fock, electrons in zip(
                    densities,
                    focks,
                    mol.nelec if separate else [mol.nelectron],
                    strict=True,
                ):
                    assert np.trace(density @ overlap) == pytest.approx(
                        electrons, abs=1e-9
                    )
                    np.testing.assert_allclose(
                        density @ overlap @ density,
                        (1 if separate else 2) * density,
                        atol=1e-8,
                        rtol=0,
                    )
                    np.testing.assert_allclose(
                        fock @ density @ overlap,
                        overlap @ density @ fock,
                        atol=1e-8,
                        rtol=0,
                    )
                if step == 0:
                    seeds[name] = result.density
                outputs.append(result)
            for result in outputs[1:]:
                assert result.energy == pytest.approx(outputs[0].energy, abs=1e-9)
                np.testing.assert_allclose(
                    result.density, outputs[0].density, atol=1e-8, rtol=0
                )
                if forces:
                    np.testing.assert_allclose(
                        result.forces, outputs[0].forces, atol=1e-8, rtol=0
                    )
                    np.testing.assert_allclose(
                        result.forces.sum(axis=0), 0, atol=1e-9, rtol=0
                    )
                else:
                    assert result.forces is None


def test_independent_fitted_empty_beta_and_failed_replay(
    monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    """An empty spin is solved explicitly; a failed replay leaves sources usable."""
    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("H", (0, 0, 0))]
    spec = FockBuildSpec.hf(
        "unrestricted", coulomb="density_fitted", exchange="density_fitted"
    )
    with (
        NativeAO(atoms, basis="def2-svp", multiplicity=2) as basis,
        FockPlan(basis, spec, device="cuda", device_budget_bytes=8 << 20) as device,
        FockPlan(basis, spec, device="cpu") as oracle,
    ):
        expected = oracle.solve(**CONTROLS)
        path = tmp_path / "empty-beta.jsonl"
        monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(path))
        result = device.solve(**CONTROLS)
        monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
        components = aggregate_host(read_host_trace(path))
        assert not components["eigensolves_by_reason"]
        assert (
            calls(components, "device_eigensolves_by_reason", "iteration")
            == 2 * result.iterations
        )
        assert calls(components, "device_eigensolves_by_reason", "final_fock") == 2
        np.testing.assert_array_equal(result.density[1], 0)
        np.testing.assert_allclose(result.forces, expected.forces, atol=1e-8, rtol=0)
        assert result.energy == pytest.approx(expected.energy, abs=1e-9)
        with pytest.raises(RuntimeError, match="did not converge"):
            device.solve(max_iterations=1, **CONTROLS)
        recovered = device.solve(initial_density=result.density, **CONTROLS)
        assert recovered.energy == pytest.approx(result.energy, abs=1e-9)
        np.testing.assert_allclose(recovered.density, result.density, atol=1e-8, rtol=0)
        with pytest.raises(MemoryError):
            FockPlan(basis, spec, device="cuda", device_budget_bytes=1)

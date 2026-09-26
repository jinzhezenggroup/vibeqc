"""Complete public CUDA hybrid forces against independent moving-grid PySCF.

Run with VIBEQC_HYBRID_FORCE_CUDA_TEST=1 in a finite Slurm GPU allocation.
Each case checks the converged endpoint, both reconverged finite-difference
steps, source accounting, and reuse of the prepared force owner.
"""

import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_HYBRID_FORCE_CUDA_TEST") != "1",
    reason="explicit Slurm CUDA hybrid-force gate",
)


def _record_evidence(name: str, payload: dict) -> None:
    directory = os.environ.get("VIBEQC_HYBRID_FORCE_EVIDENCE")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(
            json.dumps({"slurm_job": os.environ["SLURM_JOB_ID"], **payload}, indent=2)
            + "\n"
        )


@pytest.fixture(scope="module", autouse=True)
def pinned_reference() -> None:
    """Pin the independent MGGA worker semantics used by this acceptance gate."""
    import pyscf
    from pyscf.dft import libxc

    assert pyscf.__version__ == "2.14.0"
    assert libxc.libxc_version() == "7.0.0"


@pytest.mark.parametrize(
    "options",
    ({"precision": "auto"}, {"density_fitting": "cuda"}, {"host_unfused": True}),
)
def test_global_hybrid_force_does_not_inherit_unqualified_execution(
    options: dict,
) -> None:
    from vibeqc import Calculator, GridSpec, KsOptions

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    options = dict(options)
    schedule = "host_unfused" if options.pop("host_unfused", False) else "device_fused"
    calc = Calculator(
        method="pbe0-rks",
        device="cuda",
        ks_options=KsOptions(grid=GridSpec(), xc_schedule=schedule),
        **options,
    )
    assert "forces" not in calc._capabilities.supported_properties
    with pytest.raises((ValueError, NotImplementedError)):
        calc.singlepoint(
            [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))], properties=("energy", "forces")
        )


@pytest.mark.parametrize("name", ("PBE0", "B3LYP", "M06-2X", "MN15", "PBE0-alias"))
@pytest.mark.parametrize("spin", ("rks", "uks"))
def test_public_cuda_global_hybrid_force(name: str, spin: str) -> None:
    from test_dft_complete_cpu import independent_global_hybrid_gradient
    from test_dft_complete_cuda import no_cpu_derivatives
    from vibeqc import Calculator, GridSpec, KsOptions
    from vibeqc._dft_gradient import StationaryKsState
    from vibeqc_compiler.dft import NativeAO
    from vibeqc_compiler.method import MethodSpec, resolve_method

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [
        ("H", (0.13, -0.21, -1.3)),
        ("H", (-0.08, 0.16, 0.24)),
        *(([("H", (0.18, -0.04, 1.51))]) if spin == "uks" else []),
    ]
    multiplicity = 2 if spin == "uks" else 1
    method = f"{name.lower()}-{spin}"
    composition = None
    reference_xc = name
    if name == "PBE0-alias":
        # An admitted hybrid deliberately uses a semilocal public label.
        # Both energy and force must consume the owner's actual MethodIR.
        method = f"pbe-{spin}"
        reference_xc = "PBE0"
        composition = resolve_method(
            MethodSpec(
                "PBE0-label-independence",
                (("GGA_X_PBE", Fraction(3, 4)), ("GGA_C_PBE", Fraction(1))),
                exact_exchange=Fraction(1, 4),
            ),
            spin="polarized" if spin == "uks" else "unpolarized",
        )
    calc = Calculator(
        method=method,
        device="cuda",
        basis="sto-3g",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=32, angular_polar=10, angular_azimuth=20),
            composition=composition,
        ),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=200,
    )
    assert "forces" in calc._capabilities.supported_properties
    with calc.prepare_batch(
        [atoms], multiplicities=[multiplicity], warm_start=True
    ) as batch:
        with no_cpu_derivatives():
            public = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
        with NativeAO(atoms, multiplicity=multiplicity) as basis:
            primitive_count = basis.nprimitive
            state = StationaryKsState.from_native(batch, basis)
            try:
                reference_energy, reference_gradient = (
                    independent_global_hybrid_gradient(
                        basis,
                        state,
                        method,
                        xc=reference_xc,
                    )
                )
            finally:
                state._source.close()
        assert public.executed_backend == "cuda" and public.converged
        assert public.energy == pytest.approx(reference_energy, abs=2e-8)
        np.testing.assert_allclose(
            public.forces, -reference_gradient, atol=2e-7, rtol=0
        )
        np.testing.assert_allclose(public.forces.sum(axis=0), 0, atol=1e-9, rtol=0)
        with no_cpu_derivatives():
            replay = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
        np.testing.assert_allclose(replay.forces, public.forces, atol=2e-8, rtol=0)
        assert batch._stationary_cuda_execution._lease.executions == 2
        # Work contains a complete second quartet traversal for exact exchange;
        # the bounded implementation makes no fused-J/K performance claim.
        assert "exact_exchange" in batch._stationary_cuda_execution.sources.source_names
        per_execution = (
            2 * primitive_count**4
            + (len(atoms) + 2) * primitive_count**2
            + len(atoms) * (len(atoms) - 1) // 2
        )
        assert (
            batch._stationary_cuda_execution.sources.metrics()["primitive_records"]
            == 2 * per_execution
        )
        moved_xyz = np.asarray([position for _, position in atoms])
        moved_xyz[-1] += (0.02, -0.01, 0.03)
        with no_cpu_derivatives():
            moved = batch.execute(
                coordinates=(moved_xyz,), properties=("energy", "forces"), strict=True
            ).items[0]
            fresh = calc.singlepoint(
                [
                    (atom[0], position)
                    for atom, position in zip(atoms, moved_xyz, strict=True)
                ],
                multiplicity=multiplicity,
                properties=("energy", "forces"),
            )
        np.testing.assert_allclose(moved.forces, fresh.forces, atol=2e-8, rtol=0)
        geometry_reuse_error = float(np.max(np.abs(moved.forces - fresh.forces)))
        assert batch._stationary_cuda_execution._lease.refreshes == 1
        artifacts = {
            str(artifact.library): artifact.metadata["binary_sha256"]
            for artifact in batch._stationary_cuda_execution.artifacts
        }

    xyz = np.asarray([position for _, position in atoms])
    direction = np.array(
        [[0.13, -0.07, 0.11], [-0.05, 0.17, 0.03], [0.09, 0.02, -0.14]]
    )[: len(atoms)]
    estimates = []
    for step in (3e-4, 1e-4):
        energies = []
        for sign in (1, -1):
            moved = [
                (atom[0], position)
                for atom, position in zip(
                    atoms, xyz + sign * step * direction, strict=True
                )
            ]
            energies.append(
                calc.singlepoint(
                    moved, multiplicity=multiplicity, properties=("energy",)
                ).energy
            )
        estimates.append((energies[0] - energies[1]) / (2 * step))
    analytic = -float(np.sum(public.forces * direction))
    assert abs(estimates[-1] - estimates[-2]) < 1e-6
    assert abs(estimates[-1] - analytic) < 1e-6
    _record_evidence(
        f"{name.lower()}-{spin}",
        {
            "method": method,
            "reference_xc": reference_xc,
            "energy_error": abs(public.energy - reference_energy),
            "gradient_max_error": float(
                np.max(np.abs(public.forces + reference_gradient))
            ),
            "translation_residual": float(np.max(np.abs(public.forces.sum(axis=0)))),
            "primitive_records_per_execution": per_execution,
            "replay_max_error": float(np.max(np.abs(replay.forces - public.forces))),
            "geometry_reuse_max_error": geometry_reuse_error,
            "finite_difference": estimates,
            "analytic_directional_derivative": analytic,
            "artifacts": artifacts,
        },
    )


@pytest.mark.parametrize("spin", ("rks", "uks"))
def test_force_coverage_does_not_bypass_native_composition_admission(spin: str) -> None:
    """The generic pullback does not promote unqualified CUDA SCF fractions."""
    from vibeqc import Calculator, GridSpec, KsOptions
    from vibeqc_compiler.method import MethodSpec, resolve_method

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    composition = resolve_method(
        MethodSpec(
            "PBE50-unqualified-cuda-test",
            (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
            exact_exchange=Fraction(1, 2),
        ),
        spin="polarized" if spin == "uks" else "unpolarized",
    )
    calc = Calculator(
        method=f"pbe-{spin}",
        device="cuda",
        ks_options=KsOptions(grid=GridSpec(), composition=composition),
    )
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    with pytest.raises(NotImplementedError):
        calc.singlepoint(
            atoms,
            charge=1 if spin == "uks" else 0,
            multiplicity=2 if spin == "uks" else 1,
            properties=("energy", "forces"),
        )


def test_public_cuda_hybrid_p_shell_oracle() -> None:
    """Exercise non-s AO derivatives and Coulomb/exchange density permutations."""
    from test_dft_complete_cpu import ATOMS, independent_global_hybrid_gradient
    from test_dft_complete_cuda import no_cpu_derivatives
    from vibeqc import Calculator, GridSpec, KsOptions
    from vibeqc._dft_gradient import StationaryKsState
    from vibeqc_compiler.dft import NativeAO

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    calc = Calculator(
        method="b3lyp-rks",
        device="cuda",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
        ),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=200,
    )
    with calc.prepare_batch([ATOMS]) as batch, NativeAO(ATOMS) as basis:
        with no_cpu_derivatives():
            result = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
        state = StationaryKsState.from_native(batch, basis)
        try:
            energy, gradient = independent_global_hybrid_gradient(
                basis, state, "b3lyp-rks", xc="B3LYP"
            )
        finally:
            state._source.close()
        assert result.energy == pytest.approx(energy, abs=2e-8)
        np.testing.assert_allclose(result.forces, -gradient, atol=2e-7, rtol=0)
        _record_evidence(
            "b3lyp-rks-water",
            {
                "method": "b3lyp-rks",
                "energy_error": abs(result.energy - energy),
                "gradient_max_error": float(np.max(np.abs(result.forces + gradient))),
            },
        )

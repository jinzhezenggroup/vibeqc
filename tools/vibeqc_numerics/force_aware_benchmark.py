#!/usr/bin/env python3
"""Reproducible NUM03 force-aware grid/screening audit.

The strict endpoint is an offline oracle for *measurement*.  Its runtime is
excluded from adaptive-policy cost unless the policy actually tightens to the
strict level.  Paired endpoint cost, however, is charged to the policy because
the empirical estimator requires that work.

Fixed-density records are a diagnostic of XC quadrature only.  They hold the AO
density matrix fixed while rebuilding the AO basis and molecular grid under
nuclear displacements, so AO-center, grid-point and Becke partition motion enter
the finite difference while density relaxation does not.  They are never mixed
with the reconverged full-endpoint force error used for acceptance.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

TRANSITION_FAMILIES = (
    "coarse-to-standard-v1",
    "standard-to-strict-v1",
)

from vibeqc import (
    AdaptiveNumericsPolicy,
    Calculator,
    KsOptions,
    NumericalEstimate,
    NumericalLevel,
    NumericalTargetModel,
    ObservableDelta,
    PairedCalibrationSample,
    PairedDifferenceEstimator,
    Primitive,
    Shell,
    TargetErrorBudget,
    method_capabilities,
)
from vibeqc._dft_gradient import StationaryKsState
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft import GridPolicy, MolecularGrid, NativeAO
from vibeqc_compiler.dft.spatial import SpatialPolicy
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid
from vibeqc_compiler.xc.contractions import ContractionProgram


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    method: str
    atoms: tuple[tuple[str, tuple[float, float, float]], ...]
    charge: int = 0
    multiplicity: int = 1
    basis: Any = "sto-3g"
    representation: str = "cartesian"
    role: str = "holdout"
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Endpoint:
    level: str
    energy: float
    forces: tuple[tuple[float, float, float], ...]
    seconds: float
    converged: bool
    iterations: int
    backend: str
    process_peak_rss_kib_after: int | None
    gpu_resident_mib_after: float | None
    resource_diagnostics: dict[str, Any] | None
    target_model_id: str
    grid_id: str
    screening_tolerance: float
    force_route: str = "internal-stationary-gradient-diagnostic"
    public_force_available: bool = False
    force_work: dict[str, Any] | None = None


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _process_peak_rss_kib() -> int | None:
    """Return the OS high-water RSS where the standard library exposes it."""
    try:
        import resource
    except ImportError:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _gpu_resident_mib() -> float | None:
    """Current allocation for this PID; deliberately not labelled as a peak."""
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    total = 0.0
    found = False
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            total += float(fields[1])
            found = True
    return total if found else None


@lru_cache(maxsize=1)
def _cuda_force_compiler() -> Any:
    """Resolve the explicit CUDA compiler used by the stationary force diagnostic."""
    from vibeqc.profiles import find_nvcc
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.cuda_target import cuda_target_info

    configured = os.environ.get("CUDACXX")
    nvcc = Path(configured) if configured else find_nvcc()
    if nvcc is None:
        candidate = Path("/usr/local/cuda-12.9/bin/nvcc")
        nvcc = candidate if candidate.is_file() else None
    if nvcc is None:
        raise RuntimeError("NUM03 CUDA force diagnostic requires an explicit NVCC")
    target = os.environ.get("VIBEQC_NUM03_CUDA_TARGET", "sm_89")
    return CudaCompilerAdapter(nvcc, cuda_target_info(target), compile_timeout=900)


def _stationary_energy_and_forces(
    case: Case, calc: Calculator, device: str
) -> tuple[Any, np.ndarray, dict[str, Any], str, Any]:
    """Evaluate energy publicly and forces through the existing internal diagnostic."""
    with (
        calc.prepare_batch(
            [case.atoms],
            charges=[case.charge],
            multiplicities=[case.multiplicity],
            warm_start=False,
        ) as batch,
        NativeAO(
            case.atoms,
            basis=case.basis,
            representation=case.representation,
            charge=case.charge,
            multiplicity=case.multiplicity,
        ) as basis,
    ):
        item = batch.execute(strict=True, properties=("energy",)).items[0]
        state = StationaryKsState.from_native(batch, basis)
        if device == "cuda":
            from vibeqc._stationary_cuda import complete_rks_cuda_gradient_diagnostic

            diagnostic = complete_rks_cuda_gradient_diagnostic(
                state,
                basis,
                compiler=_cuda_force_compiler(),
                cache=Path(
                    os.environ.get(
                        "VIBEQC_NUM03_STATIONARY_CACHE",
                        ".cache/num03-stationary-cuda",
                    )
                ),
                tile_points=137,
                primitive_tile=29,
                integral_terms=17,
            )
            route = diagnostic.execution
        elif device == "cpu":
            from vibeqc._stationary_cpu import complete_rks_gradient_diagnostic

            diagnostic = complete_rks_gradient_diagnostic(
                state,
                basis,
                cache=Path(
                    os.environ.get(
                        "VIBEQC_NUM03_STATIONARY_CACHE",
                        ".cache/num03-stationary-cpu",
                    )
                ),
                execution="native",
                tile_points=137,
                primitive_tile=29,
                integral_terms=17,
            )
            route = diagnostic.execution
        else:
            raise ValueError("NUM03 force diagnostic supports cpu or cuda")
        # Diagnostic publishes dE/dR; physical forces negate exactly once.
        forces = -np.asarray(diagnostic.gradient, dtype=float)
        return (
            item,
            forces,
            _jsonable(dict(diagnostic.work)),
            route,
            _jsonable(batch.resource_diagnostics),
        )


def _diffuse_oh_basis() -> tuple[Shell, ...]:
    payload = json.loads(
        (
            ROOT / "benchmarks/results/density-candidates/inputs/oh_diffuse.json"
        ).read_text()
    )
    return tuple(
        Shell(
            int(shell["atom_index"]),
            int(shell["angular_momentum"]),
            tuple(Primitive(float(e), float(c)) for e, c in shell["primitives"]),
        )
        for shell in payload["inputs"]["shells"]
    )


def _hf_diffuse_basis() -> tuple[Shell, ...]:
    """Checked-in def2-SVP for H/F plus explicit stable diffuse s/p primitives."""
    pack = json.loads((ROOT / "python/vibeqc/data/basis_pack.json").read_text())[
        "bases"
    ]["def2-svp"]["elements"]
    shells: list[Shell] = []
    for atom_index, atomic_number in enumerate((1, 9)):
        for source in pack[str(atomic_number)]:
            angular = source["angular_momentum"]
            if isinstance(angular, list):
                if len(angular) != 1:
                    raise ValueError(
                        "NUM03 HF diffuse fixture requires one-l shell records"
                    )
                angular = angular[0]
            shells.append(
                Shell(
                    atom_index,
                    int(angular),
                    tuple(
                        Primitive(float(exponent), float(coefficient))
                        for exponent, coefficient in zip(
                            source["exponents"], source["coefficients"], strict=True
                        )
                    ),
                )
            )
    # qz qualification found the explicit 0.02 bohr^-2 Gaussian branch stable.
    # The exact exponent is part of the serialized Shell identity; no augmented
    # basis name is invented.
    shells.extend(
        (
            Shell(0, 0, (Primitive(0.02, 1.0),)),
            Shell(1, 0, (Primitive(0.02, 1.0),)),
            Shell(1, 1, (Primitive(0.02, 1.0),)),
        )
    )
    return tuple(shells)


def cases() -> tuple[Case, ...]:
    water = (
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 1.43233673, 1.10715266)),
        ("H", (0.0, -1.43233673, 1.10715266)),
    )
    return (
        Case(
            "h2_train_rks",
            "h2",
            "pbe-rks",
            (("H", (0.0, 0.0, -0.70)), ("H", (0.0, 0.0, 0.70))),
            role="train",
            tags=("rks",),
        ),
        Case(
            "water_train_rks",
            "water",
            "pbe-rks",
            water,
            role="train",
            tags=("rks", "asymmetric-force-pattern"),
        ),
        Case(
            "ch3_train_uks",
            "ch3",
            "pbe-uks",
            (
                ("C", (0.0, 0.0, 0.0)),
                ("H", (2.05, 0.0, 0.0)),
                ("H", (-1.025, 1.775, 0.0)),
                ("H", (-1.025, -1.775, 0.0)),
            ),
            multiplicity=2,
            role="train",
            tags=("uks", "open-shell"),
        ),
        Case(
            "oh_diffuse_holdout_uks",
            "oh-diffuse",
            "pbe-uks",
            (("O", (0.0, 0.0, 0.0)), ("H", (0.1, 0.2, 1.82))),
            multiplicity=2,
            basis=_diffuse_oh_basis(),
            representation="spherical",
            role="negative",
            tags=("uks", "open-shell", "diffuse"),
        ),
        Case(
            "lih_stretched_holdout_rks",
            "lih-stretched",
            "pbe-rks",
            (("Li", (0.0, 0.0, -3.0)), ("H", (0.0, 0.0, 3.0))),
            role="negative",
            tags=("rks", "stretched-small-gap-proxy"),
        ),
        Case(
            "hf_diffuse_holdout_rks",
            "hf-diffuse",
            "pbe-rks",
            (("H", (0.0, 0.0, -0.85)), ("F", (0.0, 0.0, 0.85))),
            basis=_hf_diffuse_basis(),
            representation="spherical",
            role="holdout",
            tags=("rks", "diffuse", "qz-stable"),
        ),
        Case(
            "lih_small_gap_holdout_rks",
            "lih-small-gap",
            "pbe-rks",
            (("Li", (0.0, 0.0, -2.5)), ("H", (0.0, 0.0, 2.5))),
            role="holdout",
            tags=("rks", "stretched", "small-gap", "qz-stable"),
        ),
        Case(
            "water_changed_geometry",
            "water",
            "pbe-rks",
            (
                water[0],
                ("H", (0.13, 1.36, 1.19)),
                ("H", (-0.08, -1.51, 1.02)),
            ),
            role="geometry-perturbation",
            tags=("rks", "changed-geometry"),
        ),
    )


def level_contracts(method: str) -> tuple[tuple[NumericalLevel, Any], ...]:
    standard = GridPolicy("standard").resolve(method)
    tight = GridPolicy("tight").resolve(method)
    coarse = replace(
        standard,
        radial_points=36,
        angular_polar=12,
        angular_azimuth=24,
    )
    values = (
        (
            NumericalLevel(
                "coarse", 0, 1e-7, canonical_hash(dataclasses.asdict(coarse))
            ),
            coarse,
        ),
        (
            NumericalLevel(
                "standard", 1, 1e-10, canonical_hash(dataclasses.asdict(standard))
            ),
            standard,
        ),
        (
            NumericalLevel(
                "strict",
                2,
                1e-14,
                canonical_hash(dataclasses.asdict(tight)),
                strict=True,
            ),
            tight,
        ),
    )
    if len({item[0].grid_identity for item in values}) != 3:
        raise AssertionError("NUM03 benchmark requires three distinct grid levels")
    return values


def _target_model(
    case: Case, calculator: Calculator, grid_spec: Any
) -> NumericalTargetModel:
    metadata = calculator.basis_metadata(
        case.atoms, charge=case.charge, multiplicity=case.multiplicity
    )
    basis_id = metadata["orbital"]["mathematical_identity"]
    from vibeqc.ks import resolve_ks_options

    resolved = resolve_ks_options(case.method, KsOptions(grid=grid_spec))
    functional = resolved.functional
    geometry_id = canonical_hash(
        {
            "atoms": case.atoms,
            "charge": case.charge,
            "multiplicity": case.multiplicity,
            "units": "Bohr",
        }
    )
    return NumericalTargetModel(
        case.method,
        geometry_id,
        basis_id,
        functional.identity,
        canonical_hash(dataclasses.asdict(grid_spec)),
    )


def run_endpoint(
    case: Case,
    level: NumericalLevel,
    grid_spec: Any,
    device: str,
    *,
    target_grid_spec: Any,
) -> Endpoint:
    calc = Calculator(
        method=case.method,
        basis=case.basis,
        basis_representation=case.representation,
        device=device,
        max_iterations=200,
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
        screening_tolerance=level.screening_tolerance,
        ks_options=KsOptions(grid=grid_spec),
    )
    # Every candidate is measured against one fixed strict mathematical target;
    # changing the execution grid must not silently change the target identity.
    target = _target_model(case, calc, target_grid_spec)
    started = time.perf_counter()
    result, forces, force_work, force_route, resources = _stationary_energy_and_forces(
        case, calc, device
    )
    elapsed = time.perf_counter() - started
    if not result.converged:
        raise RuntimeError("DFT benchmark endpoint did not converge")
    public_force = "forces" in method_capabilities(case.method).supported_properties
    return Endpoint(
        level.name,
        float(result.energy),
        tuple(tuple(float(v) for v in row) for row in forces),
        elapsed,
        bool(result.converged),
        int(result.iterations),
        str(result.executed_backend),
        _process_peak_rss_kib(),
        _gpu_resident_mib(),
        resources,
        target.identity,
        level.grid_identity,
        level.screening_tolerance,
        force_route,
        public_force,
        force_work,
    )


def axis_sweeps(case: Case, strict: Endpoint, device: str) -> dict[str, Any]:
    """Separate screening-only and grid-only relaxed endpoint errors.

    VibeQC's public DFT method contract maps a zero/nonpositive screening
    request to its 1e-12 default, so this benchmark does not mislabel zero as
    "unscreened".  The strict reference is an explicit positive 1e-14
    tight-screening endpoint.  A literal unscreened DFT route remains outside
    the current public method contract and is reported as such.
    """
    contracts = level_contracts(case.method)
    strict_grid = contracts[-1][1]
    strict_screening = contracts[-1][0].screening_tolerance
    rows: dict[str, Any] = {
        "reference": {
            "grid_id": contracts[-1][0].grid_identity,
            "screening_tolerance": strict_screening,
            "semantics": "tight-reference; not labelled unscreened",
            "literal_unscreened_public_dft_available": False,
            "reason": (
                "DFT method descriptor maps nonpositive screening to 1e-12; "
                "Calculator also rejects nonpositive screening"
            ),
        },
        "screening_only": [],
        "grid_only": [],
    }
    for index, tolerance in enumerate((1e-7, 1e-10)):
        level = NumericalLevel(
            f"screen-{tolerance:.0e}",
            index,
            tolerance,
            contracts[-1][0].grid_identity,
        )
        endpoint = run_endpoint(
            case,
            level,
            strict_grid,
            device,
            target_grid_spec=strict_grid,
        )
        rows["screening_only"].append(
            {
                "endpoint": _jsonable(endpoint),
                "actual_vs_strict": _jsonable(
                    ObservableDelta.between(
                        endpoint.energy,
                        endpoint.forces,
                        strict.energy,
                        strict.forces,
                    )
                ),
            }
        )
    rows["screening_only"].append(
        {
            "endpoint": _jsonable(strict),
            "actual_vs_strict": _jsonable(
                ObservableDelta.between(
                    strict.energy, strict.forces, strict.energy, strict.forces
                )
            ),
        }
    )
    for index, (_, grid_spec) in enumerate(contracts[:-1]):
        level = NumericalLevel(
            f"grid-{contracts[index][0].name}",
            index,
            strict_screening,
            contracts[index][0].grid_identity,
        )
        endpoint = run_endpoint(
            case,
            level,
            grid_spec,
            device,
            target_grid_spec=strict_grid,
        )
        rows["grid_only"].append(
            {
                "endpoint": _jsonable(endpoint),
                "actual_vs_strict": _jsonable(
                    ObservableDelta.between(
                        endpoint.energy,
                        endpoint.forces,
                        strict.energy,
                        strict.forces,
                    )
                ),
            }
        )
    rows["grid_only"].append(
        {
            "endpoint": _jsonable(strict),
            "actual_vs_strict": _jsonable(
                ObservableDelta.between(
                    strict.energy, strict.forces, strict.energy, strict.forces
                )
            ),
        }
    )
    return rows


def strict_frontier_gap(case: Case, device: str) -> dict[str, Any]:
    """Measure the occupied/virtual KS frontier gap from one strict stationary state."""
    strict_level, strict_grid = level_contracts(case.method)[-1]
    calc = Calculator(
        method=case.method,
        basis=case.basis,
        basis_representation=case.representation,
        device=device,
        max_iterations=200,
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
        screening_tolerance=strict_level.screening_tolerance,
        ks_options=KsOptions(grid=strict_grid),
    )
    with (
        calc.prepare_batch(
            [case.atoms],
            charges=[case.charge],
            multiplicities=[case.multiplicity],
            warm_start=False,
        ) as batch,
        NativeAO(
            case.atoms,
            basis=case.basis,
            representation=case.representation,
            charge=case.charge,
            multiplicity=case.multiplicity,
        ) as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
    energies = np.atleast_2d(np.asarray(state.orbital_energies, dtype=float))
    occupations = np.atleast_2d(np.asarray(state.occupations, dtype=float))
    if energies.shape != occupations.shape:
        raise RuntimeError("stationary orbital energy/occupation shapes disagree")
    spin_rows = []
    for spin in range(energies.shape[0]):
        occupied = energies[spin][occupations[spin] > 1e-10]
        virtual = energies[spin][occupations[spin] <= 1e-10]
        if not len(occupied) or not len(virtual):
            spin_rows.append({"spin": spin, "gap_eh": None})
            continue
        homo = float(np.max(occupied))
        lumo = float(np.min(virtual))
        spin_rows.append(
            {"spin": spin, "homo_eh": homo, "lumo_eh": lumo, "gap_eh": lumo - homo}
        )
    finite_gaps = [row["gap_eh"] for row in spin_rows if row["gap_eh"] is not None]
    return {
        "case": case.name,
        "spin_channels": spin_rows,
        "minimum_gap_eh": min(finite_gaps) if finite_gaps else None,
        "measured_from_strict_stationary_state": True,
    }


@contextmanager
def _strict_stationary_context(case: Case, device: str) -> Any:
    """Keep the native snapshot alive while SCF-domain point values are consumed."""
    strict_level, strict_grid = level_contracts(case.method)[-1]
    calc = Calculator(
        method=case.method,
        basis=case.basis,
        basis_representation=case.representation,
        device=device,
        max_iterations=200,
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
        screening_tolerance=strict_level.screening_tolerance,
        ks_options=KsOptions(grid=strict_grid),
    )
    with (
        calc.prepare_batch(
            [case.atoms],
            charges=[case.charge],
            multiplicities=[case.multiplicity],
            warm_start=False,
        ) as batch,
        NativeAO(
            case.atoms,
            basis=case.basis,
            representation=case.representation,
            charge=case.charge,
            multiplicity=case.multiplicity,
        ) as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
        density = state.density[0] if case.method.endswith("rks") else state.density
        yield state, np.array(density, copy=True), state._source.functional, basis


def _scf_point_energy(
    source: Any,
    functional: Any,
    features: dict[str, np.ndarray],
    weights: np.ndarray,
) -> float:
    """Integrate one tile with the exact native SCF point regularization."""
    npoint = len(weights)
    point_values = source.evaluate_xc_points(
        functional,
        features["rho"],
        features.get("gradient", np.zeros((2, npoint, 3))),
        features.get("tau"),
    )
    value = float(np.asarray(weights) @ np.asarray(point_values["energy"]))
    if not np.isfinite(value):
        raise ArithmeticError("nonfinite SCF-domain fixed-density XC energy")
    return value


def _fixed_density_xc_energy(
    case: Case,
    atoms: tuple[tuple[str, tuple[float, float, float]], ...],
    density: np.ndarray,
    source: Any,
    functional: Any,
    grid_spec: Any,
) -> float:
    with NativeAO(
        atoms,
        basis=case.basis,
        representation=case.representation,
        charge=case.charge,
        multiplicity=case.multiplicity,
    ) as basis:
        grid = MolecularGrid(
            basis.atoms,
            grid_spec,
            charge=case.charge,
            multiplicity=case.multiplicity,
        )
        program = ContractionProgram(functional, "potential")
        energy = 0.0
        for tile in grid.tiles(256):
            jets = basis.evaluate(tile.points, program.contract.ao_order)
            features = program.features(jets, density)
            energy += _scf_point_energy(source, functional, features, tile.weights)
        return energy


def fixed_density_xc_audit(
    case: Case,
    device: str,
    *,
    step: float = 2e-4,
) -> dict[str, Any]:
    """Compare grid-only XC errors at one strict converged AO density."""
    levels = level_contracts(case.method)
    rows: dict[str, Any] = {}
    with _strict_stationary_context(case, device) as (state, density, functional, _):
        for level, grid_spec in levels:
            started = time.perf_counter()
            energy = _fixed_density_xc_energy(
                case, case.atoms, density, state._source, functional, grid_spec
            )
            gradient = np.zeros((len(case.atoms), 3))
            for atom_index in range(len(case.atoms)):
                for axis in range(3):
                    displaced = []
                    for sign in (+1.0, -1.0):
                        moved = [(symbol, list(xyz)) for symbol, xyz in case.atoms]
                        moved[atom_index][1][axis] += sign * step
                        moved_atoms = tuple(
                            (symbol, tuple(float(x) for x in xyz))
                            for symbol, xyz in moved
                        )
                        displaced.append(
                            _fixed_density_xc_energy(
                                case,
                                moved_atoms,
                                density,
                                state._source,
                                functional,
                                grid_spec,
                            )
                        )
                    gradient[atom_index, axis] = (displaced[0] - displaced[1]) / (
                        2 * step
                    )
            rows[level.name] = {
                "energy": energy,
                "gradient": gradient.tolist(),
                "seconds": time.perf_counter() - started,
                "grid_id": level.grid_identity,
                "screening_tolerance": level.screening_tolerance,
                "point_model": "native-scf-domain",
                "semantics": (
                    "fixed-AO-density XC finite difference with rebuilt "
                    "moving grid/partition and native SCF point regularization"
                ),
            }
    strict = rows["strict"]
    for level in ("coarse", "standard"):
        rows[level]["actual_vs_strict"] = _jsonable(
            ObservableDelta.between(
                rows[level]["energy"],
                rows[level]["gradient"],
                strict["energy"],
                strict["gradient"],
            )
        )
    return rows


def spatial_screening_audit(
    case: Case,
    device: str,
    *,
    cutoff: float = 1e-8,
    region_points: int = 512,
) -> dict[str, Any]:
    """Consume #234 derivative-aware AO masks without claiming an observable bound.

    SpatialTask.discarded_max certifies omitted AO value/first-derivative
    envelopes for a region.  It is a screening signal only: the independently
    measured E_xc difference below is kept separate, and complete molecular
    force error is measured by the reconverged endpoint sweeps.
    """
    _, strict_grid = level_contracts(case.method)[-1]
    with _strict_stationary_context(case, device) as (
        state,
        density,
        functional,
        basis,
    ):
        grid = MolecularGrid(
            basis.atoms,
            strict_grid,
            charge=case.charge,
            multiplicity=case.multiplicity,
        )
        off_policy = SpatialPolicy(region_points=region_points)
        screened_policy = SpatialPolicy(
            region_points=region_points,
            screening="absolute_ao_jet",
            cutoff=cutoff,
        )
        with (
            PreparedSpatialGrid(
                basis, grid, policy=off_policy, backend="cpu"
            ) as unscreened,
            PreparedSpatialGrid(
                basis, grid, policy=screened_policy, backend="cpu"
            ) as screened,
        ):
            started = time.perf_counter()
            ingredients = (
                ("rho",) if functional.ingredients == ("rho",) else ("rho", "gradient")
            )

            def masked_energy(owner: PreparedSpatialGrid) -> float:
                return sum(
                    _scf_point_energy(
                        state._source,
                        functional,
                        tile.features,
                        tile.weights,
                    )
                    for tile in owner.iter_features(
                        density, ingredients=ingredients, order=1
                    )
                )

            baseline_energy = masked_energy(unscreened)
            candidate_energy = masked_energy(screened)
            elapsed = time.perf_counter() - started
            task_rows = []
            for index, task in enumerate(screened.tasks.tasks):
                task_rows.append(
                    {
                        "region": index,
                        "point_count": len(task.point_ids),
                        "active_shell_ids": task.active_shell_ids.tolist(),
                        "active_ao_count": len(task.ao_ids),
                        "discarded_ao_count": int(task.discarded_count),
                        "discarded_max_ao_jets": task.discarded_max.tolist(),
                    }
                )
            discarded_totals = np.sum(
                [np.asarray(row["discarded_max_ao_jets"]) for row in task_rows],
                axis=0,
            )
            return {
                "cutoff": cutoff,
                "region_points": region_points,
                "screening_signal_semantics": (
                    "absolute AO value/first-derivative regional envelopes; "
                    "not an energy/force bound"
                ),
                "point_model": "native-scf-domain",
                "derivatives": ["value", "dx", "dy", "dz"],
                "task_count": len(task_rows),
                "tasks": task_rows,
                "all_omitted_regions_recorded": True,
                "cumulative_raw_discarded_max": discarded_totals.tolist(),
                "actual_fixed_density": {
                    "energy_abs": abs(candidate_energy - baseline_energy),
                    "screened_energy": candidate_energy,
                    "unscreened_energy": baseline_energy,
                },
                "mask_identity": screened.tasks.identity,
                "unscreened_mask_identity": unscreened.tasks.identity,
                "audit_seconds": elapsed,
                "complete_force_truth_source": (
                    "reconverged screening-only endpoint sweep versus tight reference"
                ),
            }


def strict_smooth_branch_fd(case: Case, device: str) -> dict[str, Any]:
    """Reconverged total-energy FD against one strict analytic force direction."""
    level, grid_spec = level_contracts(case.method)[-1]
    base = run_endpoint(case, level, grid_spec, device, target_grid_spec=grid_spec)
    direction = np.arange(1, 3 * len(case.atoms) + 1, dtype=float).reshape(-1, 3)
    direction /= np.linalg.norm(direction)
    analytic = float(np.sum(np.asarray(base.forces) * direction))
    rows = []
    for step in (1e-3, 5e-4):
        energies = []
        for sign in (+1.0, -1.0):
            atoms = []
            for index, (symbol, xyz) in enumerate(case.atoms):
                moved = np.asarray(xyz) + sign * step * direction[index]
                atoms.append((symbol, tuple(map(float, moved))))
            calc = Calculator(
                method=case.method,
                basis=case.basis,
                basis_representation=case.representation,
                device=device,
                max_iterations=200,
                energy_tolerance=1e-11,
                density_tolerance=1e-9,
                screening_tolerance=level.screening_tolerance,
                ks_options=KsOptions(grid=grid_spec),
            )
            result = calc.singlepoint(
                atoms,
                charge=case.charge,
                multiplicity=case.multiplicity,
                properties=("energy",),
            )
            if not result.converged:
                raise RuntimeError("smooth-branch FD displacement did not converge")
            energies.append(float(result.energy))
        fd_force = -(energies[0] - energies[1]) / (2 * step)
        rows.append(
            {
                "step_bohr": step,
                "fd_directional_force": fd_force,
                "analytic_directional_force": analytic,
                "absolute_error": abs(fd_force - analytic),
            }
        )
    return {
        "rows": rows,
        "switching_observed": False,
        "switching_semantics": "same explicit GridSpec at every smooth-branch displacement",
    }


def evaluate_case(
    case: Case,
    endpoints: list[Endpoint],
    estimator: PairedDifferenceEstimator,
    budget: TargetErrorBudget,
) -> dict[str, Any]:
    strict = endpoints[-1]
    actual = [
        ObservableDelta.between(row.energy, row.forces, strict.energy, strict.forces)
        for row in endpoints
    ]
    paired = [
        ObservableDelta.between(
            endpoints[index].energy,
            endpoints[index].forces,
            endpoints[index + 1].energy,
            endpoints[index + 1].forces,
        )
        for index in range(2)
    ]
    levels = tuple(pair[0] for pair in level_contracts(case.method))
    policy = AdaptiveNumericsPolicy(levels, budget)
    state = policy.initial_state()
    policy_seconds = 0.0
    charged_levels: set[int] = set()
    decisions = []
    selected = 2

    def charge_level(index: int) -> None:
        nonlocal policy_seconds
        if index not in charged_levels:
            policy_seconds += endpoints[index].seconds
            charged_levels.add(index)

    for index in range(2):
        charge_level(index)
        if index == 0:
            estimator_started = time.perf_counter()
            estimate = estimator.predict(
                case.method,
                paired[index],
                numerical_family_id=TRANSITION_FAMILIES[index],
            )
            estimator_model_seconds = time.perf_counter() - estimator_started
            evidence_kind = "empirical_paired_estimator"
        else:
            # standard -> strict compares the candidate directly with the
            # declared strict target.  This is observed error, not another
            # empirical calibration/holdout datum.
            estimate = NumericalEstimate(
                paired[index],
                "observed-standard-to-strict-v1",
                (
                    "standard candidate is paired directly with the strict target",
                    "observed pair is excluded from empirical validation aggregates",
                ),
                method=case.method,
            )
            estimator_model_seconds = 0.0
            evidence_kind = "observed_strict_pair"
        policy_seconds += estimator_model_seconds
        # The next level is the paired observation required by the estimator;
        # reuse it if a previous decision already paid for the same endpoint.
        charge_level(index + 1)
        decision = policy.decide(
            state,
            estimate,
            geometry_id=case.name,
            mask_identity=endpoints[index].grid_id,
        )
        decisions.append(
            {
                "at_level": levels[index].name,
                "numerical_family_id": TRANSITION_FAMILIES[index],
                "evidence_kind": evidence_kind,
                "action": decision.action,
                "worst_ratio": decision.worst_ratio,
                "estimate": _jsonable(estimate.delta),
                "actual": _jsonable(actual[index]),
                "predicted_pass": budget.accepts(estimate.delta),
                "actual_pass": budget.accepts(actual[index]),
                "false_success": budget.accepts(estimate.delta)
                and not budget.accepts(actual[index]),
                "overconservative": budget.accepts(actual[index])
                and not budget.accepts(estimate.delta),
                "estimator_model_seconds": estimator_model_seconds,
                "paired_endpoint_seconds": endpoints[index + 1].seconds,
            }
        )
        state = decision.state
        if decision.action in ("hold", "relax"):
            selected = index
            break
        if index == 1:
            selected = 2
    if selected == 2:
        charge_level(2)
    verification = policy.verify_final(
        reference_level=levels[-1], actual_error=actual[selected]
    )
    return {
        "case": case.name,
        "family": case.family,
        "role": case.role,
        "tags": case.tags,
        "selected_level": levels[selected].name,
        "verification": verification,
        "actual_selected": _jsonable(actual[selected]),
        "policy_seconds": policy_seconds,
        "fixed_strict_seconds": strict.seconds,
        "speedup_vs_strict": strict.seconds / policy_seconds,
        "strict_oracle_seconds_excluded_from_policy_cost": (
            0.0 if selected == 2 else strict.seconds
        ),
        "matched_actual_force_accuracy": budget.accepts(actual[selected]),
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--energy-target", type=float, default=1e-6)
    parser.add_argument("--force-target", type=float, default=1e-5)
    parser.add_argument(
        "--fixed-density-cases",
        default="h2_train_rks,water_train_rks,ch3_train_uks,hf_diffuse_holdout_rks",
    )
    parser.add_argument("--fd-case", default="water_train_rks")
    parser.add_argument(
        "--axis-cases",
        default="water_train_rks,ch3_train_uks,hf_diffuse_holdout_rks,lih_small_gap_holdout_rks",
    )
    parser.add_argument(
        "--gap-cases", default="lih_small_gap_holdout_rks,hf_diffuse_holdout_rks"
    )
    parser.add_argument(
        "--spatial-screening-cases",
        default="h2_train_rks,ch3_train_uks,hf_diffuse_holdout_rks",
    )
    parser.add_argument("--spatial-screening-cutoff", type=float, default=1e-8)
    args = parser.parse_args()

    budget = TargetErrorBudget(args.energy_target, args.force_target)
    selected_cases = cases()
    warmups: dict[str, Any] = {}
    warmed_methods: set[str] = set()
    for case in selected_cases:
        if case.role != "train" or case.method in warmed_methods:
            continue
        warmed_methods.add(case.method)
        try:
            level, grid_spec = level_contracts(case.method)[1]
            strict_grid = level_contracts(case.method)[-1][1]
            warmups[case.method] = {
                "endpoint": _jsonable(
                    run_endpoint(
                        case,
                        level,
                        grid_spec,
                        args.device,
                        target_grid_spec=strict_grid,
                    )
                ),
                "excluded_from_policy_timing": True,
                "purpose": "warm stationary force runtime/compiler cache",
            }
        except Exception as error:  # noqa: BLE001 - retain warmup failures
            warmups[case.method] = {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "excluded_from_policy_timing": True,
            }
    raw: dict[str, list[Endpoint]] = {}
    errors: dict[str, str] = {}
    negative_results: dict[str, Any] = {}
    for case in selected_cases:
        rows = []
        try:
            contracts = level_contracts(case.method)
            target_grid_spec = contracts[-1][1]
            for level, grid_spec in contracts:
                rows.append(
                    run_endpoint(
                        case,
                        level,
                        grid_spec,
                        args.device,
                        target_grid_spec=target_grid_spec,
                    )
                )
        except Exception as error:  # noqa: BLE001 - retain failed benchmark cases
            message = f"{type(error).__name__}: {error}"
            if case.role == "negative":
                negative_results[case.name] = {
                    "status": "observed_failure",
                    "error": message,
                    "tags": case.tags,
                }
            else:
                errors[case.name] = message
            continue
        raw[case.name] = rows
        if case.role == "negative":
            negative_results[case.name] = {
                "status": "unexpected_success",
                "tags": case.tags,
                "endpoints": [_jsonable(row) for row in rows],
            }

    training: list[PairedCalibrationSample] = []
    for case in selected_cases:
        if case.role != "train" or case.name not in raw:
            continue
        rows = raw[case.name]
        training.append(
            PairedCalibrationSample(
                case.family,
                case.name + ":coarse",
                case.method,
                TRANSITION_FAMILIES[0],
                ObservableDelta.between(
                    rows[0].energy, rows[0].forces, rows[1].energy, rows[1].forces
                ),
                ObservableDelta.between(
                    rows[0].energy, rows[0].forces, rows[2].energy, rows[2].forces
                ),
            )
        )
    if len({sample.family for sample in training}) < 2:
        raise RuntimeError("insufficient successful molecular families for calibration")
    estimator = PairedDifferenceEstimator.fit(training)

    holdout_samples: list[PairedCalibrationSample] = []
    observed_standard_to_strict = []
    for case in selected_cases:
        if case.role != "holdout" or case.name not in raw:
            continue
        rows = raw[case.name]
        holdout_samples.append(
            PairedCalibrationSample(
                case.family,
                case.name + ":coarse",
                case.method,
                TRANSITION_FAMILIES[0],
                ObservableDelta.between(
                    rows[0].energy, rows[0].forces, rows[1].energy, rows[1].forces
                ),
                ObservableDelta.between(
                    rows[0].energy, rows[0].forces, rows[2].energy, rows[2].forces
                ),
            )
        )
        observed = ObservableDelta.between(
            rows[1].energy, rows[1].forces, rows[2].energy, rows[2].forces
        )
        observed_standard_to_strict.append(
            {
                "case": case.name,
                "family": case.family,
                "actual": _jsonable(observed),
                "actual_pass": budget.accepts(observed),
            }
        )
    holdout = {
        TRANSITION_FAMILIES[0]: estimator.evaluate_holdout(holdout_samples, budget),
        TRANSITION_FAMILIES[1]: {
            "validation_mode": "observed_strict_pair",
            "excluded_from_empirical_validation_aggregates": True,
            "rows": observed_standard_to_strict,
            "certified": False,
        },
    }

    evaluations = []
    for case in selected_cases:
        if case.name in raw:
            evaluations.append(evaluate_case(case, raw[case.name], estimator, budget))

    axis_results: dict[str, Any] = {}
    requested_axes = set(filter(None, args.axis_cases.split(",")))
    for case in selected_cases:
        if case.name not in requested_axes or case.name not in raw:
            continue
        try:
            axis_results[case.name] = axis_sweeps(case, raw[case.name][-1], args.device)
        except Exception as error:  # noqa: BLE001 - retain axis failures
            axis_results[case.name] = {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }

    frontier_gaps: dict[str, Any] = {}
    requested_gaps = set(filter(None, args.gap_cases.split(",")))
    for case in selected_cases:
        if case.name not in requested_gaps:
            continue
        try:
            frontier_gaps[case.name] = strict_frontier_gap(case, args.device)
        except Exception as error:  # noqa: BLE001 - retain gap failures
            frontier_gaps[case.name] = {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }

    actual_force_counterexamples = []
    for case in selected_cases:
        if case.name not in raw:
            continue
        strict = raw[case.name][-1]
        for endpoint in raw[case.name][:-1]:
            actual_error = ObservableDelta.between(
                endpoint.energy, endpoint.forces, strict.energy, strict.forces
            )
            if (
                actual_error.energy_abs <= budget.energy_abs
                and actual_error.force_max_abs > budget.force_max_abs
            ):
                actual_force_counterexamples.append(
                    {
                        "case": case.name,
                        "level": endpoint.level,
                        "error": _jsonable(actual_error),
                    }
                )

    deliberate_delta = ObservableDelta(
        budget.energy_abs * 0.1,
        ((budget.force_max_abs * 10.0, 0.0, 0.0),),
    )
    deliberate_estimate = NumericalEstimate(
        deliberate_delta,
        "deliberate-small-energy-large-force",
        ("constructed acceptance counterexample",),
        method="pbe-rks",
    )
    deliberate_policy = AdaptiveNumericsPolicy(
        tuple(item[0] for item in level_contracts("pbe-rks")), budget
    )
    deliberate_decision = deliberate_policy.decide(
        deliberate_policy.initial_state(),
        deliberate_estimate,
        geometry_id="deliberate-counterexample",
    )
    deliberate_counterexample = {
        "energy_below_target": deliberate_delta.energy_abs < budget.energy_abs,
        "force_above_target": deliberate_delta.force_max_abs > budget.force_max_abs,
        "decision": deliberate_decision.action,
        "error": _jsonable(deliberate_delta),
    }

    spatial_screening: dict[str, Any] = {}
    requested_spatial = set(filter(None, args.spatial_screening_cases.split(",")))
    for case in selected_cases:
        if case.name not in requested_spatial:
            continue
        try:
            spatial_screening[case.name] = spatial_screening_audit(
                case,
                args.device,
                cutoff=args.spatial_screening_cutoff,
            )
        except Exception as error:  # noqa: BLE001 - retain screening-audit failures
            spatial_screening[case.name] = {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }

    fixed_density = {}
    requested_fixed = set(filter(None, args.fixed_density_cases.split(",")))
    for case in selected_cases:
        if case.name in requested_fixed:
            try:
                fixed_density[case.name] = fixed_density_xc_audit(case, args.device)
            except Exception as error:  # noqa: BLE001 - retain diagnostic failures
                fixed_density[case.name] = {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }

    smooth_fd: dict[str, Any]
    fd_case = next((case for case in selected_cases if case.name == args.fd_case), None)
    if fd_case is None or fd_case.name not in raw:
        smooth_fd = {"status": "skipped", "reason": "requested case unavailable"}
    else:
        try:
            smooth_fd = strict_smooth_branch_fd(fd_case, args.device)
        except Exception as error:  # noqa: BLE001 - retain FD failures
            smooth_fd = {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }

    parent = next(
        (row for row in evaluations if row["case"] == "water_train_rks"), None
    )
    changed = next(
        (row for row in evaluations if row["case"] == "water_changed_geometry"), None
    )
    switching = {
        "parent_selected": None if parent is None else parent["selected_level"],
        "changed_geometry_selected": None
        if changed is None
        else changed["selected_level"],
        "selection_switched": (
            None
            if parent is None or changed is None
            else parent["selected_level"] != changed["selected_level"]
        ),
        "reported_separately_from_smooth_branch_fd": True,
    }
    section_failures = {
        "axis_separation": sorted(
            key for key, value in axis_results.items() if value.get("status") == "error"
        ),
        "strict_frontier_gaps": sorted(
            key
            for key, value in frontier_gaps.items()
            if value.get("status") == "error"
        ),
        "derivative_aware_spatial_screening": sorted(
            key
            for key, value in spatial_screening.items()
            if value.get("status") == "error"
        ),
        "fixed_density_xc_quadrature": sorted(
            key
            for key, value in fixed_density.items()
            if value.get("status") == "error"
        ),
        "smooth_branch_finite_difference": (
            ["requested"] if smooth_fd.get("status") == "error" else []
        ),
    }
    output = {
        "schema": "vibeqc.num03-force-aware-benchmark/v1",
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "device": args.device,
        "budget": _jsonable(budget),
        "force_evidence_contract": {
            "route": "public energy solve plus internal stationary-gradient diagnostic",
            "public_method_properties": {
                method: sorted(method_capabilities(method).supported_properties)
                for method in sorted({case.method for case in selected_cases})
            },
            "public_force_publication_is_not_assumed": True,
            "force_sign": "forces = -dE/dR exactly once",
        },
        "force_runtime_warmups": warmups,
        "empirical_estimators": {
            TRANSITION_FAMILIES[0]: {
                "identity": estimator.identity,
                "training_families": estimator.training_families,
                "training_data_hash": estimator.training_data_hash,
                "energy_scale": estimator.energy_scale,
                "force_scale": estimator.force_scale,
                "certified": False,
            },
            TRANSITION_FAMILIES[1]: {
                "validation_mode": "observed_strict_pair",
                "empirical_estimator": None,
                "certified": False,
            },
        },
        "holdout": holdout,
        "evaluations": evaluations,
        "axis_separation": axis_results,
        "strict_frontier_gaps": frontier_gaps,
        "actual_small_energy_large_force_cases": actual_force_counterexamples,
        "deliberate_small_energy_large_force_counterexample": deliberate_counterexample,
        "fixed_density_xc_quadrature": fixed_density,
        "derivative_aware_spatial_screening": spatial_screening,
        "smooth_branch_finite_difference": smooth_fd,
        "switching_behavior": switching,
        "endpoints": {
            name: [_jsonable(row) for row in rows] for name, rows in raw.items()
        },
        "errors": errors,
        "negative_case_results": negative_results,
        "section_failures": section_failures,
        "performance_interpretation": {
            "claim_success_only_if_speedup_gt_one_at_matched_actual_force_accuracy": True,
            "oracle_runtime_excluded_only_when_used_for_offline_validation": True,
            "force_runtime_compilation_warmups_excluded_and_reported": True,
            "isolated_kernel_saving_is_not_endpoint_success": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(output), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output), "errors": errors}, sort_keys=True))
    if errors or any(section_failures.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

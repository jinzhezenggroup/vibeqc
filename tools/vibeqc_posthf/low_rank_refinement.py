"""Experimental staged RHF initialization followed by the original exact solve.

Only densities cross approximation boundaries. Each stage rebuilds its Fock
matrix and residual; this deliberately simple damped initializer retains no
DIIS/Krylov vectors. Final energies and optional forces come exclusively from
the existing exact FockPlan.solve, with its own fresh convergence history.
"""

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
from vibeqc.fock import FockBuildSpec, FockPlan, FockScfResult

from .coulomb_columns import CoulombColumns
from .low_rank_consumers import LowRankProvider


@dataclass(frozen=True)
class RefinementStage:
    """Conservative predetermined accuracy/rank cap and bounded SCF work.

    ``threshold`` is a pair-matrix diagonal tolerance, not an energy tolerance.
    ``residual_tolerance`` may stop this approximate stage early; it never
    certifies convergence of the original target Hamiltonian.
    """

    threshold: float
    iterations: int = 4
    maximum_rank: int | None = None
    residual_tolerance: float = 1e-5

    def __post_init__(self):
        if (
            isinstance(self.threshold, (bool, str))
            or not math.isfinite(self.threshold)
            or self.threshold < 0
            or type(self.iterations) is not int
            or self.iterations < 1
            or (
                self.maximum_rank is not None
                and (type(self.maximum_rank) is not int or self.maximum_rank < 0)
            )
            or isinstance(self.residual_tolerance, (bool, str))
            or not math.isfinite(self.residual_tolerance)
            or self.residual_tolerance <= 0
        ):
            raise ValueError("invalid predetermined refinement stage")


@dataclass(frozen=True)
class RefinedRHFResult:
    """Exact-target result with separately labeled approximate-stage diagnostics."""

    exact: FockScfResult
    stages: tuple[dict, ...]
    approximate_seconds: float
    cleanup_seconds: float
    diagnostics: dict


def _validate_target(factor, target):
    """Require the same unscreened Coulomb operator, AO basis and ensemble."""
    if not isinstance(target, FockPlan):
        raise TypeError("exact cleanup requires the existing FockPlan")
    columns = factor._columns
    if not isinstance(columns, CoulombColumns):
        raise TypeError("refinement requires the native raw Coulomb source")
    source, basis = columns.source, target.basis
    columns.check()
    if (
        target.spec != FockBuildSpec.hf(derivative_order=target.spec.derivative_order)
        or target.diagnostics["screening_tolerance"] != 0
        or basis.atoms != source.atoms
        or basis.shells != source.shells
        or basis.representation != source.representation
        or basis.charge != source.charge
        or basis.multiplicity != source.multiplicity
        or source.multiplicity != 1
        or source.electron_count % 2
        or not 0 < source.electron_count <= 2 * source.nbf
    ):
        raise ValueError(
            "cleanup must describe the same unscreened closed-shell RHF target"
        )
    return source


def solve_refined_rhf(
    factor,
    target,
    stages,
    *,
    damping=0.25,
    compute_forces=True,
    max_iterations=100,
    energy_tolerance=1e-10,
    density_tolerance=1e-8,
):
    """Build a staged low-rank seed and reconverge the original target RHF.

    The caller owns ``factor`` and ``target``. Stages tighten monotonically and
    retain the actual factor prefix. A failed final solve raises rather than
    returning an approximate energy/force as exact. Approximate initialization
    is covered by the factor's shared budget; the separately owned FockPlan
    has its own native resource controls and is reported outside that scope.
    External refinement is excluded by holding the factor lock throughout.
    """
    with factor._lock:
        factor._check()
        source = _validate_target(factor, target)
        stages = tuple(stages)
        if not stages or any(not isinstance(s, RefinementStage) for s in stages):
            raise ValueError("at least one explicit RefinementStage is required")
        if (
            isinstance(damping, bool)
            or not math.isfinite(damping)
            or not 0 <= damping < 1
        ):
            raise ValueError("damping must be finite in [0,1)")
        previous_threshold, previous_cap = math.inf, factor.rank
        for stage in stages:
            cap = (
                factor.rank_capacity
                if stage.maximum_rank is None
                else stage.maximum_rank
            )
            if (
                stage.threshold > previous_threshold
                or not previous_cap <= cap <= factor.rank_capacity
            ):
                raise ValueError(
                    "stages must tighten thresholds and preserve reserved rank prefixes"
                )
            previous_threshold, previous_cap = stage.threshold, cap
        # Conservative simultaneous numeric peak includes orthogonalization,
        # eigensolver matrices, old/new density/Fock/residual and J/K output.
        # It composes with the existing factor/native/source ownership before
        # one-electron reads or the first stage mutation.
        plan = LowRankProvider(factor)._plan(
            "RHF_initialization", 80 * source.nbf**2 + 2 * source.nbf
        )
        started = time.perf_counter()
        overlap, core = source.one_electron()
        values, vectors = np.linalg.eigh(overlap)
        if np.min(values) <= 1e-12 * np.max(values):
            raise ValueError(
                "staged RHF requires a positive well-conditioned AO overlap"
            )
        orthogonalizer = (vectors / np.sqrt(values)) @ vectors.T
        nocc = source.electron_count // 2

        def occupied_density(fock):
            _, coefficients = np.linalg.eigh(orthogonalizer @ fock @ orthogonalizer)
            occupied = orthogonalizer @ coefficients[:, :nocc]
            density = 2 * occupied @ occupied.T
            return 0.5 * (density + density.T)

        density = occupied_density(core)
        records, previous_energy = [], None
        for stage in stages:
            transition = factor.refine(stage.threshold, maximum_rank=stage.maximum_rank)
            provider = LowRankProvider(factor)
            iterations = []
            stage_start = time.perf_counter()
            # A new generation always starts with a fresh build at the SAME
            # current density, so its energy jump measures observed sensitivity
            # to refinement rather than mixing a density/geometry change.
            for step in range(stage.iterations + 1):
                result = provider.jk(density)
                fock = core + result.coulomb - 0.5 * result.exchange
                residual = (
                    orthogonalizer
                    @ (fock @ density @ overlap - overlap @ density @ fock)
                    @ orthogonalizer
                )
                norm = float(np.linalg.norm(residual))
                energy = float(0.5 * np.sum(density * (core + fock)))
                if step == 0:
                    sensitivity = (
                        None if previous_energy is None else energy - previous_energy
                    )
                iterations.append(
                    {
                        "step": step,
                        "electronic_energy": energy,
                        "orthogonal_commutator_norm": norm,
                    }
                )
                if norm <= stage.residual_tolerance or step == stage.iterations:
                    break
                updated = occupied_density(fock)
                density = (1 - damping) * updated + damping * density
                density = 0.5 * (density + density.T)
            previous_energy = energy
            records.append(
                {
                    "requested": asdict(stage),
                    "transition": asdict(transition),
                    "hamiltonian_id": provider.hamiltonian_id,
                    "history_policy": "fresh Fock/residual; no DIIS or Krylov history retained",
                    "fixed_density_energy_change_on_refinement": sensitivity,
                    "observable_certification": "unverified",
                    "iterations": iterations,
                    "iteration_seconds": time.perf_counter() - stage_start,
                    "maximum_residual_diagonal": factor.diagnostics()[
                        "maximum_residual_diagonal"
                    ],
                }
            )
        approximate_seconds = time.perf_counter() - started
        cleanup_start = time.perf_counter()
        exact = target.solve(
            initial_density=density,
            compute_forces=compute_forces,
            max_iterations=max_iterations,
            energy_tolerance=energy_tolerance,
            density_tolerance=density_tolerance,
        )
        return RefinedRHFResult(
            exact,
            tuple(records),
            approximate_seconds,
            time.perf_counter() - cleanup_start,
            {
                "target_plan_identity": target.identity,
                "target_operator_identity": factor.diagnostics()[
                    "target_operator_identity"
                ],
                "final_provider": "original unscreened exact FockPlan; fresh SCF history",
                "approximate_derivatives": "unsupported",
                "force_source": "converged exact target"
                if compute_forces
                else "not requested",
                "initialization_resource_plan": plan.to_dict(),
                "resource_scope": "factor/source/initializer only; separately owned target FockPlan uses its native controls",
                "target_resources": target.diagnostics,
            },
        )

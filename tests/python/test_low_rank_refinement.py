"""Staged approximate work cannot bypass original-Hamiltonian convergence."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from test_low_rank_source import source_for
from vibeqc.fock import FockBuildSpec, FockPlan
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft import NativeAO

from tools.vibeqc_posthf.coulomb_columns import CoulombColumns
from tools.vibeqc_posthf.low_rank import IncrementalCholesky
from tools.vibeqc_posthf.low_rank_consumers import LowRankProvider
from tools.vibeqc_posthf.low_rank_refinement import RefinementStage, solve_refined_rhf


def basis_for(source):
    return NativeAO(
        source.atoms,
        source.shells,
        representation=source.representation,
        charge=source.charge,
        multiplicity=source.multiplicity,
    )


@pytest.mark.parametrize("name", ["h2", "water", "lih"])
def test_staged_initialization_reconverges_exact_energy_density_and_forces(name):
    source, arrays = source_for(name)
    with (
        source,
        basis_for(source) as basis,
        FockPlan(basis, screening_tolerance=0) as target,
    ):
        exact = target.solve(compute_forces=True)
        columns = CoulombColumns(source)
        with IncrementalCholesky(
            columns, rank_capacity=columns.space.size, pair_tile=5
        ) as factor:
            stages = [RefinementStage(0.1, 2, 2), RefinementStage(1e-5, 4)]
            actual = solve_refined_rhf(factor, target, stages)
            np.testing.assert_allclose(
                actual.exact.energy, exact.energy, atol=1e-10, rtol=0
            )
            np.testing.assert_allclose(
                actual.exact.density, exact.density, atol=2e-7, rtol=0
            )
            np.testing.assert_allclose(
                actual.exact.forces, exact.forces, atol=2e-7, rtol=0
            )
            independent = arrays["conventional_C"]
            independent = (independent * arrays["conventional_occ"]) @ independent.T
            np.testing.assert_allclose(
                actual.exact.density, independent, atol=3e-7, rtol=0
            )
            assert actual.exact.initial_density_used
            assert actual.diagnostics["force_source"] == "converged exact target"
            assert (
                actual.stages[1]["transition"]["previous_identity"]
                == actual.stages[0]["transition"]["identity"]
            )
            assert (
                actual.stages[1]["fixed_density_energy_change_on_refinement"]
                is not None
            )
            # The target receives just a density; an old low-rank view is stale.
            view = LowRankProvider(factor)
            factor.refine(0)
            if view._identity != factor.identity:
                with pytest.raises(ValueError, match="generation changed"):
                    view.jk(actual.exact.density)


def test_refinement_rejects_target_mismatch_and_nonmonotone_stages_before_mutation():
    source, _ = source_for("h2")
    columns = CoulombColumns(source)
    with (
        source,
        basis_for(source) as basis,
        IncrementalCholesky(columns, rank_capacity=3) as factor,
    ):
        with (
            FockPlan(
                basis, FockBuildSpec.hf(coulomb="density_fitted"), screening_tolerance=0
            ) as fitted,
            pytest.raises(ValueError, match="same unscreened"),
        ):
            solve_refined_rhf(factor, fitted, [RefinementStage(0)])
        with FockPlan(basis, screening_tolerance=0) as target:
            with pytest.raises(ValueError, match="tighten"):
                solve_refined_rhf(
                    factor, target, [RefinementStage(0), RefinementStage(0.1)]
                )
            assert factor.rank == 0
            before = dict(columns.statistics)
            budget = factor.resource_plan.budget
            factor._resource_plan = replace(
                factor.resource_plan,
                budget=ResourceBudget(
                    host_bytes=factor.resource_plan.peak_bytes["host"]
                ),
            )
            with pytest.raises(MemoryError):
                solve_refined_rhf(factor, target, [RefinementStage(0)])
            assert columns.statistics == before
            factor._resource_plan = replace(factor.resource_plan, budget=budget)
            # An unsuccessful original-target cleanup must propagate, never
            # produce a successful wrapper containing approximate observables.
            with (
                patch.object(
                    target, "solve", side_effect=RuntimeError("SCF did not converge")
                ),
                pytest.raises(RuntimeError, match="did not converge"),
            ):
                solve_refined_rhf(factor, target, [RefinementStage(0.1, 1, 1)])


def test_invalid_stage_controls():
    for controls in (
        {"threshold": -1},
        {"threshold": True},
        {"threshold": 0, "iterations": 0},
        {"threshold": 0, "maximum_rank": False},
        {"threshold": 0, "residual_tolerance": float("nan")},
    ):
        with pytest.raises(ValueError):
            RefinementStage(**controls)

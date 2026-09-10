"""Fixed-density observations connecting integral diagnostics to NUM01 evidence."""

from dataclasses import dataclass

import numpy as np
from vibeqc.accuracy import ErrorEvidence, EvidenceKind, ResolvedModel
from vibeqc_compiler.common.arrays import immutable

from .export import conventional_fock
from .low_rank_consumers import LowRankProvider


@dataclass(frozen=True)
class LowRankFixedDensityAudit:
    """Two-electron energy difference at fixed D, not a relaxed-target test."""

    target_model: ResolvedModel
    evidence: ErrorEvidence
    approximate_energy: float
    reference_energy: float
    density: np.ndarray
    diagnostics: dict


def audit_fixed_density(factor, density, *, axis_tile=2, budget=None):
    """Compare the same RHF density against the unscreened raw target provider.

    The error source is integral_factorization, not total_numerical. Different
    evaluated/target identities and fixed_density scope prevent this observation
    from certifying a relaxed target. Conditional matrix residual bounds never
    become observable bounds merely by entering the accuracy evidence API.
    """
    with factor._lock:
        factor._check()
        source = getattr(factor._columns, "source", None)
        if source is None or source.multiplicity != 1 or source.electron_count % 2:
            raise ValueError(
                "fixed-density audit requires the closed-shell native raw source"
            )
        if type(axis_tile) is not int or axis_tile < 1:
            raise ValueError("positive raw axis tile required")
        n = factor.space.nbf
        density = np.asarray(density)
        if density.dtype != np.float64 or density.shape != (n, n):
            raise ValueError("an FP64 RHF density matrix is required")
        with LowRankProvider(factor, budget=budget) as provider:
            plan = provider._plan(
                "fixed_density_audit", 40 * n * n + 8 * min(axis_tile, n) ** 4
            )
            # One captured copy is used for both operators and for publication.
            frozen = immutable(density)
            approximate = provider.jk(frozen)
            exact = conventional_fock(source, frozen, axis_tile=axis_tile)
            _, hcore = source.one_electron()
            approximate_energy = float(
                0.5
                * np.sum(frozen * (approximate.coulomb - 0.5 * approximate.exchange))
            )
            reference_energy = float(0.5 * np.sum(frozen * (exact - hcore)))
            model = ResolvedModel(
                "rhf",
                source.geometry_hash,
                source.basis_hash,
                source.electron_count,
                multiplicity=source.multiplicity,
                charge=source.charge,
                representation=source.representation,
            )
            evidence = ErrorEvidence(
                EvidenceKind.OBSERVED,
                model.identity,
                factor.hamiltonian_id,
                model.identity,
                "energy",
                "absolute",
                "Eh",
                abs(approximate_energy - reference_energy),
                abs(reference_energy),
                "integral_factorization",
                "fixed_density",
                (
                    ("raw_source", source.identity),
                    ("factor", factor.identity),
                    ("comparison", "same fixed density; two-electron RHF energy only"),
                ),
            )
            return LowRankFixedDensityAudit(
                model,
                evidence,
                approximate_energy,
                reference_energy,
                frozen,
                {
                    "resource_plan": plan.to_dict(),
                    "tensor_residual": factor.diagnostics(),
                    "observable_certification": "unverified; no relaxed solve was performed",
                },
            )

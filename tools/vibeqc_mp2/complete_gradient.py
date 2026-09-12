"""End-to-end complete MP2 gradient validation facade.

This module composes the delivered reference, provider, response and fused
derivative consumers for small systems. Dense full-MO integrals and relaxed AO
cotangents remain explicit validation boundaries; this is not the final public
bounded method implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.vibeqc_posthf.conventions import MOBlock, ovov_to_ijab
from tools.vibeqc_posthf.df import DFProvider, MetricFactor
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.mp2 import restricted_mp2
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_response import (
    CudaDFJKBackend,
    GMRESOptions,
    NativeJKBackend,
)

from .gradient import (
    canonical_energy_adjoint,
    canonical_lagrangian_weights,
    canonical_orbital_rhs,
    fused_cuda_conventional_molecular_gradient,
    fused_cuda_ri_molecular_gradient,
    solve_canonical_orbital_response,
)


@dataclass(frozen=True)
class CompleteGradientValidation:
    """Detached result and disclosed validation-only staging boundaries."""

    total_energy: float
    correlation_energy: float
    gradient: np.ndarray
    response_residual: float
    stationarity_residual: float
    reference_identity: str
    hamiltonian_id: str
    diagnostics: dict


def complete_gradient_validation(
    source,
    orbital_calculator,
    auxiliary_calculator=None,
    *,
    density_fitted=False,
    device_id=0,
    provider_budget_bytes=256 << 20,
    weight_output_budget_bytes=256 << 20,
    consumer_maximum_bytes=128 << 20,
):
    """Run a fresh CUDA RHF/MP2/Z-vector/gradient validation chain.

    Conventional response uses the existing CPU-streamed exact J/K action;
    RI response uses the existing CUDA DF J/K plan. Both solve through the
    shared #179 GMRES implementation and contract derivatives on CUDA.
    """

    for value, name in (
        (provider_budget_bytes, "provider budget"),
        (weight_output_budget_bytes, "weight output budget"),
        (consumer_maximum_bytes, "consumer budget"),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(density_fitted) is not bool or type(device_id) is not int or device_id < 0:
        raise ValueError("complete gradient validation controls are invalid")
    if source.nbf > 12:
        raise ValueError("complete gradient validation supports at most 12 AOs")
    if (
        source.multiplicity != 1
        or source.electron_count <= 0
        or source.electron_count % 2
    ):
        raise ValueError("complete gradient validation requires closed-shell RHF")
    if (
        getattr(orbital_calculator, "_device_name", None) != "cuda"
        or getattr(orbital_calculator, "_device_id", None) != device_id
    ):
        raise ValueError(
            "complete gradient validation requires the selected CUDA device"
        )
    metric = None
    response_backend = None
    if density_fitted:
        if auxiliary_calculator is None or not source.naux:
            raise ValueError(
                "RI complete gradient requires an auxiliary calculator/basis"
            )
        if (
            getattr(auxiliary_calculator, "_device_name", None) != "cuda"
            or getattr(auxiliary_calculator, "_device_id", None) != device_id
        ):
            raise ValueError("RI auxiliary calculator uses a different CUDA device")
        metric = MetricFactor.from_source(source, budget_bytes=provider_budget_bytes)
    elif auxiliary_calculator is not None:
        raise ValueError(
            "conventional complete gradient does not accept an auxiliary calculator"
        )

    reference, reference_diagnostics = export_rhf(
        source,
        backend="cuda",
        device_id=device_id,
        metric=metric,
        tolerance=1e-11,
    )
    provider = (
        DFProvider(
            reference,
            source,
            metric,
            budget_bytes=provider_budget_bytes,
        )
        if density_fitted
        else ConventionalProvider(
            reference,
            source,
            budget_bytes=provider_budget_bytes,
            backend="cpu",
            cache_policy="pin",
        )
    )
    try:
        mp2 = restricted_mp2(reference, provider)
        ovov = provider.get(MOBlock.from_spaces(reference, "ovov")).to_host()
        g = ovov_to_ijab(ovov)
        adjoint = canonical_energy_adjoint(
            g,
            reference.orbital_energies,
            reference.nocc,
            reference_identity=reference.identity,
            hamiltonian_id=reference.hamiltonian_id,
        )
        all_orbitals = tuple(range(reference.nmo))
        eri_mo = provider.get(MOBlock((all_orbitals,) * 4)).to_host()
        hcore_mo = reference.coefficients.T @ reference.hcore @ reference.coefficients
        orbital_rhs = canonical_orbital_rhs(hcore_mo, eri_mo, adjoint, reference.nocc)
        response_backend = (
            CudaDFJKBackend(
                source,
                device_id=device_id,
                metric_threshold=metric.relative_threshold,
                metric=metric,
            )
            if density_fitted
            else NativeJKBackend(
                source, budget_bytes=min(provider_budget_bytes, 64 << 20)
            )
        )
        response = solve_canonical_orbital_response(
            reference,
            response_backend,
            orbital_rhs,
            options=GMRESOptions(
                rtol=1e-11,
                atol=1e-12,
                max_iterations=200,
                max_workspace_bytes=min(provider_budget_bytes, 64 << 20),
            ),
        )
        weights = canonical_lagrangian_weights(
            hcore_mo, eri_mo, adjoint, response, reference.nocc
        )
        if density_fitted:
            gradient, contraction = fused_cuda_ri_molecular_gradient(
                reference,
                source,
                metric,
                weights,
                orbital_calculator,
                auxiliary_calculator,
                weight_output_budget_bytes=weight_output_budget_bytes,
                consumer_maximum_bytes=consumer_maximum_bytes,
                device_id=device_id,
            )
        else:
            gradient, contraction = fused_cuda_conventional_molecular_gradient(
                reference,
                source,
                weights,
                orbital_calculator,
                weight_output_budget_bytes=weight_output_budget_bytes,
                consumer_maximum_bytes=consumer_maximum_bytes,
                device_id=device_id,
            )
        return CompleteGradientValidation(
            float(reference.energy + mp2.correlation_energy),
            mp2.correlation_energy,
            immutable(gradient),
            response.residual_norm,
            weights.stationarity_residual,
            reference.identity,
            reference.hamiltonian_id,
            {
                "reference": reference_diagnostics,
                "provider": dict(provider.statistics),
                "response": dict(response_backend.statistics),
                "contraction": contraction,
                "dense_full_mo_integrals": True,
                "dense_relaxed_ao_weights": True,
                "global_derivative_tensors": False,
            },
        )
    finally:
        close = getattr(response_backend, "close", None)
        if close is not None:
            close()
        provider.close()

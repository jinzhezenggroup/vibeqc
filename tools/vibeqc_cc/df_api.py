"""Executable energy-only facade for correlation-DF RCCSD(T) (#157 C2a).

This facade deliberately keeps the first DF-RCCSD(T) scientific definition
separate from the conventional public RCCSD(T) method. The reference is a
conventional unscreened RHF determinant; density fitting is applied only to the
correlation Hamiltonian. Forces, native Calculator registration, and CUDA
promotion remain separate capabilities.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace

from tools.vibeqc_posthf.df import DFProvider, MetricFactor
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.sources import NativeSource

from .df_contract import correlation_df_reference
from .df_triples import DFCCSDTResult, solve_df_ccsdt
from .solver import SolverOptions


@dataclass(frozen=True)
class DFRCCSDTCapabilities:
    """Capabilities of the first executable correlation-DF RCCSD(T) facade."""

    method: str = "df-rccsd(t)"
    family: str = "coupled_cluster"
    available: bool = True
    supports_batch: bool = False
    supported_properties: frozenset[str] = frozenset({"energy"})
    reference_mode: str = "conventional-rhf"
    correlation_mode: str = "density-fitting"
    native_public: bool = False

    def __post_init__(self) -> None:
        if self.method != "df-rccsd(t)" or self.family != "coupled_cluster":
            raise ValueError("DF-RCCSD(T) capability identity mismatch")
        if self.supported_properties != frozenset({"energy"}):
            raise ValueError("DF-RCCSD(T) C2a is energy-only")
        if self.reference_mode != "conventional-rhf":
            raise ValueError("DF-RCCSD(T) requires a conventional RHF reference")
        if self.correlation_mode != "density-fitting":
            raise ValueError("DF-RCCSD(T) requires a correlation-DF Hamiltonian")
        if self.native_public:
            raise ValueError("DF-RCCSD(T) native Calculator promotion is not qualified")
        if self.supports_batch:
            raise ValueError("DF-RCCSD(T) C2a has no prepared-batch owner")


def df_rccsd_t_method_capabilities(
    method: str = "df-rccsd(t)",
) -> DFRCCSDTCapabilities:
    """Report the executable C2a facade without borrowing conventional forces."""

    normalized = method.lower().replace(" ", "")
    if normalized not in {"df-rccsd(t)", "df-ccsd(t)"}:
        raise ValueError(f"unknown method {method!r}")
    return DFRCCSDTCapabilities()


def _validate_controls(
    *,
    options: SolverOptions | None,
    reference_max_iterations: int,
    reference_tolerance: float,
    metric_relative_threshold: float,
    metric_budget_bytes: int,
    provider_budget_bytes: int,
    triples_max_bytes: int,
    denominator_threshold: float,
    axis_tile: int,
    auxiliary_tile: int,
) -> None:
    if options is not None and not isinstance(options, SolverOptions):
        raise TypeError("options must be SolverOptions")
    if type(reference_max_iterations) is not int or reference_max_iterations < 1:
        raise ValueError("reference_max_iterations must be a positive integer")
    if not math.isfinite(reference_tolerance) or reference_tolerance <= 0:
        raise ValueError("reference_tolerance must be positive and finite")
    if (
        not math.isfinite(metric_relative_threshold)
        or not 0 < metric_relative_threshold < 1
    ):
        raise ValueError("metric_relative_threshold must lie in (0,1)")
    if not math.isfinite(denominator_threshold) or denominator_threshold <= 0:
        raise ValueError("denominator_threshold must be positive and finite")
    for name, value in (
        ("metric_budget_bytes", metric_budget_bytes),
        ("provider_budget_bytes", provider_budget_bytes),
        ("triples_max_bytes", triples_max_bytes),
        ("axis_tile", axis_tile),
        ("auxiliary_tile", auxiliary_tile),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def df_rccsd_t_energy(
    source: NativeSource,
    *,
    options: SolverOptions | None = None,
    reference_max_iterations: int = 100,
    reference_tolerance: float = 1e-11,
    metric_relative_threshold: float = 1e-10,
    metric_budget_bytes: int = 128 << 20,
    provider_budget_bytes: int = 256 << 20,
    triples_max_bytes: int = 256 << 20,
    denominator_threshold: float = 1e-10,
    axis_tile: int = 2,
    auxiliary_tile: int = 3,
    compute_forces: bool = False,
) -> DFCCSDTResult:
    """Run conventional-RHF + correlation-DF RCCSD(T) as an energy-only facade.

    The live NativeSource must contain both orbital and auxiliary bases. The
    RHF export is deliberately conventional (metric is not supplied); only
    after that determinant is fixed is the DF metric constructed and the
    correlation Hamiltonian relabeled. This prevents an implicit DF-RHF
    reference from being mixed with the first #157 method definition.

    C2a uses the existing small native RHF export bridge, so it inherits that
    bridge's current <=12-AO qualification boundary. The factorized CCSD/(T)
    implementation itself remains the Slice-B/C1 owner.
    """

    if compute_forces:
        raise NotImplementedError(
            "DF-RCCSD(T) C2a is energy-only; DF analytic forces remain #158"
        )
    if not isinstance(source, NativeSource):
        raise TypeError("DF-RCCSD(T) requires a live NativeSource")
    source._check_open()
    if (
        source.multiplicity != 1
        or source.electron_count <= 0
        or source.electron_count % 2
    ):
        raise NotImplementedError(
            "DF-RCCSD(T) C2a supports real closed-shell all-electron RHF only"
        )
    if source.electron_count >= 2 * source.nbf:
        raise ValueError("DF-RCCSD(T) requires a nonempty virtual orbital space")
    if not source.naux or source.auxiliary_hash is None:
        raise ValueError("DF-RCCSD(T) requires an explicit auxiliary basis")

    _validate_controls(
        options=options,
        reference_max_iterations=reference_max_iterations,
        reference_tolerance=reference_tolerance,
        metric_relative_threshold=metric_relative_threshold,
        metric_budget_bytes=metric_budget_bytes,
        provider_budget_bytes=provider_budget_bytes,
        triples_max_bytes=triples_max_bytes,
        denominator_threshold=denominator_threshold,
        axis_tile=axis_tile,
        auxiliary_tile=auxiliary_tile,
    )

    # Fix the first #157 reference before building any fitted correlation state.
    reference, reference_diagnostics = export_rhf(
        source,
        backend="cpu",
        max_iterations=reference_max_iterations,
        tolerance=reference_tolerance,
        axis_tile=axis_tile,
    )
    metric = MetricFactor.from_source(
        source,
        relative_threshold=metric_relative_threshold,
        budget_bytes=metric_budget_bytes,
    )
    snapshot, contract = correlation_df_reference(reference, source, metric)

    with DFProvider(
        snapshot,
        source,
        metric,
        budget_bytes=provider_budget_bytes,
        axis_tile=axis_tile,
        auxiliary_tile=auxiliary_tile,
    ) as provider:
        result = solve_df_ccsdt(
            snapshot,
            provider,
            contract,
            options=options,
            triples_max_bytes=triples_max_bytes,
            denominator_threshold=denominator_threshold,
        )
        provider_statistics = deepcopy(provider.statistics)

    provenance = deepcopy(result.provenance)
    provenance.update(
        {
            "facade_schema": "vibeqc.df-rccsd-t.energy-facade/1",
            "method_contract": contract.record(),
            "source_identity": source.identity,
            "reference_export": deepcopy(reference_diagnostics),
            "metric": {
                "identity": metric.identity,
                "relative_threshold": metric.relative_threshold,
                "absolute_threshold": metric.absolute_threshold,
                "rank": metric.rank,
                "dimension": source.naux,
                "condition_number": metric.condition_number,
                "convention": metric.convention,
                "setup_seconds": metric.setup_seconds,
                "host_peak_bytes": metric.host_peak_bytes,
                "device_bytes": metric.device_bytes,
            },
            "df_provider_statistics": provider_statistics,
            "capability": {
                "method": "df-rccsd(t)",
                "supported_properties": ("energy",),
                "reference_mode": contract.reference_mode,
                "correlation_mode": contract.correlation_mode,
                "native_public": False,
                "supports_batch": False,
            },
        }
    )
    return replace(result, provenance=provenance)

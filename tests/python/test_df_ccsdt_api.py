# ruff: noqa: ANN201
"""Executable facade tests for #157 C2a."""

import pytest

from tools.vibeqc_cc.df_api import (
    df_rccsd_t_energy,
    df_rccsd_t_method_capabilities,
)
from tools.vibeqc_cc.solver import SolverOptions
from tools.vibeqc_posthf.fixtures import load_fixture, source_arguments
from tools.vibeqc_posthf.sources import NativeSource


def test_df_rccsdt_capability_is_explicitly_energy_only():
    caps = df_rccsd_t_method_capabilities("df-rccsd(t)")
    alias = df_rccsd_t_method_capabilities("DF-CCSD(T)")

    assert caps == alias
    assert caps.available
    assert caps.supported_properties == frozenset({"energy"})
    assert not caps.supports_batch
    assert not caps.native_public
    assert caps.reference_mode == "conventional-rhf"
    assert caps.correlation_mode == "density-fitting"

    with pytest.raises(ValueError, match="unknown method"):
        df_rccsd_t_method_capabilities("rccsd(t)")


def test_df_rccsdt_force_request_fails_before_source_validation():
    with pytest.raises(NotImplementedError, match="energy-only"):
        df_rccsd_t_energy(object(), compute_forces=True)


def test_df_rccsdt_source_facade_records_method_and_metric_boundaries():
    metadata, _arrays = load_fixture("h2")
    try:
        source = NativeSource(**source_arguments(metadata))
    except (OSError, FileNotFoundError, AttributeError) as error:
        pytest.skip(f"native post-HF bridge unavailable: {error}")

    with source:
        result = df_rccsd_t_energy(
            source,
            options=SolverOptions(
                residual_tolerance=1e-10,
                energy_tolerance=1e-12,
            ),
            metric_budget_bytes=32 << 20,
            provider_budget_bytes=32 << 20,
            triples_max_bytes=8 << 20,
        )

    assert result.converged, result.reason
    assert result.total_energy is not None
    assert result.provenance["resident_ovvv"] is False
    assert result.provenance["resident_vvvv"] is False
    assert result.provenance["full_t3"] is False

    capability = result.provenance["capability"]
    assert capability["supported_properties"] == ("energy",)
    assert capability["reference_mode"] == "conventional-rhf"
    assert capability["correlation_mode"] == "density-fitting"
    assert capability["native_public"] is False
    assert capability["supports_batch"] is False

    contract = result.provenance["method_contract"]
    metric = result.provenance["metric"]
    assert contract["reference_hamiltonian_id"] == "conventional-unscreened"
    assert contract["fock_policy"] == "preserve-conventional-rhf"
    assert contract["metric_rank"] == metric["rank"]
    assert contract["metric_dimension"] == metric["dimension"]
    assert metric["rank"] > 0
    assert metric["condition_number"] >= 1
    assert result.provenance["df_provider_statistics"]["external_reserved_bytes"] == 0

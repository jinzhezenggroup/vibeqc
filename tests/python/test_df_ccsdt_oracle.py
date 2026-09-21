# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, ANN204
"""Same-Hamiltonian DF-CCSD(T) dense-oracle tests for issue #157 slice A."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tools.cc_endpoint_fixtures import load, snapshot_from_fixture
from tools.vibeqc_cc.ccsd_t_api import rccsd_t_energy
from tools.vibeqc_cc.df_ccsdt_oracle import (
    DenseDFOracleProvider,
    correlation_df_reference,
    dense_df_oracle_from_three_index,
    dense_eri_from_three_index,
    df_fitting_error,
    prepare_same_hamiltonian_dense_oracle,
    run_dense_df_ccsdt_oracle,
)
from tools.vibeqc_posthf.df import MetricFactor
from tools.vibeqc_posthf.providers import BlockResult, ConventionalProvider


class FixtureProvider(ConventionalProvider):
    def __init__(self, snapshot, eri):
        self.snapshot = snapshot
        self.g = eri
        self.backend = "cpu"
        self.source = SimpleNamespace(_check_open=lambda: None)

    def get(self, block):
        return BlockResult(
            block,
            self.g[np.ix_(*block.slots)],
            self.snapshot.identity,
            self.snapshot.hamiltonian_id,
            {},
        )


def _factor_eri(eri):
    nmo = eri.shape[0]
    matrix = eri.reshape(nmo * nmo, nmo * nmo)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    assert values.min() > -1e-12
    keep = values > 1e-12
    return (np.sqrt(values[keep])[:, None] * vectors[:, keep].T).reshape(
        int(np.sum(keep)), nmo, nmo
    )


def _problem(name="h2o"):
    meta, arrays = load(name)
    nmo = arrays["g"].shape[0]
    source_identity = SimpleNamespace(
        electron_count=int(arrays["occ"].sum()),
        geometry_hash="df-ccsdt-geometry-" + name,
        basis_hash="df-ccsdt-basis-" + name,
    )
    reference = snapshot_from_fixture(source_identity, meta, arrays)
    b = _factor_eri(arrays["g"])
    source = SimpleNamespace(
        geometry_hash=reference.geometry_hash,
        basis_hash=reference.basis_hash,
        auxiliary_hash="df-ccsdt-aux-" + name,
        representation=reference.representation,
        nbf=nmo,
        naux=b.shape[0],
    )
    metric = MetricFactor(
        np.eye(source.naux),
        source.auxiliary_hash,
        source.geometry_hash,
        1e-12,
        source.naux,
        1e-12,
        1.0,
        "synthetic-metric-hash-" + name,
        "synthetic-metric-" + name,
        0.0,
        8 * source.naux**2,
        0,
    )
    correlated, contract = correlation_df_reference(reference, source, metric)
    return reference, correlated, contract, source, metric, b, meta, arrays


def test_contract_preserves_conventional_rhf_state_and_separates_df_identity():
    reference, correlated, contract, source, _metric, _b, _meta, _arrays = _problem(
        "h2"
    )

    assert correlated.hamiltonian_id == contract.correlation_hamiltonian_id
    assert correlated.hamiltonian_id != reference.hamiltonian_id
    assert contract.reference_identity == reference.identity
    assert contract.reference_mode == "conventional-rhf"
    assert contract.fock_policy == "preserve-conventional-rhf"
    assert contract.triples_variant == "standard-canonical"
    assert contract.metric_rank == contract.metric_dimension == source.naux
    assert contract.record()["schema"] == "vibeqc.df-ccsd-t.method/1"
    np.testing.assert_array_equal(correlated.fock, reference.fock)
    np.testing.assert_array_equal(correlated.coefficients, reference.coefficients)
    np.testing.assert_array_equal(
        correlated.orbital_energies, reference.orbital_energies
    )
    assert correlated.reference_energy == reference.reference_energy


def test_dense_df_oracle_matches_same_hamiltonian_and_auxiliary_gauge():
    reference, correlated, contract, _source, _metric, b, _meta, arrays = _problem(
        "h2o"
    )
    prepared = dense_df_oracle_from_three_index(correlated, contract, b)
    np.testing.assert_allclose(prepared.eri_mo, arrays["g"], atol=3e-12, rtol=0)

    result = run_dense_df_ccsdt_oracle(prepared)
    conventional = rccsd_t_energy(reference, FixtureProvider(reference, arrays["g"]))
    assert result.converged
    np.testing.assert_allclose(
        result.ccsd_correlation_energy,
        conventional.ccsd_correlation_energy,
        atol=2e-11,
        rtol=0,
    )
    np.testing.assert_allclose(
        result.triples_energy,
        conventional.triples_energy,
        atol=2e-11,
        rtol=0,
    )
    np.testing.assert_allclose(
        result.total_energy, conventional.total_energy, atol=2e-11, rtol=0
    )
    assert (
        result.provenance["method_contract"]["fock_policy"]
        == "preserve-conventional-rhf"
    )

    q, _ = np.linalg.qr(
        np.random.default_rng(157).normal(size=(b.shape[0], b.shape[0]))
    )
    rotated = np.einsum("RQ,Qpq->Rpq", q, b)
    rotated_prepared = dense_df_oracle_from_three_index(correlated, contract, rotated)
    rotated_result = run_dense_df_ccsdt_oracle(rotated_prepared)
    np.testing.assert_allclose(
        rotated_prepared.eri_mo, prepared.eri_mo, atol=3e-12, rtol=0
    )
    np.testing.assert_allclose(
        rotated_result.total_energy, result.total_energy, atol=2e-11, rtol=0
    )
    assert rotated_result.oracle_identity != result.oracle_identity


def test_fitting_error_is_reported_separately_and_dense_budget_is_preflighted():
    _reference, _correlated, _contract, _source, _metric, b, _meta, arrays = _problem(
        "h2o"
    )
    fitted = dense_eri_from_three_index(b[:-1])
    error = df_fitting_error(fitted, arrays["g"])
    assert error["max_abs"] > 0
    assert error["frobenius"] >= error["max_abs"]
    assert error["rms"] > 0
    required = b.nbytes + 16 * b.shape[1] ** 4
    with pytest.raises(MemoryError, match="dense DF oracle requires"):
        dense_eri_from_three_index(b, max_bytes=required - 1)


def test_prepare_oracle_extracts_b_from_df_provider_without_changing_reference(
    monkeypatch,
):
    reference, _correlated, _contract, source, metric, b, _meta, arrays = _problem("h2")

    class FakeDFProvider:
        def __init__(self, snapshot, current_source, current_metric, **settings):
            assert snapshot.hamiltonian_id == current_metric.hamiltonian_id
            assert current_source is source
            assert settings["budget_bytes"] > 0
            self.snapshot = snapshot
            self.statistics = {"transformations": 1, "peak_bytes": b.nbytes}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def three_index(self, p, q):
            assert tuple(p) == tuple(range(reference.nmo))
            assert tuple(q) == tuple(range(reference.nmo))
            return b

    from tools.vibeqc_cc import df_ccsdt_oracle as module

    monkeypatch.setattr(module, "DFProvider", FakeDFProvider)
    prepared = prepare_same_hamiltonian_dense_oracle(reference, source, metric)
    np.testing.assert_allclose(prepared.eri_mo, arrays["g"], atol=3e-12, rtol=0)
    assert prepared.snapshot.reference_energy == reference.reference_energy
    assert prepared.diagnostics["fock_policy"] == "preserve-conventional-rhf"
    assert prepared.diagnostics["df_provider_statistics"]["transformations"] == 1


def test_dense_oracle_provider_rejects_stale_identity_and_closed_access():
    _reference, correlated, contract, _source, _metric, b, _meta, _arrays = _problem(
        "h2"
    )
    prepared = dense_df_oracle_from_three_index(correlated, contract, b)
    stale = replace(contract, correlation_hamiltonian_id="density-fitting:stale")
    with pytest.raises(ValueError, match="identity mismatch"):
        DenseDFOracleProvider(correlated, prepared.eri_mo, stale)
    prepared.close()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.provider.get(
            __import__(
                "tools.vibeqc_posthf.conventions", fromlist=["MOBlock"]
            ).MOBlock.from_spaces(correlated, "ovov")
        )

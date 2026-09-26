# ruff: noqa: ANN001, ANN201
"""Factorized DF-(T) and composed DF-RCCSD(T) tests for #157 slice C."""

import numpy as np
import pytest
from test_df_ccsd_factorized import FixtureDFProvider, _method_problem

from tools.vibeqc_cc.df_ccsdt_oracle import (
    dense_df_oracle_from_three_index,
    run_dense_df_ccsdt_oracle,
)
from tools.vibeqc_cc.df_triples import (
    factorized_triples_energy,
    solve_df_ccsdt,
)
from tools.vibeqc_cc.solver import SolverOptions
from tools.vibeqc_cc.triples import triples_energy


def _block(eri, nocc, name) -> np.ndarray:
    spaces = {
        "o": tuple(range(nocc)),
        "v": tuple(range(nocc, eri.shape[0])),
    }
    return eri[np.ix_(*(spaces[axis] for axis in name))]


def test_factorized_triples_matches_dense_same_hamiltonian():
    rng = np.random.default_rng(15703)
    nocc, nvir, naux = 2, 3, 5
    nmo = nocc + nvir
    raw = rng.normal(size=(naux, nmo, nmo))
    b = 0.5 * (raw + raw.transpose(0, 2, 1))
    eri = np.einsum("Qpq,Qrs->pqrs", b, b, optimize=True)
    t1 = rng.normal(scale=0.03, size=(nocc, nvir))
    raw_t2 = rng.normal(scale=0.02, size=(nocc, nocc, nvir, nvir))
    t2 = 0.5 * (raw_t2 + raw_t2.transpose(1, 0, 3, 2))
    fov = rng.normal(scale=0.02, size=(nocc, nvir))
    eps_o = np.array([-1.1, -0.8])
    eps_v = np.array([0.2, 0.5, 0.9])

    expected = triples_energy(
        nocc,
        nvir,
        _block(eri, nocc, "ovvv"),
        _block(eri, nocc, "ovoo"),
        _block(eri, nocc, "ovov"),
        fov,
        t1,
        t2,
        eps_o,
        eps_v,
    )
    actual = factorized_triples_energy(
        b[:, :nocc, nocc:],
        b[:, nocc:, nocc:],
        _block(eri, nocc, "ovoo"),
        _block(eri, nocc, "ovov"),
        fov,
        t1,
        t2,
        eps_o,
        eps_v,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize("name", ["h2", "h2o"])
def test_factorized_df_ccsdt_matches_dense_oracle_without_ovvv_or_vvvv(name):
    snapshot, contract, b, arrays = _method_problem(name)
    dense = run_dense_df_ccsdt_oracle(
        dense_df_oracle_from_three_index(snapshot, contract, b)
    )
    provider = FixtureDFProvider(snapshot, b, arrays["g"])
    actual = solve_df_ccsdt(
        snapshot,
        provider,
        contract,
        options=SolverOptions(
            residual_tolerance=1e-10,
            energy_tolerance=1e-12,
        ),
    )

    assert actual.converged, actual.reason
    np.testing.assert_allclose(
        actual.ccsd_correlation_energy,
        dense.ccsd_correlation_energy,
        atol=2e-10,
        rtol=0,
    )
    np.testing.assert_allclose(
        actual.triples_energy,
        dense.triples_energy,
        atol=2e-10,
        rtol=0,
    )
    np.testing.assert_allclose(
        actual.total_energy, dense.total_energy, atol=3e-10, rtol=0
    )
    assert actual.provenance["resident_ovvv"] is False
    assert actual.provenance["resident_vvvv"] is False
    assert actual.provenance["full_t3"] is False

    occupied = set(range(snapshot.nocc))
    virtual = set(range(snapshot.nocc, snapshot.nmo))
    for slots in provider.calls:
        spaces = tuple(
            "v"
            if set(slot).issubset(virtual) and slot
            else "o"
            if set(slot).issubset(occupied) and slot
            else "?"
            for slot in slots
        )
        assert spaces not in (("o", "v", "v", "v"), ("v", "v", "v", "v"))


def test_factorized_triples_budget_fails_before_work():
    bov = np.zeros((1, 1, 1))
    bvv = np.zeros((1, 1, 1))
    ovoo = np.zeros((1, 1, 1, 1))
    ovov = np.zeros((1, 1, 1, 1))
    fov = np.zeros((1, 1))
    t1 = np.zeros((1, 1))
    t2 = np.zeros((1, 1, 1, 1))
    eps_o = np.array([-1.0])
    eps_v = np.array([1.0])
    with pytest.raises(MemoryError, match="temporary numeric bytes"):
        factorized_triples_energy(
            bov,
            bvv,
            ovoo,
            ovov,
            fov,
            t1,
            t2,
            eps_o,
            eps_v,
            max_bytes=1,
        )

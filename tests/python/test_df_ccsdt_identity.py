"""Bind the DF oracle to its actual orbital state and immutable numeric evidence."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_df_ccsdt_oracle import _problem

from tools.vibeqc_cc.df_ccsdt_oracle import (
    DenseDFOracleProvider,
    dense_df_oracle_from_three_index,
    run_dense_df_ccsdt_oracle,
)


@pytest.mark.parametrize("change", ("orbital_phase", "reference_energy"))
@pytest.mark.parametrize("entry", ("prepare", "provider"))
def test_same_hamiltonian_does_not_admit_a_different_snapshot(
    change: str, entry: str
) -> None:
    _, snapshot, contract, _, _, b, _, arrays = _problem("h2")
    coefficients = snapshot.coefficients.copy()
    coefficients[:, 0] *= -1
    changes = (
        {"coefficients": coefficients}
        if change == "orbital_phase"
        else {"reference_energy": snapshot.reference_energy + 0.25}
    )
    other = replace(snapshot, **changes)
    assert other.identity != snapshot.identity
    with pytest.raises(ValueError, match="identity|snapshot"):
        if entry == "prepare":
            dense_df_oracle_from_three_index(other, contract, b)
        else:
            DenseDFOracleProvider(other, arrays["g"], contract)


def test_diagnostic_mutation_cannot_rewrite_result_identity_or_provenance() -> None:
    _, snapshot, contract, _, _, b, _, _ = _problem("h2")
    prepared = dense_df_oracle_from_three_index(snapshot, contract, b)
    prepared.diagnostics["nested"] = {"trial": [1, 2]}
    first = run_dense_df_ccsdt_oracle(prepared)
    recorded = deepcopy(first.provenance)
    prepared.diagnostics["three_index_sha256"] = "tampered"
    prepared.diagnostics["eri_sha256"] = "tampered"
    prepared.diagnostics["nested"]["trial"].append(3)
    second = run_dense_df_ccsdt_oracle(prepared)
    assert second.oracle_identity == first.oracle_identity
    assert second.provenance["oracle"]["three_index_sha256"] != "tampered"
    assert second.provenance["oracle"]["eri_sha256"] != "tampered"
    assert first.provenance == recorded

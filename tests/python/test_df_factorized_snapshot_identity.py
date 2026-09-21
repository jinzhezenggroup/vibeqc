"""Factorized DF contractions must use the contract's exact reference state."""

from dataclasses import replace

import pytest
from test_df_ccsd_factorized import FixtureDFProvider, _method_problem

from tools.vibeqc_cc import df_factorized


@pytest.mark.parametrize(
    "change", ("orbital_phase", "reference_energy", "contract_binding")
)
def test_factorized_contract_rejects_other_reference_before_work(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    snapshot, contract, factors, arrays = _method_problem("h2")
    if change == "contract_binding":
        contract = replace(contract, correlation_snapshot_identity="another-state")
        other = snapshot
    else:
        coefficients = snapshot.coefficients.copy()
        coefficients[:, 0] *= -1
        changes = (
            {"coefficients": coefficients}
            if change == "orbital_phase"
            else {"reference_energy": snapshot.reference_energy + 0.25}
        )
        other = replace(snapshot, **changes)
    assert other.identity != contract.correlation_snapshot_identity
    provider = FixtureDFProvider(other, factors, arrays["g"])

    def unexpected_program(*args: object, **kwargs: object) -> None:
        pytest.fail("mismatched reference reached graph construction")

    monkeypatch.setattr(df_factorized, "_df_programs", unexpected_program)
    try:
        with pytest.raises(ValueError, match="contract/reference mismatch"):
            df_factorized.PreparedDFCCSD(other, provider, contract)
        assert provider.calls == []
        assert provider.statistics["external_reserved_bytes"] == 0
    finally:
        provider.close()

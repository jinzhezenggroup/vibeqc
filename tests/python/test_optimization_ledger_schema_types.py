"""JSON schema versions must not rely on Python's cross-type equality."""

import json

import pytest

from tools import check_compiler_optimization_ledger as ledger


@pytest.mark.parametrize("encoded", ("true", "false", "1.0", '"1"', "null", "{}"))
def test_noninteger_json_schema_version_is_rejected(encoded: str) -> None:
    payload = json.loads(ledger.LEDGER.read_text(encoding="utf-8"))
    payload["schema_version"] = json.loads(encoded)
    with pytest.raises(ledger.LedgerError, match="schema_version"):
        ledger.validate(payload)


def test_integer_version_preserves_valid_inventory() -> None:
    payload = json.loads(ledger.LEDGER.read_text(encoding="utf-8"))
    assert type(payload["schema_version"]) is int
    assert ledger.validate(payload) == {
        "groups": 8,
        "mechanisms": 31,
        "negative_evidence": 3,
    }

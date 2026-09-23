"""Fail-closed tests for the historical compiler optimization ledger."""

from __future__ import annotations

import copy
import json

import pytest

from tools import check_compiler_optimization_ledger as ledger


def _payload() -> dict[str, object]:
    return json.loads(ledger.LEDGER.read_text(encoding="utf-8"))


def test_repository_ledger_is_complete() -> None:
    assert ledger.validate(_payload()) == {
        "groups": 8,
        "mechanisms": 31,
        "negative_evidence": 3,
    }


def test_missing_historical_group_fails_closed() -> None:
    payload = _payload()
    payload["groups"] = payload["groups"][:-1]
    with pytest.raises(ledger.LedgerError, match="group inventory mismatch"):
        ledger.validate(payload)


def test_duplicate_group_id_is_rejected() -> None:
    payload = _payload()
    duplicate = copy.deepcopy(payload["groups"][0])
    payload["groups"].append(duplicate)
    with pytest.raises(ledger.LedgerError, match="duplicate group id"):
        ledger.validate(payload)


def test_invalid_historical_commit_identity_is_rejected() -> None:
    payload = _payload()
    payload["groups"][0]["historical_commits"][0] = "main"
    with pytest.raises(ledger.LedgerError, match="invalid commit identity"):
        ledger.validate(payload)


def test_owner_requires_issue_reference() -> None:
    payload = _payload()
    payload["groups"][0]["current_owners"][0]["refs"] = ["682"]
    with pytest.raises(ledger.LedgerError, match="invalid issue reference"):
        ledger.validate(payload)


def test_negative_evidence_requires_promotion_rule() -> None:
    payload = _payload()
    payload["negative_evidence"][0]["promotion_rule"] = ""
    with pytest.raises(ledger.LedgerError, match="promotion_rule"):
        ledger.validate(payload)


def test_unknown_negative_decision_is_rejected() -> None:
    payload = _payload()
    payload["negative_evidence"][0]["decision"] = "promoted"
    with pytest.raises(ledger.LedgerError, match="decision is unknown"):
        ledger.validate(payload)

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
        "adoption_cells": 64,
        "benchmark_qualified": 0,
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


def test_missing_adoption_cell_fails_closed() -> None:
    payload = _payload()
    payload["groups"][0]["adoption"][0]["methods"].remove("CCSD(T)")
    with pytest.raises(ledger.LedgerError, match="adoption cell inventory mismatch"):
        ledger.validate(payload)


def test_overlapping_adoption_cell_is_rejected() -> None:
    payload = _payload()
    duplicate = copy.deepcopy(payload["groups"][0]["adoption"][0])
    duplicate["methods"] = ["HF"]
    duplicate["backends"] = ["cpu"]
    payload["groups"][0]["adoption"].append(duplicate)
    with pytest.raises(ledger.LedgerError, match="overlaps existing adoption cell"):
        ledger.validate(payload)


def test_unknown_adoption_status_is_rejected() -> None:
    payload = _payload()
    payload["groups"][0]["adoption"][0]["status"] = "done"
    with pytest.raises(ledger.LedgerError, match="status is unknown"):
        ledger.validate(payload)


def test_represented_adoption_requires_issue_reference() -> None:
    payload = _payload()
    del payload["groups"][0]["adoption"][0]["refs"]
    with pytest.raises(ledger.LedgerError, match=r"adoption\[0\]\.refs"):
        ledger.validate(payload)


def test_benchmark_qualified_requires_retained_evidence() -> None:
    payload = _payload()
    payload["groups"][0]["adoption"][0]["status"] = "benchmark-qualified"
    with pytest.raises(ledger.LedgerError, match=r"adoption\[0\]\.evidence"):
        ledger.validate(payload)


def test_benchmark_qualified_requires_positive_regression_budget() -> None:
    payload = _payload()
    row = payload["groups"][0]["adoption"][0]
    row["status"] = "benchmark-qualified"
    row["evidence"] = ["benchmarks/results/cuda-ownership/df/README.md"]
    row["max_regression_percent"] = 0.0
    with pytest.raises(ledger.LedgerError, match="max_regression_percent"):
        ledger.validate(payload)


def test_benchmark_qualified_cells_are_counted() -> None:
    payload = _payload()
    row = payload["groups"][0]["adoption"][0]
    row["status"] = "benchmark-qualified"
    row["evidence"] = ["benchmarks/results/cuda-ownership/df/README.md"]
    row["max_regression_percent"] = 2.0
    assert ledger.validate(payload)["benchmark_qualified"] == 8


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

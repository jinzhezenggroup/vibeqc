"""Explicit JSON null cannot stand in for a required evidence-level record."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tools.check_electronic_production_paths import (
    EVIDENCE_LEVELS,
    PATH_FIELDS,
    SCHEMA,
    validate_production_path_ledger,
)

if TYPE_CHECKING:
    from pathlib import Path


def _ledger() -> dict:
    return {
        "schema": SCHEMA,
        "coverage": "pilot",
        "rows": [
            {
                "id": "probe",
                "method_family": "hf",
                "product": "energy",
                "backend": "cpu",
                "domain": "test",
                "status": "qualification-only",
                "artifact": "test artifact",
                **dict.fromkeys(PATH_FIELDS, "source.py"),
                "evidence": ["source.py"],
                "blocker": None,
                "evidence_levels": {
                    level: {
                        "state": "missing",
                        "evidence": [],
                        "reason": "Not yet retained.",
                    }
                    for level in EVIDENCE_LEVELS
                },
            }
        ],
    }


@pytest.mark.parametrize("level", EVIDENCE_LEVELS)
def test_explicit_null_evidence_level_is_rejected(tmp_path: Path, level: str) -> None:
    payload = _ledger()
    assert validate_production_path_ledger(payload, root=tmp_path) == []
    payload["rows"][0]["evidence_levels"][level] = None
    assert validate_production_path_ledger(payload, root=tmp_path) == [
        f"probe.evidence_levels.{level} must be an object"
    ]


@pytest.mark.parametrize("level", EVIDENCE_LEVELS)
def test_absent_evidence_level_keeps_one_missing_diagnostic(
    tmp_path: Path, level: str
) -> None:
    payload = _ledger()
    del payload["rows"][0]["evidence_levels"][level]
    assert validate_production_path_ledger(payload, root=tmp_path) == [
        f"probe.evidence_levels missing levels: {level}"
    ]


def test_explicit_missing_evidence_is_not_promoted(tmp_path: Path) -> None:
    payload = _ledger()
    before = repr(payload)
    assert validate_production_path_ledger(payload, root=tmp_path) == []
    assert repr(payload) == before

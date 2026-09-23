"""Regression tests for the electronic-structure production-path ledger."""

from __future__ import annotations

import copy
import json
import typing

from tools.check_electronic_production_paths import (
    DEFAULT_LEDGER,
    EVIDENCE_LEVELS,
    load_and_validate,
    validate_production_path_ledger,
)

if typing.TYPE_CHECKING:
    from pathlib import Path


def _evidence_levels() -> dict[str, object]:
    return {
        "represented": {
            "state": "present",
            "evidence": ["science.cpp"],
            "reason": None,
        },
        "compiled-cpu": {
            "state": "present",
            "evidence": ["test_energy.py"],
            "reason": None,
        },
        "compiled-cuda": {
            "state": "not-applicable",
            "evidence": [],
            "reason": "CPU fixture.",
        },
        "device-executed": {
            "state": "not-applicable",
            "evidence": [],
            "reason": "CPU fixture.",
        },
        "domain-qualified": {
            "state": "present",
            "evidence": ["test_energy.py"],
            "reason": None,
        },
        "molecular": {
            "state": "present",
            "evidence": ["test_energy.py"],
            "reason": None,
        },
        "derivative": {
            "state": "not-applicable",
            "evidence": [],
            "reason": "Energy-only fixture.",
        },
        "public": {
            "state": "present",
            "evidence": ["test_energy.py"],
            "reason": None,
        },
    }


def _fixture() -> dict[str, object]:
    return {
        "schema": "vibeqc.electronic-production-paths.v2",
        "coverage": "pilot",
        "rows": [
            {
                "id": "hf-energy-cpu-direct",
                "method_family": "hf",
                "product": "energy",
                "backend": "cpu",
                "domain": "direct-rhf",
                "status": "production",
                "public_entry": "entry.py",
                "selector": "selector.py",
                "scientific_owner": "science.cpp",
                "provider_owner": "provider.cpp",
                "execution_owner": "execute.cpp",
                "artifact": "native CPU executable",
                "state_owner": "state.hpp",
                "resource_owner": "resources.cpp",
                "evidence": ["test_energy.py"],
                "evidence_levels": _evidence_levels(),
                "blocker": None,
            }
        ],
    }


def _write_fixture_files(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
    )
    for path in (
        "entry.py",
        "selector.py",
        "science.cpp",
        "provider.cpp",
        "execute.cpp",
        "state.hpp",
        "resources.cpp",
        "test_energy.py",
    ):
        (root / path).write_text("// fixture\n", encoding="utf-8")


def test_repository_production_path_ledger_is_valid() -> None:
    payload, errors = load_and_validate(DEFAULT_LEDGER)
    assert payload["coverage"] == "pilot"
    assert errors == []
    rows = typing.cast("list[dict[str, object]]", payload["rows"])
    assert {row["method_family"] for row in rows} >= {"hf", "dft"}
    assert {row["backend"] for row in rows} >= {"cpu", "cuda"}
    for row in rows:
        levels = typing.cast("dict[str, object]", row["evidence_levels"])
        assert set(levels) == set(EVIDENCE_LEVELS)


def test_production_row_requires_existing_actual_path_anchors(tmp_path: Path) -> None:
    payload = _fixture()
    _write_fixture_files(tmp_path)
    assert validate_production_path_ledger(payload, root=tmp_path) == []

    (tmp_path / "execute.cpp").unlink()
    errors = validate_production_path_ledger(payload, root=tmp_path)
    assert errors == [
        "hf-energy-cpu-direct.execution_owner path does not exist: execute.cpp"
    ]


def test_missing_production_anchor_diagnostics_are_stably_ordered(
    tmp_path: Path,
) -> None:
    payload = _fixture()
    row = typing.cast("list[dict[str, object]]", payload["rows"])[0]
    row["public_entry"] = ""
    row["selector"] = ""
    errors = validate_production_path_ledger(payload, root=tmp_path)
    assert errors == [
        "hf-energy-cpu-direct.public_entry must be set for a production row",
        "hf-energy-cpu-direct.selector must be set for a production row",
    ]


def test_duplicate_execution_domain_is_rejected(tmp_path: Path) -> None:
    payload = _fixture()
    duplicate = copy.deepcopy(
        typing.cast("list[dict[str, object]]", payload["rows"])[0]
    )
    duplicate["id"] = "hf-energy-cpu-direct-copy"
    typing.cast("list[dict[str, object]]", payload["rows"]).append(duplicate)

    errors = validate_production_path_ledger(payload, root=tmp_path)
    assert any(
        error
        == "duplicate method/product/backend/domain row: hf / energy / cpu / direct-rhf"
        for error in errors
    )


def test_unsupported_row_requires_actionable_blocker(tmp_path: Path) -> None:
    payload = _fixture()
    row = typing.cast("list[dict[str, object]]", payload["rows"])[0]
    row["status"] = "unsupported"
    row["blocker"] = ""
    errors = validate_production_path_ledger(payload, root=tmp_path)
    assert (
        "hf-energy-cpu-direct.blocker must explain why an unsupported row is blocked"
        in errors
    )


def test_non_unsupported_row_cannot_carry_stale_blocker(tmp_path: Path) -> None:
    payload = _fixture()
    row = typing.cast("list[dict[str, object]]", payload["rows"])[0]
    row["blocker"] = "old exception"
    errors = validate_production_path_ledger(payload, root=tmp_path)
    assert errors == [
        "hf-energy-cpu-direct.blocker must be empty unless status is 'unsupported'"
    ]


def test_all_evidence_levels_are_required(tmp_path: Path) -> None:
    payload = _fixture()
    row = typing.cast("list[dict[str, object]]", payload["rows"])[0]
    levels = typing.cast("dict[str, object]", row["evidence_levels"])
    levels.pop("device-executed")
    errors = validate_production_path_ledger(payload, root=tmp_path)
    assert errors == [
        "hf-energy-cpu-direct.evidence_levels missing levels: device-executed"
    ]


def test_missing_evidence_state_requires_reason_and_no_evidence(tmp_path: Path) -> None:
    payload = _fixture()
    row = typing.cast("list[dict[str, object]]", payload["rows"])[0]
    levels = typing.cast("dict[str, dict[str, object]]", row["evidence_levels"])
    levels["domain-qualified"] = {
        "state": "missing",
        "evidence": ["test_energy.py"],
        "reason": "",
    }
    errors = validate_production_path_ledger(payload, root=tmp_path)
    assert errors == [
        "hf-energy-cpu-direct.evidence_levels.domain-qualified.evidence must be empty "
        "when state is 'missing'",
        "hf-energy-cpu-direct.evidence_levels.domain-qualified.reason must explain "
        "state 'missing'",
    ]


def test_failed_evidence_keeps_failure_anchor_and_reason(tmp_path: Path) -> None:
    payload = _fixture()
    row = typing.cast("list[dict[str, object]]", payload["rows"])[0]
    levels = typing.cast("dict[str, dict[str, object]]", row["evidence_levels"])
    levels["domain-qualified"] = {
        "state": "failed",
        "evidence": ["test_energy.py"],
        "reason": "Pinned oracle mismatch.",
    }
    assert validate_production_path_ledger(payload, root=tmp_path) == []


def test_production_backend_compile_evidence_cannot_be_silently_missing(
    tmp_path: Path,
) -> None:
    payload = _fixture()
    row = typing.cast("list[dict[str, object]]", payload["rows"])[0]
    levels = typing.cast("dict[str, dict[str, object]]", row["evidence_levels"])
    levels["compiled-cpu"] = {
        "state": "missing",
        "evidence": [],
        "reason": "No retained compile evidence.",
    }
    errors = validate_production_path_ledger(payload, root=tmp_path)
    assert errors == [
        "hf-energy-cpu-direct.evidence_levels.compiled-cpu must be present "
        "for a production row"
    ]


def test_present_evidence_requires_anchor_and_rejects_stale_reason(tmp_path: Path) -> None:
    payload = _fixture()
    row = typing.cast("list[dict[str, object]]", payload["rows"])[0]
    levels = typing.cast("dict[str, dict[str, object]]", row["evidence_levels"])
    levels["domain-qualified"] = {
        "state": "present",
        "evidence": [],
        "reason": "old blocker",
    }
    errors = validate_production_path_ledger(payload, root=tmp_path)
    assert errors == [
        "hf-energy-cpu-direct.evidence_levels.domain-qualified.evidence must be non-empty "
        "when state is 'present'",
        "hf-energy-cpu-direct.evidence_levels.domain-qualified.reason must be empty "
        "when evidence is present",
    ]


def test_load_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text("{", encoding="utf-8")
    payload, errors = load_and_validate(path, root=tmp_path)
    assert payload == {}
    assert len(errors) == 1
    assert errors[0].startswith("cannot load production-path ledger ")


def test_ledger_is_deterministic_json() -> None:
    payload = json.loads(DEFAULT_LEDGER.read_text(encoding="utf-8"))
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    reparsed = json.loads(rendered)
    assert reparsed == payload
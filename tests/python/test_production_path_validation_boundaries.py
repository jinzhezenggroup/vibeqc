"""Production anchors stay in the checkout; malformed ledgers produce diagnostics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tools.check_electronic_production_paths import (
    EVIDENCE_LEVELS,
    PATH_FIELDS,
    SCHEMA,
    load_and_validate,
    validate_production_path_ledger,
)

if TYPE_CHECKING:
    from pathlib import Path


def _ledger(root: Path) -> dict:
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (root / "anchor.py").write_text("# source anchor\n")
    not_applicable = {"compiled-cuda", "device-executed", "derivative"}
    return {
        "schema": SCHEMA,
        "coverage": "pilot",
        "rows": [
            {
                "id": "hf-cpu",
                "method_family": "hf",
                "product": "energy",
                "backend": "cpu",
                "domain": "direct-rhf",
                "status": "production",
                **dict.fromkeys(PATH_FIELDS, "anchor.py"),
                "artifact": "native executable",
                "evidence": ["anchor.py"],
                "evidence_levels": {
                    level: {
                        "state": "not-applicable"
                        if level in not_applicable
                        else "present",
                        "evidence": [] if level in not_applicable else ["anchor.py"],
                        "reason": "CPU energy-only fixture."
                        if level in not_applicable
                        else None,
                    }
                    for level in EVIDENCE_LEVELS
                },
                "blocker": None,
            }
        ],
    }


@pytest.mark.parametrize("field", (*PATH_FIELDS, "evidence"))
@pytest.mark.parametrize("spelling", ("absolute", "parent", "symlink"))
def test_production_paths_cannot_validate_an_external_file(
    tmp_path: Path, field: str, spelling: str
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    payload = _ledger(root)
    external = tmp_path / "outside.py"
    external.write_text("# not in the checkout\n")
    if spelling == "absolute":
        value = str(external)
    elif spelling == "parent":
        value = "../outside.py"
    else:
        link = root / "escape.py"
        try:
            link.symlink_to(external)
        except OSError:
            pytest.skip("symlinks unavailable")
        value = "escape.py"
    payload["rows"][0][field] = [value] if field == "evidence" else value
    errors = validate_production_path_ledger(payload, root=root)
    assert len(errors) == 1
    assert field in errors[0]
    assert "repository" in errors[0]


@pytest.mark.parametrize("field", ("coverage", "backend", "status"))
@pytest.mark.parametrize("value", ([], {}))
def test_unhashable_json_fields_return_diagnostics(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _ledger(tmp_path)
    target = payload if field == "coverage" else payload["rows"][0]
    target[field] = value
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(payload))
    _, errors = load_and_validate(path, root=tmp_path)
    assert len(errors) == 1
    assert field in errors[0]


def test_invalid_encoding_is_a_load_error(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_bytes(b"\xff")
    payload, errors = load_and_validate(path, root=tmp_path)
    assert payload == {}
    assert len(errors) == 1
    assert errors[0].startswith("cannot load production-path ledger")


def test_valid_local_anchors_and_missing_file_check_remain(tmp_path: Path) -> None:
    payload = _ledger(tmp_path)
    assert validate_production_path_ledger(payload, root=tmp_path) == []
    payload["rows"][0]["execution_owner"] = "missing.py"
    assert validate_production_path_ledger(payload, root=tmp_path) == [
        "hf-cpu.execution_owner path does not exist: missing.py"
    ]


@pytest.mark.parametrize("level", EVIDENCE_LEVELS)
@pytest.mark.parametrize("spelling", ("absolute", "parent", "symlink"))
def test_evidence_level_paths_stay_inside_repository(
    tmp_path: Path, level: str, spelling: str
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    payload = _ledger(root)
    external = tmp_path / "outside.py"
    external.write_text("# external evidence\n")
    if spelling == "absolute":
        value = str(external)
    elif spelling == "parent":
        value = "../outside.py"
    else:
        link = root / "escape.py"
        try:
            link.symlink_to(external)
        except OSError:
            pytest.skip("symlinks unavailable")
        value = "escape.py"
    payload["rows"][0]["evidence_levels"][level] = {
        "state": "present",
        "evidence": [value],
        "reason": None,
    }
    errors = validate_production_path_ledger(payload, root=root)
    assert len(errors) == 1
    assert f"evidence_levels.{level}.evidence[0]" in errors[0]
    assert "repository" in errors[0]

"""Invalid inventory input must return diagnostics rather than escape validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tools import check_direct_jk_tuning_inventory as checker

if TYPE_CHECKING:
    from typing import Any


def _fixture(root: Path) -> dict[str, Any]:
    (root / "docs").mkdir()
    (root / "note.md").write_text("synthetic #597 fixture\n", encoding="utf-8")
    (root / "constants.hpp").write_text(
        "constexpr unsigned kCandidate = 16;\n"
        "constexpr unsigned kSafety = 1;\n",
        encoding="utf-8",
    )
    (root / "policy.hpp").write_text("resolved_policy\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "issue": 597,
        "source_note": "note.md",
        "remaining_constants": [
            {
                "symbol": "kCandidate",
                "source": "constants.hpp",
                "expected_expression": "16",
                "classification": "bounded-schedule-choice",
                "state": "retained-explicit",
                "current_owner": "fixture owner",
                "intended_owner": "fixture owner",
                "retirement_condition": "fixture only",
            }
        ],
        "migrated_settings": [
            {
                "name": "fixture-policy",
                "source": "policy.hpp",
                "owner": "fixture owner",
                "required_tokens": ["resolved_policy"],
            }
        ],
        "excluded_from_tuning": [
            {
                "symbol": "kSafety",
                "source": "constants.hpp",
                "expected_expression": "1",
                "classification": "resource-safety-constraint",
                "reason": "fixture only",
            }
        ],
    }


def _write(root: Path, payload: dict[str, Any]) -> None:
    (root / checker.INVENTORY_PATH).write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _link(path: Path, target: Path) -> None:
    try:
        path.symlink_to(target)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable: {error}")


@pytest.mark.parametrize(
    "section,field",
    [
        ("remaining_constants", "classification"),
        ("remaining_constants", "state"),
        ("excluded_from_tuning", "classification"),
    ],
)
@pytest.mark.parametrize("value", [[], {}, False, None])
def test_invalid_enum_returns_field_diagnostic(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    payload = _fixture(tmp_path)
    payload[section][0][field] = value
    _write(tmp_path, payload)
    errors = checker.validate_repository(tmp_path)
    assert any(f"{section}[0].{field}" in error for error in errors)


@pytest.mark.parametrize("where", ["inventory", "source"])
def test_cyclic_symlink_returns_path_diagnostic(tmp_path: Path, where: str) -> None:
    payload = _fixture(tmp_path)
    _write(tmp_path, payload)
    path = (
        tmp_path / checker.INVENTORY_PATH
        if where == "inventory"
        else tmp_path / "constants.hpp"
    )
    path.unlink()
    _link(path, Path(path.name))
    errors = checker.validate_repository(tmp_path)
    assert errors
    assert any("cannot resolve repository path" in error for error in errors)


def test_cyclic_root_returns_diagnostic(tmp_path: Path) -> None:
    root = tmp_path / "loop"
    _link(root, Path("loop"))
    errors = checker.validate_repository(root)
    assert any("cannot resolve repository path" in error for error in errors)


def test_embedded_nul_returns_diagnostic(tmp_path: Path) -> None:
    payload = _fixture(tmp_path)
    payload["remaining_constants"][0]["source"] = "invalid\x00.hpp"
    _write(tmp_path, payload)
    assert checker.validate_repository(tmp_path)


def test_valid_fixture_preserves_success(tmp_path: Path) -> None:
    _write(tmp_path, _fixture(tmp_path))
    assert checker.validate_repository(tmp_path) == []


def test_internal_source_alias_preserves_success(tmp_path: Path) -> None:
    payload = _fixture(tmp_path)
    _link(tmp_path / "alias.hpp", Path("constants.hpp"))
    payload["remaining_constants"][0]["source"] = "alias.hpp"
    _write(tmp_path, payload)
    assert checker.validate_repository(tmp_path) == []


def test_missing_source_keeps_read_diagnostic(tmp_path: Path) -> None:
    payload = _fixture(tmp_path)
    payload["remaining_constants"][0]["source"] = "missing.hpp"
    _write(tmp_path, payload)
    errors = checker.validate_repository(tmp_path)
    assert any("cannot read" in error and "missing.hpp" in error for error in errors)

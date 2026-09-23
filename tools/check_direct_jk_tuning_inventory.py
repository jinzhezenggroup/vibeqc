"""Validate the machine-readable Direct J/K tuning ownership inventory for #597."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

INVENTORY_PATH = Path("docs/direct_jk_tuning_inventory.json")
_ALLOWED_REMAINING_CLASSIFICATIONS = {
    "bounded-schedule-choice",
    "measured-schedule-choice",
    "workload-profile-choice",
}
_ALLOWED_REMAINING_STATES = {"migration-pending", "retained-explicit"}
_ALLOWED_EXCLUDED_CLASSIFICATIONS = {
    "resource-safety-constraint",
    "scientific-numerical-policy",
}
_CONSTEXPR_RE_TEMPLATE = r"\b{symbol}\s*=\s*(?P<expression>[^;]+);"


def _safe_repository_path(
    root: Path, value: object, field: str, errors: list[str]
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field} must be a non-empty repository-relative path")
        return None
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        errors.append(f"{field} must stay inside the repository: {value!r}")
        return None
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        errors.append(f"{field} escapes the repository: {value!r}")
        return None
    return resolved


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read inventory {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append("inventory root must be a JSON object")
        return None
    return payload


def _read_text(path: Path, field: str, errors: list[str]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read {field} {path}: {exc}")
        return None


def _normalized_expression(value: str) -> str:
    return "".join(value.split())


def _require_constexpr(
    root: Path,
    entry: dict[str, Any],
    section: str,
    errors: list[str],
) -> None:
    symbol = entry.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        errors.append(f"{section} entry requires a non-empty symbol")
        return
    source = _safe_repository_path(
        root, entry.get("source"), f"{section}.{symbol}.source", errors
    )
    expected = entry.get("expected_expression")
    if not isinstance(expected, str) or not expected.strip():
        errors.append(f"{section}.{symbol}.expected_expression must be non-empty")
        return
    if source is None:
        return
    text = _read_text(source, f"{section}.{symbol}.source", errors)
    if text is None:
        return
    match = re.search(
        _CONSTEXPR_RE_TEMPLATE.format(symbol=re.escape(symbol)),
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        errors.append(f"{section}.{symbol}: declaration not found in {entry['source']}")
        return
    actual = _normalized_expression(match.group("expression"))
    wanted = _normalized_expression(expected)
    if actual != wanted:
        errors.append(
            f"{section}.{symbol}: expression drifted in {entry['source']}: "
            f"expected {expected!r}, found {match.group('expression').strip()!r}"
        )


def _require_tokens(
    root: Path,
    source_value: object,
    tokens_value: object,
    field: str,
    errors: list[str],
) -> None:
    source = _safe_repository_path(root, source_value, f"{field}.source", errors)
    if source is None:
        return
    if not isinstance(tokens_value, list) or not tokens_value:
        errors.append(f"{field}.required_tokens must be a non-empty list")
        return
    if any(not isinstance(token, str) or not token for token in tokens_value):
        errors.append(f"{field}.required_tokens must contain only non-empty strings")
        return
    text = _read_text(source, f"{field}.source", errors)
    if text is None:
        return
    for token in tokens_value:
        if token not in text:
            errors.append(
                f"{field}: required token {token!r} missing from {source_value}"
            )


def validate_repository(root: Path) -> list[str]:
    errors: list[str] = []
    inventory = _safe_repository_path(
        root, INVENTORY_PATH.as_posix(), "inventory", errors
    )
    if inventory is None:
        return errors
    payload = _load_json(inventory, errors)
    if payload is None:
        return errors

    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        errors.append("schema_version must be integer 1")
    if type(payload.get("issue")) is not int or payload["issue"] != 597:
        errors.append("issue must be integer 597")

    note = _safe_repository_path(
        root, payload.get("source_note"), "source_note", errors
    )
    if note is not None:
        note_text = _read_text(note, "source_note", errors)
        if note_text is not None and "#597" not in note_text:
            errors.append("source_note must explicitly identify #597")

    remaining = payload.get("remaining_constants")
    if not isinstance(remaining, list) or not remaining:
        errors.append("remaining_constants must be a non-empty list")
        remaining = []
    migrated = payload.get("migrated_settings")
    if not isinstance(migrated, list) or not migrated:
        errors.append("migrated_settings must be a non-empty list")
        migrated = []
    excluded = payload.get("excluded_from_tuning")
    if not isinstance(excluded, list) or not excluded:
        errors.append("excluded_from_tuning must be a non-empty list")
        excluded = []

    seen_symbols: set[str] = set()
    for index, raw_entry in enumerate(remaining):
        field = f"remaining_constants[{index}]"
        if not isinstance(raw_entry, dict):
            errors.append(f"{field} must be an object")
            continue
        entry = raw_entry
        symbol = entry.get("symbol")
        if isinstance(symbol, str) and symbol:
            if symbol in seen_symbols:
                errors.append(f"duplicate constant symbol {symbol!r}")
            seen_symbols.add(symbol)
        classification = entry.get("classification")
        if classification not in _ALLOWED_REMAINING_CLASSIFICATIONS:
            errors.append(
                f"{field}.classification is not recognized: {classification!r}"
            )
        state = entry.get("state")
        if state not in _ALLOWED_REMAINING_STATES:
            errors.append(f"{field}.state is not recognized: {state!r}")
        for required in ("current_owner", "intended_owner", "retirement_condition"):
            if not isinstance(entry.get(required), str) or not entry[required].strip():
                errors.append(f"{field}.{required} must be non-empty")
        _require_constexpr(root, entry, field, errors)
        if state == "migration-pending" and "policy_source" in entry:
            _require_tokens(
                root,
                entry.get("policy_source"),
                entry.get("required_policy_tokens"),
                f"{field}.policy_anchor",
                errors,
            )

    migrated_names: set[str] = set()
    for index, raw_entry in enumerate(migrated):
        field = f"migrated_settings[{index}]"
        if not isinstance(raw_entry, dict):
            errors.append(f"{field} must be an object")
            continue
        name = raw_entry.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"{field}.name must be non-empty")
        elif name in migrated_names:
            errors.append(f"duplicate migrated setting {name!r}")
        else:
            migrated_names.add(name)
        if (
            not isinstance(raw_entry.get("owner"), str)
            or not raw_entry["owner"].strip()
        ):
            errors.append(f"{field}.owner must be non-empty")
        token_field = f"{field}({name})" if isinstance(name, str) and name else field
        _require_tokens(
            root,
            raw_entry.get("source"),
            raw_entry.get("required_tokens"),
            token_field,
            errors,
        )

    for index, raw_entry in enumerate(excluded):
        field = f"excluded_from_tuning[{index}]"
        if not isinstance(raw_entry, dict):
            errors.append(f"{field} must be an object")
            continue
        symbol = raw_entry.get("symbol")
        if isinstance(symbol, str) and symbol:
            if symbol in seen_symbols:
                errors.append(f"duplicate constant symbol {symbol!r}")
            seen_symbols.add(symbol)
        classification = raw_entry.get("classification")
        if classification not in _ALLOWED_EXCLUDED_CLASSIFICATIONS:
            errors.append(
                f"{field}.classification is not recognized: {classification!r}"
            )
        if (
            not isinstance(raw_entry.get("reason"), str)
            or not raw_entry["reason"].strip()
        ):
            errors.append(f"{field}.reason must be non-empty")
        _require_constexpr(root, raw_entry, field, errors)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    errors = validate_repository(args.root)
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1
    payload = json.loads((args.root / INVENTORY_PATH).read_text(encoding="utf-8"))
    print(
        "Direct J/K tuning inventory validated: "
        f"remaining={len(payload['remaining_constants'])} "
        f"migrated={len(payload['migrated_settings'])} "
        f"excluded={len(payload['excluded_from_tuning'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

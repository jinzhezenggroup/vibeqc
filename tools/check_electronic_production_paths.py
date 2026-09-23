"""Validate the machine-readable electronic-structure production-path ledger."""

from __future__ import annotations

import argparse
import json
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs" / "electronic_structure_production_paths.json"
SCHEMA = "vibeqc.electronic-production-paths.v1"
STATUSES = {"production", "qualification-only", "reference", "unsupported"}
BACKENDS = {"cpu", "cuda"}

REQUIRED_FIELDS = {
    "id",
    "method_family",
    "product",
    "backend",
    "domain",
    "status",
    "public_entry",
    "selector",
    "scientific_owner",
    "provider_owner",
    "execution_owner",
    "artifact",
    "state_owner",
    "resource_owner",
    "evidence",
    "blocker",
}
PATH_FIELDS = (
    "public_entry",
    "selector",
    "scientific_owner",
    "provider_owner",
    "execution_owner",
    "state_owner",
    "resource_owner",
)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_production_path_ledger(
    payload: object,
    *,
    root: Path = ROOT,
) -> list[str]:
    """Return deterministic validation errors for one production-path ledger."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["ledger must be a JSON object"]

    if payload.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    if payload.get("coverage") not in {"pilot", "complete"}:
        errors.append("coverage must be either 'pilot' or 'complete'")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("rows must be a non-empty list")
        return errors

    full_checkout = (root / "pyproject.toml").is_file()
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str, str, str]] = set()

    for index, raw_row in enumerate(rows):
        label = f"rows[{index}]"
        if not isinstance(raw_row, dict):
            errors.append(f"{label} must be an object")
            continue

        missing = sorted(REQUIRED_FIELDS - raw_row.keys())
        if missing:
            errors.append(f"{label} missing fields: {', '.join(missing)}")
            continue

        row = typing.cast("dict[str, object]", raw_row)
        row_id = row["id"]
        if not _is_nonempty_string(row_id):
            errors.append(f"{label}.id must be a non-empty string")
            row_id_text = label
        else:
            row_id_text = typing.cast("str", row_id)
            if row_id_text in seen_ids:
                errors.append(f"duplicate row id: {row_id_text}")
            seen_ids.add(row_id_text)

        for field in ("method_family", "product", "domain", "artifact"):
            if not _is_nonempty_string(row[field]):
                errors.append(f"{row_id_text}.{field} must be a non-empty string")

        backend = row["backend"]
        if backend not in BACKENDS:
            errors.append(
                f"{row_id_text}.backend must be one of {sorted(BACKENDS)}, got {backend!r}"
            )
        status = row["status"]
        if status not in STATUSES:
            errors.append(
                f"{row_id_text}.status must be one of {sorted(STATUSES)}, got {status!r}"
            )

        key_fields = ("method_family", "product", "backend", "domain")
        if all(_is_nonempty_string(row[field]) for field in key_fields):
            key = tuple(typing.cast("str", row[field]) for field in key_fields)
            if key in seen_keys:
                errors.append(
                    f"duplicate method/product/backend/domain row: {' / '.join(key)}"
                )
            seen_keys.add(key)

        for field in PATH_FIELDS:
            value = row[field]
            if not _is_nonempty_string(value):
                if status == "production":
                    errors.append(
                        f"{row_id_text}.{field} must be set for a production row"
                    )
                continue
            path = root / typing.cast("str", value)
            if full_checkout and not path.is_file():
                errors.append(f"{row_id_text}.{field} path does not exist: {value}")

        evidence = row["evidence"]
        if not isinstance(evidence, list) or not evidence:
            if status == "production":
                errors.append(
                    f"{row_id_text}.evidence must be a non-empty list for a production row"
                )
        else:
            for evidence_index, value in enumerate(evidence):
                if not _is_nonempty_string(value):
                    errors.append(
                        f"{row_id_text}.evidence[{evidence_index}] must be a non-empty path"
                    )
                    continue
                if full_checkout and not (root / typing.cast("str", value)).is_file():
                    errors.append(
                        f"{row_id_text}.evidence[{evidence_index}] path does not exist: {value}"
                    )

        blocker = row["blocker"]
        if status == "unsupported":
            if not _is_nonempty_string(blocker):
                errors.append(
                    f"{row_id_text}.blocker must explain why an unsupported row is blocked"
                )
        elif blocker not in (None, ""):
            errors.append(
                f"{row_id_text}.blocker must be empty unless status is 'unsupported'"
            )

    return errors


def load_and_validate(
    path: Path = DEFAULT_LEDGER,
    *,
    root: Path = ROOT,
) -> tuple[dict[str, object], list[str]]:
    """Load a ledger and return its payload plus deterministic validation errors."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"cannot load production-path ledger {path}: {exc}"]
    if not isinstance(payload, dict):
        return {}, ["ledger must be a JSON object"]
    return payload, validate_production_path_ledger(payload, root=root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ledger",
        nargs="?",
        type=Path,
        default=DEFAULT_LEDGER,
        help="ledger JSON to validate",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args()

    payload, errors = load_and_validate(args.ledger)
    if args.json:
        print(
            json.dumps(
                {
                    "schema": payload.get("schema"),
                    "coverage": payload.get("coverage"),
                    "rows": len(payload.get("rows", []))
                    if isinstance(payload.get("rows"), list)
                    else 0,
                    "errors": errors,
                },
                indent=2,
            )
        )
    else:
        for error in errors:
            print(error, file=sys.stderr)
        if not errors:
            rows = payload.get("rows", [])
            print(f"Validated {len(rows)} production-path ledger rows")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

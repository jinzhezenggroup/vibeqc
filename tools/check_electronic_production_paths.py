"""Validate the machine-readable electronic-structure production-path ledger."""

from __future__ import annotations

import argparse
import json
import sys
import typing
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs" / "electronic_structure_production_paths.json"
SCHEMA = "vibeqc.electronic-production-paths.v2"
STATUSES = {"production", "qualification-only", "reference", "unsupported"}
BACKENDS = {"cpu", "cuda"}
EVIDENCE_LEVELS = (
    "represented",
    "compiled-cpu",
    "compiled-cuda",
    "device-executed",
    "domain-qualified",
    "molecular",
    "derivative",
    "public",
)
EVIDENCE_STATES = {"present", "missing", "failed", "skipped", "not-applicable"}

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
    "evidence_levels",
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


def _path_diagnostic(value: str, *, root: Path, check_exists: bool) -> str | None:
    """Require portable checkout-local anchors, including resolved symlink targets."""
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or PureWindowsPath(value).drive
        or "\\" in value
        or "\x00" in value
        or ".." in relative.parts
        or not relative.parts
    ):
        return f"path must be repository-relative without traversal: {value}"
    try:
        checkout = root.resolve()
        path = (checkout / value).resolve()
        if not path.is_relative_to(checkout):
            return f"path resolves outside repository: {value}"
        if check_exists and not path.is_file():
            return f"path does not exist: {value}"
    except (OSError, RuntimeError, ValueError):
        return f"path cannot be resolved within repository: {value}"
    return None


def _validate_evidence_levels(
    value: object,
    *,
    row_id: str,
    root: Path,
    check_exists: bool,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{row_id}.evidence_levels must be an object"]

    levels = typing.cast("dict[str, object]", value)
    expected = set(EVIDENCE_LEVELS)
    missing = sorted(expected - levels.keys())
    extra = sorted(levels.keys() - expected)
    if missing:
        errors.append(f"{row_id}.evidence_levels missing levels: {', '.join(missing)}")
    if extra:
        errors.append(
            f"{row_id}.evidence_levels has unknown levels: {', '.join(extra)}"
        )

    for level in EVIDENCE_LEVELS:
        raw = levels.get(level)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            errors.append(f"{row_id}.evidence_levels.{level} must be an object")
            continue
        item = typing.cast("dict[str, object]", raw)
        required = {"state", "evidence", "reason"}
        missing_fields = sorted(required - item.keys())
        extra_fields = sorted(item.keys() - required)
        if missing_fields:
            errors.append(
                f"{row_id}.evidence_levels.{level} missing fields: "
                + ", ".join(missing_fields)
            )
            continue
        if extra_fields:
            errors.append(
                f"{row_id}.evidence_levels.{level} has unknown fields: "
                + ", ".join(extra_fields)
            )

        state = item["state"]
        evidence = item["evidence"]
        reason = item["reason"]
        if not isinstance(state, str) or state not in EVIDENCE_STATES:
            errors.append(
                f"{row_id}.evidence_levels.{level}.state must be one of "
                f"{sorted(EVIDENCE_STATES)}, got {state!r}"
            )
            continue
        if not isinstance(evidence, list):
            errors.append(f"{row_id}.evidence_levels.{level}.evidence must be a list")
            continue

        if state in {"present", "failed"} and not evidence:
            errors.append(
                f"{row_id}.evidence_levels.{level}.evidence must be non-empty "
                f"when state is {state!r}"
            )
        if state in {"missing", "skipped", "not-applicable"} and evidence:
            errors.append(
                f"{row_id}.evidence_levels.{level}.evidence must be empty "
                f"when state is {state!r}"
            )

        if state == "present":
            if reason not in (None, ""):
                errors.append(
                    f"{row_id}.evidence_levels.{level}.reason must be empty "
                    "when evidence is present"
                )
        elif not _is_nonempty_string(reason):
            errors.append(
                f"{row_id}.evidence_levels.{level}.reason must explain state {state!r}"
            )

        for evidence_index, evidence_path in enumerate(evidence):
            if not _is_nonempty_string(evidence_path):
                errors.append(
                    f"{row_id}.evidence_levels.{level}.evidence[{evidence_index}] "
                    "must be a non-empty path"
                )
                continue
            diagnostic = _path_diagnostic(
                typing.cast("str", evidence_path),
                root=root,
                check_exists=check_exists,
            )
            if diagnostic is not None:
                errors.append(
                    f"{row_id}.evidence_levels.{level}.evidence[{evidence_index}] "
                    f"{diagnostic}"
                )
    return errors


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
    if not isinstance(payload.get("coverage"), str) or payload["coverage"] not in {
        "pilot",
        "complete",
    }:
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
        if not isinstance(backend, str) or backend not in BACKENDS:
            errors.append(
                f"{row_id_text}.backend must be one of {sorted(BACKENDS)}, got {backend!r}"
            )
        status = row["status"]
        if not isinstance(status, str) or status not in STATUSES:
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
            diagnostic = _path_diagnostic(
                typing.cast("str", value), root=root, check_exists=full_checkout
            )
            if diagnostic is not None:
                errors.append(f"{row_id_text}.{field} {diagnostic}")

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
                diagnostic = _path_diagnostic(
                    typing.cast("str", value), root=root, check_exists=full_checkout
                )
                if diagnostic is not None:
                    errors.append(
                        f"{row_id_text}.evidence[{evidence_index}] {diagnostic}"
                    )

        errors.extend(
            _validate_evidence_levels(
                row["evidence_levels"],
                row_id=row_id_text,
                root=root,
                check_exists=full_checkout,
            )
        )

        if status == "production" and isinstance(backend, str) and backend in BACKENDS:
            levels = row["evidence_levels"]
            if isinstance(levels, dict):
                required_level = f"compiled-{backend}"
                compiled = levels.get(required_level)
                if not isinstance(compiled, dict) or compiled.get("state") != "present":
                    errors.append(
                        f"{row_id_text}.evidence_levels.{required_level} "
                        "must be present for a production row"
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
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
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

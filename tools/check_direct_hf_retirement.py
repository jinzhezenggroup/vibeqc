"""Validate the issue-356 Direct-HF scientific CUDA retirement ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.report_cuda_ownership import load_ledger

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs/cuda_ownership/direct_hf_retirement.json"
OWNERSHIP = ROOT / "docs/cuda_ownership"
ALLOWED_STATUS = {
    "generated-default",
    "oracle",
    "performance-exception",
    "scientific-policy",
    "unsupported-fallback",
}
REQUIRED_FIELDS = (
    "capability",
    "current_selector",
    "generated_alternative",
    "schedule_requirements",
    "resource_profile",
    "evidence",
    "required_compiler_capabilities",
    "retirement_condition",
    "files",
)


def _direct_scientific_files(ownership: dict[str, Any]) -> set[str]:
    return {
        row["path"]
        for row in ownership["files"]
        if row["path"].startswith("src/scf/cuda/direct_") and row["role"] != "runtime"
    }


def validate_retirement_ledger(
    root: Path,
    retirement: dict[str, Any],
    ownership: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed on stale, duplicate, or newly unclassified Direct science."""
    if retirement.get("schema") != "vibeqc.direct-hf-retirement.v1":
        raise ValueError("unsupported Direct-HF retirement ledger schema")
    if retirement.get("issue") != 356:
        raise ValueError("Direct-HF retirement ledger must be bound to issue #356")
    if not retirement.get("scope"):
        raise ValueError("Direct-HF retirement ledger lacks scope")

    families = retirement.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("Direct-HF retirement ledger has no families")

    semantic_rows = {row["path"]: row for row in ownership["files"]}
    expected = _direct_scientific_files(ownership)
    seen_ids: set[str] = set()
    seen_files: dict[str, str] = {}
    statuses: dict[str, int] = {}

    for family in families:
        family_id = family.get("id")
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("Direct-HF retirement family lacks id")
        if family_id in seen_ids:
            raise ValueError(f"duplicate Direct-HF retirement family id: {family_id}")
        seen_ids.add(family_id)

        status = family.get("status")
        if status not in ALLOWED_STATUS:
            raise ValueError(f"family {family_id} has unsupported status: {status!r}")
        statuses[status] = statuses.get(status, 0) + 1

        for field in REQUIRED_FIELDS:
            value = family.get(field)
            if not value:
                raise ValueError(f"family {family_id} lacks {field}")

        evidence = family["evidence"]
        if not isinstance(evidence, list) or not all(
            isinstance(path, str) and (root / path).is_file() for path in evidence
        ):
            raise ValueError(f"family {family_id} has stale evidence paths")
        capabilities = family["required_compiler_capabilities"]
        if not isinstance(capabilities, list) or not all(
            isinstance(value, str) and value.strip() for value in capabilities
        ):
            raise ValueError(f"family {family_id} has invalid compiler capabilities")

        files = family["files"]
        if not isinstance(files, list) or not all(
            isinstance(path, str) for path in files
        ):
            raise ValueError(f"family {family_id} has invalid files")
        for path in files:
            previous = seen_files.get(path)
            if previous is not None:
                raise ValueError(
                    f"Direct-HF retirement file {path} appears in both {previous} and {family_id}"
                )
            row = semantic_rows.get(path)
            if row is None or not (root / path).is_file():
                raise ValueError(f"family {family_id} has stale ownership file: {path}")
            if row["role"] == "runtime":
                raise ValueError(
                    f"family {family_id} must not classify runtime-only ownership: {path}"
                )
            if status == "oracle" and row["role"] != "oracle":
                raise ValueError(
                    f"family {family_id} claims oracle status without semantic oracle ownership: {path}"
                )
            seen_files[path] = family_id

    actual = set(seen_files)
    if actual != expected:
        raise ValueError(
            "Direct-HF retirement coverage differs: "
            f"unclassified={sorted(expected - actual)}; stale={sorted(actual - expected)}"
        )

    return {
        "schema": retirement["schema"],
        "issue": retirement["issue"],
        "families": len(families),
        "scientific_files": len(actual),
        "statuses": dict(sorted(statuses.items())),
    }


def check(root: Path = ROOT, ledger_path: Path = DEFAULT_LEDGER) -> dict[str, Any]:
    retirement = json.loads(ledger_path.read_text())
    ownership = load_ledger(root / "docs/cuda_ownership")
    return validate_retirement_ledger(root, retirement, ownership)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check exact coverage of the issue-356 Direct-HF retirement ledger."
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    summary = check(ROOT, args.ledger)
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "Direct-HF retirement ledger: "
            f"{summary['families']} families, {summary['scientific_files']} scientific files"
        )


if __name__ == "__main__":
    main()

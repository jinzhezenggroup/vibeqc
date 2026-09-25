"""Run sharded production-domain campaigns across imported Libxc registrations.

Examples::

    python tools/qualify_libxc_production_catalog.py \
        --output .artifacts/libxc-domain/all \
        --evidence-prefix artifact://libxc-domain/all

    python tools/qualify_libxc_production_catalog.py \
        --output .artifacts/libxc-domain/shard-1 \
        --evidence-prefix artifact://libxc-domain/shard-1 \
        --shard-count 4 --shard-index 1

The tool writes one campaign JSON per attempted eligible functional plus a compact
summary. Structural blockers are summarized without fabricating campaign receipts.
No result from this tool grants capability until its exact stage evidence is
consumed by the normal capability registry.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from vibeqc_compiler.common.provenance import atomic_json, canonical_hash
from vibeqc_compiler.xc.bulk_runtime import PRODUCTION_CANDIDATE_DOMAIN
from vibeqc_compiler.xc.libxc_bulk_capabilities import (
    BulkFunctionalCapability,
    available_capabilities,
)

from tools.qualify_libxc_production_domain import qualify_functional

CATALOG_CAMPAIGN_SCHEMA = "vibeqc.libxc-production-domain-catalog-campaign/v1"
CATALOG_ROW_STATUSES = (
    "pass",
    "fail",
    "not-run",
    "structural-blocked",
    "runner-error",
)


def select_capabilities(
    *,
    names: list[str] | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    maximum: int | None = None,
) -> tuple[BulkFunctionalCapability, ...]:
    """Return a deterministic shard of the imported capability inventory."""
    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("shard_count must be a positive integer")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    if maximum is not None and (type(maximum) is not int or maximum <= 0):
        raise ValueError("maximum must be a positive integer")

    inventory = sorted(available_capabilities(), key=lambda item: item.name)
    if names:
        requested = {name.upper() for name in names}
        by_name = {item.name: item for item in inventory}
        unknown = sorted(requested - set(by_name))
        if unknown:
            raise ValueError(f"unknown imported Libxc registrations: {unknown!r}")
        inventory = [item for item in inventory if item.name in requested]

    selected = tuple(
        capability
        for index, capability in enumerate(inventory)
        if index % shard_count == shard_index
    )
    return selected if maximum is None else selected[:maximum]


def _catalog_identity(
    capabilities: tuple[BulkFunctionalCapability, ...],
    *,
    shard_count: int,
    shard_index: int,
    rtol: float,
    atol: float,
) -> str:
    return canonical_hash(
        {
            "schema": CATALOG_CAMPAIGN_SCHEMA,
            "candidate_domain": PRODUCTION_CANDIDATE_DOMAIN,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "rtol": rtol,
            "atol": atol,
            "functionals": [
                {
                    "name": item.name,
                    "capability_identity": item.identity,
                    "profile_identity": item.production_domain_profile.identity,
                }
                for item in capabilities
            ],
        }
    )


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    catalog_identity: str,
    shard_count: int,
    shard_index: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Build a deterministic compact shard summary from campaign result rows."""
    status_counts = Counter(row["status"] for row in rows)
    family_counts: dict[str, Counter[str]] = {}
    blocked_case_counts: Counter[str] = Counter()
    for row in rows:
        family_counts.setdefault(row["family"], Counter())[row["status"]] += 1
        for case_id in row.get("blocked_case_ids", ()):
            blocked_case_counts[case_id] += 1

    return {
        "schema": CATALOG_CAMPAIGN_SCHEMA,
        "identity": catalog_identity,
        "candidate_domain": PRODUCTION_CANDIDATE_DOMAIN,
        "shard": {"count": shard_count, "index": shard_index},
        "tolerance": {"rtol": rtol, "atol": atol},
        "selected_functionals": len(rows),
        "status_counts": {
            status: status_counts.get(status, 0) for status in CATALOG_ROW_STATUSES
        },
        "by_family": {
            family: {
                status: counts.get(status, 0) for status in CATALOG_ROW_STATUSES
            }
            for family, counts in sorted(family_counts.items())
        },
        "blocked_case_counts": dict(sorted(blocked_case_counts.items())),
        "functionals": {row["name"]: row for row in rows},
    }


def run_catalog(
    capabilities: tuple[BulkFunctionalCapability, ...],
    *,
    output: Path,
    evidence_prefix: str,
    pyscf_version: str,
    libxc: Any,
    shard_count: int,
    shard_index: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Run one deterministic shard and retain every eligible campaign result."""
    if not isinstance(evidence_prefix, str) or not evidence_prefix.strip():
        raise ValueError("catalog campaign requires a nonempty evidence prefix")
    prefix = evidence_prefix.rstrip("/")
    output.mkdir(parents=True, exist_ok=True)
    campaigns = output / "campaigns"
    campaigns.mkdir(parents=True, exist_ok=True)

    catalog_identity = _catalog_identity(
        capabilities,
        shard_count=shard_count,
        shard_index=shard_index,
        rtol=rtol,
        atol=atol,
    )
    rows: list[dict[str, Any]] = []
    for capability in capabilities:
        profile = capability.production_domain_profile
        row: dict[str, Any] = {
            "name": capability.name,
            "family": capability.family,
            "required_ingredients": list(capability.required_ingredients),
            "capability_identity": capability.identity,
            "profile_identity": profile.identity,
            "status": None,
            "blocker": None,
            "campaign": None,
            "blocked_case_ids": [],
        }
        if not profile.eligible:
            row["status"] = "structural-blocked"
            row["blocker"] = profile.blocker
            rows.append(row)
            continue

        campaign_path = campaigns / f"{capability.name}.json"
        evidence = f"{prefix}/campaigns/{capability.name}.json"
        try:
            payload = qualify_functional(
                capability.name,
                evidence=evidence,
                pyscf_version=pyscf_version,
                libxc=libxc,
                rtol=rtol,
                atol=atol,
            )
        except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
            row["status"] = "runner-error"
            row["blocker"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue

        atomic_json(campaign_path, payload)
        envelope = payload["stage_evidence"]
        row["status"] = envelope["status"]
        row["blocker"] = envelope["reason"]
        row["campaign"] = str(campaign_path.relative_to(output))
        row["campaign_identity"] = payload["receipt"]["identity"]
        row["execution_identity"] = payload["execution"]["identity"]
        row["blocked_case_ids"] = [
            case["case_id"]
            for case in payload["receipt"]["cases"]
            if case["status"] != "pass"
        ]
        rows.append(row)

    summary = summarize_rows(
        rows,
        catalog_identity=catalog_identity,
        shard_count=shard_count,
        shard_index=shard_index,
        rtol=rtol,
        atol=atol,
    )
    atomic_json(output / "summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-prefix", required=True)
    parser.add_argument("--name", action="append")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-functionals", type=int)
    parser.add_argument("--rtol", type=float, default=2.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument(
        "--require-all-pass",
        action="store_true",
        help="return nonzero unless every selected functional passes",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.rtol < 0.0 or args.atol < 0.0:
        raise ValueError("catalog campaign tolerances must be nonnegative")

    import pyscf
    from pyscf.dft import libxc

    capabilities = select_capabilities(
        names=args.name,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        maximum=args.max_functionals,
    )
    summary = run_catalog(
        capabilities,
        output=args.output,
        evidence_prefix=args.evidence_prefix,
        pyscf_version=pyscf.__version__,
        libxc=libxc,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        rtol=args.rtol,
        atol=args.atol,
    )
    counts = summary["status_counts"]
    print(
        f"{summary['selected_functionals']} selected; "
        + " ".join(f"{status}={counts[status]}" for status in CATALOG_ROW_STATUSES)
    )
    failures = summary["selected_functionals"] - counts["pass"]
    return 1 if args.require_all_pass and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

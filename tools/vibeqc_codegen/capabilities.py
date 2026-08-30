"""Capability metadata and structural coverage reports for shell codegen.

The mathematical shell catalog is intentionally broader than the measured
production profile.  This module keeps that distinction explicit: structural
capabilities describe what the compiler can emit, while manifest capabilities
describe which optional production wrappers were independently accepted for a
specific architecture and schedule.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .cuda_schedule import schedule_candidates
from .cuda_target import DEFAULT_CUDA_TARGET, CudaTargetInfo, cuda_target_info
from .fused_schedule import build_fused_shell_plan
from .ir import FOUR_CENTER_ERI_OPERATOR, KernelConsumer, build_integral_ir
from .shell_spec import FUSED_SHELL_SPECS, ShellClassSpec

CAPABILITY_STREAMING_FOCK = "streaming_fock"
CAPABILITY_MIXED_FOCK = "mixed_fock"
CAPABILITY_LOCAL_PACKED_STREAMING_FOCK = "local_packed_streaming_fock"

KNOWN_CAPABILITIES = frozenset(
    (
        CAPABILITY_STREAMING_FOCK,
        CAPABILITY_MIXED_FOCK,
        CAPABILITY_LOCAL_PACKED_STREAMING_FOCK,
    )
)


def normalize_capabilities(
    name: str, raw: object | None
) -> frozenset[str]:
    """Validate one manifest capability list and return a stable set.

    Capabilities are opt-in.  An omitted field therefore preserves the safe
    generic path for legacy/custom manifests instead of guessing that a
    schedule has passed an endpoint performance gate.
    """

    if raw is None:
        return frozenset()
    if not isinstance(raw, list):
        raise TypeError(f"{name} capabilities must be a list of strings")
    values: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or item not in KNOWN_CAPABILITIES:
            supported = ", ".join(sorted(KNOWN_CAPABILITIES))
            raise ValueError(
                f"{name} has unsupported capability {item!r}; "
                f"expected one of {supported}"
            )
        values.add(item)
    return frozenset(values)


@dataclass(frozen=True, slots=True)
class CapabilityCheck:
    """One structural backend or recurrence capability result."""

    supported: bool
    schedules: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Serialize the check for stable JSON reports."""

        return {
            "supported": self.supported,
            "schedules": list(self.schedules),
            "reasons": list(self.reasons),
        }


def _production_gap_payload() -> dict[str, object]:
    """Describe why an unselected class cannot enter production automatically.

    Structural source emission is deliberately weaker than a production
    promotion.  Keeping this policy in the report makes the distinction
    machine-readable instead of requiring callers to infer it from an empty
    manifest row.
    """

    return {
        "profile": None,
        "profile_match": None,
        "force": False,
        "fock": False,
        "capabilities": [],
        "status": "manifest_gap",
        "promotion_gate": "real_molecular_endpoint_and_resource_gates",
        "reason": (
            "not selected by the production manifest; endpoint/resource "
            "gates remain"
        ),
    }


@dataclass(frozen=True, slots=True)
class ShellCapabilityReport:
    """Complete structural and production coverage for one shell class."""

    spec: ShellClassSpec
    generic_fused: CapabilityCheck
    recurrences: tuple[tuple[str, CapabilityCheck], ...]
    force_derivative_orders: tuple[tuple[int, CapabilityCheck], ...]
    production: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-compatible report row."""

        return {
            "shell_class": self.spec.name,
            "angular": list(self.spec.angular),
            "component_count": self.spec.component_count,
            "pair_orders": list(self.spec.pair_orders),
            "generic_fused": self.generic_fused.to_payload(),
            "recurrences": {
                name: check.to_payload() for name, check in self.recurrences
            },
            "force_derivative_orders": {
                str(order): check.to_payload()
                for order, check in self.force_derivative_orders
            },
            "production": dict(self.production),
        }


def _check_recurrence(
    spec: ShellClassSpec,
    recurrence: str,
    target: CudaTargetInfo,
) -> CapabilityCheck:
    """Check IR legality and every target-legal emitter schedule."""

    try:
        integral = build_integral_ir(
            spec,
            consumers=(KernelConsumer.FORCE,),
            recurrence=recurrence,
        )
    except (TypeError, ValueError) as error:
        return CapabilityCheck(False, reasons=(str(error),))

    schedules: list[str] = []
    failures: dict[str, str] = {}
    try:
        candidates = schedule_candidates(integral, target)
    except (TypeError, ValueError) as error:
        return CapabilityCheck(False, reasons=(str(error),))
    for schedule in candidates:
        kind = schedule.kind.value
        try:
            # Import lazily to keep this reporting module usable by the CUDA
            # lowering itself without creating a cuda_emitter cycle.
            from .cuda_emitter import emit_shell_class_fused_cuda

            plan = build_fused_shell_plan(
                spec,
                integral=integral,
                schedule=schedule,
                target=target,
            )
            # Source emission is part of the report deliberately: a schedule
            # can be legal in ScheduleIR yet unavailable in the CUDA backend.
            emit_shell_class_fused_cuda(spec, plan)
        except (TypeError, ValueError, RuntimeError) as error:
            failures.setdefault(kind, str(error))
        else:
            if kind not in schedules:
                schedules.append(kind)
    reasons = tuple(
        f"{kind}: {reason}" for kind, reason in sorted(failures.items())
    )
    if not schedules and not reasons:
        reasons = ("no target-legal schedule is available",)
    return CapabilityCheck(bool(schedules), tuple(schedules), reasons)


def _check_force_derivative_order(
    spec: ShellClassSpec,
    order: int,
    target: CudaTargetInfo,
) -> CapabilityCheck:
    """Report one force derivative order without widening the CUDA ABI.

    The current force result ABI and all shell emitters expose first nuclear
    derivatives only.  Validate the mathematical IR first so the report
    preserves a useful distinction when a future IR accepts higher orders but
    the backend still has no Hessian/result representation.
    """

    derivative = FOUR_CENTER_ERI_OPERATOR.nuclear_derivative(order=order)
    try:
        build_integral_ir(
            spec,
            consumers=(KernelConsumer.FORCE,),
            derivative=derivative,
            recurrence="subset_wick",
        )
    except (TypeError, ValueError) as error:
        return CapabilityCheck(False, reasons=(str(error),))
    if order != 1:
        return CapabilityCheck(
            False,
            reasons=(
                (
                    "CUDA force result ABI currently exposes only order-one "
                    "derivatives"
                ),
            ),
        )
    return _check_recurrence(spec, "subset_wick", target)


def _production_index(
    manifest: Path | None,
    architecture: str,
    profile: str,
) -> dict[str, dict[str, object]]:
    """Load optional manifest state without coupling the structural module."""

    if manifest is None:
        return {}
    # Import lazily: production.py consumes capability normalization, so an
    # eager import here would create a module cycle during normal generation.
    from .production import resolve_production_profile

    resolved = resolve_production_profile(manifest, architecture, profile)
    result: dict[str, dict[str, object]] = {}
    for selection in resolved.selections:
        result[selection.spec.name] = {
            "profile": resolved.profile,
            "profile_match": resolved.match.value,
            "force": KernelConsumer.FORCE in selection.consumers,
            "fock": KernelConsumer.FOCK in selection.consumers,
            "capabilities": sorted(selection.capabilities),
            "recurrence": selection.recurrence,
            "schedule": selection.schedule.kind.value,
            "status": "manifest_selected",
            "promotion_gate": "real_molecular_endpoint_and_resource_gates",
        }
    return result


def build_capability_report(
    *,
    target: CudaTargetInfo = DEFAULT_CUDA_TARGET,
    manifest: Path | None = None,
    architecture: str | None = None,
    profile: str = "auto",
    specifications: Iterable[ShellClassSpec] = FUSED_SHELL_SPECS,
) -> dict[str, object]:
    """Build a deterministic report for every requested shell specification."""

    selected_architecture = architecture or target.architecture
    if architecture is not None:
        target = cuda_target_info(architecture)
    production = _production_index(manifest, selected_architecture, profile)
    rows = []
    for spec in specifications:
        recurrence_rows = tuple(
            (name, _check_recurrence(spec, name, target))
            for name in ("subset_wick", "rys2", "rys3", "rys4", "rys5")
        )
        derivative_rows = tuple(
            (order, _check_force_derivative_order(spec, order, target))
            for order in (1, 2)
        )
        generic = dict(recurrence_rows)["subset_wick"]
        production_row = production.get(
            spec.name,
            _production_gap_payload(),
        )
        rows.append(
            ShellCapabilityReport(
                spec=spec,
                generic_fused=generic,
                recurrences=recurrence_rows,
                force_derivative_orders=derivative_rows,
                production=production_row,
            ).to_payload()
        )
    manifest_label = None
    if manifest is not None:
        try:
            manifest_label = manifest.resolve().relative_to(
                Path(__file__).resolve().parents[2]
            ).as_posix()
        except ValueError:
            manifest_label = str(manifest)
    recurrence_supported = {
        name: sum(
            bool(row["recurrences"][name]["supported"])
            for row in rows
        )
        for name in ("subset_wick", "rys2", "rys3", "rys4", "rys5")
    }
    force_derivative_supported = {
        str(order): sum(
            bool(row["force_derivative_orders"][str(order)]["supported"])
            for row in rows
        )
        for order in (1, 2)
    }
    return {
        "schema_version": 1,
        "backend": {
            "name": "cuda",
            "architecture": target.architecture,
            "compute_capability": (
                f"{target.compute_capability_major}.{target.compute_capability_minor}"
            ),
            "generator_abi": target.generator_abi,
            "schedule_source": "schedule_candidates",
            "emitter_validation": "emit_shell_class_fused_cuda",
        },
        "architecture": target.architecture,
        "compute_capability": (
            f"{target.compute_capability_major}.{target.compute_capability_minor}"
        ),
        "manifest": manifest_label,
        "profile": profile,
        "total_shell_classes": len(rows),
        "generic_fused_supported": sum(
            bool(row["generic_fused"]["supported"]) for row in rows
        ),
        "production_selected": sum(
            bool(row["production"].get("force"))
            or bool(row["production"].get("fock"))
            for row in rows
        ),
        "recurrence_supported": recurrence_supported,
        "force_derivative_supported": force_derivative_supported,
        "shell_classes": rows,
    }


def main() -> None:
    """Emit the structural/production capability report as JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("production_shell_classes.json"),
    )
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = build_capability_report(
        architecture=arguments.architecture,
        manifest=arguments.manifest,
        profile=arguments.profile,
    )
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(output, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

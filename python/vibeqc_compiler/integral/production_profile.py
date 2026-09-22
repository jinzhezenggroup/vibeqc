"""Production manifest parsing and profile resolution for integral codegen.

This module owns production-profile schema validation and resolves accepted
kernel selections for a concrete CUDA target.  Source emission, registry
serialization, bundle writing, and compile-cost policy stay outside this owner.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, cast

from .capabilities import normalize_capabilities
from .cuda_schedule import (
    AlgebraForm,
    AlgebraFusion,
    AlgebraOrdering,
    AlgebraPlacement,
    PairOrientation,
    PairStorage,
    ScheduleIR,
    ScheduleKind,
)
from .cuda_target import CudaTargetInfo, cuda_target_info, normalize_cuda_architecture
from .fused_schedule import build_fused_shell_plan
from .ir import KernelConsumer, build_integral_ir
from .production_selection import _SUPPORTED_RECURRENCES, KernelSelection
from .shell_spec import FUSED_SHELL_SPEC_BY_NAME, ShellClassSpec

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class ProfileMatch(str, Enum):
    """How a production manifest profile was selected for a build target."""

    EXACT = "exact"
    COMPATIBLE = "compatible"
    PORTABLE = "portable"


@dataclass(frozen=True, slots=True)
class ResolvedProductionProfile:
    """Manifest resolution result for one concrete CUDA compile target."""

    target: CudaTargetInfo
    profile: str
    match: ProfileMatch
    tuned: bool
    selections: tuple[KernelSelection, ...]
    cuda_toolkit: str

    @property
    def portable(self) -> bool:
        """Return whether execution intentionally uses the generic fallback."""

        return self.match == ProfileMatch.PORTABLE


def _profile_kind(name: str, profile: dict[str, object]) -> str:
    """Return the explicit or backward-compatible profile kind."""

    configured = profile.get("kind")
    if isinstance(configured, str):
        return configured
    return "portable" if name in ("portable", "portable_cuda") else "tuned"


def _profile_compatible(
    name: str,
    profile: dict[str, object],
    architecture: str,
) -> bool:
    """Return whether a non-portable profile explicitly accepts a target."""

    if name == architecture:
        return True
    compatible = profile.get("compatible_architectures", ())
    if isinstance(compatible, list) and architecture in compatible:
        return True
    capabilities = profile.get("compatible_compute_capabilities", ())
    if not isinstance(capabilities, list):
        return False
    major, minor = cuda_target_info(architecture).compute_capability
    accepted = {
        architecture,
        f"{major}.{minor}",
        f"{major}.x",
        f"sm_{major}x",
    }
    return any(item in accepted for item in capabilities if isinstance(item, str))


def _portable_profile(
    architectures: dict[str, object],
) -> tuple[str, dict[str, object]] | None:
    """Return the single manifest portable profile, if one is declared."""

    portable = []
    for name, raw_profile in architectures.items():
        if (
            isinstance(raw_profile, dict)
            and _profile_kind(name, raw_profile) == "portable"
        ):
            portable.append((name, raw_profile))
    if len(portable) > 1:
        raise ValueError("production manifest declares multiple portable profiles")
    return portable[0] if portable else None


def _resolve_profile_payload(
    payload: dict[str, object],
    architecture: str,
    requested_profile: str,
) -> tuple[str, dict[str, object], ProfileMatch]:
    """Resolve explicit portable requests or fail-closed tuned profiles."""

    architectures = payload.get("architectures")
    if not isinstance(architectures, dict):
        raise TypeError("v2 production manifest requires architectures")

    if requested_profile in ("portable", "portable_cuda"):
        portable = _portable_profile(architectures)
        if portable is None:
            return (
                "portable_cuda",
                {"kind": "portable", "kernels": []},
                ProfileMatch.PORTABLE,
            )
        return portable[0], portable[1], ProfileMatch.PORTABLE

    if requested_profile != "auto":
        raw_profile = architectures.get(requested_profile)
        if not isinstance(raw_profile, dict):
            raise ValueError(
                f"production manifest has no profile {requested_profile!r}"
            )
        kind = _profile_kind(requested_profile, raw_profile)
        if kind == "portable":
            return requested_profile, raw_profile, ProfileMatch.PORTABLE
        if not _profile_compatible(requested_profile, raw_profile, architecture):
            raise ValueError(
                f"profile {requested_profile!r} is incompatible with {architecture}"
            )
        match = (
            ProfileMatch.EXACT
            if requested_profile == architecture
            else ProfileMatch.COMPATIBLE
        )
        return requested_profile, raw_profile, match

    exact = architectures.get(architecture)
    if isinstance(exact, dict) and _profile_kind(architecture, exact) != "portable":
        return architecture, exact, ProfileMatch.EXACT
    compatible = [
        (name, raw_profile)
        for name, raw_profile in architectures.items()
        if isinstance(raw_profile, dict)
        and _profile_kind(name, raw_profile) != "portable"
        and _profile_compatible(name, raw_profile, architecture)
    ]
    if len(compatible) > 1:
        names = ", ".join(name for name, _ in compatible)
        raise ValueError(
            f"multiple production profiles are compatible with {architecture}: {names}"
        )
    if compatible:
        return compatible[0][0], compatible[0][1], ProfileMatch.COMPATIBLE
    raise ValueError(
        f"no tuned or compatible production profile for {architecture}; "
        "request 'portable_cuda' explicitly to use the generic CUDA path"
    )


def _validate_measured_target(
    profile_name: str,
    profile: dict[str, object],
    target: CudaTargetInfo,
    match: ProfileMatch,
) -> None:
    """Reject stale exact-profile capability or generator-ABI metadata."""

    generator_abi = profile.get("generator_abi", target.generator_abi)
    if int(cast("str | int | float", generator_abi)) != target.generator_abi:
        raise ValueError(
            f"profile {profile_name!r} uses generator ABI {generator_abi}, "
            f"expected {target.generator_abi}"
        )
    if match != ProfileMatch.EXACT:
        return
    measured = profile.get("target")
    if measured is None:
        return
    if not isinstance(measured, dict):
        raise TypeError("profile target metadata must be a JSON object")
    expected: dict[str, object] = {
        "compute_capability": (
            f"{target.compute_capability_major}.{target.compute_capability_minor}"
        ),
        "warp_size": target.warp_size,
        "maximum_threads_per_block": target.maximum_threads_per_block,
        "maximum_threads_per_sm": target.maximum_threads_per_sm,
        "maximum_blocks_per_sm": target.maximum_blocks_per_sm,
        "registers_per_sm": target.registers_per_sm,
        "maximum_registers_per_thread": target.maximum_registers_per_thread,
        "shared_memory_per_block": target.shared_memory_per_block,
        "shared_memory_per_block_optin": target.shared_memory_per_block_optin,
        "shared_memory_per_sm": target.shared_memory_per_sm,
    }
    for key, value in expected.items():
        if key in measured and measured[key] != value:
            raise ValueError(
                f"profile {profile_name!r} target field {key} does not "
                f"match catalog value {value!r}"
            )


def _schedule_from_payload(payload: object) -> ScheduleIR:
    """Validate one explicit architecture-tuned schedule record."""

    if not isinstance(payload, dict):
        raise TypeError("kernel schedule must be a JSON object")
    try:
        return ScheduleIR(
            kind=ScheduleKind(payload["kind"]),
            block_threads=int(payload["block_threads"]),
            component_tile=int(payload["component_tile"]),
            tasks_per_warp=int(payload.get("tasks_per_warp", 1)),
            shared_coulomb=bool(payload.get("shared_coulomb", True)),
            pair_orientation=PairOrientation(
                payload.get("pair_orientation", PairOrientation.CANONICAL.value)
            ),
            pair_storage=PairStorage(
                payload.get("pair_storage", PairStorage.MATERIALIZED.value)
            ),
            algebra_placement=AlgebraPlacement(
                payload.get(
                    "algebra_placement",
                    AlgebraPlacement.MATERIALIZED_CSE.value,
                )
            ),
            algebra_ordering=AlgebraOrdering(
                payload.get(
                    "algebra_ordering",
                    AlgebraOrdering.TOPOLOGICAL.value,
                )
            ),
            algebra_fusion=AlgebraFusion(
                payload.get(
                    "algebra_fusion",
                    AlgebraFusion.SEPARATE.value,
                )
            ),
            algebra_form=AlgebraForm(
                payload.get(
                    "algebra_form",
                    AlgebraForm.BINARY.value,
                )
            ),
            unroll_pair_terms=bool(payload.get("unroll_pair_terms", True)),
            minimum_blocks_per_sm=int(payload.get("minimum_blocks_per_sm", 0)),
            maximum_registers=int(payload.get("maximum_registers", 0)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid production kernel schedule") from error


def _default_architecture(payload: dict[str, object]) -> str:
    """Resolve an unambiguous default architecture from a v2 manifest."""

    configured = payload.get("default_architecture")
    if isinstance(configured, str):
        return configured
    architectures = payload.get("architectures")
    if isinstance(architectures, dict) and len(architectures) == 1:
        return next(iter(architectures))
    raise ValueError(
        "multi-architecture production manifest requires default_architecture"
    )


def _recurrence_from_row(
    spec: ShellClassSpec,
    row: Mapping[str, object],
    consumers: tuple[KernelConsumer, ...],
) -> str:
    """Validate and normalize a manifest row's integral recurrence.

    Recurrence is scientific lowering intent rather than a CUDA schedule
    property.  Keeping this check at the manifest boundary makes an invalid
    force/Fock combination fail before any source generation is attempted.
    The default preserves the schema's historical subset/Wick lowering.
    """

    name = spec.name
    recurrence = row.get("recurrence", "subset_wick")
    if not isinstance(recurrence, str):
        # Manifest validation consistently reports malformed rows as
        # ``ValueError`` so callers can handle schema failures uniformly.
        raise ValueError(  # noqa: TRY004
            f"{name} recurrence must be a string"
        )
    if recurrence not in _SUPPORTED_RECURRENCES:
        supported = ", ".join(sorted(_SUPPORTED_RECURRENCES))
        raise ValueError(
            f"{name} has unsupported recurrence {recurrence!r}; "
            f"expected one of {supported}"
        )
    if (
        recurrence == "rys3"
        and name == "ppps"
        and KernelConsumer.FOCK in consumers
        and row.get("fock_schedule") is None
    ):
        raise ValueError(
            "ppps recurrence 'rys3' with a Fock consumer requires fock_schedule"
        )
    # Construct the mathematical IR at the manifest boundary so an incorrect
    # fixed-root count or force/Fock combination fails independently of CUDA
    # scheduling and without a shell-name eligibility table.
    build_integral_ir(spec, consumers, recurrence=recurrence)
    return recurrence


def _resident_force_recurrence_from_row(
    name: str,
    row: Mapping[str, object],
    consumers: tuple[KernelConsumer, ...],
) -> str | None:
    """Validate an optional force-only resident lowering beside a row.

    Resident ppps is an additional launch route, not a replacement for the
    ordinary force/Fock selection.  Keeping the opt-in on the same manifest
    row prevents duplicate shell-class metadata and leaves the existing
    force/Fock wrapper available as a safe fallback.
    """

    recurrence = row.get("resident_force_recurrence")
    if recurrence is None:
        return None
    if name != "ppps":
        raise ValueError(f"{name} does not support a resident force recurrence")
    if KernelConsumer.FORCE not in consumers:
        raise ValueError(
            f"{name} resident force recurrence requires the force consumer"
        )
    if not isinstance(recurrence, str):
        # Keep the public manifest-loading error type stable; this is a
        # schema/value failure even though the offending value has a bad type.
        raise ValueError(  # noqa: TRY004
            f"{name} resident_force_recurrence must be a string"
        )
    if recurrence != "rys3":
        raise ValueError(f"{name} resident force recurrence must be rys3")
    return recurrence


def _optional_nonnegative_number(
    name: str, row: Mapping[str, object], field: str
) -> float | int | None:
    """Validate optional compiler provenance attached to a manifest row."""

    value = row.get(field)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} {field} must be a finite non-negative number")
    return value


def _selections_from_rows(
    rows: object,
    *,
    architecture: str,
    profile: str,
    tuned: bool,
    target: CudaTargetInfo,
) -> tuple[KernelSelection, ...]:
    """Validate explicit manifest rows for one resolved build target."""

    if not isinstance(rows, list):
        raise TypeError("architecture profile requires a kernels list")
    selections = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("production kernel entry must be a JSON object")
        name = row.get("shell_class")
        if not isinstance(name, str) or name not in FUSED_SHELL_SPEC_BY_NAME:
            raise ValueError(f"unsupported production shell class {name!r}")
        if name in seen:
            raise ValueError(f"duplicate production shell class {name!r}")
        spec = FUSED_SHELL_SPEC_BY_NAME[name]
        raw_consumers = row.get("consumers", [KernelConsumer.FORCE.value])
        if not isinstance(raw_consumers, list) or not raw_consumers:
            raise ValueError(f"{name} requires a non-empty consumers list")
        try:
            consumers = tuple(KernelConsumer(item) for item in raw_consumers)
        except ValueError as error:
            raise ValueError(f"{name} has an unsupported consumer") from error
        recurrence = _recurrence_from_row(spec, row, consumers)
        capabilities = normalize_capabilities(
            name,
            row.get("capabilities"),
        )
        resident_force_recurrence = _resident_force_recurrence_from_row(
            name, row, consumers
        )
        schedule_payload = row.get("schedule")
        if schedule_payload is None:
            schedule = build_fused_shell_plan(
                spec,
                consumers=consumers,
                recurrence=recurrence,
                target=target,
            ).schedule
        else:
            schedule = _schedule_from_payload(schedule_payload)
            # Build the complete IR now so component coverage and block limits
            # fail during manifest loading rather than CUDA compilation.
            build_fused_shell_plan(
                spec,
                consumers=consumers,
                schedule=schedule,
                recurrence=recurrence,
                target=target,
            )
        fock_schedule_payload = row.get("fock_schedule")
        if fock_schedule_payload is None:
            if (
                recurrence == "rys2"
                and KernelConsumer.FOCK in consumers
                and schedule.kind == ScheduleKind.THREAD_TASKS
            ):
                # The accepted low-order force path uses scalar Rys2, while
                # its value consumer remains the compact subset/Wick mapping.
                # Derive that companion through the same compiler scheduler
                # instead of repeating an identical Fock table per class.
                fock_schedule = build_fused_shell_plan(
                    spec,
                    consumers=(KernelConsumer.FOCK,),
                    recurrence="subset_wick",
                    target=target,
                ).schedule
            else:
                fock_schedule = None
        else:
            if KernelConsumer.FOCK not in consumers:
                raise ValueError(f"{name} fock_schedule requires a Fock consumer")
            fock_schedule = _schedule_from_payload(fock_schedule_payload)
            # A fixed-root force promotion may use a very different execution
            # geometry. Validate the retained value path independently so the
            # manifest cannot silently retune Fock or inherit force recurrence.
            build_fused_shell_plan(
                spec,
                consumers=(KernelConsumer.FOCK,),
                schedule=fock_schedule,
                recurrence="subset_wick",
                target=target,
            )

        selections.append(
            KernelSelection(
                architecture=architecture,
                spec=spec,
                consumers=consumers,
                schedule=schedule,
                profile=profile,
                tuned=tuned,
                recurrence=recurrence,
                resident_force_recurrence=resident_force_recurrence,
                fock_schedule=fock_schedule,
                capabilities=capabilities,
                runtime_seconds=_optional_nonnegative_number(
                    name, row, "runtime_seconds"
                ),
                compile_seconds=_optional_nonnegative_number(
                    name, row, "compile_seconds"
                ),
                source_bytes=cast(
                    "int | None",
                    _optional_nonnegative_number(name, row, "source_bytes"),
                ),
                object_bytes=cast(
                    "int | None",
                    _optional_nonnegative_number(name, row, "object_bytes"),
                ),
            )
        )
        seen.add(name)
    return tuple(selections)


def resolve_production_profile(
    path: Path,
    architecture: str | None = None,
    profile: str = "auto",
) -> ResolvedProductionProfile:
    """Resolve one target; generic CUDA requires an explicit portable request."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("production shell manifest must be a JSON object")
    schema_version = payload.get("schema_version")
    default_architecture = (
        "sm_120" if schema_version == 1 else _default_architecture(payload)
    )
    selected_architecture = normalize_cuda_architecture(
        architecture or default_architecture
    )
    target = cuda_target_info(selected_architecture)
    if schema_version == 1:
        names = payload.get("shell_classes")
        if not isinstance(names, list) or not names:
            raise ValueError("production manifest requires shell_classes")
        acceptance = payload.get("acceptance")
        accepted_architecture = (
            acceptance.get("architecture") if isinstance(acceptance, dict) else "sm_120"
        )
        accepted_architecture = normalize_cuda_architecture(
            accepted_architecture
            if isinstance(accepted_architecture, str)
            else "sm_120"
        )
        if profile in ("portable", "portable_cuda"):
            return ResolvedProductionProfile(
                target=target,
                profile="portable_cuda",
                match=ProfileMatch.PORTABLE,
                tuned=False,
                selections=(),
                cuda_toolkit="",
            )
        if selected_architecture != accepted_architecture:
            raise ValueError(
                f"no tuned production profile for {selected_architecture}; "
                "request 'portable_cuda' explicitly to use the generic CUDA path"
            )
        if profile not in ("auto", accepted_architecture):
            raise ValueError(f"production manifest has no profile {profile!r}")
        rows = [
            {"shell_class": name, "consumers": [KernelConsumer.FORCE.value]}
            for name in names
        ]
        selections = _selections_from_rows(
            rows,
            architecture=selected_architecture,
            profile=accepted_architecture,
            tuned=True,
            target=target,
        )
        return ResolvedProductionProfile(
            target=target,
            profile=accepted_architecture,
            match=ProfileMatch.EXACT,
            tuned=True,
            selections=selections,
            cuda_toolkit="",
        )
    if schema_version != 2:
        raise ValueError("unsupported production shell manifest schema")

    profile_name, profile_payload, match = _resolve_profile_payload(
        payload,
        selected_architecture,
        profile,
    )
    _validate_measured_target(profile_name, profile_payload, target, match)
    tuned = (
        match == ProfileMatch.EXACT
        and _profile_kind(profile_name, profile_payload) == "tuned"
    )
    rows = profile_payload.get("kernels", [])
    selections = _selections_from_rows(
        rows,
        architecture=selected_architecture,
        profile=profile_name,
        tuned=tuned,
        target=target,
    )
    toolkit = profile_payload.get("cuda_toolkit", "")
    if not isinstance(toolkit, str):
        raise TypeError("profile cuda_toolkit must be a string")
    return ResolvedProductionProfile(
        target=target,
        profile=profile_name,
        match=match,
        tuned=tuned,
        selections=selections,
        cuda_toolkit=toolkit,
    )


def load_production_kernel_selections(
    path: Path,
    architecture: str | None = None,
    profile: str = "auto",
) -> tuple[KernelSelection, ...]:
    """Load safe production selections for one concrete CUDA target."""

    return resolve_production_profile(path, architecture, profile).selections


def load_production_manifest(
    path: Path,
    architecture: str | None = None,
    profile: str = "auto",
) -> tuple[ShellClassSpec, ...]:
    """Compatibility view returning classes with generated force consumers."""

    return tuple(
        selection.spec
        for selection in load_production_kernel_selections(path, architecture, profile)
        if KernelConsumer.FORCE in selection.consumers
    )


def load_production_fock_manifest(
    path: Path,
    architecture: str | None = None,
    profile: str = "auto",
) -> tuple[ShellClassSpec, ...]:
    """Compatibility view returning classes with generated Fock consumers."""

    return tuple(
        selection.spec
        for selection in load_production_kernel_selections(path, architecture, profile)
        if KernelConsumer.FOCK in selection.consumers
    )


def _profile_identifier(value: str) -> str:
    """Return a stable C/CMake identifier for a profile or architecture."""

    if re.fullmatch(r"sm_[0-9]+", value):
        return value.replace("_", "")
    identifier = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    if not identifier:
        raise ValueError("profile identifier cannot be empty")
    return identifier

"""Emit build-directory CUDA shards and host dispatch for accepted kernels."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .cuda_emitter import emit_shell_class_fused_cuda
from .cuda_schedule import (
    AlgebraFusion,
    AlgebraOrdering,
    AlgebraPlacement,
    PairOrientation,
    PairStorage,
    ScheduleIR,
    ScheduleKind,
)
from .cuda_target import (
    CudaTargetInfo,
    cuda_target_info,
    normalize_cuda_architecture,
)
from .dppp_dispatch import emit_ppps_resident_bra_rys3_cuda
from .fused_schedule import build_fused_shell_plan
from .ir import KernelConsumer, build_integral_ir
from .shell_spec import FUSED_SHELL_SPEC_BY_NAME, ShellClassSpec, shell_pair_class

_SUPPORTED_RECURRENCES = frozenset(
    ("subset_wick", "rys2", "rys3", "rys4", "rys5")
)
_STREAMING_FOCK_SHELLS = frozenset(
    (
        "ssss",
        "psss",
        "psps",
        "ppss",
        "ppps",
        "pppp",
        "dsss",
        "dsps",
        "dspp",
        "dsds",
        "dpss",
        "dpps",
        "dppp",
        "dpds",
        "dpdp",
        "ddss",
        "ddps",
        "ddpp",
        "ddds",
        "dddp",
        "dddd",
    )
)


def _supports_scalar_rys(
    spec: ShellClassSpec,
    schedule: ScheduleIR,
) -> bool:
    """Return whether the lane-local fixed-root backend can lower ``spec``.

    Each lane owns one complete shell task, so the schedule must expose one
    task per hardware lane and enough component storage for the full quartet.
    Root-count legality is a mathematical-IR concern and is checked separately.
    """

    return (
        schedule.kind == ScheduleKind.THREAD_TASKS
        and schedule.warp_size == 32
        and schedule.block_threads == 32
        and schedule.tasks_per_warp == 32
        and not schedule.shared_coulomb
        and schedule.component_tile >= spec.component_count
    )


def _supports_component_lane_rys(
    spec: ShellClassSpec,
    schedule: ScheduleIR,
) -> bool:
    """Return whether runtime-indexed component lanes can lower ``spec``.

    The current decoder has tables for s/p/d centers and supports at most a p
    shell on the fourth center. Expressing those state-table bounds directly
    avoids coupling recurrence eligibility to a list of promoted shell names.
    """

    return (
        schedule.kind == ScheduleKind.COMPONENT_LANES
        and schedule.warp_size == 32
        and schedule.block_threads >= spec.component_count
        and schedule.component_tile >= spec.component_count
        and max(spec.angular) <= 2
        and spec.angular[3] <= 1
    )


def _supports_uniform_warp_rys(
    spec: ShellClassSpec,
    schedule: ScheduleIR,
) -> bool:
    """Return whether uniform component warps can lower 32 shell tasks."""

    return (
        schedule.kind == ScheduleKind.SUBGROUP_TASKS
        and schedule.warp_size == 32
        and schedule.block_threads in (128, 256)
        and schedule.tasks_per_block == 32
        and schedule.subgroup_lanes == schedule.warp_count
        and schedule.component_tile >= spec.component_count
    )


_PRODUCTION_PRELUDE = r"""#include "scf/generated_shell_task.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <limits>

template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) values[order] = 0.0;
  // Switch low orders to stable upward recurrence before the generic cutoff;
  // this removes long alternating series from dominant s/p/d quartets.
  constexpr double series_threshold = MaximumOrder == 0 ? 1.0e-8
      : MaximumOrder == 1 ? 0.25
      : MaximumOrder == 2 ? 0.75
      : MaximumOrder == 3 ? 1.25
      : MaximumOrder == 4 ? 2.0
                          : 6.0;
  if (argument < series_threshold) {
    double term = 1.0;
    double sum = 0.0;
    for (unsigned k = 0; k < 80U; ++k) {
      sum += term /
          static_cast<double>(2U * MaximumOrder + 2U * k + 1U);
      term *= -argument / static_cast<double>(k + 1U);
      if (fabs(term) < 1.0e-18) break;
    }
    values[MaximumOrder] = sum;
    const double exponential = exp(-argument);
    for (unsigned order = MaximumOrder; order > 0U; --order) {
      values[order - 1U] =
          (2.0 * argument * values[order] + exponential) /
          static_cast<double>(2U * order - 1U);
    }
    return;
  }
  values[0] = 0.5 * sqrt(3.14159265358979323846 / argument) *
      erf(sqrt(argument));
  const double exponential = exp(-argument);
  for (unsigned order = 1; order <= MaximumOrder; ++order) {
    values[order] =
        ((2.0 * static_cast<double>(order) - 1.0) * values[order - 1U] -
         exponential) /
        (2.0 * argument);
  }
}

"""


@dataclass(frozen=True, slots=True)
class KernelSelection:
    """One architecture-tuned shell kernel selected for production."""

    architecture: str
    spec: ShellClassSpec
    consumers: tuple[KernelConsumer, ...]
    schedule: ScheduleIR
    profile: str = ""
    tuned: bool = True
    recurrence: str = "subset_wick"
    resident_force_recurrence: str | None = None
    fock_schedule: ScheduleIR | None = None

    def __post_init__(self) -> None:
        if not self.architecture.startswith("sm_"):
            raise ValueError("production architecture must use CUDA sm_ notation")
        if not self.profile:
            object.__setattr__(self, "profile", self.architecture)
        if not self.consumers:
            raise ValueError("production kernel requires at least one consumer")
        if (
            not isinstance(self.recurrence, str)
            or self.recurrence not in _SUPPORTED_RECURRENCES
        ):
            raise ValueError(f"unsupported production recurrence {self.recurrence!r}")
        # IntegralIR owns scientific recurrence legality, including the exact
        # root count implied by angular momentum and derivative order. The
        # production layer only validates whether an implemented CUDA mapping
        # can execute that already-legal recurrence.
        build_integral_ir(
            self.spec,
            self.consumers,
            recurrence=self.recurrence,
        )
        scalar_thread_tasks = _supports_scalar_rys(self.spec, self.schedule)
        if self.recurrence == "rys2" and not scalar_thread_tasks:
            raise ValueError(
                "production rys2 requires one complete scalar task per lane "
                "in a single warp"
            )
        if self.recurrence == "rys3":
            if (
                self.spec.name == "ppps"
                and KernelConsumer.FOCK in self.consumers
                and self.fock_schedule is None
            ):
                raise ValueError(
                    "production ppps rys3 with a Fock consumer requires an "
                    "independent fock_schedule"
                )
            component_lanes = _supports_component_lane_rys(self.spec, self.schedule)
            uniform_warps = _supports_uniform_warp_rys(self.spec, self.schedule)
            if not (scalar_thread_tasks or component_lanes or uniform_warps):
                raise ValueError(
                    "production rys3 requires scalar thread tasks, supported "
                    "runtime-indexed component lanes, or 32 uniform-warp tasks"
                )
        high_root_component_lanes = _supports_component_lane_rys(
            self.spec, self.schedule
        )
        high_root_uniform_warps = _supports_uniform_warp_rys(
            self.spec, self.schedule
        )
        if self.recurrence in ("rys4", "rys5") and not (
            high_root_component_lanes or high_root_uniform_warps
        ):
            raise ValueError(
                f"production {self.recurrence} requires supported "
                "runtime-indexed component lanes or 32 uniform-warp tasks"
            )
        if self.fock_schedule is not None and KernelConsumer.FOCK not in self.consumers:
            raise ValueError("a separate Fock schedule requires a Fock consumer")
        if self.resident_force_recurrence is not None:
            if self.spec.name != "ppps":
                raise ValueError(
                    "resident force recurrence is currently available only "
                    "for the ppps shell class"
                )
            if KernelConsumer.FORCE not in self.consumers:
                raise ValueError(
                    "resident force recurrence requires the force consumer"
                )
            if self.resident_force_recurrence != "rys3":
                raise ValueError("resident ppps force recurrence must be rys3")
        # The shared emitter still defines the canonical task ABI and dormant
        # force symbols for Fock-only rows.  Consumer metadata keeps those
        # symbols out of the force registry while allowing low-order bounded
        # Fock classes to avoid the generic AO-quartet fallback.


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
    """Resolve exact, compatible, portable, then synthetic generic fallback."""

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
    portable = _portable_profile(architectures)
    if portable is not None:
        return portable[0], portable[1], ProfileMatch.PORTABLE
    # An empty synthetic portable profile is the final safe fallback. It emits
    # no generated class mask, leaving the validated generic CUDA path active.
    return "portable_cuda", {"kind": "portable", "kernels": []}, ProfileMatch.PORTABLE


def _validate_measured_target(
    profile_name: str,
    profile: dict[str, object],
    target: CudaTargetInfo,
    match: ProfileMatch,
) -> None:
    """Reject stale exact-profile capability or generator-ABI metadata."""

    generator_abi = profile.get("generator_abi", target.generator_abi)
    if int(generator_abi) != target.generator_abi:
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
        raise ValueError(f"{name} recurrence must be a string")
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
        raise ValueError(f"{name} resident_force_recurrence must be a string")
    if recurrence != "rys3":
        raise ValueError(f"{name} resident force recurrence must be rys3")
    return recurrence


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
        resident_force_recurrence = _resident_force_recurrence_from_row(
            name, row, consumers
        )
        schedule_payload = row.get("schedule")
        if schedule_payload is None:
            schedule = build_fused_shell_plan(
                spec,
                consumers=consumers,
                recurrence=recurrence,
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
            )
        )
        seen.add(name)
    return tuple(selections)


def resolve_production_profile(
    path: Path,
    architecture: str | None = None,
    profile: str = "auto",
) -> ResolvedProductionProfile:
    """Resolve one target through exact, compatible, and portable profiles."""

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
        if selected_architecture != accepted_architecture or profile in (
            "portable",
            "portable_cuda",
        ):
            return ResolvedProductionProfile(
                target=target,
                profile="portable_cuda",
                match=ProfileMatch.PORTABLE,
                tuned=False,
                selections=(),
                cuda_toolkit="",
            )
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


def shell_class_index(spec: ShellClassSpec) -> int:
    """Return the production triangular quartet-class index."""

    first = shell_pair_class(*spec.angular[:2])
    second = shell_pair_class(*spec.angular[2:])
    high = max(first, second)
    low = min(first, second)
    return high * (high + 1) // 2 + low


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


def _launch_wrapper(
    spec: ShellClassSpec,
    symbol: str | None = None,
) -> str:
    """Emit a stable C ABI wrapper around one generated persistent kernel."""

    class_name = spec.name[0].upper() + spec.name[1:]
    return f"""
static_assert(sizeof(Generated{class_name}ShellTask) ==
              sizeof(vibeqc::scf::detail::GeneratedShellTask));
static_assert(alignof(Generated{class_name}ShellTask) ==
              alignof(vibeqc::scf::detail::GeneratedShellTask));
static_assert(offsetof(Generated{class_name}ShellTask, primitive_begin) ==
              offsetof(vibeqc::scf::detail::GeneratedShellTask, primitive_begin));
static_assert(offsetof(Generated{class_name}ShellTask, primitive_end) ==
              offsetof(vibeqc::scf::detail::GeneratedShellTask, primitive_end));
static_assert(offsetof(Generated{class_name}ShellTask, ao_begin) ==
              offsetof(vibeqc::scf::detail::GeneratedShellTask, ao_begin));
static_assert(offsetof(Generated{class_name}ShellTask, ao_coefficient_begin) ==
              offsetof(vibeqc::scf::detail::GeneratedShellTask,
                       ao_coefficient_begin));
static_assert(offsetof(Generated{class_name}ShellTask, density_offset) ==
              offsetof(vibeqc::scf::detail::GeneratedShellTask, density_offset));
static_assert(offsetof(Generated{class_name}ShellTask, spin_offset) ==
              offsetof(vibeqc::scf::detail::GeneratedShellTask, spin_offset));
static_assert(offsetof(Generated{class_name}ShellTask, matrix_order) ==
              offsetof(vibeqc::scf::detail::GeneratedShellTask, matrix_order));
static_assert(offsetof(Generated{class_name}ShellTask, shell_pair) ==
              offsetof(vibeqc::scf::detail::GeneratedShellTask, shell_pair));
static_assert(
    offsetof(Generated{class_name}ShellTask, reversed_shell_pair_mask) ==
    offsetof(vibeqc::scf::detail::GeneratedShellTask,
             reversed_shell_pair_mask));
static_assert(offsetof(Generated{class_name}ShellTask, shell) ==
              offsetof(vibeqc::scf::detail::GeneratedShellTask, shell));
static_assert(offsetof(Generated{class_name}ShellTask, atom) ==
              offsetof(vibeqc::scf::detail::GeneratedShellTask, atom));
static_assert(sizeof(Generated{class_name}PrimitivePairData) ==
              sizeof(vibeqc::scf::detail::GeneratedPrimitivePairData));
static_assert(alignof(Generated{class_name}PrimitivePairData) ==
              alignof(vibeqc::scf::detail::GeneratedPrimitivePairData));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, exponent_sum) ==
    offsetof(vibeqc::scf::detail::GeneratedPrimitivePairData, exponent_sum));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, reduced_exponent) ==
    offsetof(vibeqc::scf::detail::GeneratedPrimitivePairData, reduced_exponent));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, product_center) ==
    offsetof(vibeqc::scf::detail::GeneratedPrimitivePairData, product_center));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, weighted_coefficient) ==
    offsetof(vibeqc::scf::detail::GeneratedPrimitivePairData,
             weighted_coefficient));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, first_product_scale) ==
    offsetof(vibeqc::scf::detail::GeneratedPrimitivePairData,
             first_product_scale));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, second_product_scale) ==
    offsetof(vibeqc::scf::detail::GeneratedPrimitivePairData,
             second_product_scale));

extern "C" cudaError_t {symbol or f"vibeqc_launch_generated_{spec.name}"}(
    cudaStream_t stream, bool unrestricted, unsigned worker_blocks,
    const void* tasks, const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients,
    const void* atom_positions, double screening_tolerance,
    const double* schwarz_bounds, const double* density, double* forces,
    const std::uint32_t* task_count, std::uint32_t* task_head) {{
  if (worker_blocks == 0U) return cudaSuccess;
  const auto* typed_tasks =
      static_cast<const Generated{class_name}ShellTask*>(tasks);
  const auto* typed_positions =
      static_cast<const Generated{class_name}Vec3*>(atom_positions);
  const auto* typed_primitive_pairs =
      static_cast<const Generated{class_name}PrimitivePairData*>(
          primitive_pairs);
  if (unrestricted) {{
    generated_{spec.name}_shell_class_force_uhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}BlockThreads, 0, stream>>>(
        typed_tasks, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, forces, task_offset, task_count, task_head);
  }} else {{
    generated_{spec.name}_shell_class_force_rhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}BlockThreads, 0, stream>>>(
        typed_tasks, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, forces, task_offset, task_count, task_head);
  }}
  return cudaPeekAtLastError();
}}
"""


def _fock_launch_wrapper(
    spec: ShellClassSpec,
    symbol: str | None = None,
) -> str:
    """Emit the stable C ABI wrapper for one generated Fock worker."""

    class_name = spec.name[0].upper() + spec.name[1:]
    return f"""
extern "C" cudaError_t {symbol or f"vibeqc_launch_generated_{spec.name}_fock"}(
    cudaStream_t stream, bool unrestricted, unsigned worker_blocks,
    const void* tasks, const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients,
    const void* atom_positions, double screening_tolerance,
    const double* schwarz_bounds, const double* density, double* fock,
    const std::uint32_t* task_count, std::uint32_t* task_head) {{
  if (worker_blocks == 0U) return cudaSuccess;
  const auto* typed_tasks =
      static_cast<const Generated{class_name}ShellTask*>(tasks);
  const auto* typed_positions =
      static_cast<const Generated{class_name}Vec3*>(atom_positions);
  const auto* typed_primitive_pairs =
      static_cast<const Generated{class_name}PrimitivePairData*>(
          primitive_pairs);
  if (unrestricted) {{
    generated_{spec.name}_shell_class_fock_uhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}FockBlockThreads, 0, stream>>>(
        typed_tasks, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, fock, task_offset, task_count, task_head);
  }} else {{
    generated_{spec.name}_shell_class_fock_rhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}FockBlockThreads, 0, stream>>>(
        typed_tasks, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, fock, task_offset, task_count, task_head);
  }}
  return cudaPeekAtLastError();
}}
"""


def _streaming_fock_source(selection: KernelSelection) -> str:
    """Emit fixed-storage shell-pair enumeration for dominant Fock classes."""

    spec = selection.spec
    schedule = selection.fock_schedule or selection.schedule
    if (
        selection.fock_schedule is None
        and schedule.kind == ScheduleKind.SUBGROUP_TASKS
        and selection.recurrence in ("rys3", "rys4", "rys5")
    ):
        # Uniform subgroup warps are a force-only Rys experiment.  Mirror the
        # value emitter's established component-lane fallback exactly so the
        # streaming wrapper calls the Fock task with its real block geometry.
        angular_order = sum(spec.angular)
        value_state_count = (
            (angular_order + 1) * (angular_order + 2) * (angular_order + 3) // 6
        )
        block_threads = (
            (max(spec.component_count, value_state_count) + 31) // 32 * 32
        )
        schedule = ScheduleIR(
            kind=ScheduleKind.COMPONENT_LANES,
            block_threads=block_threads,
            component_tile=spec.component_count,
            tasks_per_warp=1,
            shared_coulomb=True,
            pair_orientation=schedule.pair_orientation,
            pair_storage=schedule.pair_storage,
            unroll_pair_terms=schedule.unroll_pair_terms,
            minimum_blocks_per_sm=(
                2 if selection.recurrence in ("rys4", "rys5") else 0
            ),
            warp_size=schedule.warp_size,
        )
    if spec.name not in _STREAMING_FOCK_SHELLS:
        return ""

    class_name = spec.name[0].upper() + spec.name[1:]
    first_pair_class = shell_pair_class(*spec.angular[:2])
    second_pair_class = shell_pair_class(*spec.angular[2:])
    high_pair_class = max(first_pair_class, second_pair_class)
    low_pair_class = min(first_pair_class, second_pair_class)
    prefix = f"generated_{spec.name}"
    common = f"""
/** Return the packed shell-pair ordinal for two shells in one system. */
__device__ __forceinline__ std::size_t {prefix}_stream_pair_index(
    const vibeqc::scf::detail::GeneratedShellPairStream& topology,
    std::int32_t system, std::int32_t first_shell,
    std::int32_t second_shell) {{
  const std::size_t shell_begin = static_cast<std::size_t>(
      topology.system_shell_offsets[system]);
  const std::size_t first = static_cast<std::size_t>(first_shell) - shell_begin;
  const std::size_t second =
      static_cast<std::size_t>(second_shell) - shell_begin;
  const std::size_t high = first > second ? first : second;
  const std::size_t low = first > second ? second : first;
  return static_cast<std::size_t>(topology.system_shell_pair_offsets[system]) +
      high * (high + 1U) / 2U + low;
}}

/** Apply the exact bounded RHF/UHF Fock screening predicate. */
template <bool Unrestricted>
__device__ __forceinline__ bool {prefix}_stream_survives(
    const vibeqc::scf::detail::GeneratedShellPairStream& topology,
    std::uint32_t first_pair, std::uint32_t second_pair,
    double screening_tolerance) {{
  const double quartet_bound = topology.shell_pair_bounds[first_pair] *
      topology.shell_pair_bounds[second_pair];
  if (quartet_bound < screening_tolerance) return false;
  const std::int32_t system = topology.shell_pair_systems[first_pair];
  if (topology.active != nullptr && topology.active[system] == 0U) return false;
  const std::int32_t first_shell = topology.shell_pair_first[first_pair];
  const std::int32_t second_shell = topology.shell_pair_second[first_pair];
  const std::int32_t third_shell = topology.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = topology.shell_pair_second[second_pair];
  const std::size_t ac_pair = {prefix}_stream_pair_index(
      topology, system, first_shell, third_shell);
  const std::size_t ad_pair = {prefix}_stream_pair_index(
      topology, system, first_shell, fourth_shell);
  const std::size_t bc_pair = {prefix}_stream_pair_index(
      topology, system, second_shell, third_shell);
  const std::size_t bd_pair = {prefix}_stream_pair_index(
      topology, system, second_shell, fourth_shell);
  const auto ab = topology.shell_pair_density_bounds[first_pair];
  const auto cd = topology.shell_pair_density_bounds[second_pair];
  const auto ac = topology.shell_pair_density_bounds[ac_pair];
  const auto ad = topology.shell_pair_density_bounds[ad_pair];
  const auto bc = topology.shell_pair_density_bounds[bc_pair];
  const auto bd = topology.shell_pair_density_bounds[bd_pair];
  double density_bound = fmax(ab.coulomb, cd.coulomb);
  if constexpr (Unrestricted) {{
    density_bound = fmax(
        density_bound,
        fmax(fmax(ac.exchange_alpha, ac.exchange_beta),
             fmax(ad.exchange_alpha, ad.exchange_beta)));
    density_bound = fmax(
        density_bound,
        fmax(fmax(bc.exchange_alpha, bc.exchange_beta),
             fmax(bd.exchange_alpha, bd.exchange_beta)));
  }} else {{
    const double exchange_bound = fmax(
        fmax(ac.exchange_alpha, ad.exchange_alpha),
        fmax(bc.exchange_alpha, bd.exchange_alpha));
    density_bound = fmax(density_bound, 0.5 * exchange_bound);
  }}
  return quartet_bound * density_bound >= screening_tolerance;
}}

/** Canonicalize one pair product into the stable generated task ABI. */
__device__ __forceinline__ void {prefix}_stream_populate_task(
    const vibeqc::scf::detail::GeneratedShellPairStream& topology,
    std::uint32_t first_pair, std::uint32_t second_pair,
    Generated{class_name}ShellTask& task) {{
  std::int32_t shells[4] = {{
      topology.shell_pair_first[first_pair],
      topology.shell_pair_second[first_pair],
      topology.shell_pair_first[second_pair],
      topology.shell_pair_second[second_pair],
  }};
  std::uint32_t shell_pairs[2] = {{first_pair, second_pair}};
  std::uint32_t reversed_mask = 0U;
  if (topology.shell_angular[shells[0]] <
      topology.shell_angular[shells[1]]) {{
    const std::int32_t swap = shells[0];
    shells[0] = shells[1];
    shells[1] = swap;
    reversed_mask |= 1U;
  }}
  if (topology.shell_angular[shells[2]] <
      topology.shell_angular[shells[3]]) {{
    const std::int32_t swap = shells[2];
    shells[2] = shells[3];
    shells[3] = swap;
    reversed_mask |= 2U;
  }}
  const unsigned first_class =
      static_cast<unsigned>(topology.shell_angular[shells[0]]) *
          (static_cast<unsigned>(topology.shell_angular[shells[0]]) + 1U) /
          2U +
      static_cast<unsigned>(topology.shell_angular[shells[1]]);
  const unsigned second_class =
      static_cast<unsigned>(topology.shell_angular[shells[2]]) *
          (static_cast<unsigned>(topology.shell_angular[shells[2]]) + 1U) /
          2U +
      static_cast<unsigned>(topology.shell_angular[shells[3]]);
  if (first_class < second_class) {{
    const std::int32_t first = shells[0];
    const std::int32_t second = shells[1];
    shells[0] = shells[2];
    shells[1] = shells[3];
    shells[2] = first;
    shells[3] = second;
    const std::uint32_t pair_swap = shell_pairs[0];
    shell_pairs[0] = shell_pairs[1];
    shell_pairs[1] = pair_swap;
    reversed_mask = ((reversed_mask & 1U) << 1U) |
        ((reversed_mask & 2U) >> 1U);
  }}
  const std::int32_t system = topology.shell_pair_systems[first_pair];
  const std::size_t matrix_order = topology.matrix_order;
  const std::size_t matrix_size = matrix_order * matrix_order;
  const std::size_t system_ao_begin =
      static_cast<std::size_t>(system) * matrix_order;
#pragma unroll
  for (unsigned center = 0U; center < 4U; ++center) {{
    const std::int32_t shell = shells[center];
    task.primitive_begin[center] = static_cast<std::uint64_t>(
        topology.shell_primitive_offsets[shell]);
    task.primitive_end[center] = static_cast<std::uint64_t>(
        topology.shell_primitive_offsets[shell + 1]);
    const std::size_t ao_begin = static_cast<std::size_t>(
        topology.shell_direct_ao_offsets[shell]);
    task.ao_begin[center] = static_cast<std::uint64_t>(
        ao_begin - system_ao_begin);
    task.ao_coefficient_begin[center] =
        static_cast<std::uint64_t>(ao_begin);
    task.shell[center] = static_cast<std::uint32_t>(shell);
    task.atom[center] = static_cast<std::uint32_t>(
        topology.shell_atoms[shell]);
  }}
  task.density_offset = static_cast<std::uint64_t>(
      static_cast<std::size_t>(system) * matrix_size);
  task.spin_offset = static_cast<std::uint64_t>(
      static_cast<std::size_t>(system) * 2U * matrix_size);
  task.matrix_order = topology.matrix_order;
  task.shell_pair[0] = shell_pairs[0];
  task.shell_pair[1] = shell_pairs[1];
  task.reversed_shell_pair_mask = reversed_mask;
}}
"""

    if schedule.kind == ScheduleKind.PACKED_TASKS:
        worker = f"""
template <bool Unrestricted>
__device__ __forceinline__ void {prefix}_streaming_fock(
    const vibeqc::scf::detail::GeneratedShellPairStream* topology_pointer,
    const Generated{class_name}PrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const Generated{class_name}Vec3* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head) {{
  static_assert(kGenerated{class_name}FockBlockThreads == 32U);
  __shared__ Generated{class_name}ShellTask stream_tasks[32];
  __shared__ Generated{class_name}PackedFockLaneStorage lane_storage[32];
  __shared__ std::uint32_t bra_ordinal;
  const auto& topology = *topology_pointer;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  const std::uint32_t bra_begin = topology.pair_class_offsets[
      {high_pair_class}U * stride];
  const std::uint32_t bra_end = topology.pair_class_offsets[
      {high_pair_class}U * stride + topology.batch_size];
  while (true) {{
    if (threadIdx.x == 0U) bra_ordinal = atomicAdd(bra_head, 1U);
    __syncthreads();
    if (bra_ordinal >= bra_end - bra_begin) return;
    const std::uint32_t bra_pair =
        topology.pair_order[bra_begin + bra_ordinal];
    const std::int32_t system = topology.shell_pair_systems[bra_pair];
    const std::uint32_t ket_begin = topology.pair_class_offsets[
        {low_pair_class}U * stride + system];
    const std::uint32_t ket_end = topology.pair_class_offsets[
        {low_pair_class}U * stride + system + 1U];
    for (std::uint32_t ket_base = ket_begin; ket_base < ket_end;
         ket_base += 32U) {{
      const std::uint32_t ket_ordinal = ket_base + threadIdx.x;
      bool past_schwarz_tail = ket_ordinal >= ket_end;
      std::uint32_t ket_pair = 0U;
      if (!past_schwarz_tail) {{
        ket_pair = topology.pair_order[ket_ordinal];
        past_schwarz_tail =
            topology.shell_pair_bounds[bra_pair] *
                topology.shell_pair_bounds[ket_pair] < screening_tolerance;
      }}
      // Every class/system ket segment is Schwarz-descending.  Once a whole
      // warp is below the geometry-only gate, no later ket can survive.
      if (__all_sync(0xffffffffU, past_schwarz_tail)) break;
      bool keep = !past_schwarz_tail;
      if (keep) {{
        if constexpr ({str(high_pair_class == low_pair_class).lower()}) {{
          keep = bra_pair >= ket_pair;
        }}
      }}
      if (keep) {{
        keep = {prefix}_stream_survives<Unrestricted>(
            topology, bra_pair, ket_pair, screening_tolerance);
      }}
      if (keep) {{
        {prefix}_stream_populate_task(
            topology, bra_pair, ket_pair, stream_tasks[threadIdx.x]);
        {prefix}_packed_fock_lane<Unrestricted>(
            stream_tasks, primitive_pairs, primitive_pair_offsets,
            ao_coefficients, atom_positions, screening_tolerance,
            schwarz_bounds, density, fock,
            static_cast<std::size_t>(threadIdx.x),
            lane_storage[threadIdx.x]);
      }}
      __syncthreads();
    }}
  }}
}}
"""
    elif schedule.kind == ScheduleKind.SUBGROUP_TASKS:
        tasks_per_block = schedule.tasks_per_block
        subgroup_lanes = schedule.subgroup_lanes
        subgroup_mask = (1 << subgroup_lanes) - 1
        worker = f"""
template <bool Unrestricted>
__device__ __forceinline__ void {prefix}_streaming_fock(
    const vibeqc::scf::detail::GeneratedShellPairStream* topology_pointer,
    const Generated{class_name}PrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const Generated{class_name}Vec3* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head) {{
  static_assert(kGenerated{class_name}FockBlockThreads == {schedule.block_threads}U);
  __shared__ Generated{class_name}ShellTask stream_tasks[{tasks_per_block}];
  __shared__ Generated{class_name}SubgroupFockStorage
      subgroup_storage[{tasks_per_block}];
  __shared__ std::uint32_t stream_keep[{tasks_per_block}];
  __shared__ std::uint32_t bra_ordinal;
  const unsigned subgroup = threadIdx.x / {subgroup_lanes}U;
  const unsigned lane = threadIdx.x % {subgroup_lanes}U;
  const unsigned subgroup_in_warp =
      (threadIdx.x & 31U) / {subgroup_lanes}U;
  const unsigned subgroup_mask =
      0x{subgroup_mask:08x}U << (subgroup_in_warp * {subgroup_lanes}U);
  const auto& topology = *topology_pointer;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  const std::uint32_t bra_begin = topology.pair_class_offsets[
      {high_pair_class}U * stride];
  const std::uint32_t bra_end = topology.pair_class_offsets[
      {high_pair_class}U * stride + topology.batch_size];
  while (true) {{
    if (threadIdx.x == 0U) bra_ordinal = atomicAdd(bra_head, 1U);
    __syncthreads();
    if (bra_ordinal >= bra_end - bra_begin) return;
    const std::uint32_t bra_pair =
        topology.pair_order[bra_begin + bra_ordinal];
    const std::int32_t system = topology.shell_pair_systems[bra_pair];
    const std::uint32_t ket_begin = topology.pair_class_offsets[
        {low_pair_class}U * stride + system];
    const std::uint32_t ket_end = topology.pair_class_offsets[
        {low_pair_class}U * stride + system + 1U];
    for (std::uint32_t ket_base = ket_begin; ket_base < ket_end;
         ket_base += {tasks_per_block}U) {{
      if (lane == 0U) {{
        const std::uint32_t ket_ordinal = ket_base + subgroup;
        // State 2 marks the monotonic Schwarz tail, 1 retained work, and 0
        // an exact-density or canonical-triangle rejection.
        std::uint32_t state = 2U;
        if (ket_ordinal < ket_end) {{
          const std::uint32_t ket_pair = topology.pair_order[ket_ordinal];
          const bool past_schwarz_tail =
              topology.shell_pair_bounds[bra_pair] *
                  topology.shell_pair_bounds[ket_pair] < screening_tolerance;
          bool keep = !past_schwarz_tail;
          if (keep &&
              {str(high_pair_class == low_pair_class).lower()}) {{
            keep = bra_pair >= ket_pair;
          }}
          if (keep) {{
            keep = {prefix}_stream_survives<Unrestricted>(
                topology, bra_pair, ket_pair, screening_tolerance);
          }}
          state = past_schwarz_tail ? 2U : (keep ? 1U : 0U);
          if (keep) {{
            {prefix}_stream_populate_task(
                topology, bra_pair, ket_pair, stream_tasks[subgroup]);
          }}
        }}
        stream_keep[subgroup] = state;
      }}
      __syncthreads();
      bool all_past_schwarz_tail = true;
#pragma unroll
      for (unsigned candidate = 0U; candidate < {tasks_per_block}U;
           ++candidate) {{
        all_past_schwarz_tail &= stream_keep[candidate] == 2U;
      }}
      if (all_past_schwarz_tail) break;
      if (stream_keep[subgroup] == 1U) {{
        {prefix}_subgroup_fock_task<Unrestricted>(
            stream_tasks, primitive_pairs, primitive_pair_offsets,
            ao_coefficients, atom_positions, screening_tolerance,
            schwarz_bounds, density, fock,
            static_cast<std::size_t>(subgroup), subgroup_storage[subgroup],
            lane, subgroup_mask);
      }}
      __syncthreads();
    }}
  }}
}}
"""
    elif (
        schedule.kind == ScheduleKind.COMPONENT_LANES
        and high_pair_class != low_pair_class
    ):
        # A component-lane CTA already spends the full block on one shell
        # quartet. Claim pair products directly instead of pinning one CTA to
        # a bra and serially draining its entire ket segment. This exposes the
        # same two-dimensional shell-pair concurrency used by GPU4PySCF while
        # retaining the accepted per-quartet Rys/component implementation.
        same_pair_class = str(high_pair_class == low_pair_class).lower()
        worker = f"""
template <bool Unrestricted>
__device__ __forceinline__ void {prefix}_streaming_fock(
    const vibeqc::scf::detail::GeneratedShellPairStream* topology_pointer,
    const Generated{class_name}PrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const Generated{class_name}Vec3* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* task_head) {{
  static_assert(kGenerated{class_name}FockBlockThreads ==
                {schedule.block_threads}U);
  __shared__ Generated{class_name}ShellTask stream_task[1];
  __shared__ std::uint32_t candidate_ordinal;
  __shared__ std::uint32_t stream_state;
  const auto& topology = *topology_pointer;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  while (true) {{
    if (threadIdx.x == 0U) {{
      candidate_ordinal = atomicAdd(task_head, 1U);
      std::uint64_t remaining = candidate_ordinal;
      // State 2 means the global pair-product domain is exhausted, 1 is an
      // exact retained quartet, and 0 is a screened candidate.
      stream_state = 2U;
      for (std::int32_t system = 0; system < topology.batch_size; ++system) {{
        const std::uint32_t high_begin = topology.pair_class_offsets[
            {high_pair_class}U * stride + static_cast<std::size_t>(system)];
        const std::uint32_t high_end = topology.pair_class_offsets[
            {high_pair_class}U * stride + static_cast<std::size_t>(system) + 1U];
        const std::uint32_t low_begin = topology.pair_class_offsets[
            {low_pair_class}U * stride + static_cast<std::size_t>(system)];
        const std::uint32_t low_end = topology.pair_class_offsets[
            {low_pair_class}U * stride + static_cast<std::size_t>(system) + 1U];
        const std::uint64_t high_count = high_end - high_begin;
        const std::uint64_t low_count = low_end - low_begin;
        const std::uint64_t system_candidates = {same_pair_class}
            ? high_count * (high_count + 1U) / 2U
            : high_count * low_count;
        if (remaining >= system_candidates) {{
          remaining -= system_candidates;
          continue;
        }}

        std::uint64_t high_local = 0U;
        std::uint64_t low_local = 0U;
        if constexpr ({same_pair_class}) {{
          while ((high_local + 1U) * (high_local + 2U) / 2U <= remaining) {{
            ++high_local;
          }}
          low_local = remaining - high_local * (high_local + 1U) / 2U;
        }} else {{
          high_local = remaining / low_count;
          low_local = remaining - high_local * low_count;
        }}
        const std::uint32_t bra_pair = topology.pair_order[
            high_begin + static_cast<std::uint32_t>(high_local)];
        const std::uint32_t ket_pair = topology.pair_order[
            low_begin + static_cast<std::uint32_t>(low_local)];
        const bool keep = {prefix}_stream_survives<Unrestricted>(
            topology, bra_pair, ket_pair, screening_tolerance);
        stream_state = keep ? 1U : 0U;
        if (keep) {{
          {prefix}_stream_populate_task(
              topology, bra_pair, ket_pair, stream_task[0]);
        }}
        break;
      }}
    }}
    __syncthreads();
    if (stream_state == 2U) return;
    if (stream_state == 1U) {{
      {prefix}_shell_class_fock_task<Unrestricted>(
          stream_task, primitive_pairs, primitive_pair_offsets,
          ao_coefficients, atom_positions, screening_tolerance,
          schwarz_bounds, density, fock, 0U);
    }}
    __syncthreads();
  }}
}}
"""
    else:
        worker = f"""
template <bool Unrestricted>
__device__ __forceinline__ void {prefix}_streaming_fock(
    const vibeqc::scf::detail::GeneratedShellPairStream* topology_pointer,
    const Generated{class_name}PrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const Generated{class_name}Vec3* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head) {{
  static_assert(kGenerated{class_name}FockBlockThreads ==
                {schedule.block_threads}U);
  __shared__ Generated{class_name}ShellTask stream_task[1];
  __shared__ std::uint32_t stream_state;
  __shared__ std::uint32_t bra_ordinal;
  const auto& topology = *topology_pointer;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  const std::uint32_t bra_begin = topology.pair_class_offsets[
      {high_pair_class}U * stride];
  const std::uint32_t bra_end = topology.pair_class_offsets[
      {high_pair_class}U * stride + topology.batch_size];
  while (true) {{
    if (threadIdx.x == 0U) bra_ordinal = atomicAdd(bra_head, 1U);
    __syncthreads();
    if (bra_ordinal >= bra_end - bra_begin) return;
    const std::uint32_t bra_pair =
        topology.pair_order[bra_begin + bra_ordinal];
    const std::int32_t system = topology.shell_pair_systems[bra_pair];
    const std::uint32_t ket_begin = topology.pair_class_offsets[
        {low_pair_class}U * stride + system];
    const std::uint32_t ket_end = topology.pair_class_offsets[
        {low_pair_class}U * stride + system + 1U];
    for (std::uint32_t ket_ordinal = ket_begin; ket_ordinal < ket_end;
         ++ket_ordinal) {{
      if (threadIdx.x == 0U) {{
        const std::uint32_t ket_pair = topology.pair_order[ket_ordinal];
        const bool past_schwarz_tail =
            topology.shell_pair_bounds[bra_pair] *
                topology.shell_pair_bounds[ket_pair] < screening_tolerance;
        bool keep = !past_schwarz_tail;
        if (keep && {str(high_pair_class == low_pair_class).lower()}) {{
          keep = bra_pair >= ket_pair;
        }}
        if (keep) {{
          keep = {prefix}_stream_survives<Unrestricted>(
              topology, bra_pair, ket_pair, screening_tolerance);
        }}
        stream_state = past_schwarz_tail ? 2U : (keep ? 1U : 0U);
        if (keep) {{
          {prefix}_stream_populate_task(
              topology, bra_pair, ket_pair, stream_task[0]);
        }}
      }}
      __syncthreads();
      if (stream_state == 2U) break;
      if (stream_state == 1U) {{
        {prefix}_shell_class_fock_task<Unrestricted>(
            stream_task, primitive_pairs, primitive_pair_offsets,
            ao_coefficients, atom_positions, screening_tolerance,
            schwarz_bounds, density, fock, 0U);
      }}
      __syncthreads();
    }}
  }}
}}
"""

    kernels = f"""
extern "C" __global__ __launch_bounds__(kGenerated{class_name}FockBlockThreads)
void {prefix}_shell_class_fock_rhf_streaming_kernel(
    const vibeqc::scf::detail::GeneratedShellPairStream* topology,
    const Generated{class_name}PrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const Generated{class_name}Vec3* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head) {{
  {prefix}_streaming_fock<false>(
      topology, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      bra_head);
}}

extern "C" __global__ __launch_bounds__(kGenerated{class_name}FockBlockThreads)
void {prefix}_shell_class_fock_uhf_streaming_kernel(
    const vibeqc::scf::detail::GeneratedShellPairStream* topology,
    const Generated{class_name}PrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const Generated{class_name}Vec3* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head) {{
  {prefix}_streaming_fock<true>(
      topology, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      bra_head);
}}
"""
    return common + worker + kernels


def _streaming_fock_launch_wrapper(
    spec: ShellClassSpec,
    symbol: str | None = None,
) -> str:
    """Emit the stable host wrapper for fixed-storage Fock streaming."""

    class_name = spec.name[0].upper() + spec.name[1:]
    return f"""
extern "C" cudaError_t {symbol or f"vibeqc_launch_generated_{spec.name}_streaming_fock"}(
    cudaStream_t stream, bool unrestricted, unsigned worker_blocks,
    const void* shell_pair_stream,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head) {{
  if (worker_blocks == 0U) return cudaSuccess;
  const auto* topology = static_cast<
      const vibeqc::scf::detail::GeneratedShellPairStream*>(
          shell_pair_stream);
  const auto* typed_positions =
      static_cast<const Generated{class_name}Vec3*>(atom_positions);
  const auto* typed_primitive_pairs =
      static_cast<const Generated{class_name}PrimitivePairData*>(
          primitive_pairs);
  if (unrestricted) {{
    generated_{spec.name}_shell_class_fock_uhf_streaming_kernel<<<
        worker_blocks, kGenerated{class_name}FockBlockThreads, 0, stream>>>(
        topology, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance,
        schwarz_bounds, density, fock, bra_head);
  }} else {{
    generated_{spec.name}_shell_class_fock_rhf_streaming_kernel<<<
        worker_blocks, kGenerated{class_name}FockBlockThreads, 0, stream>>>(
        topology, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance,
        schwarz_bounds, density, fock, bra_head);
  }}
  return cudaPeekAtLastError();
}}
"""


def _ppps_resident_launch_wrapper(symbol: str | None = None) -> str:
    """Emit the stable opaque-pointer C ABI for resident ppps force work.

    The generated CUDA source owns a profile-scoped descriptor type so that
    multiple architecture shards can coexist in one fat binary.  The host
    descriptor lives in ``generated_shell_task.hpp``; these size/alignment and
    field-offset assertions make a mismatch fail at AOT compilation instead
    of silently corrupting a resident launch.
    """

    return f"""
static_assert(sizeof(GeneratedPppsResidentTask) ==
              sizeof(vibeqc::scf::detail::GeneratedPppsResidentTask));
static_assert(alignof(GeneratedPppsResidentTask) ==
              alignof(vibeqc::scf::detail::GeneratedPppsResidentTask));
static_assert(offsetof(GeneratedPppsResidentTask, bra_pair) ==
              offsetof(vibeqc::scf::detail::GeneratedPppsResidentTask,
                       bra_pair));
static_assert(offsetof(GeneratedPppsResidentTask, ket_begin) ==
              offsetof(vibeqc::scf::detail::GeneratedPppsResidentTask,
                       ket_begin));
static_assert(offsetof(GeneratedPppsResidentTask, ket_count) ==
              offsetof(vibeqc::scf::detail::GeneratedPppsResidentTask,
                       ket_count));
static_assert(sizeof(GeneratedPppsPrimitivePairData) ==
              sizeof(vibeqc::scf::detail::GeneratedPrimitivePairData));
static_assert(alignof(GeneratedPppsPrimitivePairData) ==
              alignof(vibeqc::scf::detail::GeneratedPrimitivePairData));

extern "C" cudaError_t {symbol or "vibeqc_launch_ppps_resident"}(
    cudaStream_t stream, bool unrestricted, const void* resident_tasks,
    const void* ket_tasks,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* forces, unsigned block_threads,
    std::size_t task_count) {{
  if (task_count == 0U) return cudaSuccess;
  if (block_threads != 32U && block_threads != 64U &&
      block_threads != 128U && block_threads != 256U) {{
    return cudaErrorInvalidValue;
  }}
  if (task_count > static_cast<std::size_t>(
          std::numeric_limits<unsigned>::max())) return cudaErrorInvalidValue;
  const auto* typed_resident_tasks =
      static_cast<const GeneratedPppsResidentTask*>(resident_tasks);
  const auto* typed_ket_tasks =
      static_cast<const GeneratedPppsShellTask*>(ket_tasks);
  const auto* typed_positions =
      static_cast<const GeneratedPppsVec3*>(atom_positions);
  const auto* typed_primitive_pairs =
      static_cast<const GeneratedPppsPrimitivePairData*>(primitive_pairs);
  if (unrestricted) {{
    generated_ppps_resident_bra_force_uhf_kernel<<<
        static_cast<unsigned>(task_count),
        block_threads, 0, stream>>>(
        typed_resident_tasks, typed_ket_tasks, typed_primitive_pairs,
        primitive_pair_offsets, ao_coefficients, typed_positions,
        screening_tolerance, schwarz_bounds, density, forces, task_count);
  }} else {{
    generated_ppps_resident_bra_force_rhf_kernel<<<
        static_cast<unsigned>(task_count),
        block_threads, 0, stream>>>(
        typed_resident_tasks, typed_ket_tasks, typed_primitive_pairs,
        primitive_pair_offsets, ao_coefficients, typed_positions,
        screening_tolerance, schwarz_bounds, density, forces, task_count);
  }}
  return cudaPeekAtLastError();
}}
"""


def _as_selection(item: ShellClassSpec | KernelSelection) -> KernelSelection:
    """Normalize compatibility callers to the explicit production IR."""

    if isinstance(item, KernelSelection):
        return item
    plan = build_fused_shell_plan(item)
    return KernelSelection(
        architecture="sm_120",
        spec=item,
        consumers=(KernelConsumer.FORCE,),
        schedule=plan.schedule,
    )


def _emit_ppps_resident_source(selection: KernelSelection) -> str:
    """Emit only the resident ppps tail for an opted-in production row."""

    if selection.resident_force_recurrence is None:
        return ""
    # The ordinary ppps source is emitted immediately before this tail.  A
    # A subset/Wick row needs the resident Rys evaluator appended.  The scalar
    # thread-task Rys3 force worker owns that exact symbol already, while the
    # uniform-warp worker deliberately uses a schedule-qualified root symbol
    # and therefore still needs the resident evaluator beside it.
    ordinary_owns_resident_roots = (
        selection.recurrence == "rys3"
        and selection.schedule.kind == ScheduleKind.THREAD_TASKS
    )
    return emit_ppps_resident_bra_rys3_cuda(
        include_shared_definitions=False,
        include_rys3_roots=not ordinary_owns_resident_roots,
    )


def emit_production_shard(
    specifications: Iterable[ShellClassSpec | KernelSelection],
) -> str:
    """Emit one CUDA TU containing a deterministic subset of accepted classes."""

    selections = tuple(map(_as_selection, specifications))
    body = [_PRODUCTION_PRELUDE]
    for selection in selections:
        plan = build_fused_shell_plan(
            selection.spec,
            consumers=selection.consumers,
            schedule=selection.schedule,
            recurrence=selection.recurrence,
        )
        body.append(
            emit_shell_class_fused_cuda(
                selection.spec,
                plan,
                fock_schedule=selection.fock_schedule,
            )
        )
        body.append(_launch_wrapper(selection.spec))
        if KernelConsumer.FOCK in selection.consumers:
            body.append(_fock_launch_wrapper(selection.spec))
            if selection.spec.name in _STREAMING_FOCK_SHELLS:
                body.append(_streaming_fock_source(selection))
                body.append(_streaming_fock_launch_wrapper(selection.spec))
        if selection.resident_force_recurrence is not None:
            body.append(_emit_ppps_resident_source(selection))
            body.append(_ppps_resident_launch_wrapper())
    if not selections:
        body.append("// Empty deterministic shard reserved for stable CMake outputs.\n")
    return "".join(body)


def emit_registry_header(
    specifications: Iterable[ShellClassSpec | KernelSelection],
) -> str:
    """Emit production metadata and the host launch API consumed by cuda_rhf."""

    selections = tuple(map(_as_selection, specifications))
    rows = []
    for selection in selections:
        if KernelConsumer.FORCE not in selection.consumers:
            continue
        spec = selection.spec
        plan = build_fused_shell_plan(
            spec,
            consumers=selection.consumers,
            schedule=selection.schedule,
            recurrence=selection.recurrence,
        )
        consumer_mask = sum(
            1 << list(KernelConsumer).index(consumer)
            for consumer in selection.consumers
        )
        rows.append(
            f'    {{"{spec.name}", {shell_class_index(spec)}U, '
            f"{sum(spec.angular)}U, {plan.block_threads}U, "
            f"{consumer_mask}U, {plan.schedule.component_tile}U}},"
        )
    fock_rows = []
    for selection in selections:
        if KernelConsumer.FOCK not in selection.consumers:
            continue
        spec = selection.spec
        angular_order = sum(spec.angular)
        value_state_count = (
            (angular_order + 1) * (angular_order + 2) * (angular_order + 3) // 6
        )
        block_threads = ((max(spec.component_count, value_state_count) + 31) // 32) * 32
        fock_rows.append(
            f'    {{"{spec.name}", {shell_class_index(spec)}U, '
            f"{angular_order}U, {block_threads}U, 1U, "
            f"{spec.component_count}U}},"
        )
    return f"""#ifndef VIBEQC_GENERATED_SHELL_REGISTRY_HPP
#define VIBEQC_GENERATED_SHELL_REGISTRY_HPP

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace vibeqc::scf::generated {{

struct ShellKernelMetadata {{
  const char* name;
  unsigned shell_class;
  unsigned angular_order;
  unsigned block_threads;
  unsigned consumer_mask;
  unsigned component_tile;
}};

inline constexpr ShellKernelMetadata kShellKernels[] = {{
{chr(10).join(rows)}
}};
inline constexpr std::size_t kShellKernelCount =
    sizeof(kShellKernels) / sizeof(kShellKernels[0]);

inline constexpr std::array<ShellKernelMetadata, {len(fock_rows)}>
    kFockShellKernels{{{{
{chr(10).join(fock_rows)}
}}}};
inline constexpr std::size_t kFockShellKernelCount =
    kFockShellKernels.size();

/** Return the exact-class bit mask selected by VIBEQC_AOT_SHELL_CLASSES. */
std::uint64_t enabled_shell_class_mask() noexcept;

/** Return the Fock-class mask selected by VIBEQC_AOT_FOCK_SHELL_CLASSES. */
std::uint64_t enabled_fock_shell_class_mask() noexcept;

/** Launch one generated persistent kernel selected by exact class index. */
cudaError_t launch_shell_class(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* forces, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept;

/** Launch one generated coefficient-only Fock worker by exact class. */
cudaError_t launch_shell_class_fock(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept;

/** Launch one fixed-storage resident-bra Fock stream by exact class. */
cudaError_t launch_shell_class_streaming_fock(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* shell_pair_stream,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head) noexcept;

/** Launch the optional resident-bra ppps force worker. */
cudaError_t launch_ppps_resident(
    cudaStream_t stream, bool unrestricted, const void* resident_tasks,
    const void* ket_tasks, const std::int64_t* primitive_pair_offsets,
    const void* primitive_pairs, const double* ao_coefficients,
    const void* atom_positions, double screening_tolerance,
    const double* schwarz_bounds, const double* density, double* forces,
    std::size_t task_count) noexcept;

}}  // namespace vibeqc::scf::generated

#endif
"""


def emit_registry_source(
    specifications: Iterable[ShellClassSpec | KernelSelection],
) -> str:
    """Emit environment-controlled dispatch without handwritten class switches."""

    selections = tuple(map(_as_selection, specifications))
    specs = tuple(item.spec for item in selections)
    fock_specs = tuple(
        item.spec for item in selections if KernelConsumer.FOCK in item.consumers
    )
    streaming_fock_specs = tuple(
        item.spec
        for item in selections
        if KernelConsumer.FOCK in item.consumers
        and item.spec.name in _STREAMING_FOCK_SHELLS
    )
    resident_selection = next(
        (item for item in selections if item.resident_force_recurrence is not None),
        None,
    )
    declarations = "\n".join(
        f"""extern "C" cudaError_t vibeqc_launch_generated_{spec.name}(
    cudaStream_t, bool, unsigned, const void*, const std::uint32_t*,
    const std::int64_t*, const void*, const double*, const void*, double,
    const double*, const double*, double*, const std::uint32_t*,
    std::uint32_t*);"""
        for spec in specs
    )
    cases = "\n".join(
        f"""    case {shell_class_index(spec)}U:
      return vibeqc_launch_generated_{spec.name}(
          stream, unrestricted, worker_blocks, tasks, task_offset,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions,
          screening_tolerance, schwarz_bounds, density, forces, task_count,
          task_head);"""
        for spec in specs
    )
    fock_declarations = "\n".join(
        f"""extern "C" cudaError_t vibeqc_launch_generated_{spec.name}_fock(
    cudaStream_t, bool, unsigned, const void*, const std::uint32_t*,
    const std::int64_t*, const void*, const double*, const void*, double,
    const double*, const double*, double*, const std::uint32_t*,
    std::uint32_t*);"""
        for spec in fock_specs
    )
    fock_cases = "\n".join(
        f"""    case {shell_class_index(spec)}U:
      return vibeqc_launch_generated_{spec.name}_fock(
          stream, unrestricted, worker_blocks, tasks, task_offset,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density, fock,
          task_count, task_head);"""
        for spec in fock_specs
    )
    streaming_fock_declarations = "\n".join(
        f'''extern "C" cudaError_t vibeqc_launch_generated_{spec.name}_streaming_fock(
    cudaStream_t, bool, unsigned, const void*, const std::int64_t*,
    const void*, const double*, const void*, double, const double*,
    const double*, double*, std::uint32_t*);'''
        for spec in streaming_fock_specs
    )
    streaming_fock_cases = "\n".join(
        f'''    case {shell_class_index(spec)}U:
      return vibeqc_launch_generated_{spec.name}_streaming_fock(
          stream, unrestricted, worker_blocks, shell_pair_stream,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density, fock,
          bra_head);'''
        for spec in streaming_fock_specs
    )
    resident_declaration = ""
    resident_launch = "return cudaErrorNotSupported;"
    if resident_selection is not None:
        resident_declaration = (
            'extern "C" cudaError_t vibeqc_launch_ppps_resident('
            f"{_resident_launch_parameter_declaration()});"
        )
        resident_launch = (
            f"return vibeqc_launch_ppps_resident({_resident_launch_argument_list()});"
        )
    return f"""#include "vibeqc_generated_shell_registry.hpp"

#include <cstdlib>
#include <cstring>

{declarations}
{fock_declarations}
{streaming_fock_declarations}
{resident_declaration}

namespace vibeqc::scf::generated {{
namespace {{

bool selected(const char* list, const char* name) noexcept {{
  const std::size_t name_size = std::strlen(name);
  const char* cursor = list;
  while (*cursor != '\\0') {{
    while (*cursor == ',' || *cursor == ';' || *cursor == ' ' ||
           *cursor == '\\t') ++cursor;
    const char* begin = cursor;
    while (*cursor != '\\0' && *cursor != ',' && *cursor != ';' &&
           *cursor != ' ' && *cursor != '\\t') ++cursor;
    if (static_cast<std::size_t>(cursor - begin) == name_size &&
        std::strncmp(begin, name, name_size) == 0) return true;
  }}
  return false;
}}

}}  // namespace

std::uint64_t enabled_shell_class_mask() noexcept {{
  const char* selection = std::getenv("VIBEQC_AOT_SHELL_CLASSES");
  const bool all = selection == nullptr || *selection == '\\0' ||
                   std::strcmp(selection, "all") == 0;
  if (!all && std::strcmp(selection, "none") == 0) return 0;
  std::uint64_t mask = 0;
  for (const ShellKernelMetadata& kernel : kShellKernels) {{
    if (all || selected(selection, kernel.name)) {{
      mask |= std::uint64_t{{1}} << kernel.shell_class;
    }}
  }}
  return mask;
}}

std::uint64_t enabled_fock_shell_class_mask() noexcept {{
  const char* selection = std::getenv("VIBEQC_AOT_FOCK_SHELL_CLASSES");
  const bool all = selection == nullptr || *selection == '\\0' ||
                   std::strcmp(selection, "all") == 0;
  if (!all && std::strcmp(selection, "none") == 0) return 0;
  std::uint64_t mask = 0;
  for (const ShellKernelMetadata& kernel : kFockShellKernels) {{
    if (all || selected(selection, kernel.name)) {{
      mask |= std::uint64_t{{1}} << kernel.shell_class;
    }}
  }}
  return mask;
}}

cudaError_t launch_shell_class(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* forces, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept {{
  switch (shell_class) {{
{cases}
    default: return cudaErrorInvalidValue;
  }}
}}

cudaError_t launch_shell_class_fock(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept {{
  switch (shell_class) {{
{fock_cases}
    default: return cudaErrorInvalidValue;
  }}
}}

cudaError_t launch_shell_class_streaming_fock(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* shell_pair_stream,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head) noexcept {{
  switch (shell_class) {{
{streaming_fock_cases}
    default: return cudaErrorInvalidValue;
  }}
}}

cudaError_t launch_ppps_resident(
    {_resident_launch_parameter_declaration()}) noexcept {{
  {resident_launch}
}}

}}  // namespace vibeqc::scf::generated
"""


def _profile_identifier(value: str) -> str:
    """Return a stable C/CMake identifier for a profile or architecture."""

    if re.fullmatch(r"sm_[0-9]+", value):
        return value.replace("_", "")
    identifier = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    if not identifier:
        raise ValueError("profile identifier cannot be empty")
    return identifier


def _strip_emitter_includes(source: str) -> str:
    """Remove global standard includes before placing source in a namespace."""

    return re.sub(
        r"^#include <(?:cstddef|cstdint)>\n",
        "",
        source,
        flags=re.MULTILINE,
    )


def _scope_profile_identifiers(
    source: str,
    selection: KernelSelection,
    identifier: str,
) -> str:
    """Apply the emitter's two structured identifier roots to one profile.

    Generated code owns a lower-case CUDA symbol root and a CamelCase type
    root. Rewriting only those declared roots keeps profile isolation explicit
    and avoids unconstrained shell-name/string substitution.
    """

    # Profile scoping applies to generated CUDA types, never to the stable
    # runtime ABI included from ``generated_shell_task.hpp``.  Protect the
    # qualified host descriptor while rewriting the shared ``GeneratedPpps``
    # prefix, then restore it verbatim.
    host_resident_task = "vibeqc::scf::detail::GeneratedPppsResidentTask"
    host_resident_task_placeholder = "VIBEQC_STABLE_PPPS_RESIDENT_TASK_ABI"
    source = source.replace(host_resident_task, host_resident_task_placeholder)
    class_name = selection.spec.name[0].upper() + selection.spec.name[1:]
    profile_class = "".join(part.capitalize() for part in identifier.split("_"))
    return (
        source.replace(
            f"generated_{selection.spec.name}",
            f"generated_{identifier}_{selection.spec.name}",
        )
        .replace(
            f"Generated{class_name}",
            f"Generated{profile_class}{class_name}",
        )
        .replace(host_resident_task_placeholder, host_resident_task)
    )


def emit_profile_shard(
    profile: ResolvedProductionProfile,
    selections: Iterable[KernelSelection],
) -> str:
    """Emit one architecture-namespaced shard with collision-free symbols."""

    items = tuple(selections)
    identifier = _profile_identifier(profile.target.architecture)
    namespace = f"vibeqc::scf::generated::profile_{identifier}"
    body = [_PRODUCTION_PRELUDE, f"\nnamespace {namespace} {{\n"]
    for selection in items:
        plan = build_fused_shell_plan(
            selection.spec,
            consumers=selection.consumers,
            schedule=selection.schedule,
            recurrence=selection.recurrence,
            target=profile.target,
        )
        source = emit_shell_class_fused_cuda(
            selection.spec,
            plan,
            fock_schedule=selection.fock_schedule,
        )
        force_symbol = f"vibeqc_launch_{identifier}_generated_{selection.spec.name}"
        body.append(
            _scope_profile_identifiers(
                _strip_emitter_includes(source), selection, identifier
            )
        )
        force_wrapper = _scope_profile_identifiers(
            _launch_wrapper(selection.spec, force_symbol),
            selection,
            identifier,
        ).replace(
            _scope_profile_identifiers(force_symbol, selection, identifier),
            force_symbol,
        )
        body.append(force_wrapper)
        if KernelConsumer.FOCK in selection.consumers:
            fock_symbol = f"{force_symbol}_fock"
            fock_wrapper = _scope_profile_identifiers(
                _fock_launch_wrapper(
                    selection.spec,
                    fock_symbol,
                ),
                selection,
                identifier,
            ).replace(
                _scope_profile_identifiers(fock_symbol, selection, identifier),
                fock_symbol,
            )
            body.append(fock_wrapper)
            if selection.spec.name in _STREAMING_FOCK_SHELLS:
                body.append(
                    _scope_profile_identifiers(
                        _streaming_fock_source(selection), selection, identifier
                    )
                )
                streaming_symbol = f"{force_symbol}_streaming_fock"
                streaming_wrapper = _scope_profile_identifiers(
                    _streaming_fock_launch_wrapper(
                        selection.spec,
                        streaming_symbol,
                    ),
                    selection,
                    identifier,
                ).replace(
                    _scope_profile_identifiers(
                        streaming_symbol, selection, identifier
                    ),
                    streaming_symbol,
                )
                body.append(streaming_wrapper)
        if selection.resident_force_recurrence is not None:
            resident_source = _scope_profile_identifiers(
                _strip_emitter_includes(_emit_ppps_resident_source(selection)),
                selection,
                identifier,
            )
            body.append(resident_source)
            resident_symbol = f"vibeqc_launch_{identifier}_ppps_resident"
            resident_wrapper = _scope_profile_identifiers(
                _ppps_resident_launch_wrapper(resident_symbol),
                selection,
                identifier,
            ).replace(
                _scope_profile_identifiers(resident_symbol, selection, identifier),
                resident_symbol,
            )
            body.append(resident_wrapper)
    if not items:
        body.append("// Portable profile: generic CUDA kernels remain active.\n")
    body.append(f"\n}}  // namespace {namespace}\n")
    return "".join(body)


def emit_multi_registry_header(
    profiles: Iterable[ResolvedProductionProfile],
) -> str:
    """Emit immutable metadata for every independently compiled profile."""

    items = tuple(profiles)
    profile_rows = []
    kernel_rows = []
    for profile in items:
        profile_rows.append(
            f'    {{"{profile.profile}", "{profile.target.architecture}", '
            f"{profile.target.compute_capability_major}, "
            f"{profile.target.compute_capability_minor}, "
            f"{'true' if profile.tuned else 'false'}, "
            f"{'true' if profile.portable else 'false'}, "
            f"{'true' if profile.match == ProfileMatch.COMPATIBLE else 'false'}}},"
        )
        for selection in profile.selections:
            plan = build_fused_shell_plan(
                selection.spec,
                consumers=selection.consumers,
                schedule=selection.schedule,
                recurrence=selection.recurrence,
                target=profile.target,
            )
            consumer_mask = sum(
                1 << list(KernelConsumer).index(consumer)
                for consumer in selection.consumers
            )
            kernel_rows.append(
                f'    {{"{profile.target.architecture}", "{selection.spec.name}", '
                f"{shell_class_index(selection.spec)}U, {plan.block_threads}U, "
                f"{consumer_mask}U, {plan.schedule.component_tile}U}},"
            )
    return f"""#ifndef VIBEQC_GENERATED_SHELL_REGISTRY_HPP
#define VIBEQC_GENERATED_SHELL_REGISTRY_HPP

#include "scf/aot_shell_registry.hpp"

#include <array>
#include <cstddef>

namespace vibeqc::scf::generated {{

struct CompiledKernelMetadata {{
  const char* target_architecture;
  const char* name;
  unsigned shell_class;
  unsigned block_threads;
  unsigned consumer_mask;
  unsigned component_tile;
}};

inline constexpr std::array<ProfileInfo, {len(profile_rows)}> kCompiledProfiles{{{{
{chr(10).join(profile_rows)}
}}}};
inline constexpr std::size_t kCompiledProfileCount =
    kCompiledProfiles.size();

inline constexpr std::array<CompiledKernelMetadata, {len(kernel_rows)}>
    kCompiledShellKernels{{{{
{chr(10).join(kernel_rows)}
}}}};
inline constexpr std::size_t kCompiledShellKernelCount =
    kCompiledShellKernels.size();

}}  // namespace vibeqc::scf::generated

#endif
"""


def _launch_parameter_declaration() -> str:
    return """unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* output, const std::uint32_t* task_count,
    std::uint32_t* task_head"""


def _resident_launch_parameter_declaration() -> str:
    """Return the stable resident-bra launch signature shared by emitters."""

    return """cudaStream_t stream, bool unrestricted,
    const void* resident_tasks, const void* ket_tasks,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* forces, unsigned block_threads,
    std::size_t task_count"""


def _streaming_fock_launch_parameter_declaration() -> str:
    """Return the fixed-storage Fock streaming dispatch signature."""

    return """unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* shell_pair_stream,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head"""


def _launch_argument_list() -> str:
    return """stream, unrestricted, worker_blocks, tasks, task_offset,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density, output,
          task_count, task_head"""


def _resident_launch_argument_list() -> str:
    """Return arguments for forwarding a resident-bra launch."""

    return """stream, unrestricted, resident_tasks, ket_tasks,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density,
          forces, block_threads, task_count"""


def _streaming_fock_launch_argument_list() -> str:
    """Return arguments for forwarding one streaming Fock launch."""

    return """stream, unrestricted, worker_blocks, shell_pair_stream,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density, fock,
          bra_head"""


def emit_multi_registry_source(
    profiles: Iterable[ResolvedProductionProfile],
) -> str:
    """Emit one per-device profile resolver and collision-free launch table."""

    items = tuple(profiles)
    declarations = []
    helpers = []
    kernel_arrays = []
    kernel_sets = []
    launch_parameters = _launch_parameter_declaration()
    launch_arguments = _launch_argument_list()
    streaming_fock_parameters = _streaming_fock_launch_parameter_declaration()
    streaming_fock_arguments = _streaming_fock_launch_argument_list()
    for index, profile in enumerate(items):
        identifier = _profile_identifier(profile.target.architecture)
        force_cases = []
        fock_cases = []
        streaming_fock_cases = []
        resident_symbol = None
        force_names = []
        fock_names = []
        force_mask = 0
        fock_mask = 0
        for selection in profile.selections:
            shell_class = shell_class_index(selection.spec)
            plan = build_fused_shell_plan(
                selection.spec,
                consumers=selection.consumers,
                schedule=selection.schedule,
                recurrence=selection.recurrence,
                target=profile.target,
            )
            consumer_mask = sum(
                1 << list(KernelConsumer).index(consumer)
                for consumer in selection.consumers
            )
            force_symbol = f"vibeqc_launch_{identifier}_generated_{selection.spec.name}"
            declarations.append(
                f'extern "C" cudaError_t {force_symbol}('
                "cudaStream_t, bool, unsigned, const void*, const std::uint32_t*, "
                "const std::int64_t*, const void*, const double*, const void*, "
                "double, const double*, const double*, double*, "
                "const std::uint32_t*, std::uint32_t*);"
            )
            force_cases.append(
                f"    case {shell_class}U:\n"
                f"      return {force_symbol}({launch_arguments});"
            )
            # Fock-only rows deliberately retain a dormant force symbol: the
            # bounded spd path can call the same exact recurrence without a
            # generic integral fallback.  Keep those symbols out of ordinary
            # force metadata so fixed-topology plans do not reserve descriptor
            # storage for classes owned by their handwritten force routes.
            if KernelConsumer.FORCE in selection.consumers:
                force_names.append(
                    f'    {{"{selection.spec.name}", {shell_class}U, '
                    f"{sum(selection.spec.angular)}U, {plan.block_threads}U, "
                    f"{consumer_mask}U, {plan.schedule.component_tile}U}},"
                )
                force_mask |= 1 << shell_class
            if KernelConsumer.FOCK in selection.consumers:
                fock_symbol = f"{force_symbol}_fock"
                declarations.append(
                    f'extern "C" cudaError_t {fock_symbol}('
                    "cudaStream_t, bool, unsigned, const void*, "
                    "const std::uint32_t*, const std::int64_t*, const void*, "
                    "const double*, const void*, double, const double*, "
                    "const double*, double*, const std::uint32_t*, "
                    "std::uint32_t*);"
                )
                fock_cases.append(
                    f"    case {shell_class}U:\n"
                    f"      return {fock_symbol}({launch_arguments});"
                )
                if selection.spec.name in _STREAMING_FOCK_SHELLS:
                    streaming_fock_symbol = f"{force_symbol}_streaming_fock"
                    declarations.append(
                        f'extern "C" cudaError_t {streaming_fock_symbol}('
                        "cudaStream_t, bool, unsigned, const void*, "
                        "const std::int64_t*, const void*, const double*, "
                        "const void*, double, const double*, const double*, "
                        "double*, std::uint32_t*);"
                    )
                    streaming_fock_cases.append(
                        f"    case {shell_class}U:\n"
                        f"      return {streaming_fock_symbol}("
                        f"{streaming_fock_arguments});"
                    )
                fock_names.append(
                    f'    {{"{selection.spec.name}", {shell_class}U, '
                    f"{sum(selection.spec.angular)}U, {plan.block_threads}U, "
                    f"{consumer_mask}U, {plan.schedule.component_tile}U}},"
                )
                fock_mask |= 1 << shell_class
            if selection.resident_force_recurrence is not None:
                resident_symbol = f"vibeqc_launch_{identifier}_ppps_resident"
                declarations.append(
                    f'extern "C" cudaError_t {resident_symbol}('
                    f"{_resident_launch_parameter_declaration()});"
                )
        helpers.append(
            f"""cudaError_t launch_{identifier}_force({launch_parameters}) noexcept {{
  switch (shell_class) {{
{chr(10).join(force_cases)}
    default: return cudaErrorInvalidValue;
  }}
}}

cudaError_t launch_{identifier}_fock({launch_parameters}) noexcept {{
  switch (shell_class) {{
{chr(10).join(fock_cases)}
    default: return cudaErrorInvalidValue;
  }}
}}

cudaError_t launch_{identifier}_streaming_fock(
    {streaming_fock_parameters}) noexcept {{
  switch (shell_class) {{
{chr(10).join(streaming_fock_cases)}
    default: return cudaErrorNotSupported;
  }}
}}

cudaError_t launch_{identifier}_resident(
    {_resident_launch_parameter_declaration()}) noexcept {{
  """
            + (
                f"return {resident_symbol}({_resident_launch_argument_list()});"
                if resident_symbol is not None
                else "return cudaErrorNotSupported;"
            )
            + """
}
"""
        )
        kernel_arrays.append(
            f"""constexpr std::array<ShellKernelMetadata, {len(force_names)}> kForceNames{index}{{{{
{chr(10).join(force_names)}
}}}};
constexpr std::array<ShellKernelMetadata, {len(fock_names)}> kFockNames{index}{{{{
{chr(10).join(fock_names)}
}}}};
"""
        )
        kernel_sets.append(
            f"""    {{kCompiledProfiles[{index}], UINT64_C({force_mask}),
      UINT64_C({fock_mask}), kForceNames{index}.data(), kForceNames{index}.size(),
      kFockNames{index}.data(), kFockNames{index}.size(),
      launch_{identifier}_force, launch_{identifier}_fock,
      launch_{identifier}_streaming_fock,
      launch_{identifier}_resident}},"""
        )
    return f"""#include "vibeqc_generated_shell_registry.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>

{chr(10).join(declarations)}

namespace vibeqc::scf::generated {{
namespace {{

using LaunchFunction = cudaError_t (*)({_launch_parameter_declaration()}) noexcept;
using StreamingFockLaunchFunction = cudaError_t (*)(
    {_streaming_fock_launch_parameter_declaration()}) noexcept;
using ResidentLaunchFunction = cudaError_t (*)({_resident_launch_parameter_declaration()}) noexcept;

struct KernelSet {{
  ProfileInfo info;
  std::uint64_t force_mask;
  std::uint64_t fock_mask;
  const ShellKernelMetadata* force_names;
  std::size_t force_name_count;
  const ShellKernelMetadata* fock_names;
  std::size_t fock_name_count;
  LaunchFunction launch_force;
  LaunchFunction launch_fock;
  StreamingFockLaunchFunction launch_streaming_fock;
  ResidentLaunchFunction launch_resident;
}};

{chr(10).join(helpers)}
{chr(10).join(kernel_arrays)}

constexpr std::array<KernelSet, {len(items)}> kKernelSets{{{{
{chr(10).join(kernel_sets)}
}}}};

constexpr ProfileInfo kGenericProfile{{
    "generic_cuda", "portable_cuda", 0, 0, false, true, false}};

constexpr std::size_t kMaximumCachedDevices = 128;
std::array<const KernelSet*, kMaximumCachedDevices> selected_by_device{{}};
std::mutex selection_mutex;

bool selected(const char* list, const char* name) noexcept {{
  const std::size_t name_size = std::strlen(name);
  const char* cursor = list;
  while (*cursor != '\\0') {{
    while (*cursor == ',' || *cursor == ';' || *cursor == ' ' ||
           *cursor == '\t') ++cursor;
    const char* begin = cursor;
    while (*cursor != '\\0' && *cursor != ',' && *cursor != ';' &&
           *cursor != ' ' && *cursor != '\t') ++cursor;
    if (static_cast<std::size_t>(cursor - begin) == name_size &&
        std::strncmp(begin, name, name_size) == 0) return true;
  }}
  return false;
}}

const KernelSet* resolve(int major, int minor) noexcept {{
  for (const KernelSet& kernels : kKernelSets) {{
    if (kernels.info.compute_capability_major == major &&
        kernels.info.compute_capability_minor == minor) return &kernels;
  }}
  return nullptr;
}}

const KernelSet* current_kernel_set() noexcept {{
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
      static_cast<std::size_t>(device) >= kMaximumCachedDevices) return nullptr;
  std::lock_guard<std::mutex> lock(selection_mutex);
  const KernelSet*& cached = selected_by_device[static_cast<std::size_t>(device)];
  if (cached == nullptr) {{
    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess) return nullptr;
    cached = resolve(major, minor);
  }}
  return cached;
}}

std::uint64_t environment_mask(
    const char* variable, std::uint64_t available,
    const ShellKernelMetadata* names, std::size_t count) noexcept {{
  const char* selection = std::getenv(variable);
  const bool all = selection == nullptr || *selection == '\\0' ||
                   std::strcmp(selection, "all") == 0;
  if (all) return available;
  if (std::strcmp(selection, "none") == 0) return 0;
  std::uint64_t mask = 0;
  for (std::size_t index = 0; index < count; ++index) {{
    if (selected(selection, names[index].name))
      mask |= std::uint64_t{{1}} << names[index].shell_class;
  }}
  return mask & available;
}}

}}  // namespace

void select_profile_for_device(int device_id, int major, int minor) noexcept {{
  if (device_id < 0 ||
      static_cast<std::size_t>(device_id) >= kMaximumCachedDevices) return;
  std::lock_guard<std::mutex> lock(selection_mutex);
  selected_by_device[static_cast<std::size_t>(device_id)] = resolve(major, minor);
}}

const ProfileInfo& selected_profile() noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? kGenericProfile : kernels->info;
}}

const ShellKernelMetadata* selected_shell_kernels(
    std::size_t& count) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  if (kernels == nullptr) {{
    count = 0;
    return nullptr;
  }}
  count = kernels->force_name_count;
  return kernels->force_names;
}}

const ShellKernelMetadata* selected_fock_shell_kernels(
    std::size_t& count) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  if (kernels == nullptr) {{
    count = 0;
    return nullptr;
  }}
  count = kernels->fock_name_count;
  return kernels->fock_names;
}}

std::uint64_t enabled_shell_class_mask() noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? 0 : environment_mask(
      "VIBEQC_AOT_SHELL_CLASSES", kernels->force_mask,
      kernels->force_names, kernels->force_name_count);
}}

std::uint64_t enabled_fock_shell_class_mask() noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? 0 : environment_mask(
      "VIBEQC_AOT_FOCK_SHELL_CLASSES", kernels->fock_mask,
      kernels->fock_names, kernels->fock_name_count);
}}

cudaError_t launch_shell_class({_launch_parameter_declaration()}) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? cudaErrorInvalidValue : kernels->launch_force(
      shell_class, {launch_arguments});
}}

cudaError_t launch_shell_class_fock({_launch_parameter_declaration()}) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? cudaErrorInvalidValue : kernels->launch_fock(
      shell_class, {launch_arguments});
}}

cudaError_t launch_shell_class_streaming_fock(
    {_streaming_fock_launch_parameter_declaration()}) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? cudaErrorInvalidValue
                            : kernels->launch_streaming_fock(
      shell_class, {streaming_fock_arguments});
}}

cudaError_t launch_ppps_resident(
    {_resident_launch_parameter_declaration()}) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  if (kernels == nullptr) return cudaErrorInvalidValue;
  if (kernels->launch_resident == nullptr) return cudaErrorNotSupported;
  return kernels->launch_resident({_resident_launch_argument_list()});
}}

}}  // namespace vibeqc::scf::generated
"""


def write_production_bundles(
    manifest: Path,
    output_directory: Path,
    shard_count: int,
    architectures: Sequence[str],
    profile_by_architecture: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Write independent, namespaced AOT bundles and one runtime registry."""

    if shard_count < 1:
        raise ValueError("production shard count must be positive")
    normalized = tuple(
        sorted(
            {normalize_cuda_architecture(item) for item in architectures},
            key=lambda item: int(item.removeprefix("sm_")),
        )
    )
    if not normalized:
        raise ValueError("at least one CUDA architecture is required")
    requested = {
        normalize_cuda_architecture(key): value
        for key, value in (profile_by_architecture or {}).items()
    }
    profiles = tuple(
        resolve_production_profile(
            manifest,
            architecture,
            requested.get(architecture, "auto"),
        )
        for architecture in normalized
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for profile in profiles:
        identifier = _profile_identifier(profile.target.architecture)
        profile_directory = output_directory / profile.target.architecture
        profile_directory.mkdir(parents=True, exist_ok=True)
        shards = _partition_production_selections(profile.selections, shard_count)
        for index, shard in enumerate(shards):
            path = profile_directory / (
                f"vibeqc_generated_shell_{identifier}_shard_{index}.cu"
            )
            _write_if_changed(path, emit_profile_shard(profile, shard))
            outputs.append(path)
    header = output_directory / "vibeqc_generated_shell_registry.hpp"
    source = output_directory / "vibeqc_generated_shell_registry.cu"
    _write_if_changed(header, emit_multi_registry_header(profiles))
    _write_if_changed(source, emit_multi_registry_source(profiles))
    outputs.extend((header, source))
    return tuple(outputs)


def _write_if_changed(path: Path, content: str) -> None:
    """Preserve timestamps when deterministic regeneration is byte-identical."""

    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _partition_production_selections(
    selections: Iterable[KernelSelection], shard_count: int
) -> tuple[tuple[KernelSelection, ...], ...]:
    """Assign shell classes to stable translation units.

    The production shell-class index is independent of manifest membership and
    ordering.  Using it as the shard key prevents an inserted or temporarily
    disabled class from moving unrelated kernels to new CUDA source files and
    invalidating their compiler-cache entries.
    """

    shards: list[list[KernelSelection]] = [[] for _ in range(shard_count)]
    for selection in selections:
        shards[shell_class_index(selection.spec) % shard_count].append(selection)
    return tuple(tuple(shard) for shard in shards)


def write_production_bundle(
    manifest: Path,
    output_directory: Path,
    shard_count: int,
    architecture: str | None = None,
    profile: str = "auto",
) -> tuple[Path, ...]:
    """Write deterministic build artifacts and return every generated path."""

    if shard_count < 1:
        raise ValueError("production shard count must be positive")
    selections = load_production_kernel_selections(manifest, architecture, profile)
    shards = _partition_production_selections(selections, shard_count)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, shard in enumerate(shards):
        path = output_directory / f"vibeqc_generated_shell_shard_{index}.cu"
        _write_if_changed(path, emit_production_shard(shard))
        outputs.append(path)
    header = output_directory / "vibeqc_generated_shell_registry.hpp"
    source = output_directory / "vibeqc_generated_shell_registry.cu"
    _write_if_changed(header, emit_registry_header(selections))
    _write_if_changed(source, emit_registry_source(selections))
    outputs.extend((header, source))
    return tuple(outputs)

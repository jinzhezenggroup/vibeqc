"""Production kernel-selection contracts and scientific capability validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .capabilities import normalize_capabilities
from .cuda_lowering import supports_component_lane_rys
from .cuda_schedule import ScheduleIR, ScheduleKind
from .cuda_target import cuda_target_info
from .fused_schedule import build_fused_shell_plan
from .ir import IntegralIR, KernelConsumer, build_integral_ir
from .specialize import specialize_integral_ir

if TYPE_CHECKING:
    from .shell_spec import ShellClassSpec

_SUPPORTED_RECURRENCES = frozenset(("subset_wick", "rys2", "rys3", "rys4", "rys5"))


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
        and schedule.block_threads == schedule.warp_size
        and schedule.tasks_per_warp == schedule.warp_size
        and not schedule.shared_coulomb
        and schedule.component_tile >= spec.component_count
    )


def _supports_component_lane_rys(
    spec: ShellClassSpec,
    schedule: ScheduleIR,
) -> bool:
    """Return whether the backend decoder can lower ``spec`` component-wise."""

    return supports_component_lane_rys(spec, schedule)


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
    # Optional production code-shape capabilities are measured per profile
    # and persisted in the manifest.  An omitted list intentionally disables
    # optional wrappers for legacy/custom manifests.
    capabilities: frozenset[str] = frozenset()
    # Keep the mathematical request attached to a production selection.  The
    # manifest compatibility path leaves this unset and receives the
    # historical default IR, while compiler stages may supply a fully
    # explicit operator/derivative/contraction definition.
    integral: IntegralIR | None = None
    # Optional compiler provenance carried by newer manifests.  These values
    # are advisory and fall back to the structural cost model when absent.
    runtime_seconds: float | None = None
    compile_seconds: float | None = None
    source_bytes: int | None = None
    object_bytes: int | None = None

    def has_capability(self, capability: str) -> bool:
        """Return whether this profile explicitly enables one optional path."""

        return capability in self.capabilities

    def __post_init__(self) -> None:
        if not self.architecture.startswith("sm_"):
            raise ValueError("production architecture must use CUDA sm_ notation")
        if not self.profile:
            object.__setattr__(self, "profile", self.architecture)
        if not self.consumers:
            raise ValueError("production kernel requires at least one consumer")
        object.__setattr__(
            self,
            "capabilities",
            normalize_capabilities(self.spec.name, list(self.capabilities)),
        )
        if (
            not isinstance(self.recurrence, str)
            or self.recurrence not in _SUPPORTED_RECURRENCES
        ):
            raise ValueError(f"unsupported production recurrence {self.recurrence!r}")
        # IntegralIR owns scientific recurrence legality, including the exact
        # root count implied by angular momentum and derivative order. The
        # production layer only validates whether an implemented CUDA mapping
        # can execute that already-legal recurrence.  Preserve an explicit IR
        # instead of silently rebuilding the historical default.
        selected_integral = self.integral or build_integral_ir(
            self.spec,
            self.consumers,
            recurrence=self.recurrence,
        )
        if selected_integral.spec != self.spec:
            raise ValueError("production integral spec does not match selection")
        if selected_integral.recurrence != self.recurrence:
            raise ValueError(
                "production recurrence does not match the selection integral"
            )
        if selected_integral.consumers != frozenset(self.consumers):
            raise ValueError("production consumers do not match the selection integral")
        for field_name in (
            "runtime_seconds",
            "compile_seconds",
            "source_bytes",
            "object_bytes",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative number")
        scalar_thread_tasks = _supports_scalar_rys(self.spec, self.schedule)
        if (
            selected_integral.recurrence.startswith("rys")
            and selected_integral.required_rys_roots == 2
            and not scalar_thread_tasks
        ):
            raise ValueError(
                "production two-root Rys lowering requires one complete scalar "
                "task per lane in a single target warp"
            )
        if (
            scalar_thread_tasks
            and selected_integral.recurrence.startswith("rys")
            and KernelConsumer.FOCK in self.consumers
            and self.fock_schedule is None
        ):
            raise ValueError(
                "production scalar Rys force with a Fock consumer requires an "
                "independent fock_schedule"
            )
        if self.recurrence == "rys3":
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
        high_root_uniform_warps = _supports_uniform_warp_rys(self.spec, self.schedule)
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


def _selection_integral(
    selection: KernelSelection,
    *,
    consumers: tuple[KernelConsumer | str, ...] | None = None,
    recurrence: str | None = None,
) -> IntegralIR:
    """Return the mathematical IR owned by a production selection.

    Production manifests predate explicit operator metadata and therefore
    continue to synthesize the canonical four-center IR when ``integral`` is
    absent.  When a compiler stage supplies an IR, preserve its operator,
    derivative, and contraction records; only rebuild when a companion path
    intentionally changes consumers or recurrence (for example a Fock value
    plan beside a force Rys plan).
    """

    selected_recurrence = selection.recurrence if recurrence is None else recurrence
    base = selection.integral
    if base is None:
        selected_consumers = selection.consumers if consumers is None else consumers
        return build_integral_ir(
            selection.spec,
            selected_consumers,
            recurrence=selected_recurrence,
        )
    if consumers is None:
        selected_consumers = selection.consumers
    else:
        selected_consumers = tuple(KernelConsumer(item) for item in consumers)
    if (
        frozenset(selected_consumers) == base.consumers
        and selected_recurrence == base.recurrence
    ):
        return base
    if frozenset(selected_consumers) == base.consumers:
        return specialize_integral_ir(base, recurrence=selected_recurrence)
    return specialize_integral_ir(
        base,
        consumers=selected_consumers,
        recurrence=recurrence,
    )


def _as_selection(item: ShellClassSpec | KernelSelection) -> KernelSelection:
    """Normalize compatibility callers to the explicit production IR."""

    if isinstance(item, KernelSelection):
        return item
    # Historical bare-spec compatibility is intentionally isolated here.
    # New production paths resolve an explicit profile/architecture first.
    legacy_target = cuda_target_info("sm_120")
    plan = build_fused_shell_plan(item, target=legacy_target)
    return KernelSelection(
        architecture=legacy_target.architecture,
        spec=item,
        consumers=(KernelConsumer.FORCE,),
        schedule=plan.schedule,
    )

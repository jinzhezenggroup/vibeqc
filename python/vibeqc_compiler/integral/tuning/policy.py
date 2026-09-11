"""Typed schedule candidates, enumeration and production-profile eligibility.

This layer defines which candidates may be measured. Process execution and
record publication remain outside candidate construction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from ..cuda_schedule import (
    ScheduleIR,
    ScheduleKind,
    tuning_schedule_candidates,
)
from ..cuda_target import (
    DEFAULT_CUDA_TARGET,
    CudaTargetInfo,
)
from ..ir import IntegralIR, KernelConsumer, build_integral_ir
from ..production import load_production_kernel_selections
from ..shell_spec import ShellClassSpec
from .analysis import StaticAlgebraModel, _integral_signature, static_algebra_model
from .shared import _PRODUCTION_MANIFEST_PATH


@dataclass(frozen=True, slots=True)
class ScheduleTrial:
    """One shell class and one concrete schedule compiled as a unit."""

    spec: ShellClassSpec
    schedule: ScheduleIR
    consumer: KernelConsumer = KernelConsumer.FORCE
    target: CudaTargetInfo = DEFAULT_CUDA_TARGET
    # An explicit IR is optional for compatibility with the legacy CLI, but
    # when present it must remain the source of mathematical intent throughout
    # model construction and CUDA benchmark emission.
    integral: IntegralIR | None = None

    @property
    def schedule_id(self) -> str:
        """Return a stable identifier containing every tuned code-shape knob."""

        shared = "shared" if self.schedule.shared_coulomb else "recomputed"
        unroll = "unrolled" if self.schedule.unroll_pair_terms else "rolled"
        return "_".join(
            (
                self.schedule.kind.value,
                f"b{self.schedule.block_threads}",
                f"t{self.schedule.component_tile}",
                f"w{self.schedule.tasks_per_warp}",
                f"o{self.schedule.minimum_blocks_per_sm}",
                f"r{self.schedule.maximum_registers}",
                shared,
                unroll,
                self.schedule.pair_orientation.value,
                f"pairs_{self.schedule.pair_storage.value}",
                self.schedule.algebra_placement.value,
                self.schedule.algebra_ordering.value,
                self.schedule.algebra_fusion.value,
                self.schedule.algebra_form.value,
            )
        )

    @property
    def integral_suffix(self) -> str:
        """Return a stable symbol suffix for explicitly supplied IRs."""

        if self.integral is None:
            return ""
        return f"_ir{_integral_signature(self.integral)}"

    @property
    def key(self) -> str:
        """Return the cross-class report key for this trial."""

        return (
            f"{self.spec.name}:{self.consumer.value}:{self.schedule_id}"
            f"{self.integral_suffix}"
        )

    @property
    def entry_point(self) -> str:
        """Return the unique host entry used by the batch driver."""

        return (
            f"vibeqc_run_schedule_{self.spec.name}_{self.consumer.value}_"
            f"{self.schedule_id}{self.integral_suffix}"
        )

    @property
    def symbol_prefix(self) -> str:
        """Return the unique lower-case CUDA symbol prefix."""

        return (
            f"generated_{self.spec.name}_{self.consumer.value}_"
            f"{self.schedule_id}{self.integral_suffix}"
        )

    @property
    def static_model(self) -> StaticAlgebraModel:
        """Return the cached symbolic resource model for this schedule."""

        return static_algebra_model(self)


def schedule_payload(schedule: ScheduleIR) -> dict[str, object]:
    """Serialize all schedule decisions written to a v2 manifest."""

    return {
        "kind": schedule.kind.value,
        "block_threads": schedule.block_threads,
        "component_tile": schedule.component_tile,
        "tasks_per_warp": schedule.tasks_per_warp,
        "shared_coulomb": schedule.shared_coulomb,
        "pair_orientation": schedule.pair_orientation.value,
        "pair_storage": schedule.pair_storage.value,
        "algebra_placement": schedule.algebra_placement.value,
        "algebra_ordering": schedule.algebra_ordering.value,
        "algebra_fusion": schedule.algebra_fusion.value,
        "algebra_form": schedule.algebra_form.value,
        "unroll_pair_terms": schedule.unroll_pair_terms,
        "minimum_blocks_per_sm": schedule.minimum_blocks_per_sm,
        "maximum_registers": schedule.maximum_registers,
    }


@cache
def _production_fock_schedule_index(
    architecture: str,
) -> tuple[tuple[str, ScheduleIR], ...]:
    """Read explicit Fock baseline schedules from the production manifest.

    Generic schedule discovery intentionally avoids subgroup mappings for very
    large component envelopes.  A tuned manifest may still contain a
    hand-validated value-only Fock mapping for such a class.  Reusing that row
    keeps autotune comparisons honest without maintaining a second shell-name
    allowlist in Python.
    """

    selections = load_production_kernel_selections(
        _PRODUCTION_MANIFEST_PATH,
        architecture=architecture,
        profile="auto",
    )
    return tuple(
        (selection.spec.name, selection.fock_schedule)
        for selection in selections
        if KernelConsumer.FOCK in selection.consumers
        and selection.fock_schedule is not None
    )


def _known_production_fock_subgroup_schedules(
    spec: ShellClassSpec, target: CudaTargetInfo
) -> tuple[ScheduleIR, ...]:
    """Return manifest-declared Fock baselines absent from generic search."""

    schedule_by_name = dict(_production_fock_schedule_index(target.architecture))
    schedule = schedule_by_name.get(spec.name)
    if schedule is None:
        return ()
    schedule.validate_for(target)
    return (schedule,)


def supported_schedule_trials(
    spec: ShellClassSpec,
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
    target: CudaTargetInfo = DEFAULT_CUDA_TARGET,
    *,
    integral: IntegralIR | None = None,
) -> tuple[ScheduleTrial, ...]:
    """Return the schedule variants implemented by the current CUDA emitter.

    Fock trials retain the force companion because production uses one
    canonical task ABI, while timing and resource gates select only the
    requested consumer's kernels.
    """

    if any(order > 6 for order in spec.pair_orders):
        raise ValueError(f"{spec.name} is outside the current pair-order CUDA lowering")
    if any(order > 3 for order in spec.angular):
        raise ValueError(f"{spec.name} exceeds the current s/p/d/f CUDA lowering")
    selected_consumer = KernelConsumer(consumer)
    explicit_integral = integral
    if integral is None:
        consumers = (
            (KernelConsumer.FOCK, KernelConsumer.FORCE)
            if selected_consumer == KernelConsumer.FOCK
            else (KernelConsumer.FORCE,)
        )
        integral = build_integral_ir(spec, consumers)
    else:
        if integral.spec != spec:
            raise ValueError(
                "trial integral spec does not match its shell specification"
            )
        if selected_consumer not in integral.consumers:
            raise ValueError(
                f"{selected_consumer.value} trial requires its integral consumer"
            )
    schedules = [
        schedule
        for schedule in tuning_schedule_candidates(integral, target)
        if schedule.kind
        in (
            ScheduleKind.PACKED_TASKS,
            ScheduleKind.SUBGROUP_TASKS,
            ScheduleKind.SHELL_TASK,
            ScheduleKind.COMPONENT_LANES,
            ScheduleKind.TILED_COMPONENTS,
        )
    ]
    if selected_consumer == KernelConsumer.FOCK:
        schedules.extend(_known_production_fock_subgroup_schedules(spec, target))
    trials: list[ScheduleTrial] = []
    seen_schedule_ids: set[str] = set()
    for schedule in schedules:
        trial = ScheduleTrial(
            spec=spec,
            schedule=schedule,
            consumer=selected_consumer,
            target=target,
            integral=explicit_integral,
        )
        if trial.schedule_id in seen_schedule_ids:
            continue
        seen_schedule_ids.add(trial.schedule_id)
        trials.append(trial)
    return tuple(trials)

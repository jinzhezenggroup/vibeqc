"""Eligibility rules for fixed-root component-lane CUDA schedules.

Selection depends on the declared shell/operator capabilities. A source-capable
schedule is not thereby a measured or promoted production candidate."""

from __future__ import annotations

from ..cuda_schedule import (
    ScheduleIR,
    ScheduleKind,
)
from ..fused_schedule import (
    FusedShellPlan,
)
from ..ir import KernelConsumer
from ..shell_spec import (
    ShellClassSpec,
)


def supports_component_lane_rys(
    spec: ShellClassSpec,
    schedule: ScheduleIR,
) -> bool:
    """Return whether the runtime-indexed Rys decoder covers ``spec``.

    This is a backend capability, not a production promotion decision.  The
    decoder uses one lane per Cartesian component and currently has exact
    tables for s/p/d centers, with no more than a p shell on the fourth center.
    Keeping the predicate beside the lowering prevents manifest validation and
    source emission from silently growing different shell-class boundaries.
    """

    return (
        schedule.kind == ScheduleKind.COMPONENT_LANES
        and schedule.warp_size == 32
        and schedule.block_threads >= spec.component_count
        and schedule.component_tile >= spec.component_count
        and max(spec.angular) <= 2
        and spec.angular[3] <= 1
    )


def _supports_rys_component_lane_fock(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
) -> bool:
    """Return whether the runtime-indexed fixed-root Fock worker is legal.

    The value worker uses the force plan's fixed-root tables and recurrence
    state program, so legality is determined by the backend capabilities rather
    than by the set of classes that happened to be profiled first.  The
    runtime decoder currently has exact tables for s/p/d centers and allows at
    most a p shell on center four; the separate force contraction is required
    because ``IntegralIR`` derives Rys root counts from first-derivative order.
    """

    integral = plan.kernel.integral
    return (
        KernelConsumer.FORCE in integral.consumers
        and supports_component_lane_rys(spec, plan.schedule)
        and integral.recurrence in ("rys3", "rys4")
    )

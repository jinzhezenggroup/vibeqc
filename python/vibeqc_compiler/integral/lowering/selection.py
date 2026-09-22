"""Eligibility rules for fixed-root component-lane CUDA schedules.

Selection depends on the declared shell/operator capabilities. A source-capable
schedule is not thereby a measured or promoted production candidate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..cuda_schedule import (
    ScheduleIR,
    ScheduleKind,
)
from ..ir import KernelConsumer

if TYPE_CHECKING:
    from ..fused_schedule import (
        FusedShellPlan,
    )
    from ..ir import IntegralIR
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
    *,
    support_integral: IntegralIR | None = None,
) -> bool:
    """Return whether the runtime-indexed fixed-root Fock worker is legal.

    ``plan`` owns the requested Fock schedule.  A derivative-bearing
    ``support_integral`` may separately own the fixed-root tables and
    recurrence state program that the value worker reuses.  Keeping that
    lowering dependency explicit lets output pruning remove FORCE from the
    requested Fock plan without pretending the shared Rys support is dead.
    """

    integral = plan.kernel.integral if support_integral is None else support_integral
    return (
        KernelConsumer.FORCE in integral.consumers
        and supports_component_lane_rys(spec, plan.schedule)
        and integral.recurrence.startswith("rys")
        and integral.required_rys_roots in (3, 4)
    )

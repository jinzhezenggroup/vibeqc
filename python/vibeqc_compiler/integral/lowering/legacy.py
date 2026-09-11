"""Historical DPPP and resident-PPPS adapters to the general shell lowering.

These call the same implementation as generic users; no second scientific
kernel definition is maintained for the old entry-point names."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..cuda_schedule import (
    ScheduleIR,
    ScheduleKind,
)
from ..fused_schedule import (
    CoulombState,
    FusedShellPlan,
    build_fused_shell_plan,
    evaluate_fused_shell_component,
)
from ..ir import IntegralIR, build_integral_ir
from ..shell_spec import (
    DPPP_SPEC,
    FUSED_SHELL_SPEC_BY_NAME,
)
from .common import _specialize_dppp_identifiers
from .dispatch import emit_shell_class_fused_cuda
from .force_resident import _emit_ppps_resident_bra_rys3_force_consumer_cuda
from .shared import _COMPONENT_COUNT, DpppComponent


@dataclass(frozen=True, slots=True)
class DpppFusedPlan:
    """Static schedule and lookup tables for one fused ``dppp`` kernel."""

    components: tuple[DpppComponent, ...]
    coulomb_states: tuple[CoulombState, ...]
    coulomb_indices: tuple[int, ...]
    block_threads: int

    @property
    def warp_count(self) -> int:
        """Return the number of full warps used by one generated block."""

        return self.block_threads // 32


def dppp_components() -> tuple[DpppComponent, ...]:
    """Return all Cartesian components in production CCA ordering."""

    return DPPP_SPEC.components


def build_dppp_fused_plan() -> DpppFusedPlan:
    """Build the deterministic component and shared-Coulomb schedule.

    Cartesian derivative states are ordered by total degree and then by
    ``x/y/z`` degree. The dense 7x7x7 lookup table makes the generated hot
    loop a few integer operations plus one shared-memory load; invalid states
    retain ``-1`` so generator tests can audit the complete domain.
    """

    generic = build_fused_shell_plan(DPPP_SPEC)
    if len(generic.components) != _COMPONENT_COUNT:
        raise RuntimeError("dppp component schedule has an unexpected size")
    if len(generic.coulomb_states) != 84:
        raise RuntimeError("order-six Cartesian Coulomb schedule must have 84 states")
    return DpppFusedPlan(
        components=generic.components,
        coulomb_states=generic.coulomb_states,
        coulomb_indices=generic.coulomb_indices,
        block_threads=generic.block_threads,
    )


def evaluate_dppp_fused_component(
    component: DpppComponent,
    variables: Mapping[str, float],
) -> tuple[tuple[float, float, float], ...]:
    """Evaluate one component using the fused kernel's recurrence schedule.

    This host-side oracle deliberately mirrors the emitted CUDA loops rather
    than calling the symbolic expression graph. Comparing both independent
    lowerings for all 162 components catches table ordering, sign, and center
    mapping mistakes before a generated kernel is considered for production.
    """

    return evaluate_fused_shell_component(DPPP_SPEC, component, variables)


def emit_ppps_resident_bra_rys3_cuda(
    *,
    include_shared_definitions: bool = True,
    include_rys3_roots: bool = True,
    integral: IntegralIR | None = None,
) -> str:
    """Emit the scalar ``ppps`` resident-bra Rys3 worker.

    The generated kernel has a 256-thread launch bound and shared-memory
    capacity, while its task and bra-staging strides use the actual block
    dimension. Production can therefore compare 32/64/128/256-thread CTAs
    from one binary without changing scalar quartet ownership.

    ``integral`` carries explicit derivative-center and translation-recovery
    metadata into both the ordinary shared-definition prefix and the resident
    force consumer. Omitting it preserves the canonical four-center ERI
    default (centers 0, 1, and 2 independent; center 3 recovered).

    ``include_shared_definitions`` keeps the standalone correctness harness
    self-contained.  Production AOT shards already contain the ordinary ppps
    shell definitions, so they request only the resident-specific tail to
    avoid duplicate CUDA type/function definitions.  A subset/Wick ordinary
    ppps row also lacks the Rys evaluator; ``include_rys3_roots`` therefore
    controls that one additional shared dependency independently.
    """

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    selected_integral = integral or build_integral_ir(spec, recurrence="rys3")
    if selected_integral.spec != spec:
        raise ValueError("resident ppps integral spec does not match ppps")
    if selected_integral.recurrence != "rys3":
        raise ValueError("resident ppps lowering requires an rys3 integral")
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
    )
    plan = build_fused_shell_plan(spec, integral=selected_integral, schedule=schedule)
    resident_tail = _emit_ppps_resident_bra_rys3_force_consumer_cuda(
        include_rys3_roots=include_rys3_roots,
        integral=selected_integral,
    )
    if include_shared_definitions:
        source = emit_shell_class_fused_cuda(spec, plan)
        # The existing Rys thread consumer starts with the attributed roots
        # table; cut before that table so the standalone source does not
        # retain an unused 27-entry shared weight helper from the one-task
        # prototype.
        marker = """/*
 * Three-root interpolation adapted from GPU4PySCF."""
        marker_index = source.find(marker)
        if marker_index < 0:
            raise RuntimeError("generated force task marker changed unexpectedly")
        prefix = source[:marker_index]
    else:
        prefix = ""
        # The resident tail references the normal generated ppps task/cache
        # helpers.  The production shard places it directly after the normal
        # ppps emitter, where those definitions are already available.
    return prefix + _specialize_dppp_identifiers(resident_tail, spec)


def emit_dppp_fused_cuda(plan: DpppFusedPlan | None = None) -> str:
    """Emit the production-golden dppp specialization of the generic emitter."""

    if plan is None:
        generic = build_fused_shell_plan(DPPP_SPEC)
    else:
        generic = FusedShellPlan(
            kernel=build_fused_shell_plan(DPPP_SPEC).kernel,
            spec=DPPP_SPEC,
            components=plan.components,
            coulomb_states=plan.coulomb_states,
            coulomb_indices=plan.coulomb_indices,
            block_threads=plan.block_threads,
        )
    return emit_shell_class_fused_cuda(DPPP_SPEC, generic)

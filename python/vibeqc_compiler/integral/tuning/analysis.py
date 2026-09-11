"""Static algebra and liveness estimates for explicit schedule candidates.

These are model estimates, not measured compiler/device resources. Enumeration
policy consumes the model; analysis has no runtime dependency on that policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from itertools import product
from math import comb
from typing import TYPE_CHECKING

from ..cuda_schedule import (
    AlgebraForm,
    AlgebraFusion,
    AlgebraOrdering,
    AlgebraPlacement,
    ScheduleKind,
)
from ..expr import Expr, PowerLowering
from ..ir import IntegralIR, KernelConsumer, build_integral_ir
from ..shell_class import (
    ShellClassContractionKernel,
    WeightedShellContractionKernel,
    build_packed_force_geometry_algebra,
    build_shell_class_contraction_kernel,
    build_weighted_shell_contraction_kernel,
)
from ..shell_spec import AXES, ShellClassSpec

if TYPE_CHECKING:
    # Candidate types annotate analysis without importing enumeration policy.
    from .policy import ScheduleTrial


@dataclass(frozen=True, slots=True)
class StaticAlgebraModel:
    """Schedule-relevant symbolic operation and live-value estimates.

    Packed-force rows include both the weighted contraction DAG and the fixed
    backend-neutral geometry setup.  Other consumers retain their existing
    component-envelope semantics because they do not emit that geometry path.
    """

    scope: str
    algebra_placement: AlgebraPlacement
    algebra_ordering: AlgebraOrdering
    algebra_fusion: AlgebraFusion
    algebra_form: AlgebraForm
    component_count: int
    sampled_component_count: int
    recurrence_state_count: int
    root_count: int
    operation_counts: tuple[tuple[str, int], ...]
    emitted_operation_counts: tuple[tuple[str, int], ...]
    baseline_arithmetic_operation_count: int
    baseline_materialized_value_count: int
    baseline_peak_live_values: int
    arithmetic_operation_count: int
    materialized_value_count: int
    peak_live_values: int
    rematerialized_value_count: int
    reordered_value_count: int
    fma_operation_count: int

    def to_payload(self) -> dict[str, object]:
        """Serialize deterministic fields into one autotuning candidate row."""

        return {
            "scope": self.scope,
            "algebra_placement": self.algebra_placement.value,
            "algebra_ordering": self.algebra_ordering.value,
            "algebra_fusion": self.algebra_fusion.value,
            "algebra_form": self.algebra_form.value,
            "component_count": self.component_count,
            "sampled_component_count": self.sampled_component_count,
            "recurrence_state_count": self.recurrence_state_count,
            "root_count": self.root_count,
            "operation_counts": dict(self.operation_counts),
            "arithmetic_operation_count": self.arithmetic_operation_count,
            "materialized_value_count": self.materialized_value_count,
            "estimated_peak_live_values": self.peak_live_values,
            "pre_optimization": {
                "operation_counts": {
                    operation: count
                    for operation, count in self.operation_counts
                    if operation not in ("constant", "variable")
                },
                "arithmetic_operation_count": (
                    self.baseline_arithmetic_operation_count
                ),
                "materialized_value_count": (self.baseline_materialized_value_count),
                "estimated_peak_live_values": self.baseline_peak_live_values,
            },
            "post_optimization": {
                "operation_counts": dict(self.emitted_operation_counts),
                "arithmetic_operation_count": self.arithmetic_operation_count,
                "materialized_value_count": self.materialized_value_count,
                "estimated_peak_live_values": self.peak_live_values,
            },
            "inlined_value_count": (
                self.baseline_materialized_value_count - self.materialized_value_count
            ),
            "rematerialized_value_count": self.rematerialized_value_count,
            "reordered_value_count": self.reordered_value_count,
            "fma_operation_count": self.fma_operation_count,
        }


def _integral_signature(integral: IntegralIR) -> str:
    """Return a deterministic short identifier for mathematical IR intent.

    Schedule IDs intentionally describe only execution knobs.  Explicit
    operator/recovery choices must nevertheless participate in trial identity,
    otherwise two valid IRs can overwrite each other's oracle rows or CUDA
    symbols.  Serialize enum and set-like fields canonically before hashing so
    the suffix is stable across Python processes.
    """

    def coordinates(parameters) -> str | list[int]:
        selected = parameters.centers
        return selected if isinstance(selected, str) else list(selected)

    def invariants(items) -> list[dict[str, object]]:
        return [
            {
                "parameters": coordinates(item.parameters),
                "dependent_center": item.dependent_center,
            }
            for item in items
        ]

    derivative = None
    if integral.derivative is not None:
        derivative = {
            "order": integral.derivative.order,
            "parameters": coordinates(integral.derivative.parameters),
            "invariants": invariants(integral.derivative.invariants),
        }
    payload = {
        "spec": {
            "name": integral.spec.name,
            "angular": list(integral.spec.angular),
        },
        "operator": {
            "family": integral.operator.family.value,
            "centers": list(integral.operator.centers),
            "invariants": invariants(integral.operator.invariants),
        },
        "derivative": derivative,
        "contractions": [
            {
                "consumer": item.consumer.value,
                "density": sorted(model.value for model in item.density),
                "output": item.output.value,
            }
            for item in sorted(
                integral.contractions,
                key=lambda item: item.consumer.value,
            )
        ],
        "recurrence": integral.recurrence,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _balanced_component_labels(spec: ShellClassSpec) -> tuple[tuple[str, ...], ...]:
    """Return axis-balanced labels for a compact resource-stressing sample.

    Repeated-axis Cartesian labels create many rotationally equivalent DAGs.
    Keeping the labels with the smallest axis-population spread retains all
    p orientations, mixed d orientations, and the fully mixed f orientation.
    Their Cartesian product is at most 81 components for a four-center class.
    """

    selected = []
    for labels in spec.center_components:
        spreads = {
            label: max(label.count(axis) for axis in AXES)
            - min(label.count(axis) for axis in AXES)
            for label in labels
        }
        minimum_spread = min(spreads.values())
        selected.append(
            tuple(label for label in labels if spreads[label] == minimum_spread)
        )
    return tuple(selected)


def _analysis_roots(
    kernel: ShellClassContractionKernel | WeightedShellContractionKernel,
    consumer: KernelConsumer,
    *,
    integral: IntegralIR | None = None,
) -> tuple[Expr, ...]:
    """Select value or declared independent-force roots from a symbolic kernel.

    The symbolic shell kernel retains all four center gradients so it can be
    used as a correctness oracle.  Static force models, however, should count
    only the centers that the owning mathematical IR evaluates; recovered
    centers are reconstructed by the consumer and must not inflate the model.
    """

    if consumer == KernelConsumer.FOCK:
        if not isinstance(kernel, ShellClassContractionKernel):
            raise TypeError("Fock algebra analysis requires a component kernel")
        return (kernel.variables["prefactor"] * kernel.value,)
    selected = integral or build_integral_ir(
        kernel.spec,
        consumers=(consumer,),
    )
    if selected.spec != kernel.spec:
        raise ValueError("analysis integral spec does not match its symbolic kernel")
    return tuple(
        kernel.gradients[center][coordinate]
        for center in selected.independent_derivative_centers
        for coordinate in range(3)
    )


@cache
def _packed_force_geometry_analysis(pair_shift_rows: int):
    """Return static metrics for the geometry setup emitted by packed force.

    Geometry is lowered with a fixed binary/small-integer form in the CUDA
    setup helper.  Keeping this analysis separate from the schedule-dependent
    weighted contraction lets the tuner report both costs without pretending
    that geometry placement is already an execution-policy knob.
    """

    if pair_shift_rows not in (3, 4):
        raise ValueError("packed geometry analysis requires three or four shifts")
    geometry = build_packed_force_geometry_algebra()
    roots = geometry.roots_for_pair_shift_rows(pair_shift_rows)
    graph, roots = geometry.graph.apply_algebra_form(
        roots,
        AlgebraForm.BINARY,
        PowerLowering.SMALL_INTEGER,
    )
    return (
        graph.analyze_ssa(roots),
        graph.materialization_plan(roots),
    )


@cache
def _cached_static_algebra_model(
    spec: ShellClassSpec,
    consumer: KernelConsumer,
    integral: IntegralIR,
    weighted_shell: bool,
    algebra_placement: AlgebraPlacement,
    algebra_ordering: AlgebraOrdering,
    algebra_fusion: AlgebraFusion,
    algebra_form: AlgebraForm,
) -> StaticAlgebraModel:
    """Build one immutable model shared by same-class schedule variants."""

    if weighted_shell:
        kernel = build_weighted_shell_contraction_kernel(spec, integral=integral)
        graph, roots = kernel.graph.apply_algebra_form(
            _analysis_roots(kernel, consumer, integral=integral),
            algebra_form,
        )
        analysis_pairs = [
            (
                graph.analyze_ssa(roots),
                graph.materialization_plan(
                    roots,
                    algebra_placement.materialization_policy(),
                    algebra_ordering,
                    algebra_fusion,
                ),
            ),
            _packed_force_geometry_analysis(4 if spec.angular[3] != 0 else 3),
        ]
        scope = "weighted_shell_dag"
        sampled_component_count = spec.component_count
    else:
        components = tuple(product(*_balanced_component_labels(spec)))
        # Algebra normalization can return a new graph; its roots must be
        # analyzed by that owner, just as in the weighted-shell branch.
        analysis_pairs = (
            (
                graph.analyze_ssa(roots),
                graph.materialization_plan(
                    roots,
                    algebra_placement.materialization_policy(),
                    algebra_ordering,
                    algebra_fusion,
                ),
            )
            for component in components
            for component_kernel in (
                build_shell_class_contraction_kernel(
                    spec,
                    component,
                    integral=integral,
                ),
            )
            for binary_roots in (
                _analysis_roots(component_kernel, consumer, integral=integral),
            )
            for graph, roots in (
                component_kernel.graph.apply_algebra_form(
                    binary_roots,
                    algebra_form,
                ),
            )
        )
        scope = "balanced_component_sample_envelope"
        sampled_component_count = len(components)

    operation_envelope: dict[str, int] = {}
    emitted_operation_envelope: dict[str, int] = {}
    root_count = 0
    baseline_arithmetic_operation_count = 0
    baseline_materialized_value_count = 0
    baseline_peak_live_values = 0
    arithmetic_operation_count = 0
    materialized_value_count = 0
    peak_live_values = 0
    rematerialized_value_count = 0
    reordered_value_count = 0
    fma_operation_count = 0
    for pair_index, (analysis, plan) in enumerate(analysis_pairs):
        if pair_index == 0:
            # ``root_count`` describes the schedule's consumer roots.  The
            # geometry setup is accounted for in the envelopes below, but is
            # not another consumer output tensor.
            root_count = analysis.root_count
        combine = weighted_shell
        if combine:
            baseline_arithmetic_operation_count += analysis.arithmetic_operation_count
            baseline_materialized_value_count += analysis.materialized_value_count
            baseline_peak_live_values = max(
                baseline_peak_live_values,
                analysis.peak_live_values,
            )
            arithmetic_operation_count += plan.arithmetic_operation_count
            materialized_value_count += plan.materialized_value_count
            peak_live_values = max(peak_live_values, plan.peak_live_values)
        else:
            baseline_arithmetic_operation_count = max(
                baseline_arithmetic_operation_count,
                analysis.arithmetic_operation_count,
            )
            baseline_materialized_value_count = max(
                baseline_materialized_value_count,
                analysis.materialized_value_count,
            )
            baseline_peak_live_values = max(
                baseline_peak_live_values,
                analysis.peak_live_values,
            )
            arithmetic_operation_count = max(
                arithmetic_operation_count,
                plan.arithmetic_operation_count,
            )
            materialized_value_count = max(
                materialized_value_count, plan.materialized_value_count
            )
            peak_live_values = max(peak_live_values, plan.peak_live_values)
        rematerialized_value_count = max(
            rematerialized_value_count,
            plan.rematerialized_value_count,
        )
        reordered_value_count = max(
            reordered_value_count,
            plan.reordered_value_count,
        )
        fma_operation_count = max(
            fma_operation_count,
            plan.fma_operation_count,
        )
        for operation, count in analysis.operation_counts:
            operation_envelope[operation] = (
                operation_envelope.get(operation, 0) + count
                if combine
                else max(operation_envelope.get(operation, 0), count)
            )
        for operation, count in plan.operation_counts:
            emitted_operation_envelope[operation] = (
                emitted_operation_envelope.get(operation, 0) + count
                if combine
                else max(emitted_operation_envelope.get(operation, 0), count)
            )
    operation_counts = tuple(sorted(operation_envelope.items()))
    emitted_operation_counts = tuple(sorted(emitted_operation_envelope.items()))
    # The recurrence envelope belongs to the mathematical IR.  A Fock model
    # may share a force-capable plan, but its value state count remains the
    # value order rather than inheriting the force derivative order.
    maximum_order = (
        integral.maximum_coulomb_order
        if consumer == KernelConsumer.FORCE
        else integral.value_coulomb_order
    )
    return StaticAlgebraModel(
        scope=scope,
        algebra_placement=algebra_placement,
        algebra_ordering=algebra_ordering,
        algebra_fusion=algebra_fusion,
        algebra_form=algebra_form,
        component_count=spec.component_count,
        sampled_component_count=sampled_component_count,
        recurrence_state_count=comb(maximum_order + 3, 3),
        root_count=root_count,
        operation_counts=operation_counts,
        emitted_operation_counts=emitted_operation_counts,
        baseline_arithmetic_operation_count=(baseline_arithmetic_operation_count),
        baseline_materialized_value_count=baseline_materialized_value_count,
        baseline_peak_live_values=baseline_peak_live_values,
        arithmetic_operation_count=arithmetic_operation_count,
        materialized_value_count=materialized_value_count,
        peak_live_values=peak_live_values,
        rematerialized_value_count=rematerialized_value_count,
        reordered_value_count=reordered_value_count,
        fma_operation_count=fma_operation_count,
    )


def static_algebra_model(trial: ScheduleTrial) -> StaticAlgebraModel:
    """Return the symbolic static model appropriate to one schedule mapping."""

    model_integral = trial.integral or build_integral_ir(
        trial.spec,
        consumers=(trial.consumer,),
    )
    if model_integral.spec != trial.spec:
        raise ValueError("trial integral spec does not match its shell specification")
    if trial.consumer not in model_integral.consumers:
        raise ValueError(
            f"{trial.consumer.value} static model requires its integral consumer"
        )
    weighted_shell = (
        trial.consumer == KernelConsumer.FORCE
        and trial.schedule.kind == ScheduleKind.PACKED_TASKS
    )
    return _cached_static_algebra_model(
        trial.spec,
        trial.consumer,
        model_integral,
        weighted_shell,
        trial.schedule.algebra_placement,
        trial.schedule.algebra_ordering,
        trial.schedule.algebra_fusion,
        trial.schedule.algebra_form,
    )

"""Compile, validate, and rank architecture-specific CUDA schedules.

The tuner deliberately searches the bounded schedule space exposed by
``ScheduleIR`` instead of asking NVCC to discover a complete execution policy.
Every variant is emitted into an independent translation unit with unique CUDA
symbols, then all successfully compiled variants are linked into one process.
This keeps scheduler allocation overhead out of the comparison and makes the
winning schedule reproducible in the production manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import cache
from itertools import product
from math import comb
from pathlib import Path

from .batch_benchmark import KernelResources, parse_ptxas_resources
from .benchmark import (
    emit_shell_class_benchmark_cuda,
    emit_shell_class_oracle_cuda,
    emit_shell_class_resource_cuda,
)
from .cuda_adapter import CudaBenchmarkExecutor, CudaCompilerAdapter
from .cuda_schedule import (
    AlgebraForm,
    AlgebraFusion,
    AlgebraOrdering,
    AlgebraPlacement,
    PairOrientation,
    PairStorage,
    ScheduleIR,
    ScheduleKind,
    tuning_schedule_candidates,
)
from .cuda_target import (
    DEFAULT_CUDA_TARGET,
    CudaTargetInfo,
    cuda_target_info,
    normalize_cuda_architecture,
)
from .expr import Expr, PowerLowering
from .fused_schedule import build_fused_shell_plan
from .ir import IntegralIR, KernelConsumer, build_integral_ir
from .shell_class import (
    ShellClassContractionKernel,
    WeightedShellContractionKernel,
    build_packed_force_geometry_algebra,
    build_shell_class_contraction_kernel,
    build_weighted_shell_contraction_kernel,
)
from .shell_spec import AXES, FUSED_SHELL_SPEC_BY_NAME, ShellClassSpec


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
                "materialized_value_count": (
                    self.baseline_materialized_value_count
                ),
                "estimated_peak_live_values": self.baseline_peak_live_values,
            },
            "post_optimization": {
                "operation_counts": dict(self.emitted_operation_counts),
                "arithmetic_operation_count": self.arithmetic_operation_count,
                "materialized_value_count": self.materialized_value_count,
                "estimated_peak_live_values": self.peak_live_values,
            },
            "inlined_value_count": (
                self.baseline_materialized_value_count
                - self.materialized_value_count
            ),
            "rematerialized_value_count": self.rematerialized_value_count,
            "reordered_value_count": self.reordered_value_count,
            "fma_operation_count": self.fma_operation_count,
        }


@dataclass(frozen=True, slots=True)
class ScheduleTrial:
    """One shell class and one concrete schedule compiled as a unit."""

    spec: ShellClassSpec
    schedule: ScheduleIR
    consumer: KernelConsumer = KernelConsumer.FORCE
    target: CudaTargetInfo = DEFAULT_CUDA_TARGET

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
    def key(self) -> str:
        """Return the cross-class report key for this trial."""

        return f"{self.spec.name}:{self.consumer.value}:{self.schedule_id}"

    @property
    def entry_point(self) -> str:
        """Return the unique host entry used by the batch driver."""

        return (
            f"vibeqc_run_schedule_{self.spec.name}_{self.consumer.value}_"
            f"{self.schedule_id}"
        )

    @property
    def symbol_prefix(self) -> str:
        """Return the unique lower-case CUDA symbol prefix."""

        return (
            f"generated_{self.spec.name}_{self.consumer.value}_"
            f"{self.schedule_id}"
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
    weighted_shell: bool,
    algebra_placement: AlgebraPlacement,
    algebra_ordering: AlgebraOrdering,
    algebra_fusion: AlgebraFusion,
    algebra_form: AlgebraForm,
) -> StaticAlgebraModel:
    """Build one immutable model shared by same-class schedule variants."""

    integral = build_integral_ir(spec, consumers=(consumer,))

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
            _packed_force_geometry_analysis(
                4 if spec.angular[3] != 0 else 3
            ),
        ]
        scope = "weighted_shell_dag"
        sampled_component_count = spec.component_count
    else:
        components = tuple(product(*_balanced_component_labels(spec)))
        analysis_pairs = (
            (
                component_kernel.graph.analyze_ssa(roots),
                component_kernel.graph.materialization_plan(
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
    derivative_order = 1 if consumer == KernelConsumer.FORCE else 0
    maximum_order = sum(spec.angular) + derivative_order
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
        baseline_arithmetic_operation_count=(
            baseline_arithmetic_operation_count
        ),
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

    weighted_shell = (
        trial.consumer == KernelConsumer.FORCE
        and trial.schedule.kind == ScheduleKind.PACKED_TASKS
    )
    return _cached_static_algebra_model(
        trial.spec,
        trial.consumer,
        weighted_shell,
        trial.schedule.algebra_placement,
        trial.schedule.algebra_ordering,
        trial.schedule.algebra_fusion,
        trial.schedule.algebra_form,
    )


def supported_schedule_trials(
    spec: ShellClassSpec,
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
    target: CudaTargetInfo = DEFAULT_CUDA_TARGET,
) -> tuple[ScheduleTrial, ...]:
    """Return the schedule variants implemented by the current CUDA emitter.

    Fock trials retain the force companion because production uses one
    canonical task ABI, while timing and resource gates select only the
    requested consumer's kernels.
    """

    if any(order > 6 for order in spec.pair_orders):
        raise ValueError(
            f"{spec.name} is outside the current pair-order CUDA lowering"
        )
    if any(order > 3 for order in spec.angular):
        raise ValueError(
            f"{spec.name} exceeds the current s/p/d/f CUDA lowering"
        )
    selected_consumer = KernelConsumer(consumer)
    consumers = (
        (KernelConsumer.FOCK, KernelConsumer.FORCE)
        if selected_consumer == KernelConsumer.FOCK
        else (KernelConsumer.FORCE,)
    )
    integral = build_integral_ir(spec, consumers)
    return tuple(
        ScheduleTrial(
            spec=spec,
            schedule=schedule,
            consumer=selected_consumer,
            target=target,
        )
        for schedule in tuning_schedule_candidates(integral, target)
        if schedule.kind
        in (
            ScheduleKind.PACKED_TASKS,
            ScheduleKind.SUBGROUP_TASKS,
            ScheduleKind.SHELL_TASK,
            ScheduleKind.COMPONENT_LANES,
            ScheduleKind.TILED_COMPONENTS,
        )
    )


def _class_name(spec: ShellClassSpec) -> str:
    return spec.name[0].upper() + spec.name[1:]


def _identifier_suffix(schedule_id: str) -> str:
    return "".join(part.capitalize() for part in schedule_id.split("_"))


def _isolate_schedule_symbols(
    source: str,
    trial: ScheduleTrial,
    *,
    symbol_prefix: str | None = None,
    type_suffix: str | None = None,
) -> str:
    """Suffix CUDA and C++ identifiers so same-class variants can link."""

    class_name = _class_name(trial.spec)
    suffix = type_suffix or _identifier_suffix(trial.schedule_id)
    selected_prefix = symbol_prefix or trial.symbol_prefix
    source = source.replace(
        f"generated_{trial.spec.name}", selected_prefix
    )
    return source.replace(
        f"Generated{class_name}", f"Generated{class_name}{suffix}"
    )


def emit_schedule_translation_unit(
    trial: ScheduleTrial,
    *,
    task_count: int,
    primitive_count: int,
    warmups: int,
    iterations: int,
    samples: int,
    oracle_trial: ScheduleTrial | None = None,
) -> str:
    """Emit one uniquely named, independently compilable schedule benchmark."""

    consumers = (
        (KernelConsumer.FOCK, KernelConsumer.FORCE)
        if trial.consumer == KernelConsumer.FOCK
        else (KernelConsumer.FORCE,)
    )
    plan = build_fused_shell_plan(
        trial.spec,
        consumers=consumers,
        schedule=trial.schedule,
        target=trial.target,
    )
    source = emit_shell_class_benchmark_cuda(
        trial.spec,
        task_count=task_count,
        primitive_count=primitive_count,
        warmups=warmups,
        iterations=iterations,
        samples=samples,
        plan=plan,
        consumer=trial.consumer,
        benchmark_kernel_only=True,
        persistent_kernel=True,
        oracle_symbol_prefix=(
            _oracle_symbol_prefix(_oracle_schedule_trial(oracle_trial))
            if oracle_trial is not None
            else None
        ),
    )
    source = source.replace("int main() {", f'extern "C" int {trial.entry_point}() {{', 1)
    marker = r'{\"task_count\":%u'
    replacement = (
        rf'{{\"shell_class\":\"{trial.spec.name}\",'
        rf'\"schedule_id\":\"{trial.schedule_id}\",\"task_count\":%u'
    )
    if marker not in source:
        raise RuntimeError("benchmark JSON marker changed unexpectedly")
    source = source.replace(marker, replacement, 1)

    # Each translation unit contains a complete kernel and harness.  Suffix
    # both C-style and C++ type identifiers so variants of one class can link
    # into the same executable without changing the production emitter.
    return _isolate_schedule_symbols(source, trial)


def _oracle_schedule_trial(trial: ScheduleTrial) -> ScheduleTrial:
    """Canonicalize knobs that do not change the recompute oracle mapping."""

    shared_coulomb = trial.schedule.kind not in (
        ScheduleKind.PACKED_TASKS,
        ScheduleKind.SHELL_TASK,
    )
    return ScheduleTrial(
        spec=trial.spec,
        consumer=trial.consumer,
        schedule=replace(
            trial.schedule,
            shared_coulomb=shared_coulomb,
            pair_orientation=PairOrientation.CANONICAL,
            pair_storage=PairStorage.RECOMPUTED,
            unroll_pair_terms=False,
            minimum_blocks_per_sm=0,
            maximum_registers=0,
        ),
        target=trial.target,
    )


def _oracle_symbol_prefix(trial: ScheduleTrial) -> str:
    """Return a stable C symbol prefix shared by equivalent oracle mappings."""

    return (
        f"vibeqc_oracle_{trial.spec.name}_{trial.consumer.value}_"
        f"{trial.schedule.kind.value}_b{trial.schedule.block_threads}_"
        f"t{trial.schedule.component_tile}_w{trial.schedule.tasks_per_warp}"
    )


def emit_schedule_oracle_translation_unit(trial: ScheduleTrial) -> str:
    """Emit one separately compiled oracle shared by schedule code-shape knobs."""

    oracle_trial = _oracle_schedule_trial(trial)
    consumers = (
        (KernelConsumer.FOCK, KernelConsumer.FORCE)
        if oracle_trial.consumer == KernelConsumer.FOCK
        else (KernelConsumer.FORCE,)
    )
    plan = build_fused_shell_plan(
        oracle_trial.spec,
        consumers=consumers,
        schedule=oracle_trial.schedule,
        target=oracle_trial.target,
    )
    source = emit_shell_class_oracle_cuda(
        oracle_trial.spec,
        plan,
        oracle_trial.consumer,
    )
    return _isolate_schedule_symbols(
        source,
        oracle_trial,
        symbol_prefix=_oracle_symbol_prefix(oracle_trial),
        type_suffix=f"Oracle{_identifier_suffix(oracle_trial.schedule_id)}",
    )


def emit_schedule_resource_translation_unit(trial: ScheduleTrial) -> str:
    """Emit the complete production wrapper set for a measured candidate."""

    consumers = (
        (KernelConsumer.FOCK, KernelConsumer.FORCE)
        if trial.consumer == KernelConsumer.FOCK
        else (KernelConsumer.FORCE,)
    )
    plan = build_fused_shell_plan(
        trial.spec,
        consumers=consumers,
        schedule=trial.schedule,
        target=trial.target,
    )
    return _isolate_schedule_symbols(
        emit_shell_class_resource_cuda(trial.spec, plan),
        trial,
    )


def emit_schedule_driver(
    trials: Iterable[ScheduleTrial],
    architecture: str | None = None,
) -> str:
    """Emit a driver that validates its allocated GPU before benchmarking."""

    items = tuple(trials)
    selected_architecture = normalize_cuda_architecture(
        architecture
        or (items[0].target.architecture if items else DEFAULT_CUDA_TARGET.architecture)
    )
    expected_target = cuda_target_info(selected_architecture)
    declarations = "\n".join(
        f'extern "C" int {trial.entry_point}();' for trial in items
    )
    calls = "\n".join(
        f"  failures += {trial.entry_point}() != 0;" for trial in items
    )
    return rf"""#include <cuda_runtime.h>
#include <cstdio>

{declarations}

int main() {{
  const cudaError_t initialization = cudaFree(nullptr);
  if (initialization != cudaSuccess) {{
    std::fprintf(stderr, "CUDA initialization failed: %s\n",
                 cudaGetErrorString(initialization));
    return 2;
  }}
  int device = 0;
  cudaDeviceProp properties{{}};
  int driver_version = 0;
  int runtime_version = 0;
  int optin_shared_bytes = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
      cudaDriverGetVersion(&driver_version) != cudaSuccess ||
      cudaRuntimeGetVersion(&runtime_version) != cudaSuccess ||
      cudaDeviceGetAttribute(&optin_shared_bytes,
          cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess) {{
    std::fprintf(stderr, "CUDA target probe failed\n");
    return 2;
  }}
  std::printf(
      "{{\"target_probe\":true,\"device_name\":\"%s\","
      "\"device_id\":%d,\"compute_capability_major\":%d,"
      "\"compute_capability_minor\":%d,\"warp_size\":%d,"
      "\"maximum_threads_per_block\":%d,"
      "\"maximum_threads_per_sm\":%d,\"maximum_blocks_per_sm\":%d,"
      "\"registers_per_sm\":%d,\"shared_memory_per_block\":%zu,"
      "\"shared_memory_per_block_optin\":%d,"
      "\"shared_memory_per_sm\":%zu,\"sm_count\":%d,"
      "\"driver_version\":%d,\"runtime_version\":%d}}\n",
      properties.name, device, properties.major, properties.minor,
      properties.warpSize, properties.maxThreadsPerBlock,
      properties.maxThreadsPerMultiProcessor, properties.maxBlocksPerMultiProcessor,
      properties.regsPerMultiprocessor, properties.sharedMemPerBlock,
      optin_shared_bytes, properties.sharedMemPerMultiprocessor,
      properties.multiProcessorCount, driver_version, runtime_version);
  std::fflush(stdout);
  if (properties.major != {expected_target.compute_capability_major} ||
      properties.minor != {expected_target.compute_capability_minor}) {{
    std::fprintf(
        stderr,
        "compile target {selected_architecture} does not match allocated "
        "device sm_%d%d\n",
        properties.major, properties.minor);
    return 5;
  }}
  int failures = 0;
{calls}
  return failures == 0 ? 0 : 3;
}}
"""


def update_manifest_payload(
    payload: dict[str, object],
    architecture: str,
    winners: Mapping[str, ScheduleIR],
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
) -> dict[str, object]:
    """Return a v2 manifest with measured winners installed for one GPU."""

    if payload.get("schema_version") != 2:
        raise ValueError("autotuning requires a schema-v2 production manifest")
    architectures = payload.get("architectures")
    if not isinstance(architectures, dict):
        raise TypeError("production manifest requires an architectures object")
    if architecture not in architectures:
        raise ValueError(f"production manifest has no profile for {architecture}")
    profile = architectures.get(architecture)
    if not isinstance(profile, dict):
        raise TypeError("architecture profile must be a JSON object")
    kernels = profile.get("kernels")
    if not isinstance(kernels, list):
        raise TypeError("architecture profile requires a kernels list")

    selected_consumer = KernelConsumer(consumer)
    required_consumers = (
        (KernelConsumer.FOCK, KernelConsumer.FORCE)
        if selected_consumer == KernelConsumer.FOCK
        else (KernelConsumer.FORCE,)
    )
    installed = set()
    for row in kernels:
        if not isinstance(row, dict):
            raise TypeError("production kernel entry must be a JSON object")
        name = row.get("shell_class")
        if isinstance(name, str) and name in winners:
            row["schedule"] = schedule_payload(winners[name])
            raw_consumers = row.get("consumers", [])
            if not isinstance(raw_consumers, list):
                raise TypeError(f"{name} consumers must be a list")
            current = {KernelConsumer(item) for item in raw_consumers}
            current.update(required_consumers)
            row["consumers"] = [
                item.value for item in KernelConsumer if item in current
            ]
            installed.add(name)
    for name, schedule in winners.items():
        if name not in installed:
            kernels.append(
                {
                    "shell_class": name,
                    "consumers": [item.value for item in required_consumers],
                    "schedule": schedule_payload(schedule),
                }
            )
    return payload


def write_tuned_manifest(
    input_path: Path,
    output_path: Path,
    architecture: str,
    winners: Mapping[str, ScheduleIR],
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
) -> None:
    """Write architecture winners without mutating the source manifest in place."""

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    updated = update_manifest_payload(
        payload,
        architecture,
        winners,
        consumer,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(updated, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _compile_trial(
    nvcc: Path,
    architecture: str,
    directory: Path,
    trial: ScheduleTrial,
    compile_timeout: float = 300,
    artifact_suffix: str = "",
) -> dict[str, object]:
    """Compile one schedule and retain diagnostics for resource gates.

    NVCC launches ``cicc`` and ``ptxas`` children.  A timeout therefore kills
    the fresh process group rather than only the NVCC parent, otherwise large
    shell trials leave orphan compiler processes consuming CPU while later
    candidates start.
    """

    stem = f"{trial.spec.name}_{trial.schedule_id}{artifact_suffix}"
    source = directory / f"{stem}.cu"
    obj = directory / f"{stem}.o"
    compiler = CudaCompilerAdapter(
        nvcc=nvcc,
        target=cuda_target_info(architecture),
        compile_timeout=compile_timeout,
    )
    result = compiler.compile(source, obj)
    diagnostics = result.stdout + result.stderr
    marker = (
        f"{trial.symbol_prefix}_shell_class_{trial.consumer.value}_"
    )
    return {
        "key": trial.key,
        "object": obj,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "duration_seconds": result.duration_seconds,
        "diagnostics": diagnostics,
        "resources": parse_ptxas_resources(
            diagnostics,
            trial.spec.name,
            symbol_prefix=marker,
        ),
    }


def _runtime_environment(nvcc: Path) -> dict[str, str]:
    """Expose the selected CUDA runtime without changing GPU visibility."""

    environment = dict(os.environ)
    if environment.get("CUDA_VISIBLE_DEVICES") == "":
        environment.pop("CUDA_VISIBLE_DEVICES")
    library = nvcc.parent.parent / "lib64"
    previous = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        str(library) if not previous else f"{library}:{previous}"
    )
    return environment


def _tool_version(command: Path) -> str:
    """Return one compiler version line for tuning-artifact provenance."""

    try:
        result = subprocess.run(
            [str(command), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError:
        return ""
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines()]
    return next((line for line in reversed(lines) if line), "")


def _resource_rejections(
    resources: tuple[KernelResources, ...],
    *,
    consumer: KernelConsumer,
    maximum_registers: int,
    maximum_stack_bytes: int,
    maximum_shared_bytes: int,
    expected_kernel_records: int = 1,
) -> list[str]:
    """Return deterministic resource-gate failures for one schedule."""

    reasons = []
    if len(resources) != expected_kernel_records:
        reasons.append(
            f"expected {expected_kernel_records} {consumer.value} kernel records, "
            f"found {len(resources)}"
        )
    if any(item.spill_store_bytes or item.spill_load_bytes for item in resources):
        reasons.append("ptxas reported local-memory spills")
    if any(item.registers > maximum_registers for item in resources):
        reasons.append(f"register use exceeds {maximum_registers}")
    if any(item.stack_bytes > maximum_stack_bytes for item in resources):
        reasons.append(f"stack frame exceeds {maximum_stack_bytes} bytes")
    if any(item.shared_bytes > maximum_shared_bytes for item in resources):
        reasons.append(f"static shared memory exceeds {maximum_shared_bytes} bytes")
    return reasons


def _resolve_specifications(names: Iterable[str]) -> tuple[ShellClassSpec, ...]:
    """Resolve CLI shell names while preserving the requested order."""

    specifications = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        try:
            specification = FUSED_SHELL_SPEC_BY_NAME[name]
        except KeyError as error:
            choices = ", ".join(FUSED_SHELL_SPEC_BY_NAME)
            raise ValueError(
                f"unknown shell class {name!r}; choose from {choices}"
            ) from error
        specifications.append(specification)
        seen.add(name)
    if not specifications:
        raise ValueError("at least one shell class is required")
    return tuple(specifications)


def _run_autotune(arguments: argparse.Namespace) -> dict[str, object]:
    """Generate, compile, run, rank, and optionally persist schedule winners."""

    arguments.architecture = normalize_cuda_architecture(arguments.architecture)
    target = cuda_target_info(arguments.architecture)
    compiler = CudaCompilerAdapter(
        nvcc=arguments.nvcc,
        target=target,
        compile_timeout=arguments.compile_timeout,
    )
    benchmark_executor = CudaBenchmarkExecutor(
        timeout=arguments.timeout,
        local=arguments.local,
        srun=arguments.srun,
        partition=arguments.partition,
        gres=arguments.gres,
        slurm_time=arguments.slurm_time,
    )
    if arguments.max_registers is None:
        arguments.max_registers = target.tuning_maximum_registers
    if arguments.max_packed_registers is None:
        arguments.max_packed_registers = target.tuning_maximum_packed_registers
    if arguments.max_stack_bytes is None:
        arguments.max_stack_bytes = target.tuning_maximum_stack_bytes
    if arguments.max_shared_bytes is None:
        arguments.max_shared_bytes = min(
            target.tuning_maximum_shared_bytes,
            target.shared_memory_per_block,
        )
    specifications = _resolve_specifications(arguments.shell_class)
    selected_consumer = KernelConsumer(arguments.consumer)
    trials = tuple(
        trial
        for spec in specifications
        for trial in supported_schedule_trials(spec, selected_consumer, target)
    )
    work_directory_owner = None
    if arguments.work_directory is None:
        work_directory_owner = tempfile.TemporaryDirectory(
            prefix="vibeqc-shell-autotune-"
        )
        directory = Path(work_directory_owner.name)
    else:
        directory = arguments.work_directory
        directory.mkdir(parents=True, exist_ok=True)

    try:
        oracle_by_trial: dict[str, ScheduleTrial] = {}
        oracle_trials: dict[str, ScheduleTrial] = {}
        for trial in trials:
            oracle_trial = _oracle_schedule_trial(trial)
            oracle_prefix = _oracle_symbol_prefix(oracle_trial)
            oracle_by_trial[trial.key] = oracle_trial
            oracle_trials.setdefault(oracle_prefix, oracle_trial)
        for oracle_trial in oracle_trials.values():
            oracle_source = emit_schedule_oracle_translation_unit(oracle_trial)
            oracle_path = directory / (
                f"{oracle_trial.spec.name}_{oracle_trial.schedule_id}_oracle.cu"
            )
            oracle_path.write_text(oracle_source, encoding="utf-8")
        with ThreadPoolExecutor(max_workers=arguments.compile_jobs) as compile_pool:
            oracle_compile_rows = list(
                compile_pool.map(
                    lambda trial: _compile_trial(
                        arguments.nvcc,
                        arguments.architecture,
                        directory,
                        trial,
                        arguments.compile_timeout,
                        "_oracle",
                    ),
                    oracle_trials.values(),
                )
            )
        oracle_compile_by_prefix = {
            prefix: row
            for prefix, row in zip(
                oracle_trials,
                oracle_compile_rows,
                strict=True,
            )
        }

        for trial in trials:
            source = emit_schedule_translation_unit(
                trial,
                task_count=arguments.tasks,
                primitive_count=arguments.primitives,
                warmups=arguments.warmups,
                iterations=arguments.iterations,
                samples=arguments.samples,
                oracle_trial=oracle_by_trial[trial.key],
            )
            (directory / f"{trial.spec.name}_{trial.schedule_id}.cu").write_text(
                source,
                encoding="utf-8",
            )

        with ThreadPoolExecutor(max_workers=arguments.compile_jobs) as compile_pool:
            compile_rows = list(
                compile_pool.map(
                    lambda trial: _compile_trial(
                        arguments.nvcc,
                        arguments.architecture,
                        directory,
                        trial,
                        arguments.compile_timeout,
                    ),
                    trials,
                )
            )

        runnable_trials = tuple(
            trial
            for trial, row in zip(trials, compile_rows, strict=True)
            if row["returncode"] == 0
            and oracle_compile_by_prefix[
                _oracle_symbol_prefix(oracle_by_trial[trial.key])
            ]["returncode"]
            == 0
        )
        runtime_rows: dict[str, dict[str, object]] = {}
        run_returncode = None
        run_stderr = ""
        if runnable_trials:
            driver = directory / "autotune_driver.cu"
            driver.write_text(
                emit_schedule_driver(runnable_trials, arguments.architecture),
                encoding="utf-8",
            )
            executable = directory / "shell_schedule_autotune"
            compile_by_key = {
                trial.key: row
                for trial, row in zip(trials, compile_rows, strict=True)
            }
            objects = [
                Path(compile_by_key[trial.key]["object"])
                for trial in runnable_trials
            ]
            used_oracle_prefixes = {
                _oracle_symbol_prefix(oracle_by_trial[trial.key])
                for trial in runnable_trials
            }
            objects.extend(
                    Path(oracle_compile_by_prefix[prefix]["object"])
                    for prefix in sorted(used_oracle_prefixes)
                )
            link = compiler.link(driver, objects, executable)
            if link.returncode != 0:
                raise RuntimeError(link.stdout + link.stderr)
            run = benchmark_executor.run(
                executable,
                _runtime_environment(arguments.nvcc),
            )
            run_returncode = run.returncode
            run_stderr = run.stderr
            runtime_probe = None
            for line in run.stdout.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("target_probe") is True:
                    runtime_probe = row
                    continue
                shell_class = row.get("shell_class")
                consumer = row.get("consumer")
                schedule_id = row.get("schedule_id")
                if (
                    isinstance(shell_class, str)
                    and isinstance(consumer, str)
                    and isinstance(schedule_id, str)
                ):
                    runtime_rows[
                        f"{shell_class}:{consumer}:{schedule_id}"
                    ] = row

        else:
            runtime_probe = None

        candidates = []
        passing_by_class: dict[
            str,
            list[tuple[ScheduleTrial, dict[str, object], dict[str, object]]],
        ] = {}
        for trial, compile_row in zip(trials, compile_rows, strict=True):
            resources = compile_row["resources"]
            oracle_compile = oracle_compile_by_prefix[
                _oracle_symbol_prefix(oracle_by_trial[trial.key])
            ]
            maximum_registers = (
                arguments.max_packed_registers
                if trial.schedule.kind == ScheduleKind.PACKED_TASKS
                else arguments.max_registers
            )
            reasons = _resource_rejections(
                resources,
                consumer=trial.consumer,
                maximum_registers=maximum_registers,
                maximum_stack_bytes=arguments.max_stack_bytes,
                maximum_shared_bytes=arguments.max_shared_bytes,
            )
            if (
                trial.schedule.kind == ScheduleKind.SUBGROUP_TASKS
                and not arguments.allow_experimental_subgroup_winner
            ):
                reasons.append(
                    "subgroup schedules require explicit end-to-end "
                    "production acceptance"
                )
            if compile_row["timed_out"]:
                reasons.append(
                    "NVCC compilation exceeded "
                    f"{arguments.compile_timeout:g} seconds"
                )
            elif compile_row["returncode"] != 0:
                reasons.append("NVCC compilation failed")
            if oracle_compile["timed_out"]:
                reasons.append(
                    "oracle NVCC compilation exceeded "
                    f"{arguments.compile_timeout:g} seconds"
                )
            elif oracle_compile["returncode"] != 0:
                reasons.append("oracle NVCC compilation failed")
            runtime = runtime_rows.get(trial.key)
            if run_returncode == 5:
                reasons.append("compile and runtime CUDA targets differ")
            if runtime is None:
                reasons.append("schedule did not produce a runtime result")
            else:
                maximum_value = float(
                    runtime[f"maximum_{trial.consumer.value}"]
                )
                maximum_error = float(
                    runtime[f"maximum_{trial.consumer.value}_error"]
                )
                tolerance = arguments.absolute_tolerance + (
                    arguments.relative_tolerance * maximum_value
                )
                if maximum_error > tolerance:
                    reasons.append(
                        f"{trial.consumer.value} error {maximum_error:.3e} "
                        f"exceeds {tolerance:.3e}"
                    )
                if float(runtime["speedup"]) < arguments.minimum_speedup:
                    reasons.append(
                        f"speedup is below {arguments.minimum_speedup:.3f}x"
                    )
            accepted = not reasons
            row = {
                "shell_class": trial.spec.name,
                "consumer": trial.consumer.value,
                "schedule_id": trial.schedule_id,
                "schedule": schedule_payload(trial.schedule),
                "static_model": trial.static_model.to_payload(),
                "compile_succeeded": compile_row["returncode"] == 0,
                "compile_timed_out": compile_row["timed_out"],
                "compile_seconds": compile_row["duration_seconds"],
                "resources": [asdict(item) for item in resources],
                "runtime": runtime,
                "accepted": accepted,
                "rejection_reasons": reasons,
                "production_validation": None,
            }
            candidates.append(row)
            if accepted and runtime is not None:
                passing_by_class.setdefault(trial.spec.name, []).append(
                    (trial, runtime, row)
                )
            if arguments.verbose and compile_row["diagnostics"]:
                print(compile_row["diagnostics"], file=sys.stderr, end="")

        winners: dict[str, ScheduleIR] = {}
        winner_rows = []
        for spec in specifications:
            passing = passing_by_class.get(spec.name, [])
            if not passing:
                continue
            ranked = sorted(
                passing,
                key=lambda item: (
                    float(item[1]["fused_ms"]),
                    item[0].schedule_id,
                ),
            )
            for trial, runtime, candidate_row in ranked:
                resource_source = emit_schedule_resource_translation_unit(trial)
                resource_suffix = "_production_resources"
                resource_path = directory / (
                    f"{trial.spec.name}_{trial.schedule_id}{resource_suffix}.cu"
                )
                resource_path.write_text(resource_source, encoding="utf-8")
                resource_compile = _compile_trial(
                    arguments.nvcc,
                    arguments.architecture,
                    directory,
                    trial,
                    arguments.compile_timeout,
                    resource_suffix,
                )
                maximum_registers = (
                    arguments.max_packed_registers
                    if trial.schedule.kind == ScheduleKind.PACKED_TASKS
                    else arguments.max_registers
                )
                production_reasons = _resource_rejections(
                    resource_compile["resources"],
                    consumer=trial.consumer,
                    maximum_registers=maximum_registers,
                    maximum_stack_bytes=arguments.max_stack_bytes,
                    maximum_shared_bytes=arguments.max_shared_bytes,
                    expected_kernel_records=4,
                )
                if resource_compile["timed_out"]:
                    production_reasons.append(
                        "NVCC compilation exceeded "
                        f"{arguments.compile_timeout:g} seconds"
                    )
                elif resource_compile["returncode"] != 0:
                    production_reasons.append("NVCC compilation failed")
                production_validation = {
                    "compile_succeeded": resource_compile["returncode"] == 0,
                    "compile_timed_out": resource_compile["timed_out"],
                    "compile_seconds": resource_compile["duration_seconds"],
                    "resources": [
                        asdict(item) for item in resource_compile["resources"]
                    ],
                    "accepted": not production_reasons,
                    "rejection_reasons": production_reasons,
                }
                candidate_row["production_validation"] = production_validation
                if arguments.verbose and resource_compile["diagnostics"]:
                    print(
                        resource_compile["diagnostics"],
                        file=sys.stderr,
                        end="",
                    )
                if production_reasons:
                    candidate_row["accepted"] = False
                    candidate_row["rejection_reasons"].extend(
                        f"production validation: {reason}"
                        for reason in production_reasons
                    )
                    continue
                winners[spec.name] = trial.schedule
                winner_rows.append(
                    {
                        "shell_class": spec.name,
                        "consumer": selected_consumer.value,
                        "schedule_id": trial.schedule_id,
                        "schedule": schedule_payload(trial.schedule),
                        "static_model": trial.static_model.to_payload(),
                        "runtime": runtime,
                        "production_validation": production_validation,
                    }
                )
                break

        if arguments.manifest_output is not None and winners:
            write_tuned_manifest(
                arguments.manifest,
                arguments.manifest_output,
                arguments.architecture,
                winners,
                selected_consumer,
            )

        return {
            "schema_version": 1,
            "architecture": arguments.architecture,
            "consumer": selected_consumer.value,
            "nvcc": str(arguments.nvcc),
            "target": target.to_payload(),
            "toolchain": {
                "nvcc": _tool_version(arguments.nvcc),
                "ptxas": _tool_version(arguments.nvcc.with_name("ptxas")),
                "generator_abi": target.generator_abi,
            },
            "single_gpu_process": True,
            "runtime": {
                "local": arguments.local,
                "srun": None if arguments.local else arguments.srun,
                "partition": None if arguments.local else arguments.partition,
                "gres": None if arguments.local else arguments.gres,
                "returncode": run_returncode,
                "stderr": run_stderr,
                "device": runtime_probe,
            },
            "gates": {
                "minimum_speedup": arguments.minimum_speedup,
                "absolute_tolerance": arguments.absolute_tolerance,
                "relative_tolerance": arguments.relative_tolerance,
                "maximum_registers": arguments.max_registers,
                "maximum_packed_registers": arguments.max_packed_registers,
                "maximum_stack_bytes": arguments.max_stack_bytes,
                "maximum_shared_bytes": arguments.max_shared_bytes,
                "compile_timeout_seconds": arguments.compile_timeout,
                "spills_allowed": False,
                "experimental_subgroup_winners_allowed": (
                    arguments.allow_experimental_subgroup_winner
                ),
            },
            "winners": winner_rows,
            "candidates": candidates,
            "oracles": [
                {
                    "symbol_prefix": prefix,
                    "compile_succeeded": row["returncode"] == 0,
                    "compile_timed_out": row["timed_out"],
                    "compile_seconds": row["duration_seconds"],
                }
                for prefix, row in oracle_compile_by_prefix.items()
            ],
        }
    finally:
        if work_directory_owner is not None:
            work_directory_owner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nvcc",
        type=Path,
        default=Path(
            os.environ.get("VIBEQC_NVCC", shutil.which("nvcc") or "nvcc")
        ),
    )
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--srun", default="srun")
    parser.add_argument("--partition", default="main")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument(
        "--slurm-time",
        default="00:10:00",
        help="finite Slurm allocation time used for the benchmark process",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="run directly on an already allocated/visible GPU",
    )
    parser.add_argument("--shell-class", action="append", required=True)
    parser.add_argument(
        "--consumer",
        choices=tuple(item.value for item in KernelConsumer),
        default=KernelConsumer.FORCE.value,
    )
    parser.add_argument("--tasks", type=int, default=512)
    parser.add_argument("--primitives", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--compile-jobs", type=int, default=2)
    parser.add_argument("--compile-timeout", type=float, default=300.0)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-10)
    parser.add_argument("--relative-tolerance", type=float, default=2.0e-10)
    parser.add_argument("--max-registers", type=int)
    parser.add_argument("--max-packed-registers", type=int)
    parser.add_argument("--max-stack-bytes", type=int)
    parser.add_argument("--max-shared-bytes", type=int)
    parser.add_argument(
        "--allow-experimental-subgroup-winner",
        action="store_true",
        help=(
            "allow subgroup schedules to become manifest winners after the "
            "synthetic gate; normally they require a separate end-to-end "
            "production acceptance"
        ),
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("production_shell_classes.json"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="write winners into this schema-v2 manifest path",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    if arguments.compile_jobs < 1:
        parser.error("--compile-jobs must be positive")
    if arguments.compile_timeout <= 0:
        parser.error("--compile-timeout must be positive")
    report = _run_autotune(arguments)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(output, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output, encoding="utf-8")
    if len(report["winners"]) != len(set(arguments.shell_class)):
        raise SystemExit(4)


if __name__ == "__main__":
    main()

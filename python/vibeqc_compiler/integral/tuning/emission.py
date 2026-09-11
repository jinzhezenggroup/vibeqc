"""Emit isolated trial/oracle/resource translation units and benchmark drivers.

Symbol isolation keeps same-class schedules linkable while preserving the
mathematical and schedule identities selected by policy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from ..benchmark import (
    emit_shell_class_benchmark_cuda,
    emit_shell_class_oracle_cuda,
    emit_shell_class_resource_cuda,
)
from ..cuda_schedule import (
    PairOrientation,
    PairStorage,
    ScheduleKind,
)
from ..cuda_target import (
    DEFAULT_CUDA_TARGET,
    cuda_target_info,
    normalize_cuda_architecture,
)
from ..fused_schedule import build_fused_shell_plan
from ..ir import KernelConsumer, build_integral_ir
from ..shell_spec import ShellClassSpec
from .policy import ScheduleTrial


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
    if trial.integral is not None:
        # Explicit mathematical IRs can share every execution knob while
        # still requiring different generated C++ types.  Keep their type
        # names disjoint as well as their C entry points and kernel symbols.
        suffix += _identifier_suffix(trial.integral_suffix)
    selected_prefix = symbol_prefix or trial.symbol_prefix
    source = source.replace(f"generated_{trial.spec.name}", selected_prefix)
    return source.replace(f"Generated{class_name}", f"Generated{class_name}{suffix}")


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

    integral = trial.integral or build_integral_ir(
        trial.spec,
        consumers=(
            (KernelConsumer.FOCK, KernelConsumer.FORCE)
            if trial.consumer == KernelConsumer.FOCK
            else (KernelConsumer.FORCE,)
        ),
    )
    plan = build_fused_shell_plan(
        trial.spec,
        integral=integral,
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
    source = source.replace(
        "int main() {", f'extern "C" int {trial.entry_point}() {{', 1
    )
    marker = r"{\"task_count\":%u"
    replacement = (
        rf"{{\"shell_class\":\"{trial.spec.name}\","
        rf"\"schedule_id\":\"{trial.schedule_id}\","
        rf"\"trial_key\":\"{trial.key}\",\"task_count\":%u"
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
        integral=trial.integral,
    )


def _oracle_symbol_prefix(trial: ScheduleTrial) -> str:
    """Return a stable C symbol prefix shared by equivalent oracle mappings."""

    return (
        f"vibeqc_oracle_{trial.spec.name}_{trial.consumer.value}_"
        f"{trial.schedule.kind.value}_b{trial.schedule.block_threads}_"
        f"t{trial.schedule.component_tile}_w{trial.schedule.tasks_per_warp}"
        f"{trial.integral_suffix}"
    )


def emit_schedule_oracle_translation_unit(trial: ScheduleTrial) -> str:
    """Emit one separately compiled oracle shared by schedule code-shape knobs."""

    oracle_trial = _oracle_schedule_trial(trial)
    integral = oracle_trial.integral or build_integral_ir(
        oracle_trial.spec,
        consumers=(
            (KernelConsumer.FOCK, KernelConsumer.FORCE)
            if oracle_trial.consumer == KernelConsumer.FOCK
            else (KernelConsumer.FORCE,)
        ),
    )
    plan = build_fused_shell_plan(
        oracle_trial.spec,
        integral=integral,
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

    integral = trial.integral or build_integral_ir(
        trial.spec,
        consumers=(
            (KernelConsumer.FOCK, KernelConsumer.FORCE)
            if trial.consumer == KernelConsumer.FOCK
            else (KernelConsumer.FORCE,)
        ),
    )
    plan = build_fused_shell_plan(
        trial.spec,
        integral=integral,
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
    calls = "\n".join(f"  failures += {trial.entry_point}() != 0;" for trial in items)
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

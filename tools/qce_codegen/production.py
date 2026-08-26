"""Emit build-directory CUDA shards and host dispatch for accepted kernels."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .dppp_dispatch import emit_shell_class_fused_cuda
from .fused_schedule import build_fused_shell_plan
from .ir import (
    KernelConsumer,
    PairOrientation,
    PairStorage,
    ScheduleIR,
    ScheduleKind,
)
from .shell_spec import FUSED_SHELL_SPEC_BY_NAME, ShellClassSpec, shell_pair_class

_PRODUCTION_PRELUDE = r"""#include "scf/generated_shell_task.hpp"

#include <cuda_runtime.h>
#include <cmath>

template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) values[order] = 0.0;
  if (argument < 6.0) {
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

    def __post_init__(self) -> None:
        if not self.architecture.startswith("sm_"):
            raise ValueError("production architecture must use CUDA sm_ notation")
        if not self.consumers:
            raise ValueError("production kernel requires at least one consumer")
        # Force owns the canonical task ABI. Fock may share that source, but a
        # value-only entry cannot yet be emitted without its force companion.
        if KernelConsumer.FORCE not in self.consumers:
            raise ValueError("current production emitter requires a force consumer")


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


def load_production_kernel_selections(
    path: Path, architecture: str | None = None
) -> tuple[KernelSelection, ...]:
    """Load explicit kernel consumers and schedules for one GPU architecture."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == 1:
        # Preserve custom manifests while the repository migrates to v2.
        names = payload.get("shell_classes")
        if not isinstance(names, list) or not names:
            raise ValueError("production manifest requires shell_classes")
        acceptance = payload.get("acceptance")
        accepted_architecture = (
            acceptance.get("architecture")
            if isinstance(acceptance, dict)
            else None
        )
        selected_architecture = architecture or (
            accepted_architecture
            if isinstance(accepted_architecture, str)
            else "sm_120"
        )
        rows = [
            {
                "shell_class": name,
                "consumers": [KernelConsumer.FORCE.value],
            }
            for name in names
        ]
    elif payload.get("schema_version") == 2:
        selected_architecture = architecture or _default_architecture(payload)
        architectures = payload.get("architectures")
        if not isinstance(architectures, dict):
            raise ValueError("v2 production manifest requires architectures")
        profile = architectures.get(selected_architecture)
        if not isinstance(profile, dict):
            raise ValueError(
                f"production manifest has no profile for {selected_architecture}"
            )
        rows = profile.get("kernels")
        if not isinstance(rows, list) or not rows:
            raise ValueError("architecture profile requires a non-empty kernels list")
    else:
        raise ValueError("unsupported production shell manifest schema")

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
        schedule_payload = row.get("schedule")
        if schedule_payload is None:
            schedule = build_fused_shell_plan(spec, consumers=consumers).schedule
        else:
            schedule = _schedule_from_payload(schedule_payload)
            # Build the complete IR now so component coverage and block limits
            # fail during manifest loading rather than CUDA compilation.
            build_fused_shell_plan(spec, consumers=consumers, schedule=schedule)
        selections.append(
            KernelSelection(
                architecture=selected_architecture,
                spec=spec,
                consumers=consumers,
                schedule=schedule,
            )
        )
        seen.add(name)
    return tuple(selections)


def shell_class_index(spec: ShellClassSpec) -> int:
    """Return the production triangular quartet-class index."""

    first = shell_pair_class(*spec.angular[:2])
    second = shell_pair_class(*spec.angular[2:])
    high = max(first, second)
    low = min(first, second)
    return high * (high + 1) // 2 + low


def load_production_manifest(
    path: Path, architecture: str | None = None
) -> tuple[ShellClassSpec, ...]:
    """Compatibility view returning the ordered shell specifications."""

    return tuple(
        selection.spec
        for selection in load_production_kernel_selections(path, architecture)
    )


def load_production_fock_manifest(
    path: Path, architecture: str | None = None
) -> tuple[ShellClassSpec, ...]:
    """Compatibility view returning classes with generated Fock consumers."""

    return tuple(
        selection.spec
        for selection in load_production_kernel_selections(path, architecture)
        if KernelConsumer.FOCK in selection.consumers
    )


def _launch_wrapper(spec: ShellClassSpec) -> str:
    """Emit a stable C ABI wrapper around one generated persistent kernel."""

    class_name = spec.name[0].upper() + spec.name[1:]
    return f"""
static_assert(sizeof(Generated{class_name}ShellTask) ==
              sizeof(qce::scf::detail::GeneratedShellTask));
static_assert(alignof(Generated{class_name}ShellTask) ==
              alignof(qce::scf::detail::GeneratedShellTask));
static_assert(offsetof(Generated{class_name}ShellTask, primitive_begin) ==
              offsetof(qce::scf::detail::GeneratedShellTask, primitive_begin));
static_assert(offsetof(Generated{class_name}ShellTask, primitive_end) ==
              offsetof(qce::scf::detail::GeneratedShellTask, primitive_end));
static_assert(offsetof(Generated{class_name}ShellTask, ao_begin) ==
              offsetof(qce::scf::detail::GeneratedShellTask, ao_begin));
static_assert(offsetof(Generated{class_name}ShellTask, ao_coefficient_begin) ==
              offsetof(qce::scf::detail::GeneratedShellTask,
                       ao_coefficient_begin));
static_assert(offsetof(Generated{class_name}ShellTask, density_offset) ==
              offsetof(qce::scf::detail::GeneratedShellTask, density_offset));
static_assert(offsetof(Generated{class_name}ShellTask, spin_offset) ==
              offsetof(qce::scf::detail::GeneratedShellTask, spin_offset));
static_assert(offsetof(Generated{class_name}ShellTask, matrix_order) ==
              offsetof(qce::scf::detail::GeneratedShellTask, matrix_order));
static_assert(offsetof(Generated{class_name}ShellTask, shell_pair) ==
              offsetof(qce::scf::detail::GeneratedShellTask, shell_pair));
static_assert(
    offsetof(Generated{class_name}ShellTask, reversed_shell_pair_mask) ==
    offsetof(qce::scf::detail::GeneratedShellTask,
             reversed_shell_pair_mask));
static_assert(offsetof(Generated{class_name}ShellTask, shell) ==
              offsetof(qce::scf::detail::GeneratedShellTask, shell));
static_assert(offsetof(Generated{class_name}ShellTask, atom) ==
              offsetof(qce::scf::detail::GeneratedShellTask, atom));
static_assert(sizeof(Generated{class_name}PrimitivePairData) ==
              sizeof(qce::scf::detail::GeneratedPrimitivePairData));
static_assert(alignof(Generated{class_name}PrimitivePairData) ==
              alignof(qce::scf::detail::GeneratedPrimitivePairData));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, exponent_sum) ==
    offsetof(qce::scf::detail::GeneratedPrimitivePairData, exponent_sum));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, reduced_exponent) ==
    offsetof(qce::scf::detail::GeneratedPrimitivePairData, reduced_exponent));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, product_center) ==
    offsetof(qce::scf::detail::GeneratedPrimitivePairData, product_center));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, weighted_coefficient) ==
    offsetof(qce::scf::detail::GeneratedPrimitivePairData,
             weighted_coefficient));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, first_product_scale) ==
    offsetof(qce::scf::detail::GeneratedPrimitivePairData,
             first_product_scale));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, second_product_scale) ==
    offsetof(qce::scf::detail::GeneratedPrimitivePairData,
             second_product_scale));

extern "C" cudaError_t qce_launch_generated_{spec.name}(
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


def _fock_launch_wrapper(spec: ShellClassSpec) -> str:
    """Emit the stable C ABI wrapper for one generated Fock worker."""

    class_name = spec.name[0].upper() + spec.name[1:]
    return f"""
extern "C" cudaError_t qce_launch_generated_{spec.name}_fock(
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
        )
        body.append(emit_shell_class_fused_cuda(selection.spec, plan))
        body.append(_launch_wrapper(selection.spec))
        if KernelConsumer.FOCK in selection.consumers:
            body.append(_fock_launch_wrapper(selection.spec))
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
        spec = selection.spec
        plan = build_fused_shell_plan(
            spec,
            consumers=selection.consumers,
            schedule=selection.schedule,
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
            (angular_order + 1)
            * (angular_order + 2)
            * (angular_order + 3)
            // 6
        )
        block_threads = (
            (max(spec.component_count, value_state_count) + 31) // 32
        ) * 32
        fock_rows.append(
            f'    {{"{spec.name}", {shell_class_index(spec)}U, '
            f"{angular_order}U, {block_threads}U, 1U, "
            f"{spec.component_count}U}},"
        )
    return f"""#ifndef QCE_GENERATED_SHELL_REGISTRY_HPP
#define QCE_GENERATED_SHELL_REGISTRY_HPP

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace qce::scf::generated {{

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

/** Return the exact-class bit mask selected by QCE_AOT_SHELL_CLASSES. */
std::uint64_t enabled_shell_class_mask() noexcept;

/** Return the Fock-class mask selected by QCE_AOT_FOCK_SHELL_CLASSES. */
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

}}  // namespace qce::scf::generated

#endif
"""


def emit_registry_source(
    specifications: Iterable[ShellClassSpec | KernelSelection],
) -> str:
    """Emit environment-controlled dispatch without handwritten class switches."""

    selections = tuple(map(_as_selection, specifications))
    specs = tuple(item.spec for item in selections)
    fock_specs = tuple(
        item.spec
        for item in selections
        if KernelConsumer.FOCK in item.consumers
    )
    declarations = "\n".join(
        f"""extern "C" cudaError_t qce_launch_generated_{spec.name}(
    cudaStream_t, bool, unsigned, const void*, const std::uint32_t*,
    const std::int64_t*, const void*, const double*, const void*, double,
    const double*, const double*, double*, const std::uint32_t*,
    std::uint32_t*);"""
        for spec in specs
    )
    cases = "\n".join(
        f"""    case {shell_class_index(spec)}U:
      return qce_launch_generated_{spec.name}(
          stream, unrestricted, worker_blocks, tasks, task_offset,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions,
          screening_tolerance, schwarz_bounds, density, forces, task_count,
          task_head);"""
        for spec in specs
    )
    fock_declarations = "\n".join(
        f"""extern "C" cudaError_t qce_launch_generated_{spec.name}_fock(
    cudaStream_t, bool, unsigned, const void*, const std::uint32_t*,
    const std::int64_t*, const void*, const double*, const void*, double,
    const double*, const double*, double*, const std::uint32_t*,
    std::uint32_t*);"""
        for spec in fock_specs
    )
    fock_cases = "\n".join(
        f"""    case {shell_class_index(spec)}U:
      return qce_launch_generated_{spec.name}_fock(
          stream, unrestricted, worker_blocks, tasks, task_offset,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density, fock,
          task_count, task_head);"""
        for spec in fock_specs
    )
    return f"""#include "qce_generated_shell_registry.hpp"

#include <cstdlib>
#include <cstring>

{declarations}
{fock_declarations}

namespace qce::scf::generated {{
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
  const char* selection = std::getenv("QCE_AOT_SHELL_CLASSES");
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
  const char* selection = std::getenv("QCE_AOT_FOCK_SHELL_CLASSES");
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

}}  // namespace qce::scf::generated
"""


def _write_if_changed(path: Path, content: str) -> None:
    """Preserve timestamps when deterministic regeneration is byte-identical."""

    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def write_production_bundle(
    manifest: Path,
    output_directory: Path,
    shard_count: int,
    architecture: str | None = None,
) -> tuple[Path, ...]:
    """Write deterministic build artifacts and return every generated path."""

    if shard_count < 1:
        raise ValueError("production shard count must be positive")
    selections = load_production_kernel_selections(manifest, architecture)
    shards = [[] for _ in range(shard_count)]
    for index, selection in enumerate(selections):
        shards[index % shard_count].append(selection)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, shard in enumerate(shards):
        path = output_directory / f"qce_generated_shell_shard_{index}.cu"
        _write_if_changed(path, emit_production_shard(shard))
        outputs.append(path)
    header = output_directory / "qce_generated_shell_registry.hpp"
    source = output_directory / "qce_generated_shell_registry.cu"
    _write_if_changed(header, emit_registry_header(selections))
    _write_if_changed(source, emit_registry_source(selections))
    outputs.extend((header, source))
    return tuple(outputs)

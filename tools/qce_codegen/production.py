"""Emit build-directory CUDA shards and host dispatch for accepted kernels."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .dppp_dispatch import emit_shell_class_fused_cuda
from .fused_schedule import build_fused_shell_plan
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


def shell_class_index(spec: ShellClassSpec) -> int:
    """Return the production triangular quartet-class index."""

    first = shell_pair_class(*spec.angular[:2])
    second = shell_pair_class(*spec.angular[2:])
    high = max(first, second)
    low = min(first, second)
    return high * (high + 1) // 2 + low


def load_production_manifest(path: Path) -> tuple[ShellClassSpec, ...]:
    """Load and validate the ordered list used by normal production builds."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported production shell manifest schema")
    names = payload.get("shell_classes")
    if not isinstance(names, list) or not names:
        raise ValueError("production manifest requires a non-empty shell_classes list")
    specifications = []
    seen = set()
    for name in names:
        if not isinstance(name, str) or name not in FUSED_SHELL_SPEC_BY_NAME:
            raise ValueError(f"unsupported production shell class {name!r}")
        if name in seen:
            raise ValueError(f"duplicate production shell class {name!r}")
        specifications.append(FUSED_SHELL_SPEC_BY_NAME[name])
        seen.add(name)
    return tuple(specifications)


def load_production_fock_manifest(path: Path) -> tuple[ShellClassSpec, ...]:
    """Load the subset whose generated sources also provide Fock workers."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported production shell manifest schema")
    names = payload.get("fock_shell_classes", [])
    if not isinstance(names, list):
        raise ValueError("fock_shell_classes must be a list")
    specifications = []
    seen = set()
    for name in names:
        if not isinstance(name, str) or name not in FUSED_SHELL_SPEC_BY_NAME:
            raise ValueError(f"unsupported production Fock shell class {name!r}")
        if name in seen:
            raise ValueError(f"duplicate production Fock shell class {name!r}")
        specifications.append(FUSED_SHELL_SPEC_BY_NAME[name])
        seen.add(name)
    return tuple(specifications)


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
    const void* tasks, const std::int64_t* primitive_pair_offsets,
    const void* primitive_pairs, const double* ao_coefficients,
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
        density, forces, task_count, task_head);
  }} else {{
    generated_{spec.name}_shell_class_force_rhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}BlockThreads, 0, stream>>>(
        typed_tasks, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, forces, task_count, task_head);
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
    const void* tasks, const std::int64_t* primitive_pair_offsets,
    const void* primitive_pairs, const double* ao_coefficients,
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
        density, fock, task_count, task_head);
  }} else {{
    generated_{spec.name}_shell_class_fock_rhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}FockBlockThreads, 0, stream>>>(
        typed_tasks, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, fock, task_count, task_head);
  }}
  return cudaPeekAtLastError();
}}
"""


def emit_production_shard(
    specifications: Iterable[ShellClassSpec],
    fock_specifications: Iterable[ShellClassSpec] = (),
) -> str:
    """Emit one CUDA TU containing a deterministic subset of accepted classes."""

    specs = tuple(specifications)
    fock_names = {spec.name for spec in fock_specifications}
    body = [_PRODUCTION_PRELUDE]
    for spec in specs:
        include_fock = spec.name in fock_names
        body.append(
            emit_shell_class_fused_cuda(spec, include_fock=include_fock)
        )
        body.append(_launch_wrapper(spec))
        if include_fock:
            body.append(_fock_launch_wrapper(spec))
    if not specs:
        body.append("// Empty deterministic shard reserved for stable CMake outputs.\n")
    return "".join(body)


def emit_registry_header(
    specifications: Iterable[ShellClassSpec],
    fock_specifications: Iterable[ShellClassSpec] = (),
) -> str:
    """Emit production metadata and the host launch API consumed by cuda_rhf."""

    specs = tuple(specifications)
    fock_specs = tuple(fock_specifications)
    rows = []
    for spec in specs:
        plan = build_fused_shell_plan(spec)
        rows.append(
            f'    {{"{spec.name}", {shell_class_index(spec)}U, '
            f"{sum(spec.angular)}U, {plan.block_threads}U}},"
        )
    fock_rows = []
    for spec in fock_specs:
        angular_order = sum(spec.angular)
        value_state_count = (
            (angular_order + 1) * (angular_order + 2) *
            (angular_order + 3) // 6
        )
        block_threads = (
            (max(spec.component_count, value_state_count) + 31) // 32
        ) * 32
        fock_rows.append(
            f'    {{"{spec.name}", {shell_class_index(spec)}U, '
            f"{angular_order}U, {block_threads}U}},"
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
}};

inline constexpr ShellKernelMetadata kShellKernels[] = {{
{chr(10).join(rows)}
}};
inline constexpr std::size_t kShellKernelCount =
    sizeof(kShellKernels) / sizeof(kShellKernels[0]);

inline constexpr std::array<ShellKernelMetadata, {len(fock_specs)}>
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
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* forces, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept;

/** Launch one generated coefficient-only Fock worker by exact class. */
cudaError_t launch_shell_class_fock(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept;

}}  // namespace qce::scf::generated

#endif
"""


def emit_registry_source(
    specifications: Iterable[ShellClassSpec],
    fock_specifications: Iterable[ShellClassSpec] = (),
) -> str:
    """Emit environment-controlled dispatch without handwritten class switches."""

    specs = tuple(specifications)
    fock_specs = tuple(fock_specifications)
    declarations = "\n".join(
        f"""extern "C" cudaError_t qce_launch_generated_{spec.name}(
    cudaStream_t, bool, unsigned, const void*, const std::int64_t*,
    const void*, const double*, const void*, double, const double*,
    const double*, double*,
    const std::uint32_t*, std::uint32_t*);"""
        for spec in specs
    )
    cases = "\n".join(
        f"""    case {shell_class_index(spec)}U:
      return qce_launch_generated_{spec.name}(
          stream, unrestricted, worker_blocks, tasks, primitive_pair_offsets,
          primitive_pairs, ao_coefficients, atom_positions,
          screening_tolerance, schwarz_bounds, density, forces, task_count,
          task_head);"""
        for spec in specs
    )
    fock_declarations = "\n".join(
        f"""extern "C" cudaError_t qce_launch_generated_{spec.name}_fock(
    cudaStream_t, bool, unsigned, const void*, const std::int64_t*,
    const void*, const double*, const void*, double, const double*,
    const double*, double*,
    const std::uint32_t*, std::uint32_t*);"""
        for spec in fock_specs
    )
    fock_cases = "\n".join(
        f"""    case {shell_class_index(spec)}U:
      return qce_launch_generated_{spec.name}_fock(
          stream, unrestricted, worker_blocks, tasks, primitive_pair_offsets,
          primitive_pairs, ao_coefficients, atom_positions,
          screening_tolerance, schwarz_bounds, density, fock, task_count,
          task_head);"""
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
    manifest: Path, output_directory: Path, shard_count: int
) -> tuple[Path, ...]:
    """Write deterministic build artifacts and return every generated path."""

    if shard_count < 1:
        raise ValueError("production shard count must be positive")
    specifications = load_production_manifest(manifest)
    fock_specifications = load_production_fock_manifest(manifest)
    force_names = {spec.name for spec in specifications}
    missing_force_sources = [
        spec.name for spec in fock_specifications if spec.name not in force_names
    ]
    if missing_force_sources:
        raise ValueError(
            "Fock shell classes must also provide the shared force source: "
            + ", ".join(missing_force_sources)
        )
    fock_names = {spec.name for spec in fock_specifications}
    shards = [[] for _ in range(shard_count)]
    for index, specification in enumerate(specifications):
        shards[index % shard_count].append(specification)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, shard in enumerate(shards):
        path = output_directory / f"qce_generated_shell_shard_{index}.cu"
        shard_fock = [spec for spec in shard if spec.name in fock_names]
        _write_if_changed(path, emit_production_shard(shard, shard_fock))
        outputs.append(path)
    header = output_directory / "qce_generated_shell_registry.hpp"
    source = output_directory / "qce_generated_shell_registry.cu"
    _write_if_changed(
        header, emit_registry_header(specifications, fock_specifications)
    )
    _write_if_changed(
        source, emit_registry_source(specifications, fock_specifications)
    )
    outputs.extend((header, source))
    return tuple(outputs)

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
    for (unsigned order = 0; order <= MaximumOrder; ++order) {
      double term = 1.0;
      double sum = 0.0;
      for (unsigned k = 0; k < 80U; ++k) {
        sum += term / static_cast<double>(2U * order + 2U * k + 1U);
        term *= -argument / static_cast<double>(k + 1U);
        if (fabs(term) < 1.0e-18) break;
      }
      values[order] = sum;
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


def _launch_wrapper(spec: ShellClassSpec) -> str:
    """Emit a stable C ABI wrapper around one generated persistent kernel."""

    class_name = spec.name[0].upper() + spec.name[1:]
    return f"""
static_assert(sizeof(Generated{class_name}ShellTask) ==
              sizeof(qce::scf::detail::GeneratedShellTask));
static_assert(alignof(Generated{class_name}ShellTask) ==
              alignof(qce::scf::detail::GeneratedShellTask));

extern "C" cudaError_t qce_launch_generated_{spec.name}(
    cudaStream_t stream, bool unrestricted, unsigned worker_blocks,
    const void* tasks, const double* primitive_exponents,
    const double* primitive_coefficients, const double* ao_coefficients,
    const void* atom_positions, double screening_tolerance,
    const double* schwarz_bounds, const double* density, double* forces,
    const std::uint32_t* task_count, std::uint32_t* task_head) {{
  if (worker_blocks == 0U) return cudaSuccess;
  const auto* typed_tasks =
      static_cast<const Generated{class_name}ShellTask*>(tasks);
  const auto* typed_positions =
      static_cast<const Generated{class_name}Vec3*>(atom_positions);
  if (unrestricted) {{
    generated_{spec.name}_shell_class_force_uhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}BlockThreads, 0, stream>>>(
        typed_tasks, primitive_exponents, primitive_coefficients,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, forces, task_count, task_head);
  }} else {{
    generated_{spec.name}_shell_class_force_rhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}BlockThreads, 0, stream>>>(
        typed_tasks, primitive_exponents, primitive_coefficients,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, forces, task_count, task_head);
  }}
  return cudaPeekAtLastError();
}}
"""


def emit_production_shard(specifications: Iterable[ShellClassSpec]) -> str:
    """Emit one CUDA TU containing a deterministic subset of accepted classes."""

    specs = tuple(specifications)
    body = [_PRODUCTION_PRELUDE]
    for spec in specs:
        body.append(emit_shell_class_fused_cuda(spec))
        body.append(_launch_wrapper(spec))
    if not specs:
        body.append("// Empty deterministic shard reserved for stable CMake outputs.\n")
    return "".join(body)


def emit_registry_header(specifications: Iterable[ShellClassSpec]) -> str:
    """Emit production metadata and the host launch API consumed by cuda_rhf."""

    specs = tuple(specifications)
    rows = []
    for spec in specs:
        plan = build_fused_shell_plan(spec)
        rows.append(
            f'    {{"{spec.name}", {shell_class_index(spec)}U, '
            f"{sum(spec.angular)}U, {plan.block_threads}U}},"
        )
    return f"""#ifndef QCE_GENERATED_SHELL_REGISTRY_HPP
#define QCE_GENERATED_SHELL_REGISTRY_HPP

#include <cuda_runtime_api.h>

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

/** Return the exact-class bit mask selected by QCE_AOT_SHELL_CLASSES. */
std::uint64_t enabled_shell_class_mask() noexcept;

/** Launch one generated persistent kernel selected by exact class index. */
cudaError_t launch_shell_class(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const double* primitive_exponents, const double* primitive_coefficients,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* forces, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept;

}}  // namespace qce::scf::generated

#endif
"""


def emit_registry_source(specifications: Iterable[ShellClassSpec]) -> str:
    """Emit environment-controlled dispatch without handwritten class switches."""

    specs = tuple(specifications)
    declarations = "\n".join(
        f"""extern "C" cudaError_t qce_launch_generated_{spec.name}(
    cudaStream_t, bool, unsigned, const void*, const double*, const double*,
    const double*, const void*, double, const double*, const double*, double*,
    const std::uint32_t*, std::uint32_t*);"""
        for spec in specs
    )
    cases = "\n".join(
        f"""    case {shell_class_index(spec)}U:
      return qce_launch_generated_{spec.name}(
          stream, unrestricted, worker_blocks, tasks, primitive_exponents,
          primitive_coefficients, ao_coefficients, atom_positions,
          screening_tolerance, schwarz_bounds, density, forces, task_count,
          task_head);"""
        for spec in specs
    )
    return f"""#include "qce_generated_shell_registry.hpp"

#include <cstdlib>
#include <cstring>

{declarations}

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

cudaError_t launch_shell_class(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const double* primitive_exponents, const double* primitive_coefficients,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* forces, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept {{
  switch (shell_class) {{
{cases}
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
    shards = [[] for _ in range(shard_count)]
    for index, specification in enumerate(specifications):
        shards[index % shard_count].append(specification)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, shard in enumerate(shards):
        path = output_directory / f"qce_generated_shell_shard_{index}.cu"
        _write_if_changed(path, emit_production_shard(shard))
        outputs.append(path)
    header = output_directory / "qce_generated_shell_registry.hpp"
    source = output_directory / "qce_generated_shell_registry.cu"
    _write_if_changed(header, emit_registry_header(specifications))
    _write_if_changed(source, emit_registry_source(specifications))
    outputs.extend((header, source))
    return tuple(outputs)

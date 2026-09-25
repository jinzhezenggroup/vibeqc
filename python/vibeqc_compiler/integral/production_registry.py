"""Production registry serialization for generated integral kernels.

This leaf owns registry metadata and dispatch source text only.  Production
source emission and filesystem/bundle orchestration consume these serializers
without being imported back into this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibeqc_compiler.common.cuda_target import cuda_target_info

from .capabilities import CAPABILITY_MIXED_FOCK, CAPABILITY_STREAMING_FOCK
from .cuda_schedule import ScheduleKind
from .fused_schedule import build_fused_shell_plan
from .ir import KernelConsumer
from .production_cost import shell_class_index
from .production_profile import (
    ProfileMatch,
    ResolvedProductionProfile,
    _profile_identifier,
)
from .production_selection import (
    KernelSelection,
    _as_selection,
    _selection_integral,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .shell_spec import ShellClassSpec


def _stable_selection_order(
    specifications: Iterable[ShellClassSpec | KernelSelection],
) -> tuple[KernelSelection, ...]:
    """Materialize selections in canonical class order for registry sources."""

    return tuple(
        sorted(
            map(_as_selection, specifications),
            key=lambda item: (shell_class_index(item.spec), item.spec.name),
        )
    )


def _fock_tasks_per_claim(selection: KernelSelection) -> int:
    """Expose packed queue width without changing other schedules' grid policy.

    Packed value workers claim a warp of independent shell tasks at once.
    Their host capacity bound counts tasks, not CTAs. A conservative width of
    one preserves existing launch policy for component and subgroup consumers,
    including implicit Rys Fock schedules that differ from the force schedule.
    """
    schedule = selection.fock_schedule or selection.schedule
    if schedule.kind == ScheduleKind.PACKED_TASKS:
        return schedule.tasks_per_block
    return 1


def emit_registry_header(
    specifications: Iterable[ShellClassSpec | KernelSelection],
) -> str:
    """Emit production metadata and the host launch API consumed by cuda_rhf."""

    selections = _stable_selection_order(specifications)
    rows = []
    for selection in selections:
        if KernelConsumer.FORCE not in selection.consumers:
            continue
        spec = selection.spec
        integral = _selection_integral(selection)
        plan = build_fused_shell_plan(
            spec,
            integral=integral,
            schedule=selection.schedule,
            target=cuda_target_info(selection.architecture),
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
    mixed_fock_rows = []
    for selection in selections:
        if KernelConsumer.FOCK not in selection.consumers:
            continue
        spec = selection.spec
        angular_order = sum(spec.angular)
        value_state_count = (
            (angular_order + 1) * (angular_order + 2) * (angular_order + 3) // 6
        )
        block_threads = ((max(spec.component_count, value_state_count) + 31) // 32) * 32
        row = (
            f'    {{"{spec.name}", {shell_class_index(spec)}U, '
            f"{angular_order}U, {block_threads}U, 1U, "
            f"{spec.component_count}U, {_fock_tasks_per_claim(selection)}U}},"
        )
        fock_rows.append(row)
        if selection.has_capability(CAPABILITY_MIXED_FOCK):
            mixed_fock_rows.append(row)
    return f"""#ifndef VIBEQC_GENERATED_SHELL_REGISTRY_HPP
#define VIBEQC_GENERATED_SHELL_REGISTRY_HPP

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace vibeqc::scf::generated {{

struct ShellKernelMetadata {{
  const char* name;
  unsigned shell_class;
  unsigned angular_order;
  unsigned block_threads;
  unsigned consumer_mask;
  unsigned component_tile;
  // Packed Fock claim width; other consumers retain the conservative grid.
  unsigned fock_tasks_per_claim{{1}};
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

inline constexpr std::array<ShellKernelMetadata, {len(mixed_fock_rows)}>
    kMixedFockShellKernels{{{{
{chr(10).join(mixed_fock_rows)}
}}}};
inline constexpr std::size_t kMixedFockShellKernelCount =
    kMixedFockShellKernels.size();

/** Return the exact-class bit mask selected by VIBEQC_AOT_SHELL_CLASSES. */
std::uint64_t enabled_shell_class_mask() noexcept;

/** Return the Fock-class mask selected by VIBEQC_AOT_FOCK_SHELL_CLASSES. */
std::uint64_t enabled_fock_shell_class_mask() noexcept;

/** Return the generated mixed-Fock capability mask. */
std::uint64_t enabled_mixed_fock_shell_class_mask() noexcept;

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

/** Launch one generated mixed-precision Fock worker by exact class. */
cudaError_t launch_shell_class_mixed_fock(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept;

/** Launch one fixed-storage resident-bra Fock stream by exact class. */
cudaError_t launch_shell_class_streaming_fock(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* shell_pair_stream,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, bool mixed_precision_enabled,
    double fp64_threshold, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head,
    unsigned long long* fp64_work_count,
    unsigned long long* fp32_work_count) noexcept;

/** Launch the optional resident-bra ppps force worker. */
cudaError_t launch_ppps_resident(
    cudaStream_t stream, bool unrestricted, const void* resident_tasks,
    const void* ket_tasks, const std::int64_t* primitive_pair_offsets,
    const void* primitive_pairs, const double* ao_coefficients,
    const void* atom_positions, double screening_tolerance,
    const double* schwarz_bounds, const double* density, double* forces,
    std::size_t task_count) noexcept;

}}  // namespace vibeqc::scf::generated

#endif
"""


def emit_registry_source(
    specifications: Iterable[ShellClassSpec | KernelSelection],
) -> str:
    """Emit environment-controlled dispatch without handwritten class switches."""

    selections = _stable_selection_order(specifications)
    specs = tuple(item.spec for item in selections)
    fock_specs = tuple(
        item.spec for item in selections if KernelConsumer.FOCK in item.consumers
    )
    mixed_fock_specs = tuple(
        item.spec
        for item in selections
        if KernelConsumer.FOCK in item.consumers
        and item.has_capability(CAPABILITY_MIXED_FOCK)
    )
    streaming_fock_specs = tuple(
        item.spec
        for item in selections
        if KernelConsumer.FOCK in item.consumers
        and item.has_capability(CAPABILITY_STREAMING_FOCK)
    )
    resident_selection = next(
        (item for item in selections if item.resident_force_recurrence is not None),
        None,
    )
    declarations = "\n".join(
        f"""extern "C" cudaError_t vibeqc_launch_generated_{spec.name}(
    cudaStream_t, bool, unsigned, const void*, const std::uint32_t*,
    const std::int64_t*, const void*, const double*, const void*, double,
    const double*, const double*, double*, const std::uint32_t*,
    std::uint32_t*);"""
        for spec in specs
    )
    cases = "\n".join(
        f"""    case {shell_class_index(spec)}U:
      return vibeqc_launch_generated_{spec.name}(
          stream, unrestricted, worker_blocks, tasks, task_offset,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions,
          screening_tolerance, schwarz_bounds, density, forces, task_count,
          task_head);"""
        for spec in specs
    )
    fock_declarations = "\n".join(
        f"""extern "C" cudaError_t vibeqc_launch_generated_{spec.name}_fock(
    cudaStream_t, bool, unsigned, const void*, const std::uint32_t*,
    const std::int64_t*, const void*, const double*, const void*, double,
    const double*, const double*, double*, const std::uint32_t*,
    std::uint32_t*);"""
        for spec in fock_specs
    )
    fock_cases = "\n".join(
        f"""    case {shell_class_index(spec)}U:
      return vibeqc_launch_generated_{spec.name}_fock(
          stream, unrestricted, worker_blocks, tasks, task_offset,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density, fock,
          task_count, task_head);"""
        for spec in fock_specs
    )
    mixed_fock_declarations = "\n".join(
        f"""extern "C" cudaError_t vibeqc_launch_generated_{spec.name}_mixed_fock(
    cudaStream_t, bool, unsigned, const void*, const std::uint32_t*,
    const std::int64_t*, const void*, const double*, const void*, double,
    const double*, const double*, double*, const std::uint32_t*,
    std::uint32_t*);"""
        for spec in mixed_fock_specs
    )
    mixed_fock_cases = "\n".join(
        f"""    case {shell_class_index(spec)}U:
      return vibeqc_launch_generated_{spec.name}_mixed_fock(
          stream, unrestricted, worker_blocks, tasks, task_offset,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density, fock,
          task_count, task_head);"""
        for spec in mixed_fock_specs
    )
    streaming_fock_declarations = "\n".join(
        f"""extern "C" cudaError_t vibeqc_launch_generated_{spec.name}_streaming_fock(
    cudaStream_t, bool, unsigned, const void*, const std::int64_t*,
    const void*, const double*, const void*, double, bool, double,
    const double*, const double*, double*, std::uint32_t*,
    unsigned long long*, unsigned long long*);"""
        for spec in streaming_fock_specs
    )
    streaming_fock_cases = "\n".join(
        f"""    case {shell_class_index(spec)}U:
      return vibeqc_launch_generated_{spec.name}_streaming_fock(
          stream, unrestricted, worker_blocks, shell_pair_stream,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions, screening_tolerance, mixed_precision_enabled,
          fp64_threshold, schwarz_bounds, density, fock, bra_head,
          fp64_work_count, fp32_work_count);"""
        for spec in streaming_fock_specs
    )
    resident_declaration = ""
    resident_launch = "return cudaErrorNotSupported;"
    if resident_selection is not None:
        resident_declaration = (
            'extern "C" cudaError_t vibeqc_launch_ppps_resident('
            f"{_resident_launch_parameter_declaration()});"
        )
        resident_launch = (
            f"return vibeqc_launch_ppps_resident({_resident_launch_argument_list()});"
        )
    return f"""#include "vibeqc_generated_shell_registry.hpp"

#include <cstdlib>
#include <cstring>

{declarations}
{fock_declarations}
{mixed_fock_declarations}
{streaming_fock_declarations}
{resident_declaration}

namespace vibeqc::scf::generated {{
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
  const char* selection = std::getenv("VIBEQC_AOT_SHELL_CLASSES");
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
  const char* selection = std::getenv("VIBEQC_AOT_FOCK_SHELL_CLASSES");
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

std::uint64_t enabled_mixed_fock_shell_class_mask() noexcept {{
  const char* selection =
      std::getenv("VIBEQC_AOT_MIXED_FOCK_SHELL_CLASSES");
  const bool all = selection == nullptr || *selection == '\\0' ||
                   std::strcmp(selection, "all") == 0;
  if (!all && std::strcmp(selection, "none") == 0) return 0;
  std::uint64_t mask = 0;
  for (const ShellKernelMetadata& kernel : kMixedFockShellKernels) {{
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

cudaError_t launch_shell_class_mixed_fock(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept {{
  switch (shell_class) {{
{mixed_fock_cases}
    default: return cudaErrorInvalidValue;
  }}
}}

cudaError_t launch_shell_class_streaming_fock(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* shell_pair_stream,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, bool mixed_precision_enabled,
    double fp64_threshold, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head,
    unsigned long long* fp64_work_count,
    unsigned long long* fp32_work_count) noexcept {{
  switch (shell_class) {{
{streaming_fock_cases}
    default: return cudaErrorInvalidValue;
  }}
}}

cudaError_t launch_ppps_resident(
    {_resident_launch_parameter_declaration()}) noexcept {{
  {resident_launch}
}}

}}  // namespace vibeqc::scf::generated
"""


def emit_multi_registry_header(
    profiles: Iterable[ResolvedProductionProfile],
) -> str:
    """Emit immutable metadata for every independently compiled profile."""

    items = tuple(profiles)
    profile_rows = []
    kernel_rows = []
    for profile in items:
        profile_rows.append(
            f'    {{"{profile.profile}", "{profile.target.architecture}", '
            f"{profile.target.compute_capability_major}, "
            f"{profile.target.compute_capability_minor}, "
            f"{'true' if profile.tuned else 'false'}, "
            f"{'true' if profile.portable else 'false'}, "
            f"{'true' if profile.match == ProfileMatch.COMPATIBLE else 'false'}}},"
        )
        for selection in _stable_selection_order(profile.selections):
            integral = _selection_integral(selection)
            plan = build_fused_shell_plan(
                selection.spec,
                integral=integral,
                schedule=selection.schedule,
                target=profile.target,
            )
            consumer_mask = sum(
                1 << list(KernelConsumer).index(consumer)
                for consumer in selection.consumers
            )
            kernel_rows.append(
                f'    {{"{profile.target.architecture}", "{selection.spec.name}", '
                f"{shell_class_index(selection.spec)}U, {plan.block_threads}U, "
                f"{consumer_mask}U, {plan.schedule.component_tile}U}},"
            )
    return f"""#ifndef VIBEQC_GENERATED_SHELL_REGISTRY_HPP
#define VIBEQC_GENERATED_SHELL_REGISTRY_HPP

#include "scf/aot_shell_registry.hpp"

#include <array>
#include <cstddef>

namespace vibeqc::scf::generated {{

struct CompiledKernelMetadata {{
  const char* target_architecture;
  const char* name;
  unsigned shell_class;
  unsigned block_threads;
  unsigned consumer_mask;
  unsigned component_tile;
}};

inline constexpr std::array<ProfileInfo, {len(profile_rows)}> kCompiledProfiles{{{{
{chr(10).join(profile_rows)}
}}}};
inline constexpr std::size_t kCompiledProfileCount =
    kCompiledProfiles.size();

inline constexpr std::array<CompiledKernelMetadata, {len(kernel_rows)}>
    kCompiledShellKernels{{{{
{chr(10).join(kernel_rows)}
}}}};
inline constexpr std::size_t kCompiledShellKernelCount =
    kCompiledShellKernels.size();

}}  // namespace vibeqc::scf::generated

#endif
"""


def _launch_parameter_declaration() -> str:
    return """unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* output, const std::uint32_t* task_count,
    std::uint32_t* task_head"""


def _resident_launch_parameter_declaration() -> str:
    """Return the stable resident-bra launch signature shared by emitters."""

    return """cudaStream_t stream, bool unrestricted,
    const void* resident_tasks, const void* ket_tasks,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* forces, unsigned block_threads,
    std::size_t task_count"""


def _streaming_fock_launch_parameter_declaration() -> str:
    """Return the fixed-storage Fock streaming dispatch signature."""

    return """unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* shell_pair_stream,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, bool mixed_precision_enabled,
    double fp64_threshold, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head,
    unsigned long long* fp64_work_count,
    unsigned long long* fp32_work_count"""


def _launch_argument_list() -> str:
    return """stream, unrestricted, worker_blocks, tasks, task_offset,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density, output,
          task_count, task_head"""


def _resident_launch_argument_list() -> str:
    """Return arguments for forwarding a resident-bra launch."""

    return """stream, unrestricted, resident_tasks, ket_tasks,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density,
          forces, block_threads, task_count"""


def _streaming_fock_launch_argument_list() -> str:
    """Return arguments for forwarding one streaming Fock launch."""

    return """stream, unrestricted, worker_blocks, shell_pair_stream,
          primitive_pair_offsets, primitive_pairs, ao_coefficients,
          atom_positions, screening_tolerance, mixed_precision_enabled,
          fp64_threshold, schwarz_bounds, density, fock, bra_head,
          fp64_work_count, fp32_work_count"""


def emit_multi_registry_source(
    profiles: Iterable[ResolvedProductionProfile],
) -> str:
    """Emit one per-device profile resolver and collision-free launch table."""

    items = tuple(profiles)
    declarations = []
    helpers = []
    kernel_arrays = []
    kernel_sets = []
    launch_parameters = _launch_parameter_declaration()
    launch_arguments = _launch_argument_list()
    streaming_fock_parameters = _streaming_fock_launch_parameter_declaration()
    streaming_fock_arguments = _streaming_fock_launch_argument_list()
    for index, profile in enumerate(items):
        identifier = _profile_identifier(profile.target.architecture)
        force_cases = []
        fock_cases = []
        mixed_fock_cases = []
        streaming_fock_cases = []
        resident_symbol = None
        force_names = []
        fock_names = []
        mixed_fock_names = []
        force_mask = 0
        fock_mask = 0
        mixed_fock_mask = 0
        for selection in _stable_selection_order(profile.selections):
            shell_class = shell_class_index(selection.spec)
            integral = _selection_integral(selection)
            plan = build_fused_shell_plan(
                selection.spec,
                integral=integral,
                schedule=selection.schedule,
                target=profile.target,
            )
            consumer_mask = sum(
                1 << list(KernelConsumer).index(consumer)
                for consumer in selection.consumers
            )
            force_symbol = f"vibeqc_launch_{identifier}_generated_{selection.spec.name}"
            declarations.append(
                f'extern "C" cudaError_t {force_symbol}('
                "cudaStream_t, bool, unsigned, const void*, const std::uint32_t*, "
                "const std::int64_t*, const void*, const double*, const void*, "
                "double, const double*, const double*, double*, "
                "const std::uint32_t*, std::uint32_t*);"
            )
            force_cases.append(
                f"    case {shell_class}U:\n"
                f"      return {force_symbol}({launch_arguments});"
            )
            # Fock-only rows deliberately retain a dormant force symbol: the
            # bounded spd path can call the same exact recurrence without a
            # generic integral fallback.  Keep those symbols out of ordinary
            # force metadata so fixed-topology plans do not reserve descriptor
            # storage for classes owned by their handwritten force routes.
            if KernelConsumer.FORCE in selection.consumers:
                force_names.append(
                    f'    {{"{selection.spec.name}", {shell_class}U, '
                    f"{sum(selection.spec.angular)}U, {plan.block_threads}U, "
                    f"{consumer_mask}U, {plan.schedule.component_tile}U}},"
                )
                force_mask |= 1 << shell_class
            if KernelConsumer.FOCK in selection.consumers:
                fock_symbol = f"{force_symbol}_fock"
                declarations.append(
                    f'extern "C" cudaError_t {fock_symbol}('
                    "cudaStream_t, bool, unsigned, const void*, "
                    "const std::uint32_t*, const std::int64_t*, const void*, "
                    "const double*, const void*, double, const double*, "
                    "const double*, double*, const std::uint32_t*, "
                    "std::uint32_t*);"
                )
                fock_cases.append(
                    f"    case {shell_class}U:\n"
                    f"      return {fock_symbol}({launch_arguments});"
                )
                if selection.has_capability(CAPABILITY_MIXED_FOCK):
                    mixed_fock_symbol = f"{force_symbol}_mixed_fock"
                    declarations.append(
                        f'extern "C" cudaError_t {mixed_fock_symbol}('
                        "cudaStream_t, bool, unsigned, const void*, "
                        "const std::uint32_t*, const std::int64_t*, const void*, "
                        "const double*, const void*, double, const double*, "
                        "const double*, double*, const std::uint32_t*, "
                        "std::uint32_t*);"
                    )
                    mixed_fock_cases.append(
                        f"    case {shell_class}U:\n"
                        f"      return {mixed_fock_symbol}({launch_arguments});"
                    )
                    mixed_fock_names.append(
                        f'    {{"{selection.spec.name}", {shell_class}U, '
                        f"{sum(selection.spec.angular)}U, {plan.block_threads}U, "
                        f"{consumer_mask}U, {plan.schedule.component_tile}U}},"
                    )
                    mixed_fock_mask |= 1 << shell_class
                if selection.has_capability(CAPABILITY_STREAMING_FOCK):
                    streaming_fock_symbol = f"{force_symbol}_streaming_fock"
                    declarations.append(
                        f'extern "C" cudaError_t {streaming_fock_symbol}('
                        "cudaStream_t, bool, unsigned, const void*, "
                        "const std::int64_t*, const void*, const double*, "
                        "const void*, double, bool, double, const double*, "
                        "const double*, double*, std::uint32_t*, "
                        "unsigned long long*, unsigned long long*);"
                    )
                    streaming_fock_cases.append(
                        f"    case {shell_class}U:\n"
                        f"      return {streaming_fock_symbol}("
                        f"{streaming_fock_arguments});"
                    )
                fock_names.append(
                    f'    {{"{selection.spec.name}", {shell_class}U, '
                    f"{sum(selection.spec.angular)}U, {plan.block_threads}U, "
                    f"{consumer_mask}U, {plan.schedule.component_tile}U, "
                    f"{_fock_tasks_per_claim(selection)}U}},"
                )
                fock_mask |= 1 << shell_class
            if selection.resident_force_recurrence is not None:
                resident_symbol = f"vibeqc_launch_{identifier}_ppps_resident"
                declarations.append(
                    f'extern "C" cudaError_t {resident_symbol}('
                    f"{_resident_launch_parameter_declaration()});"
                )
        helpers.append(
            f"""cudaError_t launch_{identifier}_force({launch_parameters}) noexcept {{
  switch (shell_class) {{
{chr(10).join(force_cases)}
    default: return cudaErrorInvalidValue;
  }}
}}

cudaError_t launch_{identifier}_fock({launch_parameters}) noexcept {{
  switch (shell_class) {{
{chr(10).join(fock_cases)}
    default: return cudaErrorInvalidValue;
  }}
}}

cudaError_t launch_{identifier}_mixed_fock({launch_parameters}) noexcept {{
  switch (shell_class) {{
{chr(10).join(mixed_fock_cases)}
    default: return cudaErrorInvalidValue;
  }}
}}

cudaError_t launch_{identifier}_streaming_fock(
    {streaming_fock_parameters}) noexcept {{
  switch (shell_class) {{
{chr(10).join(streaming_fock_cases)}
    default: return cudaErrorNotSupported;
  }}
}}

cudaError_t launch_{identifier}_resident(
    {_resident_launch_parameter_declaration()}) noexcept {{
  """
            + (
                f"return {resident_symbol}({_resident_launch_argument_list()});"
                if resident_symbol is not None
                else "return cudaErrorNotSupported;"
            )
            + """
}
"""
        )
        kernel_arrays.append(
            f"""constexpr std::array<ShellKernelMetadata, {len(force_names)}> kForceNames{index}{{{{
{chr(10).join(force_names)}
}}}};
constexpr std::array<ShellKernelMetadata, {len(fock_names)}> kFockNames{index}{{{{
{chr(10).join(fock_names)}
}}}};
constexpr std::array<ShellKernelMetadata, {len(mixed_fock_names)}> kMixedFockNames{index}{{{{
{chr(10).join(mixed_fock_names)}
}}}};
"""
        )
        kernel_sets.append(
            f"""    {{kCompiledProfiles[{index}], UINT64_C({force_mask}),
      UINT64_C({fock_mask}), UINT64_C({mixed_fock_mask}),
      kForceNames{index}.data(), kForceNames{index}.size(),
      kFockNames{index}.data(), kFockNames{index}.size(),
      kMixedFockNames{index}.data(), kMixedFockNames{index}.size(),
      launch_{identifier}_force, launch_{identifier}_fock,
      launch_{identifier}_mixed_fock,
      launch_{identifier}_streaming_fock,
      launch_{identifier}_resident}},"""
        )
    return f"""#include "vibeqc_generated_shell_registry.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>

{chr(10).join(declarations)}

namespace vibeqc::scf::generated {{
namespace {{

using LaunchFunction = cudaError_t (*)({_launch_parameter_declaration()}) noexcept;
using StreamingFockLaunchFunction = cudaError_t (*)(
    {_streaming_fock_launch_parameter_declaration()}) noexcept;
using ResidentLaunchFunction = cudaError_t (*)({_resident_launch_parameter_declaration()}) noexcept;

struct KernelSet {{
  ProfileInfo info;
  std::uint64_t force_mask;
  std::uint64_t fock_mask;
  std::uint64_t mixed_fock_mask;
  const ShellKernelMetadata* force_names;
  std::size_t force_name_count;
  const ShellKernelMetadata* fock_names;
  std::size_t fock_name_count;
  const ShellKernelMetadata* mixed_fock_names;
  std::size_t mixed_fock_name_count;
  LaunchFunction launch_force;
  LaunchFunction launch_fock;
  LaunchFunction launch_mixed_fock;
  StreamingFockLaunchFunction launch_streaming_fock;
  ResidentLaunchFunction launch_resident;
}};

{chr(10).join(helpers)}
{chr(10).join(kernel_arrays)}

constexpr std::array<KernelSet, {len(items)}> kKernelSets{{{{
{chr(10).join(kernel_sets)}
}}}};

constexpr ProfileInfo kGenericProfile{{
    "generic_cuda", "portable_cuda", 0, 0, false, true, false}};

constexpr std::size_t kMaximumCachedDevices = 128;
std::array<const KernelSet*, kMaximumCachedDevices> selected_by_device{{}};
std::mutex selection_mutex;

bool selected(const char* list, const char* name) noexcept {{
  const std::size_t name_size = std::strlen(name);
  const char* cursor = list;
  while (*cursor != '\\0') {{
    while (*cursor == ',' || *cursor == ';' || *cursor == ' ' ||
           *cursor == '\t') ++cursor;
    const char* begin = cursor;
    while (*cursor != '\\0' && *cursor != ',' && *cursor != ';' &&
           *cursor != ' ' && *cursor != '\t') ++cursor;
    if (static_cast<std::size_t>(cursor - begin) == name_size &&
        std::strncmp(begin, name, name_size) == 0) return true;
  }}
  return false;
}}

const KernelSet* resolve(int major, int minor) noexcept {{
  for (const KernelSet& kernels : kKernelSets) {{
    if (kernels.info.compute_capability_major == major &&
        kernels.info.compute_capability_minor == minor) return &kernels;
  }}
  return nullptr;
}}

const KernelSet* current_kernel_set() noexcept {{
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
      static_cast<std::size_t>(device) >= kMaximumCachedDevices) return nullptr;
  std::lock_guard<std::mutex> lock(selection_mutex);
  const KernelSet*& cached = selected_by_device[static_cast<std::size_t>(device)];
  if (cached == nullptr) {{
    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess) return nullptr;
    cached = resolve(major, minor);
  }}
  return cached;
}}

std::uint64_t environment_mask(
    const char* variable, std::uint64_t available,
    const ShellKernelMetadata* names, std::size_t count) noexcept {{
  const char* selection = std::getenv(variable);
  const bool all = selection == nullptr || *selection == '\\0' ||
                   std::strcmp(selection, "all") == 0;
  if (all) return available;
  if (std::strcmp(selection, "none") == 0) return 0;
  std::uint64_t mask = 0;
  for (std::size_t index = 0; index < count; ++index) {{
    if (selected(selection, names[index].name))
      mask |= std::uint64_t{{1}} << names[index].shell_class;
  }}
  return mask & available;
}}

}}  // namespace

void select_profile_for_device(int device_id, int major, int minor) noexcept {{
  if (device_id < 0 ||
      static_cast<std::size_t>(device_id) >= kMaximumCachedDevices) return;
  std::lock_guard<std::mutex> lock(selection_mutex);
  selected_by_device[static_cast<std::size_t>(device_id)] = resolve(major, minor);
}}

const ProfileInfo& selected_profile() noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? kGenericProfile : kernels->info;
}}

const ShellKernelMetadata* selected_shell_kernels(
    std::size_t& count) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  if (kernels == nullptr) {{
    count = 0;
    return nullptr;
  }}
  count = kernels->force_name_count;
  return kernels->force_names;
}}

const ShellKernelMetadata* selected_fock_shell_kernels(
    std::size_t& count) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  if (kernels == nullptr) {{
    count = 0;
    return nullptr;
  }}
  count = kernels->fock_name_count;
  return kernels->fock_names;
}}

std::uint64_t enabled_shell_class_mask() noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? 0 : environment_mask(
      "VIBEQC_AOT_SHELL_CLASSES", kernels->force_mask,
      kernels->force_names, kernels->force_name_count);
}}

std::uint64_t enabled_fock_shell_class_mask() noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? 0 : environment_mask(
      "VIBEQC_AOT_FOCK_SHELL_CLASSES", kernels->fock_mask,
      kernels->fock_names, kernels->fock_name_count);
}}

std::uint64_t enabled_mixed_fock_shell_class_mask() noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? 0 : environment_mask(
      "VIBEQC_AOT_MIXED_FOCK_SHELL_CLASSES", kernels->mixed_fock_mask,
      kernels->mixed_fock_names, kernels->mixed_fock_name_count);
}}

cudaError_t launch_shell_class({_launch_parameter_declaration()}) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? cudaErrorInvalidValue : kernels->launch_force(
      shell_class, {launch_arguments});
}}

cudaError_t launch_shell_class_fock({_launch_parameter_declaration()}) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? cudaErrorInvalidValue : kernels->launch_fock(
      shell_class, {launch_arguments});
}}

cudaError_t launch_shell_class_mixed_fock(
    {_launch_parameter_declaration()}) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? cudaErrorInvalidValue
                            : kernels->launch_mixed_fock(
      shell_class, {launch_arguments});
}}

cudaError_t launch_shell_class_streaming_fock(
    {_streaming_fock_launch_parameter_declaration()}) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  return kernels == nullptr ? cudaErrorInvalidValue
                            : kernels->launch_streaming_fock(
      shell_class, {streaming_fock_arguments});
}}

cudaError_t launch_ppps_resident(
    {_resident_launch_parameter_declaration()}) noexcept {{
  const KernelSet* kernels = current_kernel_set();
  if (kernels == nullptr) return cudaErrorInvalidValue;
  if (kernels->launch_resident == nullptr) return cudaErrorNotSupported;
  return kernels->launch_resident({_resident_launch_argument_list()});
}}

}}  // namespace vibeqc::scf::generated
"""

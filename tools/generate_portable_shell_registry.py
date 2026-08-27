"""Generate the zero-AOT registry used by portable CUDA builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write_if_changed(path: Path, content: str) -> None:
    """Preserve timestamps when deterministic regeneration is byte-identical."""

    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def emit_portable_registry_header(
    requested_architecture: str,
    profile: str = "portable_cuda",
) -> str:
    """Emit the generated-shell API with empty, standard-conforming metadata."""

    architecture_literal = json.dumps(requested_architecture)
    profile_literal = json.dumps(profile)
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
}};

inline constexpr std::array<ShellKernelMetadata, 0> kShellKernels{{}};
inline constexpr std::size_t kShellKernelCount = kShellKernels.size();

inline constexpr std::array<ShellKernelMetadata, 0> kFockShellKernels{{}};
inline constexpr std::size_t kFockShellKernelCount =
    kFockShellKernels.size();

inline constexpr char kProductionShellProfile[] = {profile_literal};
inline constexpr char kRequestedCudaArchitecture[] = {architecture_literal};
inline constexpr bool kProductionShellProfileTuned = false;

/** Portable CUDA uses the validated generic direct-J/K and force paths. */
std::uint64_t enabled_shell_class_mask() noexcept;

/** Portable CUDA has no generated coefficient-only Fock workers. */
std::uint64_t enabled_fock_shell_class_mask() noexcept;

/** No generated force kernel is available in the portable profile. */
cudaError_t launch_shell_class(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* forces, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept;

/** No generated Fock kernel is available in the portable profile. */
cudaError_t launch_shell_class_fock(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept;

}}  // namespace vibeqc::scf::generated

#endif
"""


def emit_portable_registry_source() -> str:
    """Emit masks and launch stubs for a build with no generated shell kernels."""

    return """#include "vibeqc_generated_shell_registry.hpp"

namespace vibeqc::scf::generated {

std::uint64_t enabled_shell_class_mask() noexcept {
  return 0;
}

std::uint64_t enabled_fock_shell_class_mask() noexcept {
  return 0;
}

cudaError_t launch_shell_class(
    unsigned, cudaStream_t, bool, unsigned, const void*,
    const std::uint32_t*, const std::int64_t*, const void*, const double*,
    const void*, double, const double*, const double*, double*,
    const std::uint32_t*, std::uint32_t*) noexcept {
  return cudaErrorNotSupported;
}

cudaError_t launch_shell_class_fock(
    unsigned, cudaStream_t, bool, unsigned, const void*,
    const std::uint32_t*, const std::int64_t*, const void*, const double*,
    const void*, double, const double*, const double*, double*,
    const std::uint32_t*, std::uint32_t*) noexcept {
  return cudaErrorNotSupported;
}

}  // namespace vibeqc::scf::generated
"""


def write_portable_bundle(
    output_directory: Path,
    shard_count: int,
    requested_architecture: str,
    profile: str = "portable_cuda",
) -> tuple[Path, ...]:
    """Write deterministic empty shards plus the portable registry API."""

    if shard_count < 1:
        raise ValueError("portable shard count must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index in range(shard_count):
        path = output_directory / f"vibeqc_generated_shell_shard_{index}.cu"
        _write_if_changed(
            path,
            "// Portable CUDA profile: generated shell AOT is disabled.\n",
        )
        outputs.append(path)
    header = output_directory / "vibeqc_generated_shell_registry.hpp"
    source = output_directory / "vibeqc_generated_shell_registry.cu"
    _write_if_changed(
        header,
        emit_portable_registry_header(requested_architecture, profile),
    )
    _write_if_changed(source, emit_portable_registry_source())
    outputs.extend((header, source))
    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--requested-architecture", required=True)
    parser.add_argument("--profile", default="portable_cuda")
    arguments = parser.parse_args()
    write_portable_bundle(
        arguments.output_directory,
        arguments.shards,
        arguments.requested_architecture,
        arguments.profile,
    )


if __name__ == "__main__":
    main()

"""Compiler ownership for the retained resident-PSSS Direct-HF schedule.

The 128-thread/64-bra-pair baseline is accepted performance evidence, not CUDA
semantics.  Keep it in compiler policy so native runtime code consumes one
generated schedule definition while #597 can later replace the baseline through
target/workload profile selection without another handwritten CUDA constant.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cuda_schedule import ScheduleIR, ScheduleKind
from .shell_spec import PSSS_SPEC


@dataclass(frozen=True, slots=True)
class ResidentPsssScheduleProfile:
    """Accepted resident-PSSS execution profile."""

    schedule: ScheduleIR
    maximum_bra_primitive_pairs: int


def resident_psss_schedule_profile() -> ResidentPsssScheduleProfile:
    """Return the accepted baseline through the shared CUDA schedule vocabulary."""

    return ResidentPsssScheduleProfile(
        schedule=ScheduleIR(
            kind=ScheduleKind.THREAD_TASKS,
            block_threads=128,
            component_tile=PSSS_SPEC.component_count,
            tasks_per_warp=32,
            shared_coulomb=False,
            minimum_blocks_per_sm=4,
            warp_size=32,
        ),
        maximum_bra_primitive_pairs=64,
    )


def emit_direct_resident_psss_schedule_header() -> str:
    """Emit the native ABI constants from compiler-owned schedule policy."""

    profile = resident_psss_schedule_profile()
    schedule = profile.schedule
    return f"""#pragma once

#include <cstddef>

// Generated from compiler-owned Direct-HF schedule policy. Do not edit this
// build artifact; change direct_resident_schedule.py or the selected profile.
namespace vibeqc::scf::cuda_execution {{

inline constexpr unsigned kResidentPsssThreads = {schedule.block_threads}U;
inline constexpr unsigned kResidentPsssMinimumBlocksPerSm =
    {schedule.minimum_blocks_per_sm}U;
inline constexpr std::size_t kResidentPsssMaximumBraPrimitivePairs =
    {profile.maximum_bra_primitive_pairs}U;

static_assert(kResidentPsssThreads % {schedule.warp_size}U == 0U);

}}  // namespace vibeqc::scf::cuda_execution
"""

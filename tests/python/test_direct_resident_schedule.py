"""Regression guards for compiler-owned resident-PSSS schedule policy."""

from pathlib import Path

from vibeqc_compiler.integral.cuda_schedule import ScheduleKind
from vibeqc_compiler.integral.direct_resident_schedule import (
    emit_direct_resident_psss_schedule_header,
    resident_psss_schedule_profile,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_resident_psss_profile_uses_shared_schedule_ir() -> None:
    profile = resident_psss_schedule_profile()
    schedule = profile.schedule

    assert schedule.kind == ScheduleKind.THREAD_TASKS
    assert schedule.block_threads == 128
    assert schedule.component_tile == 3
    assert schedule.tasks_per_warp == 32
    assert schedule.minimum_blocks_per_sm == 4
    assert profile.maximum_bra_primitive_pairs == 64


def test_resident_psss_header_is_compiler_owned() -> None:
    source = emit_direct_resident_psss_schedule_header()

    assert "kResidentPsssThreads = 128U;" in source
    assert "kResidentPsssMinimumBlocksPerSm =" in source
    assert "kResidentPsssMaximumBraPrimitivePairs =" in source
    assert "static_assert(kResidentPsssThreads % 32U == 0U);" in source


def test_native_resident_psss_consumers_have_no_duplicate_schedule_constants() -> None:
    constants = (REPOSITORY_ROOT / "src/scf/cuda/direct_constants.hpp").read_text(
        encoding="utf-8"
    )
    assert "constexpr unsigned kResidentPsssThreads = 128;" not in constants
    assert (
        "constexpr std::size_t kResidentPsssMaximumBraPrimitivePairs = 64;"
        not in constants
    )

    for path in (
        "src/scf/cuda/topology.cpp",
        "src/scf/cuda_rhf.cpp",
        "src/scf/cuda/direct_angular_force.cu",
    ):
        source = (REPOSITORY_ROOT / path).read_text(encoding="utf-8")
        assert '#include "generated_direct_resident_psss_schedule.cuh"' in source

    angular_force = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_angular_force.cu"
    ).read_text(encoding="utf-8")
    assert (
        "__launch_bounds__(kResidentPsssThreads, kResidentPsssMinimumBlocksPerSm)"
        in angular_force
    )

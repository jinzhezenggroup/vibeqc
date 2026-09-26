"""Packed Fock grids must count task claims rather than individual tasks."""

import json
from dataclasses import replace
from pathlib import Path

from vibeqc_compiler.integral.cuda_schedule import ScheduleIR, ScheduleKind
from vibeqc_compiler.integral.production import (
    emit_multi_registry_header,
    emit_multi_registry_source,
    emit_registry_header,
    resolve_production_profile,
)


def test_fock_claim_metadata_follows_the_selected_value_schedule() -> None:
    """Both registries expose packed width, including separate Fock policies."""
    root = Path(__file__).resolve().parents[2]
    profile = resolve_production_profile(
        root / "python/vibeqc_compiler/integral/production_shell_classes.json",
        "sm_120",
    )
    packed = next(x for x in profile.selections if x.spec.name == "psss")
    # An explicit component-lane value schedule must not inherit the force
    # schedule's 32-task claim. It processes one complete shell task per CTA.
    component = replace(
        packed,
        fock_schedule=ScheduleIR(
            kind=ScheduleKind.COMPONENT_LANES,
            block_threads=32,
            component_tile=3,
            tasks_per_warp=1,
            shared_coulomb=True,
        ),
    )
    for selection, width in ((packed, 32), (component, 1)):
        header = emit_registry_header((selection,))
        simple_rows = header.split("kFockShellKernels", 1)[1].split("}};", 1)[0]
        assert f'{{"psss", 1U, 1U, 32U, 1U, 3U, {width}U}}' in simple_rows
        source = emit_multi_registry_source(
            (replace(profile, selections=(selection,)),)
        )
        profile_rows = source.split("kFockNames0", 1)[1].split("}};", 1)[0]
        assert f'{{"psss", 1U, 1U, 32U, 1U, 3U, {width}U}}' in profile_rows


def test_profiled_fock_materialization_reaches_generated_registry(
    tmp_path: Path,
) -> None:
    """A measured profile may select streaming without a handwritten class switch."""

    root = Path(__file__).resolve().parents[2]
    source = root / "python/vibeqc_compiler/integral/production_shell_classes.json"
    payload = json.loads(source.read_text())
    kernels = payload["architectures"]["sm_120"]["kernels"]
    psss_row = next(row for row in kernels if row["shell_class"] == "psss")
    psss_row["fock_route"] = "streaming"
    profiled = tmp_path / "production_shell_classes.json"
    profiled.write_text(json.dumps(payload))

    resolved = resolve_production_profile(profiled, "sm_120")
    psss = next(x for x in resolved.selections if x.spec.name == "psss")
    assert psss.fock_route == "streaming"
    header = emit_registry_header((psss,))
    assert (
        "inline constexpr std::uint64_t kPreferredStreamingFockShellClassMask =\n"
        "    2ULL;"
    ) in header

    selected_profile = replace(resolved, selections=(psss,))
    multi_header = emit_multi_registry_header((selected_profile,))
    multi_source = emit_multi_registry_source((selected_profile,))
    assert "preferred_streaming_fock_shell_class_mask() noexcept" in multi_header
    assert (
        "{kCompiledProfiles[0], UINT64_C(0),\n"
        "      UINT64_C(2), UINT64_C(0),\n"
        "      UINT64_C(2),"
    ) in multi_source
    assert (
        "return kernels == nullptr ? 0 : kernels->preferred_streaming_fock_mask;"
    ) in multi_source

    baseline = resolve_production_profile(source, "sm_120")
    baseline_psss = next(x for x in baseline.selections if x.spec.name == "psss")
    assert baseline_psss.fock_route == "paged"
    baseline_header = emit_registry_header((baseline_psss,))
    assert (
        "inline constexpr std::uint64_t kPreferredStreamingFockShellClassMask =\n"
        "    0ULL;"
    ) in baseline_header

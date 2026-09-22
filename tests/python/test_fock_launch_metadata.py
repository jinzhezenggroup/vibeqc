"""Packed Fock grids must count task claims rather than individual tasks."""

from dataclasses import replace
from pathlib import Path

from vibeqc_compiler.integral.cuda_schedule import ScheduleIR, ScheduleKind
from vibeqc_compiler.integral.production import (
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

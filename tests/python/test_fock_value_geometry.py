"""Contracts for value-only generated Fock geometry specialization."""

from pathlib import Path

from vibeqc_compiler.integral import (
    PSSS_SPEC,
    KernelConsumer,
    build_fused_shell_plan,
    cuda_target_info,
    emit_shell_class_fused_cuda,
)
from vibeqc_compiler.integral.production import load_production_kernel_selections

ROOT = Path(__file__).resolve().parents[2]


def test_psss_fock_geometry_prunes_force_only_state() -> None:
    manifest = ROOT / "python/vibeqc_compiler/integral/production_shell_classes.json"
    selection = next(
        item
        for item in load_production_kernel_selections(manifest, "sm_120")
        if item.spec.name == "psss"
    )
    plan = build_fused_shell_plan(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK,),
        schedule=selection.schedule,
        recurrence=selection.recurrence,
        target=cuda_target_info("sm_120"),
    )
    source = emit_shell_class_fused_cuda(
        PSSS_SPEC, plan, fock_schedule=selection.fock_schedule
    )
    helper = source.split("generated_psss_make_fock_primitive_geometry", maxsplit=1)[
        1
    ].split("/**", maxsplit=1)[0]

    assert "boys_values<1>" in helper
    assert "product_scales" not in helper
    assert "decay_gradients" not in helper
    assert "generated_psss_make_primitive_geometry(" in source
    assert "generated_psss_make_fock_primitive_geometry(" in source

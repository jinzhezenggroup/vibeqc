"""Rys quadrature table and generated root-emitter codegen tests."""

from __future__ import annotations

import pytest
from vibeqc_compiler.integral import (
    FUSED_SHELL_SPEC_BY_NAME,
    ScheduleIR,
    ScheduleKind,
    build_fused_shell_plan,
    cuda_target_info,
    emit_ppps_rys3_root_body_cuda,
    emit_rys2_roots_cuda,
    emit_rys3_roots_cuda,
    emit_rys4_roots_cuda,
    emit_rys5_roots_cuda,
    emit_shell_class_fused_cuda,
    rys2_table_roots_weights,
    rys3_roots_weights,
    rys3_table_roots_weights,
    rys4_roots_weights,
    rys4_table_roots_weights,
    rys5_roots_weights,
    rys5_table_roots_weights,
    rys_boys_values,
)

TEST_CUDA_TARGET = cuda_target_info("sm_120")


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 25.0, 80.0))
def test_gpu4pyscf_rys2_table_matches_moment_oracle(argument: float) -> None:
    """Verify the attributed low-order table used by four production shells."""

    roots, weights = rys2_table_roots_weights(argument)
    moments = rys_boys_values(argument, 4)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=3.0e-11, abs=3.0e-14)


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999))
def test_rys3_rule_reproduces_first_six_boys_moments(argument: float) -> None:
    """Treat the host eigensolve only as a high-accuracy Rys3 oracle."""

    roots, weights = rys3_roots_weights(argument)
    assert all(0.0 < root < 1.0 for root in roots)
    assert all(weight > 0.0 for weight in weights)
    moments = rys_boys_values(argument, 6)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=3.0e-13, abs=3.0e-14)


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999, 25.0, 50.0, 80.0))
def test_gpu4pyscf_rys3_table_matches_moment_oracle(argument: float) -> None:
    """Verify the attributed nroots=3 table before CUDA emission."""

    roots, weights = rys3_table_roots_weights(argument)
    moments = rys_boys_values(argument, 6)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=3.0e-11, abs=3.0e-14)


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999))
def test_rys4_rule_reproduces_first_eight_boys_moments(argument: float) -> None:
    """Treat the host eigensolve only as a high-accuracy Rys4 oracle."""

    roots, weights = rys4_roots_weights(argument)
    assert all(0.0 < root < 1.0 for root in roots)
    assert all(weight > 0.0 for weight in weights)
    moments = rys_boys_values(argument, 8)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=8.0e-12, abs=5.0e-14)


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999, 25.0, 55.0, 80.0))
def test_gpu4pyscf_rys4_table_matches_moment_oracle(argument: float) -> None:
    """Verify the attributed nroots=4 slice before CUDA integration."""

    roots, weights = rys4_table_roots_weights(argument)
    moments = rys_boys_values(argument, 8)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=4.0e-11, abs=5.0e-13)


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999))
def test_rys5_rule_reproduces_first_ten_boys_moments(argument: float) -> None:
    """Treat the host eigensolve only as a high-accuracy Rys5 oracle."""

    roots, weights = rys5_roots_weights(argument)
    assert all(0.0 < root < 1.0 for root in roots)
    assert all(weight > 0.0 for weight in weights)
    moments = rys_boys_values(argument, 10)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=3.0e-11, abs=8.0e-13)


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999, 25.0, 60.0, 80.0))
def test_gpu4pyscf_rys5_table_matches_moment_oracle(argument: float) -> None:
    """Verify the attributed nroots=5 slice before CUDA integration."""

    roots, weights = rys5_table_roots_weights(argument)
    moments = rys_boys_values(argument, 10)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=5.0e-10, abs=5.0e-12)


@pytest.mark.parametrize(
    "argument",
    (
        3.0e-7 - 1.0e-14,
        3.0e-7,
        3.0e-7 + 1.0e-14,
        2.5 - 1.0e-12,
        2.5,
        2.5 + 1.0e-12,
        55.0 - 1.0e-10,
        55.0,
        55.0 + 1.0e-10,
    ),
)
def test_gpu4pyscf_rys4_table_is_accurate_across_branch_boundaries(
    argument: float,
) -> None:
    """Cover the small-x, interpolation-interval, and large-x boundaries."""

    roots, weights = rys4_table_roots_weights(argument)
    moments = rys_boys_values(argument, 8)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=4.0e-11, abs=5.0e-13)


def test_dppp_rys4_cuda_emits_only_the_fixed_root_slice() -> None:
    """Keep Rys4 tables compact enough for generated CUDA compilation."""

    roots = emit_rys4_roots_cuda()
    assert "Copyright 2021-2024 The PySCF Developers" in roots
    assert "generated_dppp_rys4_rw[4480]" in roots
    assert "generated_dppp_rys4_roots" in roots
    assert "series < 8U" in roots
    assert "argument > 55.0" in roots


def test_dddp_rys5_cuda_emits_only_the_fixed_root_slice() -> None:
    """Keep Rys5 tables compact enough for generated CUDA compilation."""

    roots = emit_rys5_roots_cuda()
    assert "Copyright 2021-2024 The PySCF Developers" in roots
    assert "generated_dddp_rys5_rw[5600]" in roots
    assert "generated_dddp_rys5_roots" in roots
    assert "series < 10U" in roots
    assert "argument > 60.0" in roots


def test_low_order_rys2_cuda_emits_only_the_fixed_root_slice() -> None:
    """Keep the shared two-root table compact and attributed."""

    roots = emit_rys2_roots_cuda()
    assert "Copyright 2021-2024 The PySCF Developers" in roots
    assert "generated_low_order_rys2_rw[2240]" in roots
    assert "series < 4U" in roots
    assert "argument > 45.0" in roots


def test_ppps_rys_cuda_emits_compact_state_program_and_attributed_table() -> None:
    """Prevent the direct recurrence from regressing into a scalar DAG."""

    roots = emit_rys3_roots_cuda()
    body = emit_ppps_rys3_root_body_cuda()
    assert "Copyright 2021-2024 The PySCF Developers" in roots
    assert "generated_ppps_rys3_rw[3360]" in roots
    assert "generated_ppps_rys3_roots" in roots
    assert 80 <= body.count("double rys_state_") <= 93
    assert body.count("rys_state_") > 69
    assert body.count("const double component_density_weight") == 27
    assert body.count("force_") == 243
    assert "boys_" not in body
    assert "component_gradient" not in body

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=12,
    )
    plan = build_fused_shell_plan(
        spec, schedule=schedule, recurrence="rys3", target=TEST_CUDA_TARGET
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    assert "generated_ppps_rys3_force_task" in source
    assert "generated_ppps_rys3_roots" in source
    assert "component_weights[kGeneratedPppsComponentCount][32]" in source
    assert "generated_ppps_scalar_thread" not in source

    dpss_spec = FUSED_SHELL_SPEC_BY_NAME["dpss"]
    dpss_schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=dpss_spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    dpss_source = emit_shell_class_fused_cuda(
        dpss_spec,
        build_fused_shell_plan(
            dpss_spec,
            schedule=dpss_schedule,
            recurrence="rys3",
            target=TEST_CUDA_TARGET,
        ),
    )
    assert "generated_dpss_rys3_force_task" in dpss_source
    assert "generated_dpss_rys3_roots" in dpss_source
    assert "generated_ppps_rys3_roots" not in dpss_source

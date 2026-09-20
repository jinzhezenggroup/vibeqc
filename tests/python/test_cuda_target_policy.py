"""CUDA architecture catalog and explicit-target contracts."""

from __future__ import annotations

import inspect

import pytest
from vibeqc_compiler.integral import (
    CUDA_TARGETS,
    PSSS_SPEC,
    CudaKernelIR,
    build_fused_shell_plan,
    build_integral_ir,
    cuda_target_info,
    schedule_candidates,
)
from vibeqc_compiler.integral.autotune import (
    emit_schedule_driver,
    supported_schedule_trials,
)
from vibeqc_compiler.integral.capabilities import build_capability_report


@pytest.mark.parametrize("architecture", sorted(CUDA_TARGETS))
def test_static_cuda_catalog_has_no_device_topology(architecture: str):
    """Architecture metadata must not impersonate one concrete GPU."""

    target = cuda_target_info(architecture)
    assert target.sm_count is None
    with pytest.raises(ValueError, match="runtime-probed"):
        target.require_sm_count()

    integral = build_integral_ir(PSSS_SPEC)
    assert schedule_candidates(integral, target)


def test_runtime_probe_owns_sm_count():
    """A concrete topology appears only after a device probe enriches a target."""

    static = cuda_target_info("sm_120")
    runtime = static.with_runtime_probe(sm_count=170)

    assert static.sm_count is None
    assert runtime.require_sm_count() == 170
    assert runtime.to_payload()["sm_count"] == 170
    with pytest.raises(ValueError, match="SM count must be positive"):
        static.with_runtime_probe(sm_count=0)


def test_generic_cuda_schedule_apis_have_no_sm120_default():
    """Generic scheduling cannot silently select a concrete CUDA architecture."""

    assert (
        inspect.signature(schedule_candidates).parameters["target"].default
        is inspect.Parameter.empty
    )
    assert (
        inspect.signature(CudaKernelIR).parameters["target"].default
        is inspect.Parameter.empty
    )

    with pytest.raises(ValueError, match="CUDA target must be explicit"):
        build_fused_shell_plan(PSSS_SPEC)
    with pytest.raises(ValueError, match="CUDA target must be explicit"):
        supported_schedule_trials(PSSS_SPEC)
    with pytest.raises(ValueError, match="explicit CUDA target or architecture"):
        build_capability_report()
    with pytest.raises(ValueError, match="explicit CUDA architecture"):
        emit_schedule_driver(())


def test_explicit_offline_architecture_remains_supported():
    """Offline reports and drivers can select an architecture without a device probe."""

    report = build_capability_report(
        architecture="sm_80",
        specifications=(PSSS_SPEC,),
    )
    source = emit_schedule_driver((), architecture="sm_80")

    assert report["architecture"] == "sm_80"
    assert "compile target sm_80" in source


def test_capability_target_and_architecture_must_agree():
    """Supplying two explicit target identities cannot silently replace either one."""

    with pytest.raises(ValueError, match="target and architecture disagree"):
        build_capability_report(
            target=cuda_target_info("sm_80"),
            architecture="sm_90",
            specifications=(PSSS_SPEC,),
        )


@pytest.mark.parametrize("count", [True, 1.5, float("nan"), float("inf")])
def test_probed_sm_count_is_a_finite_integer(count):
    target = cuda_target_info("sm_120")
    with pytest.raises(ValueError, match="SM count must be positive"):
        target.with_runtime_probe(sm_count=count)

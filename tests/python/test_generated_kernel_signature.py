"""Issue #673 Slice C: generated kernel signature and launch pruning."""

from pathlib import Path

import pytest
from vibeqc_compiler.integral import production
from vibeqc_compiler.integral.production import (
    _streaming_fock_internal_signature,
    _streaming_fock_launch_parameter_declaration,
    _streaming_fock_launch_wrapper,
    _streaming_fock_source,
    load_production_kernel_selections,
)
from vibeqc_compiler.integral.signature import (
    GeneratedKernelArgument,
    GeneratedKernelSignature,
)

_PRODUCTION_MANIFEST = Path(production.__file__).with_name(
    "production_shell_classes.json"
)


def _selection(name: str) -> production.KernelSelection:
    return next(
        selection
        for selection in load_production_kernel_selections(
            _PRODUCTION_MANIFEST, "sm_120"
        )
        if selection.spec.name == name
    )


def test_signature_manifest_drives_declaration_and_forwarding_order() -> None:
    signature = GeneratedKernelSignature(
        (
            GeneratedKernelArgument("const void*", "input", "typed_input"),
            GeneratedKernelArgument("double*", "output"),
        )
    )

    assert signature.names == ("input", "output")
    assert signature.parameter_list() == ("    const void* input,\n    double* output")
    assert signature.argument_list() == "input, output"
    assert signature.argument_list(wrapper=True) == "typed_input, output"
    pruned = signature.without("input")
    assert pruned.names == ("output",)
    diagnostics = pruned.pruning_diagnostics(signature)
    assert diagnostics["schema"] == "vibeqc.compiler.signature-pruning.v1"
    assert diagnostics["parameters_before"] == ["input", "output"]
    assert diagnostics["parameters_after"] == ["output"]
    assert diagnostics["removed_parameters"] == ["input"]
    assert diagnostics["parameter_count_before"] == 2
    assert diagnostics["parameter_count_after"] == 1
    assert diagnostics["reason"] == "compile-time liveness"
    with pytest.raises(ValueError, match="unknown generated arguments"):
        signature.without("missing")


def test_nonmixed_streaming_fock_prunes_dead_internal_arguments() -> None:
    selection = _selection("ssss")
    signature = _streaming_fock_internal_signature(selection)

    assert "mixed_precision_enabled" not in signature.names
    assert "fp64_threshold" not in signature.names
    assert "fp32_work_count" not in signature.names
    assert "fp64_work_count" in signature.names

    source = _streaming_fock_source(selection)
    assert "mixed_precision_enabled" not in source
    assert "fp64_threshold" not in source
    assert "fp32_work_count" not in source
    assert "fp64_work_count" in source


def test_stable_streaming_wrapper_adapts_to_pruned_internal_signature() -> None:
    selection = _selection("ssss")
    public_parameters = _streaming_fock_launch_parameter_declaration()
    wrapper = _streaming_fock_launch_wrapper(selection)

    assert "mixed_precision_enabled" in public_parameters
    assert "fp64_threshold" in public_parameters
    assert "fp32_work_count" in public_parameters
    assert "bool mixed_precision_enabled" in wrapper
    assert "double fp64_threshold" in wrapper
    assert "unsigned long long* fp32_work_count" in wrapper

    launches = wrapper.split("<<<", maxsplit=1)[1]
    assert "mixed_precision_enabled" not in launches
    assert "fp64_threshold" not in launches
    assert "fp32_work_count" not in launches


def test_mixed_streaming_fock_retains_live_precision_arguments() -> None:
    selection = _selection("dsds")
    signature = _streaming_fock_internal_signature(selection)

    assert "mixed_precision_enabled" in signature.names
    assert "fp64_threshold" in signature.names
    assert "fp32_work_count" in signature.names

    source = _streaming_fock_source(selection)
    assert "mixed_precision_enabled" in source
    assert "fp64_threshold" in source
    assert "fp32_work_count" in source


def test_component_lane_signature_keeps_internal_head_name_and_wrapper_adapter() -> (
    None
):
    selection = _selection("ddpp")
    signature = _streaming_fock_internal_signature(selection)

    assert "task_head" in signature.names
    assert "bra_head" not in signature.names
    assert "task_head" in signature.parameter_list()
    assert "task_head" in signature.argument_list()
    assert "bra_head" in signature.argument_list(wrapper=True)

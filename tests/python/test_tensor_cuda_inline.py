"""Output-demand TensorIR lowering for generated CUDA consumers."""

from fractions import Fraction

import pytest
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    exp,
    input_tensor,
    reduce_sum,
)
from vibeqc_compiler.tensor.cuda_inline import (
    exact_cuda_literal,
    lower_inline_cuda_output,
)


def _optional_diagnostic_program() -> Program:
    spin = Index("s", IndexSpace("spin", "spin", 2))
    vector = TensorSpec((spin,), role="input")
    production = input_tensor("density", vector)
    diagnostic = input_tensor("diagnostic_only", vector)
    return Program(
        {
            "force": reduce_sum(production, axes=(0,)),
            "diagnostic": reduce_sum(exp(diagnostic), axes=(0,)),
        }
    )


def test_inline_cuda_applies_output_demand_before_consumer_lowering() -> None:
    program = _optional_diagnostic_program()
    lowered = lower_inline_cuda_output(
        program, output="force", bindings={"density": ("d0", "d1")}
    )

    assert lowered.expression == "(d0 + d1)"
    assert lowered.required_inputs == ("density",)
    assert lowered.original_logical_hash == program.logical_hash
    assert lowered.specialization_logical_hash != program.logical_hash
    assert lowered.optimized_logical_hash == lowered.specialization_logical_hash
    assert len(lowered.optimizer_identity) == 64
    assert lowered.pruning_diagnostics["requested_outputs"] == ["force"]
    assert lowered.pruning_diagnostics["removed_outputs"] == ["diagnostic"]
    assert lowered.pruning_diagnostics["removed_inputs"] == ["diagnostic_only"]


def test_inline_cuda_fails_closed_when_requested_branch_is_not_supported() -> None:
    program = _optional_diagnostic_program()
    with pytest.raises(ValueError, match="unsupported TensorIR inline CUDA op: exp"):
        lower_inline_cuda_output(
            program,
            output="diagnostic",
            bindings={"diagnostic_only": ("q0", "q1")},
        )


def test_inline_cuda_rejects_missing_live_input_binding() -> None:
    program = _optional_diagnostic_program()
    with pytest.raises(ValueError, match="missing inline CUDA input binding: density"):
        lower_inline_cuda_output(program, output="force", bindings={})


def test_exact_cuda_literal_and_scalar_binding_contract() -> None:
    assert exact_cuda_literal(2) == "2.0"
    assert exact_cuda_literal(Fraction(1, 3)) == "(1.0/3.0)"
    scalar = input_tensor("scalar", TensorSpec(role="input"))
    program = Program({"value": scalar})
    lowered = lower_inline_cuda_output(
        program, output="value", bindings={"scalar": "x"}
    )
    assert lowered.expression == "x"
    with pytest.raises(ValueError, match="binding size"):
        lower_inline_cuda_output(
            program, output="value", bindings={"scalar": ("x", "y")}
        )

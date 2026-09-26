"""Output-demand TensorIR lowering for generated CUDA consumers."""

from fractions import Fraction
from pathlib import Path

import pytest
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
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


def test_inline_spin_contraction_is_bounded_and_keeps_pairing() -> None:
    for size in (2, 65):
        spin = Index("s", IndexSpace("spin", "spin", size))
        spec = TensorSpec((spin,), role="input")
        left, right = input_tensor("left", spec), input_tensor("right", spec)
        program = Program(
            {"weight": einsum("s,s->", left, right, coefficient=Fraction(-3, 7))}
        )
        bindings = {
            "left": tuple(f"a{i}" for i in range(size)),
            "right": tuple(f"b{i}" for i in range(size)),
        }
        if size > 64:
            with pytest.raises(ValueError, match="64-term work bound"):
                lower_inline_cuda_output(program, output="weight", bindings=bindings)
        else:
            lowered = lower_inline_cuda_output(
                program, output="weight", bindings=bindings
            )
            # Only arithmetic emitted above from this fixed, trusted test graph.
            assert eval(  # noqa: S307
                lowered.expression,
                {"__builtins__": {}},
                {"a0": 2, "a1": 3, "b0": 5, "b1": 7},
            ) == pytest.approx(-93 / 7)


def test_inline_cuda_applies_output_demand_before_consumer_lowering() -> None:
    program = _optional_diagnostic_program()
    lowered = lower_inline_cuda_output(
        program, output="force", bindings={"density": ("d0", "d1")}
    )

    assert lowered.expression == "((d0) + (d1))"
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
    assert lowered.expression == "(x)"
    with pytest.raises(ValueError, match="binding size"):
        lower_inline_cuda_output(
            program, output="value", bindings={"scalar": ("x", "y")}
        )


@pytest.mark.parametrize(
    ("left_binding", "right_binding", "expected"),
    (
        ("a + b", "c", 20.0),
        ("a - b", "c", -4.0),
        ("a > b ? a : b", "c + a", 18.0),
    ),
)
def test_inline_cuda_preserves_bound_expression_precedence(
    tmp_path: Path,
    left_binding: str,
    right_binding: str,
    expected: float,
) -> None:
    """Compile the emitted scalar, independently checking arbitrary bindings."""
    import shutil
    import subprocess

    from vibeqc_compiler.tensor import einsum

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    index = Index("t", IndexSpace("singleton", "batch", 1))
    spec = TensorSpec((index,), role="input")
    left, right = input_tensor("left", spec), input_tensor("right", spec)
    program = Program({"value": einsum("t,t->t", left, right)})
    lowered = lower_inline_cuda_output(
        program,
        output="value",
        bindings={"left": left_binding, "right": right_binding},
    )
    source, executable = tmp_path / "binding.cpp", tmp_path / "binding"
    source.write_text(
        "#include <iostream>\nint main() { const double a=2,b=3,c=4; "
        + "std::cout << ("
        + lowered.expression
        + "); }\n"
    )
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(executable)], check=True
    )
    result = subprocess.run(
        [str(executable)], text=True, capture_output=True, check=True
    )
    assert float(result.stdout) == expected

"""Serialize two-, three-, and four-center contracts without running a backend."""

# Source-tree CLI bootstrap; importing the compiler needs no native runtime.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
from pathlib import Path

from vibeqc_compiler.integral import (
    BasisShell,
    CenterBinding,
    IntegralIR,
    OperatorSpec,
    RawBlock,
    ShellSignature,
    TensorLayout,
    TranslationInvariant,
    WeightDescriptor,
    WeightedDerivative,
    integral_to_payload,
    query_integral_capability,
)


def example_integrals() -> dict[str, IntegralIR]:
    """Small contracts showing orbital, auxiliary, and arbitrary-weight intent."""
    examples = {}
    for name, family, roles, derivative in (
        ("two_center_value", "overlap", ("orbital", "orbital"), False),
        (
            "three_center_derivative",
            "three_center_eri",
            ("orbital", "orbital", "auxiliary"),
            True,
        ),
        ("four_center_external_weight", "four_center_eri", ("orbital",) * 4, True),
    ):
        centers = tuple(range(len(roles)))
        # Centers 0 and 1 share one atom but remain separate position variables.
        signature = ShellSignature(
            tuple(BasisShell(i, i, int(i == 0), role) for i, role in enumerate(roles)),
            tuple(CenterBinding(i, max(i - 1, 0)) for i in centers),
        )
        operator = OperatorSpec(family, centers, (TranslationInvariant(),))
        deriv = operator.nuclear_derivative() if derivative else None
        if name == "four_center_external_weight":
            consumer = WeightedDerivative(
                WeightDescriptor(
                    "cc_lagrangian_tile",
                    TensorLayout(signature.tensor_indices, signature.component_shape),
                    sign=1,
                    prefactor=0.5,
                ),
                TensorLayout(("atom", "xyz"), (3, 3)),
                memory_budget_bytes=4096,
                output="atomic_force",
                output_sign=-1,
            )
        else:
            indices = (
                ("center", "xyz") if derivative else ()
            ) + signature.tensor_indices
            shape = (
                (len(centers), 3) if derivative else ()
            ) + signature.component_shape
            consumer = RawBlock(TensorLayout(indices, shape), memory_budget_bytes=4096)
        examples[name] = IntegralIR(signature, operator, deriv, (consumer,))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="write JSON to this file (default: stdout)"
    )
    arguments = parser.parse_args()
    payload = {
        name: {
            "integral": integral_to_payload(integral),
            "cuda_lowering": query_integral_capability(integral).to_payload(),
        }
        for name, integral in example_integrals().items()
    }
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(content, end="")
    else:
        arguments.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()

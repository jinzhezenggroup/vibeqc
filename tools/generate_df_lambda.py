"""Emit retained native DF Lambda actions from shared TensorIR AD."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from generativeqc_compiler.cc.df_lambda import retained_response_programs

from tools.generate_df_ccsd_core import programs as core_programs
from tools.generate_df_ccsd_hoisted import contraction_query
from tools.generate_df_ccsd_native import programs as virtual_programs
from tools.generate_rccsd_native import (
    REPRESENTATIVE,
    _cuda_program,
    _required_function,
)


def output_type(name: str) -> str:
    return (
        "DeviceParameterOutput"
        if name.startswith("parameter_")
        else "DeviceLambdaOutputs"
    )


def header() -> str:
    """Exact runtime-shape capacities and scientific identities; no GPU probe."""
    lines = [
        "// Generated DF Lambda capacities; do not edit.",
        "#pragma once",
        '#include "generated_df_ccsd_cpu.hpp"',
        "namespace generativeqc::cc::generated::dflambda {",
        "using df::checked_add; using df::checked_mul; using df::checked_product;",
    ]
    retained = retained_response_programs(*REPRESENTATIVE)
    virtual = virtual_programs("cuda")
    for prefix in ("", "independent_"):
        identity = hashlib.sha256(
            json.dumps(
                {
                    "core": retained[prefix + "transpose"].logical_hash,
                    "virtual": virtual["amplitude_vjp"].logical_hash,
                    "composition": "core plus every Q amplitude VJP; dense pair-projected Frobenius metric",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        lines.append(
            f'inline constexpr const char* {prefix}operator_hash="{identity}";'
        )
    for name, program in retained.items():
        lines += [
            f'inline constexpr const char* {name}_hash="{program.logical_hash}";',
            f"inline constexpr std::size_t {name}_operations={sum(n.op != 'input' for n in program.live_nodes)};",
            _required_function(program, name + "_arena_elements"),
            contraction_query(program, name + "_contraction_terms"),
        ]
    lines.append(
        contraction_query(core_programs()["replay"], "replay_contraction_terms")
    )
    for name, program in virtual.items():
        lines.append(
            contraction_query(program, "virtual_" + name + "_contraction_terms")
        )
    return "\n".join([*lines, "}", ""])


def cuda_header() -> str:
    return "\n".join(
        [
            "// Generated DF Lambda declarations; do not edit.",
            "#pragma once",
            '#include "generated_df_lambda.hpp"',
            '#include "generated_df_ccsd_core_cuda.cuh"',
            "namespace generativeqc::cc::generated::dflambda {",
            "using CudaState = dfcore::CudaState;",
            "// Caller clears the sticky flag at each complete core-plus-Q action boundary.",
            *(
                f"{output_type(name)} run_{name}_cuda(CudaState& state);"
                for name in retained_response_programs(*REPRESENTATIVE)
            ),
            "}",
            "",
        ]
    )


def cuda_source() -> str:
    lines = [
        '#include "generated_df_lambda_cuda.cuh"',
        "namespace generativeqc::cc::generated::dflambda {",
    ]
    for name, program in retained_response_programs(*REPRESENTATIVE).items():
        inputs = {
            n.attrs["name"]: "s." + n.attrs["name"]
            for n in program.live_nodes
            if n.op == "input"
        }
        lines += [
            _cuda_program(
                program,
                name,
                output_type(name),
                input_overrides=inputs,
                reset_error=False,
            ),
            f"{output_type(name)} run_{name}_cuda(CudaState& state) {{ return run_{name}(state); }}",
        ]
    return "\n".join([*lines, "}", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, emit in (
        (".hpp", header),
        ("_cuda.cuh", cuda_header),
        ("_cuda.cu", cuda_source),
    ):
        (args.output_dir / ("generated_df_lambda" + suffix)).write_text(emit())


if __name__ == "__main__":
    main()

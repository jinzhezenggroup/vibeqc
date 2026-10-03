"""Emit the retained RCCSD equations consuming accumulated DF virtual terms.

The native solver supplies every Q contribution before invoking this core.
Primary iterations and expanded replay share existing mathematical inventories;
no ovvv/vvvv input or intermediate is admitted in either generated program.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.tensor import Program, optimize, reassociate_einsums

from tools.generate_rccsd_native import (
    INPUT_NAMES,
    REPRESENTATIVE,
    _cpu_function,
    _cuda_program,
    _required_function,
    iteration_program,
)

INPUTS = tuple(k for k in INPUT_NAMES if k not in ("ovvv", "vvvv")) + (
    "df_virtual_singles",
    "df_virtual_doubles",
)


def programs() -> dict[str, Program]:
    """Use distinct shared/expanded inventories and bounded contraction trees."""
    result = {}
    for name, source in (
        (
            "iteration",
            iteration_program(*REPRESENTATIVE, external_virtual_correction=True),
        ),
        (
            "replay",
            build_ccsd_program(
                *REPRESENTATIVE,
                form="expanded",
                diagnostics=False,
                external_virtual_correction=True,
            ),
        ),
    ):
        prepared = optimize(
            reassociate_einsums(source, max_intermediate_axes={"virtual": 2})
        )
        if any(
            sum(i.space.kind == "virtual" for i in n.spec.indices) > 2
            for n in prepared.live_nodes
        ):
            raise ValueError("DF core reconstructed an omitted virtual block")
        result[name] = Program(
            prepared.outputs,
            provenance={
                **prepared.provenance,
                "native_execution_order": "dependencies",
            },
        )
    return result


def cpu_header() -> str:
    """Runtime-shape core actions; outputs borrow the supplied arena."""
    lines = [
        "// Generated retained DF RCCSD equations; do not edit.",
        "#pragma once",
        '#include "generated_df_ccsd_cpu.hpp"',
        "namespace generativeqc::cc::generated::dfcore {",
        "using df::checked_add; using df::checked_mul; using df::checked_product;",
        "struct Inputs {",
        *[f"  const double* {name}{{}};" for name in INPUTS],
        "};",
        "struct IterationOutputs { double energy; const double *r1, *r2, *next_t1, *next_t2; };",
        "struct ReplayOutputs { double energy; const double *r1, *r2; };",
    ]
    for name, program in programs().items():
        lines += [
            f'inline constexpr const char* {name}_equation_hash = "{program.logical_hash}";',
            f"inline constexpr std::size_t {name}_operation_count = {sum(n.op != 'input' for n in program.live_nodes)};",
            _required_function(program, f"{name}_arena_elements"),
            _cpu_function(
                program,
                f"run_{name}_cpu",
                "IterationOutputs" if name == "iteration" else "ReplayOutputs",
                input_overrides={key: f"inputs.{key}" for key in INPUTS},
            ),
        ]
    return "\n".join(
        [*lines, "}  // namespace generativeqc::cc::generated::dfcore", ""]
    )


def cuda_header() -> str:
    """Borrow the conventional solver state, adding accumulated corrections."""
    return """// Generated DF RCCSD core device declarations; do not edit.
#pragma once
#include "cc/cuda_solver_support.cuh"
namespace generativeqc::cc::generated::dfcore {
struct CudaState : generated::CudaState {
  const double *df_virtual_singles{}, *df_virtual_doubles{};
};
// The owner clears the sticky arithmetic flag before the entire Q sum.
DeviceIterationOutputs run_iteration_cuda(CudaState& state);
DeviceReplayOutputs run_replay_cuda(CudaState& state);
}  // namespace generativeqc::cc::generated::dfcore
"""


def cuda_source() -> str:
    """Preserve errors from all preceding auxiliary actions until host acceptance."""
    lines = [
        '#include "generated_df_ccsd_core_cpu.hpp"',
        '#include "generated_df_ccsd_core_cuda.cuh"',
        "namespace generativeqc::cc::generated::dfcore {",
    ]
    for name, program in programs().items():
        output = (
            "DeviceIterationOutputs" if name == "iteration" else "DeviceReplayOutputs"
        )
        lines += [
            _cuda_program(
                program,
                name,
                output,
                input_overrides={key: f"s.{key}" for key in INPUTS},
                reset_error=False,
            ),
            f"{output} run_{name}_cuda(CudaState& state) {{ return run_{name}(state); }}",
        ]
    return "\n".join(
        [*lines, "}  // namespace generativeqc::cc::generated::dfcore", ""]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, producer in (
        ("generated_df_ccsd_core_cpu.hpp", cpu_header),
        ("generated_df_ccsd_core_cuda.cuh", cuda_header),
        ("generated_df_ccsd_core_cuda.cu", cuda_source),
    ):
        (args.output_dir / name).write_text(producer())


if __name__ == "__main__":
    main()

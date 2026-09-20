"""Compile the generated TensorIR ordinary/resident static-data ABI without a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    constant,
    input_tensor,
    multiply,
    scatter_add,
)
from vibeqc_compiler.tensor.cuda_execute import compile_cuda
from vibeqc_compiler.tensor.cuda_plan import plan_cuda
from vibeqc_compiler.tensor.cuda_resident import compile_resident


def _program() -> Program:
    source_space = IndexSpace("compile_source", "batch", 5)
    target_space = IndexSpace("compile_target", "batch", 3)
    source = Index("s", source_space)
    target = Index("t", target_space)
    values = input_tensor(
        "values",
        TensorSpec((source,), dtype="float64", role="input", differentiable=True),
    )
    weights = constant(
        ("1.25", "-0.5", "2.0", "0.75", "-1.5"),
        TensorSpec((source,), dtype="float64", role="constant"),
    )
    weighted = multiply(values, weights)
    reduced = scatter_add(weighted, 0, (0, 0, 1, 2, 2), target)
    return Program({"result": reduced})


def _check_source(path: Path) -> None:
    source = path.read_text()
    if "tensor_static_initialize" not in source:
        raise RuntimeError("external TensorIR source lacks static-data ABI")
    if (
        "static const double constant_" in source
        or "static const I index_data_" in source
    ):
        raise RuntimeError(
            "external TensorIR source still embeds static payload literals"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", required=True, type=Path)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()

    compiler = CudaCompilerAdapter(args.nvcc, cuda_target_info(args.architecture))
    plan = plan_cuda(_program(), compiler.target)
    ordinary = compile_cuda(plan, compiler, args.cache / "ordinary")
    resident = compile_resident(plan, compiler, args.cache / "resident")

    if plan.static_data_bytes <= 0 or plan.host_bytes < plan.static_data_bytes:
        raise RuntimeError("TensorIR static-data resource accounting is incomplete")
    for artifact, filename in (
        (ordinary, "program.cu"),
        (resident, "resident.cu"),
    ):
        static_path = artifact.library.parent / "static.bin"
        if not static_path.is_file():
            raise RuntimeError("compiled TensorIR artifact lacks static.bin")
        if static_path.stat().st_size != artifact.metadata.get("static_data_bytes"):
            raise RuntimeError("compiled TensorIR static-data size mismatch")
        _check_source(artifact.library.parent / filename)

    print(
        json.dumps(
            {
                "plan": plan.identity,
                "static_data_bytes": plan.static_data_bytes,
                "ordinary_source_bytes": ordinary.metadata["generated_source_bytes"],
                "ordinary_compile_seconds": ordinary.metadata["compile_seconds"],
                "resident_compile_seconds": resident.metadata["compile_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

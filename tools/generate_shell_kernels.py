#!/usr/bin/env python3
"""Generate or inspect shell-class-specific CUDA derivative kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qce_codegen.shell_class import (
    D_COMPONENTS,
    build_dppp_component_kernel,
    build_dppp_contraction_kernel,
    build_psss_kernel,
    emit_dppp_component_cuda,
    emit_dppp_contraction_cuda,
    emit_psss_cuda,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell-class", choices=("psss", "dppp"), default="psss")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--d-component", choices=D_COMPONENTS, default="xx")
    parser.add_argument(
        "--p-components",
        default="xxx",
        help="three x/y/z component labels for the dppp p shells",
    )
    parser.add_argument(
        "--lowering",
        choices=("full", "factored"),
        default="full",
        help="emit full primitive algebra or component algebra using shared geometry",
    )
    parser.add_argument(
        "--format", choices=("cuda", "stats"), default="cuda"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write generated output to this path instead of standard output",
    )
    arguments = parser.parse_args()

    if arguments.shell_class == "psss":
        kernel = build_psss_kernel(arguments.axis)
        component_metadata = {"axis": arguments.axis}
    else:
        if len(arguments.p_components) != 3 or any(
            axis not in "xyz" for axis in arguments.p_components
        ):
            parser.error("--p-components must contain exactly three x/y/z labels")
        kernel = (
            build_dppp_component_kernel(
                arguments.d_component, tuple(arguments.p_components)
            )
            if arguments.lowering == "full"
            else build_dppp_contraction_kernel(
                arguments.d_component, tuple(arguments.p_components)
            )
        )
        component_metadata = {
            "d_component": arguments.d_component,
            "lowering": arguments.lowering,
            "p_components": arguments.p_components,
        }
    if arguments.format == "cuda":
        output = (
            emit_psss_cuda(kernel)
            if arguments.shell_class == "psss"
            else (
                emit_dppp_component_cuda(kernel)
                if arguments.lowering == "full"
                else emit_dppp_contraction_cuda(kernel)
            )
        )
    else:
        roots = [kernel.value]
        if hasattr(kernel, "boys_argument"):
            roots.append(kernel.boys_argument)
        roots.extend(item for center in kernel.gradients for item in center)
        output = json.dumps(
            {
                **component_metadata,
                "operation_counts": kernel.graph.operation_counts(roots),
                "reachable_nodes": len(kernel.graph.topological_order(roots)),
                "shell_class": arguments.shell_class,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    if arguments.output is None:
        print(output, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

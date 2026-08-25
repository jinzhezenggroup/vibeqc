#!/usr/bin/env python3
"""Generate or inspect shell-class-specific CUDA derivative kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qce_codegen.shell_class import build_psss_kernel, emit_psss_cuda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell-class", choices=("psss",), default="psss")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument(
        "--format", choices=("cuda", "stats"), default="cuda"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write generated output to this path instead of standard output",
    )
    arguments = parser.parse_args()

    kernel = build_psss_kernel(arguments.axis)
    if arguments.format == "cuda":
        output = emit_psss_cuda(kernel)
    else:
        roots = [kernel.value, kernel.boys_argument]
        roots.extend(item for center in kernel.gradients for item in center)
        output = json.dumps(
            {
                "axis": arguments.axis,
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

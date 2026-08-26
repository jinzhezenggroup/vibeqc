"""Generate or inspect shell-class-specific CUDA derivative kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qce_codegen.dppp_dispatch import (
    build_dppp_fused_plan,
    emit_dppp_fused_cuda,
    emit_shell_class_fused_cuda,
)
from qce_codegen.fused_schedule import build_fused_shell_plan
from qce_codegen.low_order_force import emit_psps_weighted_force_cuda
from qce_codegen.production import write_production_bundle
from qce_codegen.shell_class import (
    build_dppp_component_kernel,
    build_dppp_contraction_kernel,
    build_psss_kernel,
    emit_dppp_component_cuda,
    emit_dppp_contraction_cuda,
    emit_psss_cuda,
)
from qce_codegen.shell_spec import DDPS_SPEC, DPDS_SPEC, DPPP_SPEC, PSPS_SPEC

FUSED_SPECS = {
    "dppp": DPPP_SPEC,
    "dpds": DPDS_SPEC,
    "ddps": DDPS_SPEC,
    "psps": PSPS_SPEC,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shell-class",
        choices=("psss", *FUSED_SPECS),
        default="psss",
    )
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument(
        "--d-component",
        choices=DPPP_SPEC.center_components[0],
        default="xx",
    )
    parser.add_argument(
        "--p-components",
        default="xxx",
        help="three x/y/z component labels for the dppp p shells",
    )
    parser.add_argument(
        "--lowering",
        choices=("full", "factored", "fused"),
        default="full",
        help=(
            "emit full primitive algebra, one factored component, or a "
            "complete cooperative shell-class force kernel"
        ),
    )
    parser.add_argument(
        "--format", choices=("cuda", "stats"), default="cuda"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write generated output to this path instead of standard output",
    )
    parser.add_argument(
        "--production-manifest",
        type=Path,
        help="generate production CUDA shards from an accepted-class manifest",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="build directory for --production-manifest artifacts",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=4,
        help="stable CUDA translation-unit count for production generation",
    )
    arguments = parser.parse_args()

    if arguments.production_manifest is not None:
        if arguments.output_directory is None:
            parser.error("--production-manifest requires --output-directory")
        if arguments.output is not None:
            parser.error("--output cannot be combined with --production-manifest")
        write_production_bundle(
            arguments.production_manifest,
            arguments.output_directory,
            arguments.shards,
        )
        return
    if arguments.output_directory is not None:
        parser.error("--output-directory requires --production-manifest")

    if arguments.shell_class == "psss":
        kernel = build_psss_kernel(arguments.axis)
        component_metadata = {"axis": arguments.axis}
    elif arguments.shell_class == "dppp":
        if len(arguments.p_components) != 3 or any(
            axis not in "xyz" for axis in arguments.p_components
        ):
            parser.error("--p-components must contain exactly three x/y/z labels")
        if arguments.lowering == "full":
            kernel = build_dppp_component_kernel(
                arguments.d_component, tuple(arguments.p_components)
            )
        elif arguments.lowering == "factored":
            kernel = build_dppp_contraction_kernel(
                arguments.d_component, tuple(arguments.p_components)
            )
        else:
            kernel = None
        component_metadata = {"lowering": arguments.lowering}
        if arguments.lowering != "fused":
            component_metadata.update(
                {
                    "d_component": arguments.d_component,
                    "p_components": arguments.p_components,
                }
            )
    else:
        if arguments.lowering != "fused":
            parser.error(
                f"{arguments.shell_class} currently supports only the fused lowering"
            )
        kernel = None
        component_metadata = {"lowering": "fused"}
    if arguments.format == "cuda":
        if arguments.shell_class == "psss":
            output = emit_psss_cuda(kernel)
        elif arguments.lowering == "full":
            output = emit_dppp_component_cuda(kernel)
        elif arguments.lowering == "factored":
            output = emit_dppp_contraction_cuda(kernel)
        elif arguments.shell_class == "dppp":
            output = emit_dppp_fused_cuda()
        elif arguments.shell_class == "psps":
            output = emit_psps_weighted_force_cuda()
        else:
            output = emit_shell_class_fused_cuda(
                FUSED_SPECS[arguments.shell_class]
            )
    else:
        if arguments.lowering == "fused":
            if arguments.shell_class == "dppp":
                plan = build_dppp_fused_plan()
                source = emit_dppp_fused_cuda(plan)
                block_threads = plan.block_threads
            elif arguments.shell_class == "psps":
                plan = build_fused_shell_plan(PSPS_SPEC)
                source = emit_psps_weighted_force_cuda()
                block_threads = 256
            elif arguments.shell_class in FUSED_SPECS:
                specification = FUSED_SPECS[arguments.shell_class]
                plan = build_fused_shell_plan(specification)
                source = emit_shell_class_fused_cuda(specification, plan)
                block_threads = plan.block_threads
            else:
                parser.error("psss does not support the fused lowering")
            output = json.dumps(
                {
                    **component_metadata,
                    "block_threads": block_threads,
                    "component_count": len(plan.components),
                    "coulomb_state_count": len(plan.coulomb_states),
                    "shell_class": arguments.shell_class,
                    "source_bytes": len(source.encode("utf-8")),
                    "source_lines": source.count("\n"),
                    "warp_count": plan.warp_count,
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
            if arguments.output is None:
                print(output, end="")
            else:
                arguments.output.parent.mkdir(parents=True, exist_ok=True)
                arguments.output.write_text(output, encoding="utf-8")
            return
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

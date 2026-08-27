"""Generate or inspect shell-class-specific CUDA derivative kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vibeqc_codegen.cuda_emitter import emit_shell_class_fused_cuda
from vibeqc_codegen.fused_schedule import build_fused_shell_plan
from vibeqc_codegen.ir import KernelConsumer
from vibeqc_codegen.low_order_force import (
    PPSS_BLOCK_THREADS,
    PSPS_BLOCK_THREADS,
    emit_ppss_weighted_force_cuda,
    emit_psps_weighted_force_cuda,
)
from vibeqc_codegen.production import (
    write_production_bundle,
    write_production_bundles,
)
from vibeqc_codegen.shell_class import (
    build_dppp_component_kernel,
    build_dppp_contraction_kernel,
    build_psss_kernel,
    emit_dppp_component_cuda,
    emit_dppp_contraction_cuda,
    emit_psss_cuda,
)
from vibeqc_codegen.shell_spec import DPPP_SPEC, FUSED_SHELL_SPEC_BY_NAME

FUSED_SPECS = FUSED_SHELL_SPEC_BY_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shell-class",
        choices=tuple(FUSED_SPECS),
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
    parser.add_argument(
        "--architecture",
        help="select an architecture profile from a v2 production manifest",
    )
    parser.add_argument(
        "--target-architecture",
        action="append",
        help=(
            "concrete CUDA compile target for a multi-profile bundle; repeat "
            "for every fat-binary architecture"
        ),
    )
    parser.add_argument(
        "--profile",
        default="auto",
        help="auto, portable, sm_XX, or a named manifest profile",
    )
    parser.add_argument(
        "--profile-map",
        action="append",
        default=[],
        metavar="SM_XX=PROFILE",
        help="override profile resolution for one --target-architecture",
    )
    parser.add_argument(
        "--consumer",
        action="append",
        choices=tuple(item.value for item in KernelConsumer),
        help=(
            "generated fused consumer; repeat to emit both Fock values and "
            "analytic forces (force is the default)"
        ),
    )
    arguments = parser.parse_args()

    if arguments.production_manifest is not None:
        if arguments.output_directory is None:
            parser.error("--production-manifest requires --output-directory")
        if arguments.output is not None:
            parser.error("--output cannot be combined with --production-manifest")
        if arguments.target_architecture:
            profile_by_architecture = {}
            for item in arguments.profile_map:
                architecture, separator, profile = item.partition("=")
                if not separator or not architecture or not profile:
                    parser.error("--profile-map must use SM_XX=PROFILE syntax")
                profile_by_architecture[architecture] = profile
            if arguments.profile != "auto":
                for architecture in arguments.target_architecture:
                    profile_by_architecture.setdefault(
                        architecture, arguments.profile
                    )
            write_production_bundles(
                arguments.production_manifest,
                arguments.output_directory,
                arguments.shards,
                arguments.target_architecture,
                profile_by_architecture,
            )
        else:
            write_production_bundle(
                arguments.production_manifest,
                arguments.output_directory,
                arguments.shards,
                arguments.architecture,
                arguments.profile,
            )
        return
    if arguments.output_directory is not None:
        parser.error("--output-directory requires --production-manifest")

    consumers = tuple(
        KernelConsumer(item)
        for item in (arguments.consumer or (KernelConsumer.FORCE.value,))
    )
    if KernelConsumer.FOCK in consumers and KernelConsumer.FORCE not in consumers:
        # The current shared emitter always retains the force oracle alongside
        # a Fock pilot. This prevents a value-only plan from truncating the
        # raised Coulomb states needed by the emitted force functions.
        consumers = (*consumers, KernelConsumer.FORCE)

    if arguments.shell_class == "psss" and arguments.lowering != "fused":
        if arguments.lowering != "full":
            parser.error("psss supports the full pilot or fused shell lowering")
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
        if arguments.shell_class == "psss" and arguments.lowering != "fused":
            output = emit_psss_cuda(kernel)
        elif arguments.lowering == "full":
            output = emit_dppp_component_cuda(kernel)
        elif arguments.lowering == "factored":
            output = emit_dppp_contraction_cuda(kernel)
        elif arguments.shell_class in ("ppss", "psps"):
            if consumers != (KernelConsumer.FORCE,):
                parser.error(
                    f"the weighted {arguments.shell_class} emitter is force-only"
                )
            output = {
                "ppss": emit_ppss_weighted_force_cuda,
                "psps": emit_psps_weighted_force_cuda,
            }[arguments.shell_class]()
        else:
            specification = FUSED_SPECS[arguments.shell_class]
            output = emit_shell_class_fused_cuda(
                specification,
                build_fused_shell_plan(
                    specification, consumers=consumers
                ),
            )
    else:
        if arguments.lowering == "fused":
            specification = FUSED_SPECS[arguments.shell_class]
            plan = build_fused_shell_plan(
                specification, consumers=consumers
            )
            if arguments.shell_class in ("ppss", "psps"):
                if consumers != (KernelConsumer.FORCE,):
                    parser.error(
                        f"the weighted {arguments.shell_class} emitter is force-only"
                    )
                block_threads, emitter = {
                    "ppss": (PPSS_BLOCK_THREADS, emit_ppss_weighted_force_cuda),
                    "psps": (PSPS_BLOCK_THREADS, emit_psps_weighted_force_cuda),
                }[arguments.shell_class]
                source = emitter()
            else:
                source = emit_shell_class_fused_cuda(specification, plan)
                block_threads = plan.block_threads
            output = json.dumps(
                {
                    **component_metadata,
                    "block_threads": block_threads,
                    "component_count": len(plan.components),
                    "coulomb_state_count": len(plan.coulomb_states),
                    "consumers": sorted(
                        item.value for item in plan.kernel.integral.consumers
                    ),
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

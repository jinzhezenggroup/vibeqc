"""Emit one qualified stationary CUDA force translation unit or primitive shard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.integral.first_derivative_native import emit_first_derivative_cuda
from vibeqc_compiler.integral.first_derivative_schedule import (
    CUDA_REQUESTS_PER_UNIT,
    derivative_cuda_sources,
)
from vibeqc_compiler.method.stationary_cuda import (
    QUALIFIED_SPD_AOT_SHARD_WIDTH,
    QUALIFIED_SPD_AOT_SHARDS,
    QUALIFIED_SPD_COMPONENTS,
    emit_stationary_aot_cuda,
    emit_stationary_component_aot_wrapper_cuda,
    qualified_sp_requests,
)

from tools.generate_df_kernels import write_if_changed


def _component_aot_sources() -> tuple[str, ...]:
    units = derivative_cuda_sources(QUALIFIED_SPD_COMPONENTS)
    if (
        CUDA_REQUESTS_PER_UNIT != QUALIFIED_SPD_AOT_SHARD_WIDTH
        or len(units) != QUALIFIED_SPD_AOT_SHARDS
    ):
        raise RuntimeError("stationary s/p/d AOT shard contract drift")
    return tuple(source for _, source in units)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--functional", type=int, choices=(0, 1, 2))
    parser.add_argument("--spin", choices=("unpolarized", "polarized"))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--component-domain", choices=("sp", "spd"), default="sp")
    parser.add_argument("--shard-index", type=int)
    args = parser.parse_args()

    if args.component_domain == "sp":
        if args.shard_index is not None:
            parser.error("--shard-index requires --component-domain spd")
        if args.functional is None or args.spin is None:
            parser.error("--functional and --spin are required for stationary wrappers")
        primitive_source = emit_first_derivative_cuda(qualified_sp_requests())
        source = emit_stationary_aot_cuda(
            args.functional,
            primitive_source=primitive_source,
            spin=args.spin,
            iterations=args.iterations,
        )
    elif args.shard_index is not None:
        if args.functional is not None or args.spin is not None:
            parser.error(
                "component primitive shards are shared across functionals/spins"
            )
        sources = _component_aot_sources()
        if not 0 <= args.shard_index < QUALIFIED_SPD_AOT_SHARDS:
            parser.error(f"--shard-index must be in [0,{QUALIFIED_SPD_AOT_SHARDS - 1}]")
        source = sources[args.shard_index]
    else:
        if args.functional is None or args.spin is None:
            parser.error("--functional and --spin are required for stationary wrappers")
        source = emit_stationary_component_aot_wrapper_cuda(
            args.functional, spin=args.spin, iterations=args.iterations
        )

    write_if_changed(args.output, source)


if __name__ == "__main__":
    main()

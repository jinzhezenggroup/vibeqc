"""Generate the shared raw DF CUDA value header and its operator inventory."""

# Source-tree CLI bootstrap; importing the compiler needs no native runtime.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vibeqc_compiler.integral.df_cuda import df_program_inventory, emit_df_values_cuda


def write_if_changed(path: Path, text: str) -> None:
    """Retain object-cache reuse when an unrelated generator file changes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--derivatives", action="store_true")
    parser.add_argument("--policy-output", type=Path)
    parser.add_argument("--schedule-output", type=Path)
    args = parser.parse_args()
    if args.schedule_output and not args.derivatives:
        parser.error("--schedule-output requires --derivatives")
    if args.derivatives:
        from vibeqc_compiler.integral.df_derivatives_cuda import (
            df_derivative_inventory,
            emit_df_derivatives_cuda,
        )

        emitter, inventory = emit_df_derivatives_cuda, df_derivative_inventory
    else:
        emitter, inventory = emit_df_values_cuda, df_program_inventory
    source = emitter()
    write_if_changed(args.output, source)
    if args.policy_output:
        from vibeqc_compiler.integral.df_policy import emit_df_policy_cuda

        write_if_changed(
            args.policy_output, emit_df_policy_cuda(derivatives=args.derivatives)
        )
    if args.schedule_output:
        from vibeqc_compiler.integral.df_policy import emit_df_derivative_schedule_cuda

        write_if_changed(args.schedule_output, emit_df_derivative_schedule_cuda())
    if args.inventory:
        payload = {
            **inventory(),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        write_if_changed(
            args.inventory, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()

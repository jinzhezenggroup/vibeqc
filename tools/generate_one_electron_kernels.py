"""Generate S/T/V CUDA primitive helpers and their scientific IR inventory."""

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

from vibeqc_compiler.integral.one_electron_cpu import emit_one_electron_st_cpu
from vibeqc_compiler.integral.one_electron_cuda import (
    emit_one_electron_values_cuda,
    one_electron_program_inventory,
)
from vibeqc_compiler.integral.one_electron_derivatives_cuda import (
    emit_one_electron_derivatives_cuda,
    one_electron_derivative_inventory,
)
from vibeqc_compiler.integral.one_electron_policy_cuda import (
    emit_one_electron_policy_cuda,
    one_electron_policy_inventory,
)

from tools.generate_df_kernels import write_if_changed


def main():
    """Keep generated code out of source control and preserve unchanged mtimes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--derivatives", action="store_true")
    parser.add_argument("--policy-output", type=Path)
    parser.add_argument("--cpu-st-output", type=Path)
    args = parser.parse_args()
    if args.derivatives and args.policy_output:
        parser.error("--policy-output applies only to values")
    if not args.output and not args.cpu_st_output:
        parser.error("at least one generated output is required")
    if args.inventory and not args.output:
        parser.error("--inventory requires --output")
    policy_source = emit_one_electron_policy_cuda() if args.policy_output else None
    if policy_source is not None:
        write_if_changed(args.policy_output, policy_source)
    if args.cpu_st_output:
        write_if_changed(args.cpu_st_output, emit_one_electron_st_cpu())
    source = None
    if args.output:
        source = (
            emit_one_electron_derivatives_cuda()
            if args.derivatives
            else emit_one_electron_values_cuda()
        )
        write_if_changed(args.output, source)
    if args.inventory:
        assert source is not None
        payload = {
            **(
                one_electron_derivative_inventory()
                if args.derivatives
                else one_electron_program_inventory()
            ),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        if policy_source is not None:
            payload["execution_policy"] = {
                **one_electron_policy_inventory(),
                "source_sha256": hashlib.sha256(policy_source.encode()).hexdigest(),
            }
        write_if_changed(
            args.inventory, json.dumps(payload, sort_keys=True, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()

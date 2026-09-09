"""Generate S/T/V CUDA primitive helpers and their scientific IR inventory."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_df_kernels import write_if_changed
from tools.vibeqc_codegen.one_electron_cuda import (
    emit_one_electron_values_cuda,
    one_electron_program_inventory,
)


def main():
    """Keep generated code out of source control and preserve unchanged mtimes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    source = emit_one_electron_values_cuda()
    write_if_changed(args.output, source)
    if args.inventory:
        payload = {
            **one_electron_program_inventory(),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        write_if_changed(
            args.inventory, json.dumps(payload, sort_keys=True, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()

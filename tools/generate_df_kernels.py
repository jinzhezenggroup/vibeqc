"""Generate the shared raw DF CUDA value header and its operator inventory."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.vibeqc_codegen.df_cuda import df_program_inventory, emit_df_values_cuda


def write_if_changed(path: Path, text: str) -> None:
    """Retain object-cache reuse when an unrelated generator file changes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    source = emit_df_values_cuda()
    write_if_changed(args.output, source)
    if args.inventory:
        payload = {
            **df_program_inventory(),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        write_if_changed(
            args.inventory, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()

"""Write the installed identity record for one stationary CUDA AOT artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.method.stationary_cuda import (
    stationary_aot_contract_identity,
    stationary_aot_plan_identity,
)


def _code_objects(values: list[str]) -> list[dict[str, str]]:
    result = set()
    for value in values:
        number = value.removesuffix("-real").removesuffix("-virtual")
        if not number.isdigit():
            raise ValueError(f"invalid stationary CUDA architecture: {value}")
        kinds = (
            ("cubin",)
            if value.endswith("-real")
            else ("ptx",)
            if value.endswith("-virtual")
            else ("cubin", "ptx")
        )
        result.update((f"sm_{number}", kind) for kind in kinds)
    return [
        {"architecture": architecture, "kind": kind}
        for architecture, kind in sorted(result)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--functional", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--spin", choices=("unpolarized", "polarized"), required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--architecture", action="append", default=[])
    parser.add_argument("--compile-architecture", action="append", default=[])
    args = parser.parse_args()
    code_objects = _code_objects(args.compile_architecture)
    architectures = sorted({item["architecture"] for item in code_objects})
    if not architectures or architectures != sorted(set(args.architecture)):
        raise ValueError("stationary AOT target/code-object architecture mismatch")
    payload = {
        "schema": "vibeqc.stationary-cuda-aot.v2",
        "functional": args.functional,
        "spin": args.spin,
        "plan_identity": stationary_aot_plan_identity(args.functional, spin=args.spin),
        "partition_iterations": args.iterations,
        "architectures": architectures,
        "compile_architectures": sorted(set(args.compile_architecture)),
        "code_objects": code_objects,
        "source_identity": canonical_hash(args.source.read_text()),
        "source_sha256": file_hash(args.source),
        "contract_identity": stationary_aot_contract_identity(
            args.functional, spin=args.spin, iterations=args.iterations
        ),
        "binary_sha256": file_hash(args.library),
        "binary_bytes": args.library.stat().st_size,
        "compile_contract": {
            "fp64": True,
            "fmad": False,
            "relaxed_constexpr": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if not args.output.exists() or args.output.read_text() != text:
        args.output.write_text(text)


if __name__ == "__main__":
    main()

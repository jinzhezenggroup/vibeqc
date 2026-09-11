"""Replay CPU tensor fragments and register the shared CG01 numerical evidence."""

from __future__ import annotations

# Source-tree CLI bootstrap; importing the compiler needs no native runtime.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vibeqc_compiler.tensor import PASSES, Program, execute, optimize, rewrite
from vibeqc_compiler.tensor.examples import example_cases

from tools.vibeqc_validation.schema import (
    GATES,
    block_error,
    canonical_hash,
    file_hash,
    new_evidence,
    outcome,
    validate_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def _arrays(inputs) -> dict:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": value.tolist(),
        }
        for name, value in sorted(inputs.items())
    }


def run_examples(*, seed: int = 145, equations_dir: Path | None = None) -> list[dict]:
    """Check every rewrite against loops, preserving replay artifacts on request.

    This records CPU numerical evidence only. It makes no compiled-source,
    GPU, complete-method, endpoint, memory-performance, or promotion claim.
    """
    source_files = {
        str(path.relative_to(ROOT)): file_hash(path)
        for path in sorted((ROOT / "python/vibeqc_compiler/tensor").glob("*.py"))
    }
    source_files[str(Path(__file__).resolve().relative_to(ROOT))] = file_hash(__file__)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    records = []
    if equations_dir is not None:
        equations_dir.mkdir(parents=True, exist_ok=True)
    for case in example_cases(seed):
        program = case.program
        arrays = _arrays(case.inputs)
        record = new_evidence(
            tier="cpu",
            subject=f"TensorIR/{case.name}",
            inputs_hash=canonical_hash(
                {"equation": program.logical_hash, "inputs": arrays}
            ),
        )
        record.update(
            revision=revision,
            backend_selected="numpy-cpu-interpreter",
            device={
                "kind": "cpu",
                "machine": platform.machine(),
                "system": platform.system(),
            },
            hardware=outcome("pass"),
            toolchain={"python": platform.python_version(), "numpy": np.__version__},
            settings={
                "seed": seed,
                "dtype": "float64",
                "dirty": dirty,
                "scope": "tensor fragments; no complete CCSD/MP2 method",
                "gates": GATES["integral_fp64"],
                "reference": "explicit coordinate loops",
                "reference_hash": canonical_hash(case.reference.tolist()),
                "source_files": source_files,
            },
        )
        record["hashes"].update(
            equation=program.logical_hash,
            ir=canonical_hash(program.to_payload()),
            source=canonical_hash(source_files),
        )
        record["hash_reasons"]["schedule"] = (
            "CPU reference interpreter has no lowered execution schedule"
        )
        record["stages"]["representation"] = outcome("pass")
        variants = {"original": program, "replay": Program.loads(program.dumps())}
        current = program
        for pass_name in PASSES:
            current = rewrite(current, pass_name)
            variants[f"after_{pass_name}"] = current
        variants["optimized_replay"] = Program.loads(optimize(program).dumps())
        retained = {}
        for name, variant in variants.items():
            execution = execute(variant, case.inputs)
            record["block_errors"][name] = block_error(
                execution.outputs["value"], case.reference, **GATES["integral_fp64"]
            )
            retained[name] = execution.logical_retained_bytes
        record["settings"]["logical_retained_bytes"] = retained
        passed = all(error["passed"] for error in record["block_errors"].values())
        record["stages"]["numerical"] = outcome(
            "pass" if passed else "fail",
            None if passed else "explicit-loop numerical gate failed",
        )
        for stage in ("source", "compilation", "endpoint", "production"):
            record["stages"][stage] = outcome(
                "not-run", "CPU equation interpreter; this stage is outside CG08"
            )
        if equations_dir is not None:
            payload = {
                "schema": "vibeqc.tensor.example",
                "schema_version": 1,
                "program": program.to_payload(),
                "inputs": arrays,
                "reference": case.reference.tolist(),
                "packing": case.packing.to_payload() if case.packing else None,
            }
            (equations_dir / f"{case.name}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
        validate_evidence(record)
        records.append(record)
    return records


def main() -> int:
    """CLI for repeatable CPU checks and optional self-contained example export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=145, help="fixed NumPy fixture seed (default: 145)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the list of CG01 evidence records as JSON (otherwise stdout)",
    )
    parser.add_argument(
        "--equations-dir",
        type=Path,
        help="also export each program, inputs, loop reference, and packing map",
    )
    args = parser.parse_args()
    records = run_examples(seed=args.seed, equations_dir=args.equations_dir)
    encoded = json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return (
        0 if all(r["stages"]["numerical"]["status"] == "pass" for r in records) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Replay RCCSD A equations and register per-block CG01 numerical evidence."""

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
from pathlib import Path

import numpy as np
from vibeqc_compiler.tensor import Program, execute, optimize

from tools.vibeqc_cc import amplitude_layouts, build_program
from tools.vibeqc_cc.inventory import TERMS
from tools.vibeqc_cc.oracle import dense_feeds, homogeneous_groups, random_case
from tools.vibeqc_validation.schema import (
    block_error,
    canonical_hash,
    file_hash,
    new_evidence,
    outcome,
    validate_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def load_references(path):
    """Fail closed on changed values, reference version or upstream identity."""
    data = json.loads(Path(path).read_text())
    upstream = json.loads((ROOT / "tools/vibeqc_cc/source_manifest.json").read_text())
    if (
        data["schema"] != "vibeqc.rccsd.fixed-amplitude-reference"
        or data["version"] != 1
        or data["pyscf"] != "2.14.0"
        or data["upstream"] != upstream
        or data["cases_hash"] != canonical_hash(data["cases"])
    ):
        raise ValueError("invalid CC reference provenance or cases hash")
    for case in data["cases"]:
        if case["inputs_hash"] != canonical_hash(case["inputs"]):
            raise ValueError("CC reference input hash mismatch")
    return data


def equation_artifact(nocc=2, nvir=2):
    program = build_program(nocc, nvir)
    return {
        "program": program.to_payload(),
        "packing": [p.to_payload() for p in amplitude_layouts(nocc, nvir)],
        "inventory_hash": canonical_hash(TERMS),
        "terms": {
            row[0]: {"definition": row, "hash": canonical_hash(row)} for row in TERMS
        },
    }


def run(output, references):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_references(references)
    cases = []
    f, g, x, y = random_case()
    cases.append(("determinant-groups", (f, g, x, y), homogeneous_groups(f, g, x, y)))
    for case in data["cases"]:
        arrays = tuple(np.array(case["inputs"][k]) for k in ("fock", "eri", "t1", "t2"))
        cases.append(
            (
                "pyscf-" + case["name"],
                arrays,
                {
                    "correlation_energy": case["correlation_energy"],
                    "singles_residual": case["updates"][0]["residual"],
                },
            )
        )
    source_paths = [
        *sorted((ROOT / "tools/vibeqc_cc").glob("*.py")),
        *sorted((ROOT / "python/vibeqc_compiler/tensor").glob("*.py")),
        ROOT / "tools/validate_cc.py",
        ROOT / "tools/generate_cc_references.py",
    ]
    sources = {
        str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in source_paths
    }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    records = []
    for name, arrays, expected in cases:
        f, g, x, y = arrays
        program = build_program(*x.shape)
        feeds = dense_feeds(f, g, x, y)
        inputs = {k: v.tolist() for k, v in feeds.items()}
        record = new_evidence(
            tier="cpu", subject="RCCSD-A/" + name, inputs_hash=canonical_hash(inputs)
        )
        record.update(
            revision=revision,
            backend_selected="numpy-cpu-interpreter",
            device={"kind": "cpu", "machine": platform.machine()},
            hardware=outcome("pass"),
            toolchain={"python": platform.python_version(), "numpy": np.__version__},
            settings={
                "scope": "fixed-amplitude energy and T1 only",
                "source_files": sources,
                "reference_file_sha256": file_hash(references),
                "upstream": data["upstream"],
                "reference": "independent determinant polynomial"
                if name.startswith("determinant")
                else "pinned PySCF 2.14.0",
                "dirty": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"], cwd=ROOT, text=True
                    ).strip()
                ),
            },
        )
        record["hashes"].update(
            equation=program.logical_hash,
            ir=canonical_hash(program.to_payload()),
            source=canonical_hash(sources),
        )
        record["hash_reasons"]["schedule"] = (
            "CPU equation interpreter; no compiled CC schedule"
        )
        for variant, p in (
            ("original", program),
            ("replay", Program.loads(program.dumps())),
            ("optimized", optimize(program)),
        ):
            values = execute(p, feeds).outputs
            for key, ref in expected.items():
                error = block_error(
                    values[key], np.asarray(ref), atol=1e-11, rtol=1e-10
                )
                absolute = float(np.max(np.abs(values[key] - ref)))
                error["absolute_gate"] = 1e-8 if "energy" in key else 1e-9
                error["passed"] = error["passed"] and absolute <= error["absolute_gate"]
                record["block_errors"][variant + "/" + key] = error
        passed = all(v["passed"] for v in record["block_errors"].values())
        record["stages"]["representation"] = outcome("pass")
        record["stages"]["numerical"] = outcome(
            "pass" if passed else "fail",
            None if passed else "independent equation gate failed",
        )
        for stage in ("source", "compilation", "endpoint", "production"):
            record["stages"][stage] = outcome(
                "not-run", "not a compiled CC method or converged CCSD endpoint"
            )
        validate_evidence(record)
        records.append(record)
        replay = {
            **equation_artifact(*x.shape),
            "inputs": inputs,
            "reference": {k: np.asarray(v).tolist() for k, v in expected.items()},
        }
        (output / (name + ".json")).write_text(
            json.dumps(replay, sort_keys=True, allow_nan=False) + "\n"
        )
    (output / "numerical.json").write_text(
        json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    records = run(args.output, args.references)
    passed = all(r["stages"]["numerical"]["status"] == "pass" for r in records)
    print(f"RCCSD A: {len(records)} evidence records, numerical_pass={passed}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

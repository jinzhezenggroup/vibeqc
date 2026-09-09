"""Replay full RCCSD equations and export independent CG01 evidence (slice B)."""

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from tools.validate_cc import load_references
from tools.vibeqc_cc.doubles import DEFINITIONS, build_ccsd_program
from tools.vibeqc_cc.oracle import DeterminantOracle, dense_feeds, random_case
from tools.vibeqc_tensor import execute
from tools.vibeqc_validation.schema import (
    block_error,
    canonical_hash,
    file_hash,
    new_evidence,
    outcome,
    write_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = ROOT / "tests/reference_data/cc/rccsd-b.json"
    references = load_references(reference_path)
    cases = []
    for case in references["cases"]:
        arrays = tuple(np.array(case["inputs"][k]) for k in ("fock", "eri", "t1", "t2"))
        expected = {
            **case["intermediates"],
            "correlation_energy": case["correlation_energy"],
            "singles_residual": case["updates"][0]["residual"],
            "doubles_residual": case["updates"][0]["residual2"],
        }
        cases.append(("pyscf-" + case["name"], arrays, expected))
    arrays = random_case(2, 3)
    e, r1, r2 = DeterminantOracle(arrays[0], arrays[1], 2).evaluate_full(*arrays[2:])
    cases.append(
        (
            "determinant",
            arrays,
            {"correlation_energy": e, "singles_residual": r1, "doubles_residual": r2},
        )
    )
    sources = {
        str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p)
        for p in [
            *sorted((ROOT / "tools/vibeqc_cc").glob("*.py")),
            *sorted((ROOT / "tools/vibeqc_tensor").glob("*.py")),
            ROOT / "tools/generate_cc_references.py",
            Path(__file__).resolve(),
        ]
    }
    records = []
    for name, arrays, expected in cases:
        x = arrays[2]
        feeds = dense_feeds(*arrays)
        record = new_evidence(
            tier="cpu",
            subject="RCCSD-B/" + name,
            inputs_hash=canonical_hash({k: v.tolist() for k, v in feeds.items()}),
        )
        record.update(
            revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            backend_selected="numpy-cpu-interpreter",
            device={"kind": "cpu", "machine": platform.machine()},
            hardware=outcome("pass"),
            toolchain={"python": platform.python_version(), "numpy": np.__version__},
            settings={
                "source_files": sources,
                "reference_file_sha256": file_hash(reference_path),
                "reference_cases_hash": references["cases_hash"],
                "reference_outputs_hash": canonical_hash(
                    {k: np.asarray(v).tolist() for k, v in expected.items()}
                ),
                "upstream": references["upstream"],
                "dirty": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"], cwd=ROOT, text=True
                    ).strip()
                ),
                "forms": {},
                "scope": "full residuals at fixed amplitudes; not solver convergence",
            },
        )
        for form in ("expanded", "shared", "optimized"):
            p = build_ccsd_program(*x.shape, form=form)
            values = execute(p, feeds).outputs
            record["settings"]["forms"][form] = p.logical_hash
            for k, expected_value in expected.items():
                error = block_error(
                    values[k], np.asarray(expected_value), atol=1e-11, rtol=1e-10
                )
                error["passed"] &= error["max_absolute_error"] <= (
                    1e-8 if k == "correlation_energy" else 1e-9
                )
                record["block_errors"][form + "/" + k] = error
        record["hashes"].update(
            equation=p.logical_hash,
            ir=canonical_hash(p.to_payload()),
            source=canonical_hash(sources),
        )
        record["hash_reasons"]["schedule"] = "CPU interpreter only"
        record["stages"]["representation"] = outcome("pass")
        passed = all(e["passed"] for e in record["block_errors"].values())
        record["stages"]["numerical"] = outcome(
            "pass" if passed else "fail",
            None if passed else "full residual gate failed",
        )
        write_evidence(output / (name + ".json"), record)
        records.append(record)
    p = build_ccsd_program(2, 3)
    artifact = {
        "program": p.to_payload(),
        "inventory": DEFINITIONS,
        "inventory_hash": canonical_hash(DEFINITIONS),
        "expanded_hash": build_ccsd_program(2, 3, form="expanded").logical_hash,
    }
    (output / "equations-2o3v.json").write_text(
        json.dumps(artifact, sort_keys=True) + "\n"
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = run(args.output)
    passed = all(r["stages"]["numerical"]["status"] == "pass" for r in records)
    print(f"Full RCCSD: {len(records)} independent records; passed={passed}")
    raise SystemExit(0 if passed else 1)

"""Record CPU A1 fixture-consumer evidence; never claim native or GPU acceptance."""

import argparse
import json
import platform
import runpy
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.vibeqc_mp2 import PreparedMP2Energy
from tools.vibeqc_validation.schema import (
    block_error,
    canonical_hash,
    file_hash,
    new_evidence,
    outcome,
    validate_evidence,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    helpers = runpy.run_path(str(ROOT / "tests/python/test_mp2_energy.py"))
    files = [
        *sorted((ROOT / "tools/vibeqc_mp2").glob("*.py")),
        ROOT / "tests/python/test_mp2_energy.py",
        Path(__file__).resolve(),
    ]
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in files}
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    records = []
    for name in ("h2", "water", "lih", "f_heh"):
        s, source, meta, arrays = helpers["fixture"](name)
        expected = helpers["spin_components"](s, arrays["conventional_mo"])
        with PreparedMP2Energy(
            s, source, occupied_tile=2, virtual_tile=3, axis_tile=4
        ) as p:
            result = p.execute()
            equations = {
                str(shape): program.to_payload()
                for shape, program in p._programs.items()
            }
        record = new_evidence(
            tier="cpu",
            subject=f"MP2-A1-internal-fixture/{name}",
            inputs_hash=meta["array_hash"],
        )
        record.update(
            revision=revision,
            backend_selected="numpy-cpu-interpreter",
            hardware=outcome("pass"),
            device={
                "kind": "cpu",
                "system": platform.system(),
                "machine": platform.machine(),
            },
            toolchain={"python": platform.python_version(), "numpy": np.__version__},
            settings={
                "dirty": dirty,
                "scope": "internal consumer of dense test-only AO source; no public/native/GPU acceptance",
                "reference_versions": meta["versions"],
                "source_files": sources,
                "reference": "pinned PySCF MO integrals plus independent explicit spin summation",
                "result": asdict(result),
                "equations": equations,
                "budget_scope": "numeric capacity estimate; not measured process RSS or native peak",
            },
        )
        record["hashes"].update(
            equation=canonical_hash(result.equation_hashes),
            ir=canonical_hash(equations),
            source=canonical_hash(sources),
        )
        record["hash_reasons"]["schedule"] = (
            "CPU interpreter; no lowered native schedule executed"
        )
        record["block_errors"] = {
            "OS_SS": block_error(
                [result.opposite_spin, result.same_spin],
                expected,
                atol=1e-11,
                rtol=1e-10,
            ),
            "correlation": block_error(
                result.correlation_energy,
                meta["records"]["conventional"]["correlation_energy"],
                atol=1e-9,
                rtol=0,
            ),
        }
        record["stages"]["representation"] = outcome("pass")
        passed = all(v["passed"] for v in record["block_errors"].values())
        record["stages"]["numerical"] = outcome(
            "pass" if passed else "fail",
            None if passed else "independent fixture gate failed",
        )
        record["solver_trace_reason"] = (
            "supplied validated snapshot; MP2 is noniterative"
        )
        validate_evidence(record)
        records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return (
        0 if all(r["stages"]["numerical"]["status"] == "pass" for r in records) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

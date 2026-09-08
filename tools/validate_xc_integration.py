"""Archive fixed-density CPU errors and every density-matrix finite-difference step."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import numpy as np
from vibeqc.profiles import atomic_json, canonical_hash, file_hash

from tools.vibeqc_dft import NativeAO
from tools.vibeqc_dft.fixtures import basis_arguments
from tools.vibeqc_validation.schema import (
    block_error,
    new_evidence,
    outcome,
    validate_evidence,
)
from tools.vibeqc_xc import FixedDensityXC, functional
from tools.vibeqc_xc.integration_fixtures import CASES, load_integration_fixture


def run(output):
    records = []
    for name in ("LDA_XC_PW", "PBE"):
        for layout, spin in (
            ("total", "unpolarized"),
            ("total", "polarized"),
            ("spin", "polarized"),
        ):
            integrator = FixedDensityXC(functional(name, spin=spin))
            for case in CASES:
                meta, data, grid = load_integration_fixture(case)
                density = data[f"density_{layout}"]
                with NativeAO(**basis_arguments(meta)) as basis:
                    result = integrator.integrate(basis, grid, density, tile_points=7)
                    prefix = f"{name}_{layout}"
                    errors = {
                        "energy": block_error(
                            [result.energy],
                            data[f"{prefix}_energy"],
                            atol=1e-11,
                            rtol=1e-10,
                        ),
                        "potential": block_error(
                            result.potential,
                            data[f"{prefix}_potential"],
                            atol=1e-11,
                            rtol=1e-10,
                        ),
                    }
                    variations = []
                    if case == "water":
                        for a, b in ((0, 0), (1, 4)):
                            for channel in range(2) if layout == "spin" else (None,):
                                direction = np.zeros_like(density)
                                matrix = (
                                    direction if channel is None else direction[channel]
                                )
                                matrix[a, b] = matrix[b, a] = 1
                                analytic = float(np.sum(result.potential * direction))
                                samples = []
                                for step, limit in zip(
                                    (1e-3, 3e-4, 1e-4), (2e-5, 2e-6, 3e-7), strict=True
                                ):
                                    plus = integrator.integrate(
                                        basis,
                                        grid,
                                        density + step * direction,
                                        tile_points=11,
                                    )
                                    minus = integrator.integrate(
                                        basis,
                                        grid,
                                        density - step * direction,
                                        tile_points=11,
                                    )
                                    fd = (plus.energy - minus.energy) / (2 * step)
                                    samples.append(
                                        {
                                            "step_density": step,
                                            "derivative": fd,
                                            "analytic": analytic,
                                            "absolute_error": abs(fd - analytic),
                                            "limit": limit,
                                            "passed": abs(fd - analytic) < limit,
                                        }
                                    )
                                variations.append(
                                    {
                                        "ao_indices": [a, b],
                                        "spin_channel": channel,
                                        "samples": samples,
                                    }
                                )
                    records.append(
                        {
                            "case": case,
                            "functional": name,
                            "spin": spin,
                            "density_layout": layout,
                            "identity": result.identity,
                            "points": result.points,
                            "tiles": result.tiles,
                            "errors": errors,
                            "variations": variations,
                        }
                    )
    passed = all(
        e["passed"] for row in records for e in row["errors"].values()
    ) and all(
        s["passed"] for row in records for v in row["variations"] for s in v["samples"]
    )
    report = new_evidence(
        tier="cpu",
        subject="DFT03 fixed-density XC; no SCF",
        inputs_hash=canonical_hash([r["identity"] for r in records]),
    )
    report["revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = [
        "tools/vibeqc_xc/integration.py",
        "tools/vibeqc_xc/potential.py",
        "tools/validate_xc_integration.py",
        "tools/vibeqc_xc/integration_fixtures.py",
        "tests/python/test_xc_integration.py",
    ]
    report["source_files"] = {path: file_hash(ROOT / path) for path in paths}
    report["hashes"]["source"] = canonical_hash(report["source_files"])
    report["backend_selected"] = "cpu"
    report["device"] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    report["toolchain"] = {
        "python": sys.version,
        "numpy": np.__version__,
        "library_sha256": file_hash(Path(os.environ["VIBEQC_LIBRARY"])),
    }
    report["settings"] = {
        "scope": "fixed explicit grids; interior-v1; no clipping, SCF, forces or GPU",
        "tile_points": 7,
        "fd_tile_points": 11,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    report["records"] = records
    report["stages"]["numerical"] = outcome(
        "pass" if passed else "fail",
        None if passed else "independent numerical gate failed",
    )
    validate_evidence(report)
    atomic_json(output, report)
    print(f"{len(records)} independent comparisons; pass={passed}; evidence={output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)

"""Validate complete generated DF shell blocks against independent libcint values.

Run this manual numerical tier inside Slurm with --local, or let the shared
benchmark adapter request the finite GPU allocation. Isolated fixture timings
do not promote a production DF source or establish endpoint improvement.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyscf

from tools.vibeqc_codegen.cuda_adapter import CudaBenchmarkExecutor, CudaCompilerAdapter
from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_codegen.df_cuda import df_program_inventory, emit_df_values_cuda
from tools.vibeqc_validation.df_values import df_value_matrix, make_df_value_fixture
from tools.vibeqc_validation.df_values_cuda import emit_df_value_driver
from tools.vibeqc_validation.f_shell import cuobjdump_resources
from tools.vibeqc_validation.f_shell_numerics import numerical_error
from tools.vibeqc_validation.schema import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--threads", type=int, nargs="+", default=[64, 128, 256], choices=(64, 128, 256)
    )
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = cuda_target_info(args.architecture)
    fixtures = df_value_matrix()
    # Contraction counts are runtime extents, independent of shell angular IR.
    fixtures.extend(
        [
            make_df_value_fixture((1, 2), primitive_lengths=(9, 7)),
            make_df_value_fixture((0, 1, 0), primitive_lengths=(9, 7, 5)),
        ]
    )
    inputs = np.concatenate([f.records for f in fixtures])
    input_path = directory / "inputs.bin"
    input_path.write_bytes(
        b"VQDF1421" + struct.pack("<Q", len(inputs)) + inputs.tobytes()
    )
    header, source, obj, driver, executable = [
        directory / name
        for name in ("df_values.cuh", "fixture.cu", "fixture.o", "link.cu", "fixture")
    ]
    header.write_text(emit_df_values_cuda())
    source.write_text(emit_df_value_driver(target.architecture))
    driver.write_text("// Host main and the CUDA fixture are in the compiled object.\n")
    compiler = CudaCompilerAdapter(args.nvcc.resolve(), target, compile_timeout=600)
    compiled = compiler.compile(source, obj)
    (directory / "ptxas.txt").write_text(compiled.stdout + compiled.stderr)
    if compiled.returncode:
        raise RuntimeError("DF numerical fixture compilation failed; see ptxas.txt")
    linked = compiler.link(driver, [obj], executable)
    (directory / "link.txt").write_text(linked.stdout + linked.stderr)
    if linked.returncode:
        raise RuntimeError("DF numerical fixture link failed")
    dump = subprocess.run(
        [str(args.nvcc.with_name("cuobjdump")), "--dump-resource-usage", str(obj)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    (directory / "cuobjdump.txt").write_text(dump.stdout)
    executor = CudaBenchmarkExecutor(
        timeout=300,
        local=args.local,
        partition="main",
        gres="gpu:5090:1",
        slurm_time="00:05:00",
    )
    report = {
        "schema": "vibeqc.df_value_validation",
        "version": 1,
        "tier": "isolated_native_values",
        "production_promoted": False,
        "source_hash": file_hash(header),
        "driver_hash": file_hash(source),
        "object_hash": file_hash(obj),
        "object_bytes": obj.stat().st_size,
        "source_bytes": header.stat().st_size,
        "compile": asdict(compiled),
        "resources": cuobjdump_resources(dump.stdout),
        "reference_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
        },
        "programs": len(df_program_inventory()["programs"]),
        "fixture_count": len(fixtures),
        "primitive_count": len(inputs),
        "projection_location": "host, independent PySCF transform",
        "runs": [],
    }
    for threads in args.threads:
        output = directory / f"values-{threads}.bin"
        run = subprocess.run(
            [*executor.command(executable), str(input_path), str(output), str(threads)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        (directory / f"runtime-{threads}.log").write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError("DF numerical fixture execution failed")
        values = np.fromfile(output, dtype="<f8")
        if values.shape != (len(inputs),):
            raise ValueError("DF numerical fixture output is incomplete")
        rows, offset = [], 0
        for fixture in fixtures:
            contracted = fixture.contract(
                values[offset : offset + len(fixture.records)]
            )
            offset += len(fixture.records)
            rows.append(
                {
                    "name": fixture.name,
                    "inputs_hash": fixture.input_hash,
                    "shape": list(fixture.reference.shape),
                    "inputs": fixture.inputs,
                    "cartesian": numerical_error(
                        contracted, fixture.reference, atol=2e-10, rtol=2e-10
                    ),
                    "spherical": numerical_error(
                        fixture.spherical(contracted),
                        fixture.spherical_reference,
                        atol=2e-10,
                        rtol=2e-10,
                    ),
                }
            )
        report["runs"].append(
            {
                "runtime": json.loads(run.stdout),
                "fixtures": rows,
                "passed": all(
                    r[k]["passed"] for r in rows for k in ("cartesian", "spherical")
                ),
            }
        )
    report["passed"] = all(run["passed"] for run in report["runs"])
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "fixtures": len(fixtures),
                "primitive_count": len(inputs),
                "report": str(directory / "report.json"),
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

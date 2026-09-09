"""Validate generated CUDA S/T/V blocks independently of HF convergence.

Compilation, resources and raw timings use the shared CUDA acceptance adapters.
The numerical runner requests a finite Slurm allocation unless --local is used
inside an existing allocation. These isolated timings do not promote HF paths.
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
from tools.vibeqc_codegen.one_electron_cuda import emit_one_electron_values_cuda
from tools.vibeqc_validation.f_shell import cuobjdump_resources
from tools.vibeqc_validation.one_electron_cuda import emit_one_electron_value_driver
from tools.vibeqc_validation.one_electron_values import one_electron_value_matrix
from tools.vibeqc_validation.schema import block_error, file_hash


def main():
    """Archive complete inputs, exact source/binary identity and every block error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.local and not os.environ.get("SLURM_JOB_ID"):
        parser.error("--local requires an existing Slurm GPU allocation")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    fixtures = one_electron_value_matrix()
    inputs = np.concatenate([f.records for f in fixtures])
    input_path = directory / "inputs.bin"
    input_path.write_bytes(
        b"VQOE1401" + struct.pack("<Q", len(inputs)) + inputs.tobytes()
    )
    header, source, obj, driver, executable = [
        directory / name
        for name in (
            "generated_one_electron_values.cuh",
            "fixture.cu",
            "fixture.o",
            "link.cu",
            "fixture",
        )
    ]
    header.write_text(emit_one_electron_values_cuda())
    source.write_text(emit_one_electron_value_driver(args.architecture))
    driver.write_text("// Numerical main is in fixture.o.\n")
    compiler = CudaCompilerAdapter(
        args.nvcc.resolve(), cuda_target_info(args.architecture), compile_timeout=900
    )
    compiled = compiler.compile(source, obj)
    (directory / "ptxas.txt").write_text(compiled.stdout + compiled.stderr)
    if compiled.returncode:
        raise RuntimeError("one-electron fixture compilation failed; see ptxas.txt")
    linked = compiler.link(driver, [obj], executable)
    (directory / "link.txt").write_text(linked.stdout + linked.stderr)
    if linked.returncode:
        raise RuntimeError("one-electron fixture link failed; see link.txt")
    dump = subprocess.run(
        [str(args.nvcc.with_name("cuobjdump")), "--dump-resource-usage", str(obj)],
        capture_output=True,
        text=True,
        check=True,
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
        "schema": "vibeqc.one_electron_value_validation",
        "version": 1,
        "production_promoted": False,
        "source_hash": file_hash(header),
        "driver_hash": file_hash(source),
        "object_hash": file_hash(obj),
        "input_hash": file_hash(input_path),
        "source_bytes": header.stat().st_size,
        "object_bytes": obj.stat().st_size,
        "compile": asdict(compiled),
        "resources": cuobjdump_resources(dump.stdout),
        "reference_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
        },
        "runs": [],
    }
    for threads in (64, 128):
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
            raise RuntimeError("one-electron fixture failed; see runtime log")
        values = np.fromfile(output, dtype="<f8").reshape(len(inputs), 3)
        rows, offset = [], 0
        for fixture in fixtures:
            contracted = fixture.contract(
                values[offset : offset + len(fixture.records)]
            )
            offset += len(fixture.records)
            spherical = fixture.spherical(contracted)
            for i, name in enumerate(("S", "T", "V")):
                rows.append(
                    {
                        "name": fixture.inputs["name"] + "/" + name,
                        "inputs": fixture.inputs,
                        "input_hash": fixture.input_hash,
                        "cartesian": block_error(
                            contracted[i], fixture.reference[i], atol=1e-11, rtol=3e-12
                        ),
                        "spherical": block_error(
                            spherical[i],
                            fixture.spherical_reference[i],
                            atol=1e-11,
                            rtol=3e-12,
                        ),
                    }
                )
        report["runs"].append({"runtime": json.loads(run.stdout), "fixtures": rows})
    report["passed"] = all(
        row[k]["passed"]
        for run in report["runs"]
        for row in run["fixtures"]
        for k in ("cartesian", "spherical")
    )
    (directory / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "fixtures": len(fixtures),
                "primitive_count": len(inputs),
                "report": str(directory / "report.json"),
            }
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

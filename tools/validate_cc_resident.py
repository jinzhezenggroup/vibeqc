"""GPU resident-vs-ordinary parity validator for #149 B.
Run with: export VIBEQC_TENSOR_CUDA_TEST=1 VIBEQC_TENSOR_ARCH=sm_90 VIBEQC_NVCC=/usr/local/cuda/bin/nvcc SCRATCH=/path/to/scratch/issue-0149-b && source $SCRATCH/venv/bin/activate && export PYTHONPATH=$SCRATCH/repo/python:$SCRATCH/repo && python tools/validate_cc_resident.py --output $SCRATCH/parity-results --cache $SCRATCH/cache --nvcc /usr/local/cuda/bin/nvcc --architecture sm_90 --compile-only
"""

import argparse
import json
import os
import platform
import sys as _compiler_sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# In installed/repo mode, find the source tree
for candidate in (ROOT, Path(os.environ.get("SCRATCH", "")) / "repo"):
    if (candidate / "tools/vibeqc_cc").is_dir():
        ROOT = candidate.resolve()
        break

_compiler_sys.path.insert(0, str(ROOT / "python"))

import numpy as np
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor.cuda_execute import tensor_source_identity
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
from vibeqc_compiler.tensor.cuda_resident import PreparedResident, compile_resident

from tools.validate_cc import load_references
from tools.vibeqc_cc.cuda import PreparedRCCSDResidual, rccsd_program
from tools.vibeqc_cc.oracle import dense_feeds
from tools.vibeqc_validation.schema import block_error, file_hash


def run(
    output: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
    *,
    compile_only: typing.Any = False,
) -> typing.Any:
    output.mkdir(parents=True, exist_ok=True)
    reference_path = ROOT / "tests/reference_data/cc/rccsd-b.json"
    references = load_references(reference_path)

    manifest = {
        "scope": "#149 B resident-vs-ordinary GPU parity",
        "tensor_source_identity": tensor_source_identity(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "reference_sha256": file_hash(reference_path),
        "compiler_target": compiler.target.to_payload(),
        "runtime_device": None,  # populated by first non-compile run
        "cases": [],
    }

    for case in references["cases"]:
        arrays = [
            np.asarray(case["inputs"][k], dtype=np.float64)
            for k in ("fock", "eri", "t1", "t2")
        ]
        feeds = dense_feeds(*arrays)
        shape = arrays[2].shape
        assert np.max(np.abs(arrays[3])) > 0, "need nonzero t2"
        program = rccsd_program(*shape, trace=True)
        unit = plan_cuda(
            program,
            compiler.target,
            library_bytes=0,
            schedule=TensorSchedule(tile_m=1, tile_n=1, tile_k=1),
        )
        budgets = (("normal", 256 << 20, 4 << 20), ("minimum", unit.peak_bytes, 0))
        print(f"  {case['name']} shape={shape}", flush=True)

        for bname, budget, ws in budgets:
            record = {"case": case["name"], "budget": bname}
            p = plan_cuda(program, compiler.target, max_bytes=budget, library_bytes=ws)

            if compile_only:
                a = compile_resident(p, compiler, cache)
                print(f"    [{bname}] compiled {a.metadata['key'][:16]}", flush=True)
                record["compiled"] = True
                manifest["cases"].append(record)
                continue

            # 1. Ordinary host-staged baseline
            print(f"    [{bname}] ordinary...", flush=True, end=" ")
            with PreparedRCCSDResidual(
                *shape,
                compiler,
                cache,
                trace=True,
                max_bytes=budget,
                library_bytes=ws,
            ) as ordinary:
                ordinary_result = ordinary.execute(feeds)
            ordinary_outputs = {
                k: np.asarray(v) for k, v in ordinary_result.outputs.items()
            }
            print("OK", flush=True)

            # 2. Compile + run resident
            artifact = compile_resident(p, compiler, cache)
            record["resident_artifact"] = {
                "key": artifact.metadata["key"],
                "binary_sha256": artifact.metadata["binary_sha256"],
            }
            if manifest["runtime_device"] is None:
                manifest["runtime_device"] = getattr(ordinary.executor, "device", None)
            record["runtime_device"] = manifest["runtime_device"]
            parity = []
            with PreparedResident(p, artifact) as resident:
                resident.upload(feeds)
                for run_i, profile in enumerate((False, True, False)):
                    leases, metrics = resident.run(profile=profile)
                    res = {
                        name: resident.download(lease) for name, lease in leases.items()
                    }
                    checks = {}
                    for name, ord_val in ordinary_outputs.items():
                        checks[name] = block_error(
                            res[name], ord_val, atol=1e-11, rtol=1e-10
                        )
                    ok = all(
                        c["max_absolute_error"] <= (1e-8 if "energy" in k else 1e-9)
                        for k, c in checks.items()
                    )
                    parity.append(
                        {
                            "run": run_i,
                            "profiled": profile,
                            "passed": ok,
                            "metrics": metrics,
                        }
                    )
                # stale lease: save lease from run N, then invalidate with
                # another upload/run, and assert the old lease is rejected.
                stale_ok = True
                if leases:
                    old_lease = next(iter(leases.values()))
                    resident.upload(feeds)  # advances generation
                    new_leases, _ = resident.run()
                    try:
                        resident.download(old_lease)
                        stale_ok = False
                    except RuntimeError:
                        pass
                    # Also verify the new (current) leases work
                    for name, lease in new_leases.items():
                        resident.download(lease)

            # 3. Error propagation
            bad = {k: np.asarray(v, copy=True) for k, v in feeds.items()}
            bad["t2"][0, 0, 0, 0] = np.inf
            nonfinite_ok = True
            with PreparedResident(p, artifact) as rej:
                try:
                    rej.upload(bad)
                    rej.run()
                    nonfinite_ok = False
                except (RuntimeError, ValueError):
                    pass

            missing_ok = True
            with PreparedResident(p, artifact) as rej:
                rej.upload({"t1": feeds["t1"]})
                try:
                    rej.run()
                    missing_ok = False
                except ValueError:
                    pass

            # 4. Repeated runs
            repeat_ok = True
            with PreparedResident(p, artifact) as reuser:
                reuser.upload(feeds)
                r1, _ = reuser.run()
                first = {name: reuser.download(lease) for name, lease in r1.items()}
                reuser.upload(feeds)
                r2, _ = reuser.run()
                second = {name: reuser.download(lease) for name, lease in r2.items()}
                for name, first_val in first.items():
                    c = block_error(second[name], first_val, atol=1e-11, rtol=1e-10)
                    if c["max_absolute_error"] > 1e-9:
                        repeat_ok = False

            tests_passed = all([stale_ok, nonfinite_ok, missing_ok, repeat_ok])
            all_passed = all(r["passed"] for r in parity) and tests_passed

            record["resident_artifact"] = {
                "key": artifact.metadata["key"],
                "binary_sha256": artifact.metadata["binary_sha256"],
            }
            record["parity"] = parity
            record["tests"] = {
                "stale_lease": stale_ok,
                "nonfinite": nonfinite_ok,
                "missing_input": missing_ok,
                "repeat": repeat_ok,
            }
            record["passed"] = all_passed
            manifest["cases"].append(record)

            fname = f"{case['name']}-{bname}.json"
            (output / fname).write_text(json.dumps(record, indent=2) + "\n")
            print(f"    [{bname}] PASSED={all_passed}", flush=True)

    manifest["passed"] = bool(manifest["cases"]) and all(
        c.get("passed", False)
        if not c.get("compiled", False)
        else c.get("compiled", False)
        for c in manifest["cases"]
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    report = run(
        args.output,
        CudaCompilerAdapter(args.nvcc, cuda_target_info(args.architecture)),
        args.cache,
        compile_only=args.compile_only,
    )
    raise SystemExit(0 if args.compile_only or report.get("passed") else 1)

"""Validate production CUDA resource bounds and emit a reproducible receipt.

Run inside a Slurm GPU allocation. This checks scoped owned capacities and
scientific results; it does not interpret device free memory as owned usage.
"""

import argparse
import ctypes
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
from vibeqc import (
    Calculator,
    ResourceBudget,
    ResourcePlan,
    ResourceSession,
    _native,
    plan_resources,
)
from vibeqc.autotune import source_identity
from vibeqc.profiles import file_hash, find_nvcc

from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter
from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    input_tensor,
    reduce_sum,
)
from tools.vibeqc_tensor.resources import tensor_resource_choices

H2 = [(1, (0, 0, -0.7)), (1, (0, 0, 0.7))]
WATER = [(8, (0, 0, 0)), (1, (1.43, 0, 1.11)), (1, (-1.43, 0, 1.11))]


def hf_case(method, mode):
    """Measure all retained ragged buckets across matching cold/warm histories."""
    systems = [H2, WATER, H2]
    fitted = mode != "direct"
    options = {
        "method": method,
        "density_fitting": "cuda" if fitted else "none",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    reference = Calculator(device="cuda", **options)
    request = reference._resource_request(systems)
    candidate = next(
        c
        for c in request.candidates
        if c.mode == ("resident" if mode == "direct" else mode)
    )
    selected = ResourcePlan(
        ResourceBudget(), (request,), (("hf", candidate.name),), "feasible"
    )
    budget = ResourceBudget(
        host_bytes=selected.peak_bytes["host"],
        device_bytes=selected.peak_bytes["device"],
    )
    calculator = Calculator(device="cuda", resource_budget=budget, **options)
    native_budget = int(
        dict(candidate.decisions).get("density_fitting_memory_budget_bytes", 0)
    )
    ordinary = Calculator(
        device="cuda", density_fitting_memory_budget_bytes=native_budget, **options
    )
    cpu = Calculator(**{**options, "density_fitting": "cpu" if fitted else "none"})
    oracle = [cpu.singlepoint(atoms) for atoms in systems]
    replays = []
    ledger = None
    try:
        with (
            calculator.prepare_batch(systems) as batch,
            ordinary.prepare_batch(systems) as baseline,
        ):
            assert dict(batch.resource_plan.selections)["hf"] == candidate.name
            for _ in range(2):
                actual, expected = (
                    batch.execute(strict=True).items,
                    baseline.execute(strict=True).items,
                )
                energy_error = max(
                    abs(a.energy - b.energy)
                    for a, b in zip(actual, expected, strict=True)
                )
                cpu_error = max(
                    abs(a.energy - b.energy)
                    for a, b in zip(actual, oracle, strict=True)
                )
                force_error = max(
                    float(np.max(np.abs(a.forces - b.forces)))
                    for a, b in zip(actual, expected, strict=True)
                )
                cpu_force_error = max(
                    float(np.max(np.abs(a.forces - b.forces)))
                    for a, b in zip(actual, oracle, strict=True)
                )
                assert energy_error < 1e-10 and cpu_error < 1e-9
                assert force_error < 1e-9 and cpu_force_error < 2e-8
                observation = batch.resource_diagnostics["observation"]
                tracked = observation["device_ledger"]
                assert (
                    0
                    < tracked["live_bytes"]
                    <= tracked["peak_bytes"]
                    <= tracked["limit_bytes"]
                )
                assert tracked["rejected_allocations"] == 0
                if fitted:
                    assert all(
                        d.streamed == (mode == "recomputed")
                        for d in batch.last_density_fitting_metric_diagnostics()
                    )
                replays.append(
                    {
                        "observation": observation,
                        "maximum_energy_error": energy_error,
                        "maximum_cpu_energy_error": cpu_error,
                        "maximum_force_error": force_error,
                        "maximum_cpu_force_error": cpu_force_error,
                    }
                )
            accepted = batch.resource_plan.to_dict()
            # Keep only the observation handle while native cache destruction
            # releases buffers, making charge release independently observable.
            ledger, batch._resource_ledger = batch._resource_ledger, None
        after_destroy = ledger.to_dict()["live_bytes"]
        assert after_destroy == 0
    finally:
        if ledger is not None:
            ledger.close()
    minimum_host = min(
        ResourcePlan(
            ResourceBudget(), (request,), (("hf", c.name),), "feasible"
        ).peak_bytes["host"]
        for c in request.candidates
    )
    infeasible = plan_resources([request], ResourceBudget(host_bytes=minimum_host - 1))
    assert infeasible.status == "infeasible"
    return {
        "case": f"{method}-{mode}",
        "plan": accepted,
        "replays": replays,
        "live_device_bytes_after_destroy": after_destroy,
        "infeasible": infeasible.to_dict(),
    }


def shared_case(compiler, cache):
    """Retain HF and TensorIR simultaneously under one constrained budget."""
    calculator = Calculator(device="cuda")
    hf = calculator._resource_request([H2])
    index = Index("i", IndexSpace("axis", "batch", 8192))
    x = input_tensor("x", TensorSpec((index,), role="input"))
    program = Program(
        {
            f"r{i}": reduce_sum(add(x, x, coefficients=(1, i + 1)), (0,))
            for i in range(6)
        }
    )
    tensor = tensor_resource_choices(program, compiler.target)
    smallest = min((p for _, p in tensor.plans), key=lambda p: p.device_bytes)
    hf_plan = plan_resources([hf], ResourceBudget()).require_feasible()
    budget = ResourceBudget(
        device_bytes=hf_plan.peak_bytes["device"] + smallest.device_bytes,
        host_bytes=hf_plan.peak_bytes["host"] + smallest.host_bytes,
    )
    plan = plan_resources([hf, tensor.request], budget).require_feasible()
    assert tensor.selected(plan).schedule.recompute
    with ResourceSession(
        plan,
        {
            "hf": lambda p: calculator.prepare_batch([H2], resource_plan=p),
            "tensor": tensor.factory(compiler, cache),
        },
    ) as session:
        session.advance(0)
        native = session.provider("hf")
        energy = native.execute(strict=True).items[0].energy
        assert abs(energy - Calculator().singlepoint(H2).energy) < 1e-10
        feeds = {"x": np.linspace(-0.25, 0.75, 8192)}
        result = session.provider("tensor").execute(feeds)
        error = max(
            abs(float(result.outputs[f"r{i}"]) - (i + 2) * feeds["x"].sum())
            for i in range(6)
        )
        assert error < 1e-8
        observation = native.resource_diagnostics["observation"]
        device_peak_bound = (
            observation["device_ledger"]["peak_bytes"]
            + result.metrics["tracked_device_bytes"]
        )
        assert device_peak_bound <= plan.peak_bytes["device"]
        return {
            "case": "hf-tensor",
            "plan": plan.to_dict(),
            "hf_observation": observation,
            "tensor_metrics": result.metrics,
            "observed_device_peak_upper_bound": device_peak_bound,
            "maximum_tensor_error": error,
            "allocation_fallbacks": session.fallbacks,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_120")
    args = parser.parse_args()
    build = args.build.resolve()
    cache = (build / "CMakeCache.txt").read_text()
    for required in (
        "VIBEQC_ENABLE_CUDA:BOOL=ON",
        "VIBEQC_CUDA_FAST_COMPILE:BOOL=OFF",
        "CMAKE_BUILD_TYPE:STRING=Release",
    ):
        if required not in cache:
            raise ValueError(
                f"resource receipts require production configuration: {required}"
            )
    library = build / "libvibeqc.so"
    os.environ["VIBEQC_LIBRARY"] = str(library)
    os.environ["VIBEQC_PROFILE"] = "off"
    native = _native.load_library(device="cpu")
    native.vibeqc_get_source_identity.restype = ctypes.c_char_p
    native_source = native.vibeqc_get_source_identity().decode()
    if native_source != source_identity(Path(__file__).resolve().parents[2]):
        raise ValueError("rebuild the native library after the final source edits")
    records = []
    for method in ("rhf", "uhf"):
        for mode in ("direct", "resident", "recomputed"):
            record = hf_case(method, mode)
            records.append(record)
            print(f"validated {record['case']}", flush=True)
    nvcc = find_nvcc()
    if nvcc is None:
        raise RuntimeError("set VIBEQC_NVCC to the production CUDA compiler")
    compiler = CudaCompilerAdapter(nvcc, cuda_target_info(args.architecture))
    records.append(shared_case(compiler, args.cache))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "platform": platform.platform(),
                "library_sha256": file_hash(library),
                "source_identity": native_source,
                "production_configuration": [
                    line
                    for line in cache.splitlines()
                    if line.startswith(
                        (
                            "CMAKE_BUILD_TYPE:",
                            "CMAKE_CUDA_",
                            "CMAKE_CXX_FLAGS",
                            "VIBEQC_",
                        )
                    )
                ],
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ),
                "scope": "owned HF CUDA allocations and TensorIR arrays/provider retention; sum of observed owner peaks bounds their joint live usage; runtime/library exclusions remain explicit in each plan",
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

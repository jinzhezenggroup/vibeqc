"""Reproduce #469 CPU scalar/AVX2 lane-lowering measurements."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cpu_target import (
    AVX2_FMA_TARGET,
    AVX512F_FMA_TARGET,
    GENERIC_CPU_TARGET,
)
from vibeqc_compiler.integral.cpu_lane_execute import (
    compile_first_derivative_cpu_lane,
)
from vibeqc_compiler.integral.cpu_schedule import default_cpu_schedule
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir


def _flags() -> set[str]:
    try:
        return set(Path("/proc/cpuinfo").read_text().lower().split())
    except OSError:
        return set()


def _angular(value: str) -> tuple[int, int, int, int]:
    if len(value) != 4 or any(letter not in "spdf" for letter in value):
        raise argparse.ArgumentTypeError(
            "shell class must be four letters from s/p/d/f"
        )
    return (
        "spdf".index(value[0]),
        "spdf".index(value[1]),
        "spdf".index(value[2]),
        "spdf".index(value[3]),
    )


def _native_function(artifact):
    owner = ctypes.CDLL(str(artifact.native.library))
    function = owner.vibeqc_first_sum_cpu_lane_v1
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    return owner, function


def _median_ms(function, records, output, samples):
    for _ in range(10):
        assert (
            function(
                records.ctypes.data,
                len(records),
                records.shape[1],
                output.ctypes.data,
                output.size,
            )
            == 0
        )
    values = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        assert (
            function(
                records.ctypes.data,
                len(records),
                records.shape[1],
                output.ctypes.data,
                output.size,
            )
            == 0
        )
        values.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell", type=_angular, default=_angular("fdps"))
    parser.add_argument("--records", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.records < 1 or args.samples < 1:
        raise ValueError("records and samples must be positive")
    executable = shutil.which("c++")
    if executable is None:
        raise RuntimeError("a C++ compiler is required")
    cache = (args.cache or Path(tempfile.mkdtemp(prefix="vibeqc469-"))).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    ir = build_weighted_eri_ir(args.shell)
    compiler = CppCompilerAdapter(Path(executable))
    flags = _flags()
    targets = [GENERIC_CPU_TARGET]
    if platform.machine().lower() in ("x86_64", "amd64"):
        targets.extend((AVX2_FMA_TARGET, AVX512F_FMA_TARGET))

    rng = np.random.default_rng(469)
    stride = 17
    records = np.empty((args.records, stride), dtype=np.float64)
    records[:, :4] = rng.uniform(0.2, 2.0, size=(args.records, 4))
    records[:, 4:16] = rng.normal(scale=0.7, size=(args.records, 12))
    records[:, 16] = rng.normal(size=args.records)
    output = np.empty(13, dtype=np.float64)
    baseline = None
    reference = None
    results = []
    keepalive = []
    for target in targets:
        start = time.perf_counter()
        artifact = compile_first_derivative_cpu_lane(
            ir,
            compiler,
            cache / target.name,
            component_indices=(0,),
            target=target,
            schedule=default_cpu_schedule(target),
        )
        cold = time.perf_counter() - start
        supported = target.vector_lanes == 1 or all(
            feature in flags for feature in target.features
        )
        median = None
        maximum_error = None
        if supported:
            owner, function = _native_function(artifact)
            keepalive.append(owner)
            median = _median_ms(function, records, output, args.samples)
            if reference is None:
                reference = output.copy()
                baseline = median
                maximum_error = 0.0
            else:
                maximum_error = float(np.max(np.abs(output - reference)))
        sources = tuple(
            (cache / target.name / "generated-cpu-lane-sources").glob("*.cpp")
        )
        results.append(
            {
                "target": target.name,
                "features": target.features,
                "vector_lanes": target.vector_lanes,
                "cold_compile_seconds": cold,
                "generated_source_bytes": sum(path.stat().st_size for path in sources),
                "shared_library_bytes": artifact.native.library.stat().st_size,
                "executed": supported,
                "median_ms": median,
                "speedup_vs_generic": (
                    baseline / median
                    if supported and baseline is not None and median is not None
                    else None
                ),
                "maximum_absolute_difference_vs_generic": maximum_error,
            }
        )
    print(
        json.dumps(
            {
                "schema": "vibeqc.issue469.cpu-simd.v1",
                "shell": args.shell,
                "component_index": 0,
                "records": args.records,
                "samples": args.samples,
                "machine": platform.machine(),
                "targets": results,
                "numeric_policy": (
                    "strict libm special functions per lane; vector recurrence; "
                    "-ffp-contract=off; no fast-math"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Reproduce #351 bounded full-shell CPU codegen measurements.

The generated timing includes one complete d-p-s-s value plus all twelve
shell-center derivatives. The independent RawSource timing is value-only and
must not be interpreted as a matched endpoint performance comparison.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
import typing
from pathlib import Path

import numpy as np
from vibeqc import Primitive, Shell
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.integral.first_derivatives_execute import (
    FirstDerivativeShellEvaluator,
    compile_first_derivative_shell,
)
from vibeqc_compiler.integral.weight_pullback import normalized_radial_primitives
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir

from tools.vibeqc_posthf.sources import NativeSource


def _fixture() -> typing.Any:
    coordinates = np.array(
        [
            [0.13, -0.31, 0.24],
            [-0.43, 0.27, 0.51],
            [0.68, -0.14, -0.22],
            [-0.21, 0.48, -0.63],
        ]
    )
    specs = (
        ((0.60, 1.0), (0.22, 0.35)),
        ((0.80, 1.0),),
        ((1.10, 1.0),),
        ((0.90, 1.0),),
    )
    angular = (2, 1, 0, 0)
    primitives = tuple(
        normalized_radial_primitives(l, shell)
        for l, shell in zip(angular, specs, strict=True)
    )
    shells = tuple(
        Shell(
            atom,
            l,
            tuple(Primitive(exponent, coefficient) for exponent, coefficient in shell),
        )
        for atom, (l, shell) in enumerate(zip(angular, specs, strict=True))
    )
    return coordinates, specs, primitives, shells


def _median_ms(function: typing.Any, samples: typing.Any) -> typing.Any:
    for _ in range(5):
        function()
    timings = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        function()
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--tile-size", type=int, default=5)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")
    executable = shutil.which("c++")
    if executable is None:
        raise RuntimeError("a C++ compiler is required")
    cache = args.cache or Path(tempfile.mkdtemp(prefix="vibeqc351-"))
    cache = cache.resolve()
    if args.cache is not None and cache.exists() and any(cache.iterdir()):
        raise ValueError("--cache must be empty for a cold-compile measurement")
    cache.mkdir(parents=True, exist_ok=True)
    ir = build_weighted_eri_ir((2, 1, 0, 0))

    start = time.perf_counter()
    artifact = compile_first_derivative_shell(
        ir,
        CppCompilerAdapter(Path(executable)),
        cache,
        tile_size=args.tile_size,
    )
    cold_compile = time.perf_counter() - start
    evaluator = FirstDerivativeShellEvaluator(
        artifact, record_capacity=2, budget_bytes=8 << 20
    )
    coordinates, _, primitives, shells = _fixture()
    generated_ms = _median_ms(
        lambda: evaluator.contract(primitives, coordinates), args.samples
    )
    with NativeSource([(1, xyz) for xyz in coordinates], basis=shells) as source:
        sizes = tuple(source.shell_sizes)
        offsets = tuple(int(x) for x in np.cumsum((0, *sizes[:-1]), dtype=np.int64))
        oracle_ms = _median_ms(
            lambda: source._read("four_center_eri", offsets, sizes), args.samples
        )
    sources = tuple((cache / "generated-sources").glob("*.cpp"))
    payload = {
        "cold_compile_seconds": cold_compile,
        "program_identity": artifact.program_identity,
        "tile_component_counts": [
            len(tile.component_indices) for tile in artifact.tiles
        ],
        "generated_source_bytes": sum(path.stat().st_size for path in sources),
        "shared_library_bytes_sum": sum(
            tile.native.library.stat().st_size for tile in artifact.tiles
        ),
        "numeric_peak_bytes": evaluator.numeric_bytes,
        "generated_value_plus_12_derivatives_median_ms": generated_ms,
        "independent_oracle_value_only_median_ms": oracle_ms,
        "samples": args.samples,
        "comparison_scope": (
            "oracle timing is value-only; this is compile/resource evidence, "
            "not a matched endpoint promotion comparison"
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

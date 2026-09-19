"""Reproduce #471 bounded CPU schedule autotuning with an independent oracle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from vibeqc import Primitive, Shell
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.integral.cpu_tune import (
    CpuTuneLimits,
    tune_cpu_first_derivative_shell,
    write_cpu_tuning_manifest,
)
from vibeqc_compiler.integral.weight_pullback import (
    normalized_cartesian_components,
    normalized_radial_primitives,
)
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir

from tools.vibeqc_posthf.sources import NativeSource


def _fixture():
    coordinates = np.array(
        [
            [0.13, -0.31, 0.24],
            [-0.43, 0.27, 0.51],
            [0.68, -0.14, -0.22],
            [-0.21, 0.48, -0.63],
        ]
    )
    angular = (1, 0, 0, 0)
    specs = (
        (
            (2.10, 0.12),
            (1.45, -0.08),
            (0.98, 0.21),
            (0.66, 0.31),
            (0.44, -0.11),
            (0.29, 0.27),
            (0.18, 0.13),
            (0.10, -0.04),
        ),
        ((1.30, 0.61), (0.62, 0.29), (0.31, -0.08), (0.14, 0.05)),
        ((1.10, 0.72), (0.48, 0.22), (0.19, -0.06)),
        ((0.93, 0.81), (0.21, -0.08)),
    )
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
    return angular, coordinates, specs, primitives, shells


def _independent_reference(angular, coordinates, shells):
    ir = build_weighted_eri_ir(angular)
    factors = np.array(
        [
            scale
            for _, scale in normalized_cartesian_components(
                angular,
                np.ones(ir.signature.component_shape),
            )
        ]
    )
    with NativeSource([(1, xyz) for xyz in coordinates], basis=shells) as source:
        sizes = tuple(source.shell_sizes)
        offsets = tuple(
            int(value) for value in np.cumsum((0, *sizes[:-1]), dtype=np.int64)
        )
        value = source._read("four_center_eri", offsets, sizes).reshape(-1)
        slices = tuple(
            slice(offset, offset + size)
            for offset, size in zip(offsets, sizes, strict=True)
        )
        derivative = (
            source.integral_derivatives()["eri"][(slice(None), *slices)]
            .reshape(12, -1)
            .T
        )
    normalized = np.column_stack((value, derivative))
    return normalized / factors[:, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--maximum-candidates", type=int, default=16)
    args = parser.parse_args()
    if args.repeats < 5:
        raise ValueError("at least five repeats are required")
    executable = shutil.which("c++")
    if executable is None:
        raise RuntimeError("a C++ compiler is required")
    native_library = os.environ.get("VIBEQC_LIBRARY")
    if not native_library or not Path(native_library).is_file():
        raise RuntimeError("VIBEQC_LIBRARY must name the fresh independent CPU library")
    cache = (args.cache or Path(tempfile.mkdtemp(prefix="vibeqc471-"))).resolve()
    cache.mkdir(parents=True, exist_ok=True)

    angular, coordinates, specs, primitives, shells = _fixture()
    ir = build_weighted_eri_ir(angular)
    reference = _independent_reference(angular, coordinates, shells)
    reference_identity = canonical_hash(
        {
            "schema": "vibeqc.issue471.reference.v1",
            "oracle": "native-dynamic-jet-four-center-values-and-derivatives",
            "native_library_sha256": file_hash(Path(native_library)),
            "angular": angular,
            "coordinates": coordinates.tolist(),
            "primitive_specs": specs,
        }
    )
    limits = CpuTuneLimits(
        maximum_candidates=args.maximum_candidates,
        repeats=args.repeats,
        parallel_tasks=8,
        parallel_workers=(1, 2, 4),
    )
    result = tune_cpu_first_derivative_shell(
        ir,
        CppCompilerAdapter(Path(executable)),
        cache,
        primitives=primitives,
        centers=coordinates,
        independent_reference=reference,
        reference_identity=reference_identity,
        limits=limits,
    )
    result["benchmark"] = {
        "schema": "vibeqc.issue471.benchmark.v1",
        "shell": "psss",
        "primitive_record_count": int(np.prod([len(shell) for shell in primitives])),
        "native_library_sha256": file_hash(Path(native_library)),
        "thread_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_cpu_tuning_manifest(args.output, result)
    print(encoded, end="")


if __name__ == "__main__":
    main()

"""Compilation and bounded execution for CPU lane integral kernels."""

from __future__ import annotations

import ctypes as ct
import math
import os
import tempfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cpu_target import CpuTargetInfo
from vibeqc_compiler.common.cuda_runtime import CudaArtifact
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash, file_hash

from .cpu_lane import (
    cpu_lane_component_identity,
    emit_first_components_cpu_lanes,
)
from .cpu_schedule import CpuScheduleIR
from .first_derivatives_execute import first_derivative_component_tiles
from .first_derivatives_native import validate_first_components
from .ir import IntegralIR


@dataclass(frozen=True)
class CompiledFirstDerivativeCpuLane:
    native: CudaArtifact
    integral: IntegralIR
    component_indices: tuple[int, ...]
    target: CpuTargetInfo
    schedule: CpuScheduleIR
    program_identity: str

    def validate(self) -> None:
        validate_first_components(self.integral, self.component_indices)
        self.schedule.validate_for(self.target)
        identity = self.native.metadata["identity"]
        expected = cpu_lane_component_identity(
            self.integral,
            self.component_indices,
            self.target,
            self.schedule,
        )
        flags = tuple(identity.get("flags", ()))
        if (
            expected != self.program_identity
            or canonical_hash(identity) != self.native.metadata["key"]
            or identity.get("backend") != "cpu"
            or file_hash(self.native.library) != self.native.metadata["binary_sha256"]
            or any(option not in flags for option in self.target.compiler_options)
        ):
            raise ValueError("CPU lane derivative artifact identity mismatch")


def first_derivative_cpu_lane_shell_identity(
    integral,
    target,
    schedule,
    *,
    tile_size=64,
) -> str:
    tiles = first_derivative_component_tiles(integral, tile_size=tile_size)
    return canonical_hash(
        {
            "schema": "vibeqc.first-derivative-shell.cpu-lanes.v1",
            "target": target.to_payload(),
            "schedule": schedule.to_payload(),
            "tile_size": tile_size,
            "tiles": tuple(
                cpu_lane_component_identity(integral, tile, target, schedule)
                for tile in tiles
            ),
        }
    )


@dataclass(frozen=True)
class CompiledFirstDerivativeCpuLaneShell:
    integral: IntegralIR
    tiles: tuple[CompiledFirstDerivativeCpuLane, ...]
    target: CpuTargetInfo
    schedule: CpuScheduleIR
    tile_size: int
    program_identity: str

    def validate(self) -> None:
        self.schedule.validate_for(self.target)
        expected_tiles = first_derivative_component_tiles(
            self.integral, tile_size=self.tile_size
        )
        if (
            tuple(tile.component_indices for tile in self.tiles) != expected_tiles
            or any(tile.integral != self.integral for tile in self.tiles)
            or any(tile.target != self.target for tile in self.tiles)
            or any(tile.schedule != self.schedule for tile in self.tiles)
            or self.program_identity
            != first_derivative_cpu_lane_shell_identity(
                self.integral,
                self.target,
                self.schedule,
                tile_size=self.tile_size,
            )
        ):
            raise ValueError("CPU lane full-shell identity mismatch")
        for tile in self.tiles:
            tile.validate()


def compile_first_derivative_cpu_lane(
    integral,
    compiler,
    cache,
    *,
    component_indices,
    target,
    schedule,
):
    if not isinstance(compiler, CppCompilerAdapter):
        raise TypeError("CPU lane execution requires an explicit C++ compiler")
    if not isinstance(target, CpuTargetInfo) or not isinstance(schedule, CpuScheduleIR):
        raise TypeError("CPU lane compilation requires target and schedule records")
    indices = tuple(component_indices)
    schedule.validate_for(target)
    source = emit_first_components_cpu_lanes(
        integral,
        indices,
        target,
        schedule,
    )
    directory = Path(cache).resolve() / "generated-cpu-lane-sources"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (canonical_hash({"source": source}) + ".cpp")
    if not path.exists():
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as temp:
            temp.write(source)
            name = temp.name
        try:
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    elif path.read_text() != source:
        raise ValueError("CPU lane source cache integrity failure")
    headers = (asset_path("src/integrals/first_component_cpu_lane_runtime.hpp"),)
    root = headers[0].parents[2]
    native = compile_runtime(
        compiler,
        Path(cache) / "native-cpu-lanes",
        path,
        headers=headers,
        options=(
            "-ffp-contract=off",
            *target.compiler_options,
            f"-I{root / 'src'}",
        ),
    )
    artifact = CompiledFirstDerivativeCpuLane(
        native=native,
        integral=integral,
        component_indices=indices,
        target=target,
        schedule=schedule,
        program_identity=cpu_lane_component_identity(
            integral, indices, target, schedule
        ),
    )
    artifact.validate()
    return artifact


def compile_first_derivative_cpu_lane_shell(
    integral,
    compiler,
    cache,
    *,
    target,
    schedule,
    tile_size=64,
):
    schedule.validate_for(target)
    tiles = first_derivative_component_tiles(integral, tile_size=tile_size)
    compiled = tuple(
        compile_first_derivative_cpu_lane(
            integral,
            compiler,
            cache,
            component_indices=tile,
            target=target,
            schedule=schedule,
        )
        for tile in tiles
    )
    artifact = CompiledFirstDerivativeCpuLaneShell(
        integral=integral,
        tiles=compiled,
        target=target,
        schedule=schedule,
        tile_size=tile_size,
        program_identity=first_derivative_cpu_lane_shell_identity(
            integral,
            target,
            schedule,
            tile_size=tile_size,
        ),
    )
    artifact.validate()
    return artifact


class FirstDerivativeCpuLaneEvaluator:
    """Execute one scalar/SIMD component tile with bounded record storage."""

    def __init__(self, artifact, *, record_capacity=128, budget_bytes=1 << 20):
        if type(record_capacity) is not int or record_capacity < 1:
            raise ValueError("record capacity must be a positive integer")
        if type(budget_bytes) is not int or budget_bytes < 1:
            raise ValueError("CPU lane budget must be a positive integer")
        if not isinstance(artifact, CompiledFirstDerivativeCpuLane):
            raise TypeError("expected a compiled CPU lane derivative artifact")
        artifact.validate()
        self.artifact = artifact
        self.nexponent = len(artifact.integral.signature.shells)
        self.ncenter = len(artifact.integral.operator.centers)
        self.stride = self.nexponent + 3 * self.ncenter + 1
        self.shape = (
            len(artifact.component_indices),
            1 + 3 * self.ncenter,
        )
        lanes = artifact.target.vector_lanes
        native_stack = (
            (self.stride - 1) * lanes
            + lanes
            + self.shape[1] * lanes
            + math.prod(self.shape)
        )
        self.numeric_bytes = 8 * (
            record_capacity * self.stride + 4 * math.prod(self.shape) + native_stack
        )
        if self.numeric_bytes > budget_bytes:
            raise ValueError("CPU lane derivative numeric budget exceeded")
        self.library = ct.CDLL(str(artifact.native.library))
        self.library.vibeqc_first_cpu_lane_identity_v1.restype = ct.c_char_p
        self.library.vibeqc_first_cpu_target_v1.restype = ct.c_char_p
        if (
            self.library.vibeqc_first_cpu_lane_identity_v1().decode()
            != artifact.program_identity
            or self.library.vibeqc_first_cpu_target_v1().decode()
            != artifact.target.name
        ):
            raise ValueError("CPU lane compiled program identity mismatch")
        self.run = self.library.vibeqc_first_sum_cpu_lane_v1
        self.run.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_void_p,
            ct.c_size_t,
        ]
        self.run.restype = ct.c_int
        self.record_capacity = record_capacity

    def contract(self, primitives, centers):
        if len(primitives) != self.nexponent or any(not shell for shell in primitives):
            raise ValueError(
                "nonempty primitive lists must match the compiled shell tuple"
            )
        if any(np.iscomplexobj(shell) for shell in primitives):
            raise ValueError("primitive coefficients and exponents must be real")
        centers = np.asarray(centers)
        if (
            centers.shape != (self.ncenter, 3)
            or np.iscomplexobj(centers)
            or not np.isfinite(centers).all()
        ):
            raise ValueError("CPU lane centers must be finite real xyz values")
        records = np.empty((self.record_capacity, self.stride))
        chunk = np.empty(self.shape)
        result = np.zeros(self.shape)
        count = 0

        def flush(size):
            status = self.run(
                records.ctypes.data,
                size,
                self.stride,
                chunk.ctypes.data,
                chunk.size,
            )
            if status:
                exc = ValueError if status == 1 else FloatingPointError
                raise exc(f"generated CPU lane derivative failed with status {status}")
            with np.errstate(over="raise", invalid="raise"):
                np.add(result, chunk, out=result)

        for combination in product(*primitives):
            exponents, coefficients = zip(*combination, strict=True)
            records[count, : self.nexponent] = exponents
            records[count, self.nexponent : -1] = centers.ravel()
            records[count, -1] = math.prod(coefficients)
            count += 1
            if count == self.record_capacity:
                flush(count)
                count = 0
        if count:
            flush(count)
        return result


class FirstDerivativeCpuLaneShellEvaluator:
    """Execute a complete Cartesian shell through scalar or SIMD lane tiles."""

    def __init__(self, artifact, *, record_capacity=128, budget_bytes=4 << 20):
        if not isinstance(artifact, CompiledFirstDerivativeCpuLaneShell):
            raise TypeError("expected a compiled CPU lane shell artifact")
        artifact.validate()
        self.artifact = artifact
        self.shape = (
            artifact.integral.signature.component_count,
            1 + 3 * len(artifact.integral.operator.centers),
        )
        self.evaluators = tuple(
            FirstDerivativeCpuLaneEvaluator(
                tile,
                record_capacity=record_capacity,
                budget_bytes=budget_bytes,
            )
            for tile in artifact.tiles
        )
        self.numeric_bytes = 8 * math.prod(self.shape) + max(
            evaluator.numeric_bytes for evaluator in self.evaluators
        )
        if self.numeric_bytes > budget_bytes:
            raise ValueError("CPU lane full-shell numeric budget exceeded")

    def contract(self, primitives, centers):
        result = np.empty(self.shape)
        for tile, evaluator in zip(
            self.artifact.tiles,
            self.evaluators,
            strict=True,
        ):
            result[list(tile.component_indices)] = evaluator.contract(
                primitives,
                centers,
            )
        return result

"""Explicit CPU compilation and bounded primitive streaming for first derivatives.

No PySCF, public runtime, global integral Jacobian or GPU probing is involved.
Caller-owned primitive lists and fixed atom/density contractions stay outside.
"""

import ctypes as ct
import math
import os
import tempfile
import typing
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cuda_runtime import CudaArtifact
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash, file_hash

from .first_derivatives_native import (
    emit_first_components,
    first_component_identity,
    validate_first_components,
)
from .ir import IntegralIR


@dataclass(frozen=True)
class CompiledFirstDerivative:
    native: CudaArtifact
    integral: IntegralIR
    component_indices: tuple[int, ...]
    program_identity: str

    def validate(self) -> None:
        validate_first_components(self.integral, self.component_indices)
        identity = self.native.metadata["identity"]
        if (
            first_component_identity(self.integral, self.component_indices)
            != self.program_identity
            or canonical_hash(identity) != self.native.metadata["key"]
            or identity.get("backend") != "cpu"
            or file_hash(self.native.library) != self.native.metadata["binary_sha256"]
        ):
            raise ValueError("first derivative artifact identity mismatch")


def first_derivative_component_tiles(
    integral: typing.Any, *, tile_size: typing.Any = 64
) -> typing.Any:
    """Partition one complete Cartesian shell into deterministic bounded tiles."""
    if type(tile_size) is not int or not 1 <= tile_size <= 64:
        raise ValueError("first derivative tile size must be an integer from one to 64")
    validate_first_components(integral, (0,))
    count = integral.signature.component_count
    return tuple(
        tuple(range(start, min(start + tile_size, count)))
        for start in range(0, count, tile_size)
    )


def first_derivative_shell_identity(
    integral: typing.Any, *, tile_size: typing.Any = 64
) -> typing.Any:
    """Hash the complete ordered tile set without creating a second science IR."""
    tiles = first_derivative_component_tiles(integral, tile_size=tile_size)
    return canonical_hash(
        {
            "schema": "vibeqc.first-derivative-shell.cpu.v1",
            "tile_size": tile_size,
            "tiles": tuple(first_component_identity(integral, tile) for tile in tiles),
        }
    )


@dataclass(frozen=True)
class CompiledFirstDerivativeShell:
    """Complete Cartesian shell assembled from independently bounded programs."""

    integral: IntegralIR
    tiles: tuple[CompiledFirstDerivative, ...]
    tile_size: int
    program_identity: str

    def validate(self) -> None:
        expected = first_derivative_component_tiles(
            self.integral, tile_size=self.tile_size
        )
        if (
            tuple(tile.component_indices for tile in self.tiles) != expected
            or any(tile.integral != self.integral for tile in self.tiles)
            or self.program_identity
            != first_derivative_shell_identity(self.integral, tile_size=self.tile_size)
        ):
            raise ValueError("full-shell first derivative tile identity mismatch")
        for tile in self.tiles:
            tile.validate()


def compile_first_derivative(
    integral: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
    *,
    component_indices: typing.Any,
) -> typing.Any:
    if not isinstance(compiler, CppCompilerAdapter):
        raise TypeError("first component execution requires an explicit CPU compiler")
    indices = tuple(component_indices)
    source = emit_first_components(integral, indices)
    directory = Path(cache).resolve() / "generated-sources"
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
        raise ValueError("first derivative source cache integrity failure")
    names = (
        "src/integrals/first_component_runtime.hpp",
        "src/integrals/eri_geometry.hpp",
        "src/integrals/range_moments.hpp",
    )
    headers = tuple(asset_path(name) for name in names)
    root = headers[0].parents[2]
    native = compile_runtime(
        compiler,
        Path(cache) / "native",
        path,
        headers=headers,
        options=("-ffp-contract=off", f"-I{root / 'src'}"),
    )
    return CompiledFirstDerivative(
        native, integral, indices, first_component_identity(integral, indices)
    )


def compile_first_derivative_shell(
    integral: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
    *,
    tile_size: typing.Any = 64,
) -> typing.Any:
    """Compile every Cartesian component without creating an unbounded DAG."""
    tiles = first_derivative_component_tiles(integral, tile_size=tile_size)
    compiled = tuple(
        compile_first_derivative(
            integral, compiler, cache, component_indices=component_indices
        )
        for component_indices in tiles
    )
    artifact = CompiledFirstDerivativeShell(
        integral,
        compiled,
        tile_size,
        first_derivative_shell_identity(integral, tile_size=tile_size),
    )
    artifact.validate()
    return artifact


class FirstDerivativeEvaluator:
    """One compiled raw tile with a fixed-size primitive record buffer.

    At most record_capacity records and three result tiles coexist. Native
    publication storage holds one result tile plus 13 scalar outputs. Caller
    metadata, compiler/cache, shared-library mappings, generated-kernel call
    stacks/scalar temporaries and Python object overhead are explicit exclusions
    from numeric_bytes. Calls do not retain any molecular integral data.
    """

    def __init__(
        self,
        artifact: typing.Any,
        *,
        record_capacity: typing.Any = 128,
        budget_bytes: typing.Any = 1 << 20,
    ) -> None:
        if type(record_capacity) is not int or record_capacity < 1:
            raise ValueError("record capacity must be a positive integer")
        if type(budget_bytes) is not int or budget_bytes < 1:
            raise ValueError("first component budget must be a positive integer")
        if not isinstance(artifact, CompiledFirstDerivative):
            raise TypeError("expected a compiled first-derivative artifact")
        artifact.validate()
        self.artifact = artifact
        self.nexponent = len(artifact.integral.signature.shells)
        self.ncenter = len(artifact.integral.operator.centers)
        self.stride = self.nexponent + 3 * self.ncenter + 1
        self.shape = (len(artifact.component_indices), 1 + 3 * self.ncenter)
        self.numeric_bytes = 8 * (
            record_capacity * self.stride + 4 * math.prod(self.shape) + 13
        )
        if self.numeric_bytes > budget_bytes:
            raise ValueError("first derivative numeric budget exceeded")
        self.library = ct.CDLL(str(artifact.native.library))
        self.library.vibeqc_first_identity_v1.restype = ct.c_char_p
        if (
            self.library.vibeqc_first_identity_v1().decode()
            != artifact.program_identity
        ):
            raise ValueError("first derivative compiled program identity mismatch")
        self.run = self.library.vibeqc_first_sum_v1
        self.run.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_void_p,
            ct.c_size_t,
        ]
        self.run.restype = ct.c_int
        self.record_capacity = record_capacity

    def contract(self, primitives: typing.Any, centers: typing.Any) -> typing.Any:
        """Sum radially normalized primitives; return raw Cartesian components."""
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
            raise ValueError("first derivative centers must be finite real xyz values")
        records = np.empty((self.record_capacity, self.stride))
        chunk = np.empty(self.shape)
        result = np.zeros(self.shape)
        count = 0

        def flush(count: typing.Any) -> None:
            status = self.run(
                records.ctypes.data, count, self.stride, chunk.ctypes.data, chunk.size
            )
            if status:
                exc = ValueError if status == 1 else FloatingPointError
                raise exc(
                    f"generated first derivative failed with native status {status}"
                )
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


class FirstDerivativeShellEvaluator:
    """Execute every Cartesian component with bounded peak generated work."""

    def __init__(
        self,
        artifact: typing.Any,
        *,
        record_capacity: typing.Any = 128,
        budget_bytes: typing.Any = 4 << 20,
    ) -> None:
        if not isinstance(artifact, CompiledFirstDerivativeShell):
            raise TypeError("expected a compiled full-shell first-derivative artifact")
        artifact.validate()
        self.artifact = artifact
        self.shape = (
            artifact.integral.signature.component_count,
            1 + 3 * len(artifact.integral.operator.centers),
        )
        self.evaluators = tuple(
            FirstDerivativeEvaluator(
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
            raise ValueError("full-shell first derivative numeric budget exceeded")

    def contract(self, primitives: typing.Any, centers: typing.Any) -> typing.Any:
        """Return the complete shell in canonical Cartesian component order."""
        result = np.empty(self.shape)
        for tile, evaluator in zip(self.artifact.tiles, self.evaluators, strict=True):
            result[list(tile.component_indices)] = evaluator.contract(
                primitives, centers
            )
        return result

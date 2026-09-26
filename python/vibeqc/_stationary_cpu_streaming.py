"""Bounded native enumeration over compiler-owned component derivative dispatch."""

import ctypes as ct
import typing
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.first_derivative_schedule import (
    COMPONENT_LABELS,
    COMPONENT_OPERATORS,
    derivative_dispatch_table,
)

from ._stationary_cpu_components import ComponentPrimitiveExecutor

DoublePointer = ct.POINTER(ct.c_double)
IndexPointer = ct.POINTER(ct.c_int64)
Dispatch = ct.CFUNCTYPE(ct.c_int, ct.c_uint, DoublePointer, ct.c_size_t, DoublePointer)


class CompiledComponentExecutor(ComponentPrimitiveExecutor):
    """Retain generated library lifetimes and reuse one fixed primitive buffer."""

    def __init__(
        self,
        basis: typing.Any,
        cache: str | Path,
        primitive_tile: int,
        compiler: CppCompilerAdapter,
    ) -> None:
        from ._stationary_cpu import _publish_source

        if type(primitive_tile) is not int or not 1 <= primitive_tile <= 4096:
            raise ValueError("primitive tile must be an integer in [1,4096]")
        super().__init__(basis, cache, primitive_tile, compiler)
        labels = np.full((len(self.aos), 3), -1, dtype=np.int64)
        for i, expansion in enumerate(self.expansions):
            for j, (label, _) in enumerate(expansion):
                labels[i, j] = COMPONENT_LABELS.index(label)
        self.labels = np.frombuffer(labels.tobytes(), dtype=np.int64).reshape(
            labels.shape
        )
        domain = tuple(sorted({c for terms in self.expansions for c, _ in terms}))
        bindings = np.asarray(derivative_dispatch_table(domain), dtype=np.int64)
        self.bindings = np.frombuffer(bindings.tobytes(), dtype=np.int64).reshape(
            bindings.shape
        )
        self.dispatch = (Dispatch * len(self.libraries))(
            *(
                ct.cast(lib.vibeqc_first_derivative_cpu, Dispatch)
                for lib in self.libraries
            )
        )
        header = asset_path("src/integrals/first_derivative_component_runtime.hpp")
        source = '#include "integrals/first_derivative_component_runtime.hpp"\n'
        path = Path(cache) / (canonical_hash(source) + ".cpp")
        _publish_source(path, source)
        artifact = compile_runtime(
            compiler,
            cache,
            path,
            headers=(header,),
            options=("-ffp-contract=off", f"-I{header.parents[1]}"),
        )
        self.runtime = ct.CDLL(str(artifact.library))
        self.contract = self.runtime.vibeqc_component_contract_cpu
        self.contract.argtypes = [
            DoublePointer,
            ct.c_size_t,
            DoublePointer,
            ct.c_size_t,
            DoublePointer,
            ct.c_size_t,
            IndexPointer,
            IndexPointer,
            ct.c_size_t,
            ct.c_size_t,
            ct.POINTER(Dispatch),
            ct.c_size_t,
            ct.c_uint,
            IndexPointer,
            ct.c_double,
            ct.c_int64,
            DoublePointer,
            ct.c_size_t,
            DoublePointer,
            ct.POINTER(ct.c_uint64),
        ]
        self.contract.restype = ct.c_int
        self.arguments = (
            self.centers.ctypes.data_as(DoublePointer),
            len(self.centers),
            self.primitives.ctypes.data_as(DoublePointer),
            len(self.primitives),
            self.aos.ctypes.data_as(DoublePointer),
            len(self.aos),
            self.labels.ctypes.data_as(IndexPointer),
            self.bindings.ctypes.data_as(IndexPointer),
            len(self.bindings),
            len(COMPONENT_LABELS),
            self.dispatch,
            len(self.libraries),
        )
        self.compilation_work["component_dispatch_bytes"] = (
            self.labels.nbytes + self.bindings.nbytes + ct.sizeof(self.dispatch)
        )

    def integral(
        self,
        operator: str,
        indices: tuple[int, ...],
        weight: float,
        nucleus: int | None = None,
    ) -> tuple[list[int], np.ndarray]:
        if operator not in COMPONENT_OPERATORS:
            raise ValueError("unsupported component derivative operator")
        op = COMPONENT_OPERATORS.index(operator)
        rank = 4 if op == 3 else 2
        if len(indices) != rank or any(
            not isinstance(i, (int, np.integer)) or not 0 <= i < len(self.aos)
            for i in indices
        ):
            raise ValueError("invalid component AO indices")
        if (
            op == 2
            and (
                not isinstance(nucleus, (int, np.integer))
                or not 0 <= nucleus < len(self.centers)
            )
        ) or (op != 2 and nucleus is not None):
            raise ValueError("invalid component nucleus")
        ids = np.asarray(indices, dtype=np.int64)
        result, work = np.empty((4, 3)), ct.c_uint64()
        if self.contract(
            *self.arguments,
            op,
            ids.ctypes.data_as(IndexPointer),
            weight,
            -1 if nucleus is None else nucleus,
            self.buffer.ctypes.data_as(DoublePointer),
            len(self.buffer),
            result.ctypes.data_as(DoublePointer),
            ct.byref(work),
        ):
            raise ArithmeticError("generated CPU component derivative failed")
        self.records += work.value
        owners = [int(self.aos[i, 0]) for i in indices]
        if nucleus is not None:
            owners.append(nucleus)
        return owners, result[: len(owners)]

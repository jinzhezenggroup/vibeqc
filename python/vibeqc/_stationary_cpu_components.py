"""Bounded multicomponent public-AO consumer of generated primitive gradients."""

import ctypes as ct
import typing
from itertools import product
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.integral.first_derivative_schedule import (
    derivative_binding,
    derivative_sources,
)


class ComponentPrimitiveExecutor:
    """Stream every public component weight; share only compiled scalar kernels."""

    def __init__(
        self,
        basis: typing.Any,
        cache: str | Path,
        primitive_tile: int,
        compiler: CppCompilerAdapter,
    ) -> None:
        from ._stationary_cpu import _compile_primitive_library

        if primitive_tile < 1:
            raise ValueError("primitive tile must be positive")
        if any(shell.angular_momentum > 2 for shell in basis.shells):
            raise NotImplementedError("component executor supports s/p/d bases only")
        start = 3 * basis.natom
        self.centers = basis.packed[:start].reshape(-1, 3)
        self.primitives = basis.packed[start : start + 2 * basis.nprimitive].reshape(
            -1, 2
        )
        self.aos = basis.packed[start + 2 * basis.nprimitive :].reshape(-1, 16)
        if any(row[3] not in (1, 2, 3) for row in self.aos):
            raise ValueError("public AOs require one to three Cartesian components")
        self.expansions = tuple(
            tuple(
                (
                    "".join(
                        axis * int(power)
                        for axis, power in zip("xyz", row[4 + 4 * t : 7 + 4 * t])
                    ),
                    row[7 + 4 * t],
                )
                for t in range(int(row[3]))
            )
            for row in self.aos
        )
        domain = tuple(
            sorted({c for expansion in self.expansions for c, _ in expansion})
        )
        self.libraries, self.calls = [], {}
        sources = derivative_sources(domain)
        for requests, source in sources:
            library, call = _compile_primitive_library(source, cache, compiler)
            self.libraries.append(library)
            for kind, request in enumerate(requests):
                self.calls[request] = (call, kind)
        self.compilation_work = {
            "primitive_compiled_kernels": len(self.calls),
            "primitive_translation_units": len(sources),
            "primitive_generated_source_bytes": sum(
                len(s.encode("utf-8")) for _, s in sources
            ),
            "primitive_largest_source_bytes": max(
                len(s.encode("utf-8")) for _, s in sources
            ),
        }
        self.buffer = np.zeros((primitive_tile, 17))
        self.records = 0

    def _run(self, request: tuple[str, tuple[str, ...]], count: int) -> np.ndarray:
        call, kind = self.calls[request]
        out = np.empty((4, 3))
        if call(
            kind,
            self.buffer.ctypes.data_as(ct.POINTER(ct.c_double)),
            count,
            out.ctypes.data_as(ct.POINTER(ct.c_double)),
        ):
            raise ArithmeticError("generated CPU component derivative failed")
        self.records += count
        return out

    def integral(
        self,
        operator: str,
        indices: tuple[int, ...],
        weight: float,
        nucleus: int | None = None,
    ) -> tuple[list[int], np.ndarray]:
        rows = self.aos[list(indices)]
        rank = len(indices)
        owners = [int(row[0]) for row in rows]
        if nucleus is not None:
            owners.append(nucleus)
        ranges = [range(int(row[1]), int(row[1] + row[2])) for row in rows]
        result = np.zeros((len(owners), 3))
        for terms in product(*(self.expansions[i] for i in indices)):
            binding = derivative_binding(operator, tuple(c for c, _ in terms))
            self.buffer.fill(0)
            self.buffer[:, 4 : 4 + 3 * len(owners)] = self.centers[owners][
                np.ix_(binding.centers, binding.axes)
            ].reshape(-1)
            norm = weight * np.prod([coefficient for _, coefficient in terms])
            count = 0
            # Do not fold ordered weights or assume a density symmetry. Only
            # exponent/coordinate slots change to reuse a compiled primitive.
            for ids in product(*ranges):
                primitives = self.primitives[list(ids)]
                self.buffer[count, :rank] = primitives[list(binding.centers[:rank]), 0]
                self.buffer[count, 16] = norm * np.prod(primitives[:, 1])
                count += 1
                if count == len(self.buffer):
                    result[np.ix_(binding.centers, binding.axes)] += self._run(
                        binding.request, count
                    )[: len(owners)]
                    count = 0
            if count:
                result[np.ix_(binding.centers, binding.axes)] += self._run(
                    binding.request, count
                )[: len(owners)]
        return owners, result

    def nuclear(self, a: int, b: int, charges: np.ndarray) -> np.ndarray:
        self.buffer.fill(0)
        self.buffer[0, :2] = charges[[a, b]]
        self.buffer[0, 4:10] = self.centers[[a, b]].reshape(-1)
        self.buffer[0, 16] = 1
        return self._run(("nuclear", ()), 1)[:2]

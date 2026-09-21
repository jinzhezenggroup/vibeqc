"""Bounded CPU bridge from stationary RSH exchange weights to the #249 provider."""

from __future__ import annotations

import math
import typing
from itertools import product
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.integral.ir import four_center_eri_operator
from vibeqc_compiler.integral.range_separation import CoulombKernel
from vibeqc_compiler.integral.shell_spec import ShellClassSpec
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir
from vibeqc_compiler.integral.weighted_eri_execute import (
    PreparedWeightedEri,
    compile_weighted_eri,
)
from vibeqc_compiler.method.spec import RangeSeparatedExchangePrimitive


class RangeExchangePrimitiveExecutor:
    """Bind stationary RSH weights to generated SR/LR ERI derivatives."""

    def __init__(
        self,
        basis: typing.Any,
        cache: str | Path,
        primitive_tile: int,
        compiler: CppCompilerAdapter,
    ) -> None:
        if not isinstance(compiler, CppCompilerAdapter):
            raise TypeError("range exchange requires an explicit CPU C++ compiler adapter")
        if type(primitive_tile) is not int or primitive_tile < 1:
            raise ValueError("range exchange primitive tile must be positive")
        if any(shell.angular_momentum > 2 for shell in basis.shells):
            raise NotImplementedError("range exchange CPU bridge supports s/p/d bases only")

        start = 3 * basis.natom
        self.centers = basis.packed[:start].reshape(-1, 3)
        self.primitives = basis.packed[
            start : start + 2 * basis.nprimitive
        ].reshape(-1, 2)
        self.aos = basis.packed[start + 2 * basis.nprimitive :].reshape(-1, 16)
        if any(int(row[3]) not in (1, 2, 3) for row in self.aos):
            raise NotImplementedError(
                "range exchange public AOs require one to three Cartesian components"
            )
        self.expansions = tuple(
            tuple(
                (
                    "".join(
                        axis * int(power)
                        for axis, power in zip("xyz", row[4 + 4 * t : 7 + 4 * t])
                    ),
                    float(row[7 + 4 * t]),
                )
                for t in range(int(row[3]))
            )
            for row in self.aos
        )
        self.cache = Path(cache)
        self.primitive_tile = primitive_tile
        self.compiler = compiler
        self._plans: dict[tuple[typing.Any, ...], PreparedWeightedEri] = {}
        self.records = 0

    @property
    def compilation_work(self) -> dict[str, int]:
        return {
            "range_exchange_compiled_plans": len(self._plans),
            "range_exchange_primitive_records": self.records,
        }

    @staticmethod
    def _family(primitive: RangeSeparatedExchangePrimitive) -> str:
        return {
            "short-range": "short_range",
            "long-range": "long_range",
        }[primitive.operator]

    def _prepared(
        self,
        primitive: RangeSeparatedExchangePrimitive,
        angular: tuple[int, int, int, int],
        component_index: int,
    ) -> PreparedWeightedEri:
        spec = ShellClassSpec("".join("spdf"[value] for value in angular), angular)
        begin = 64 * (component_index // 64)
        subset = tuple(range(begin, min(begin + 64, spec.component_count)))
        key = (primitive.operator, primitive.omega, angular, begin)
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        operator = four_center_eri_operator(
            CoulombKernel(self._family(primitive), float(primitive.omega))
        )
        integral = build_weighted_eri_ir(angular, operator=operator)
        artifact = compile_weighted_eri(
            integral,
            self.compiler,
            self.cache,
            component_indices=subset,
        )
        plan = PreparedWeightedEri(
            artifact,
            record_capacity=self.primitive_tile,
            tile_capacity=1,
        )
        self._plans[key] = plan
        return plan

    def integral(
        self,
        primitive: RangeSeparatedExchangePrimitive,
        indices: tuple[int, int, int, int],
        weight: float,
    ) -> tuple[list[int], np.ndarray]:
        if type(primitive) is not RangeSeparatedExchangePrimitive:
            raise TypeError("range exchange executor requires a MethodIR range primitive")
        if len(indices) != 4:
            raise ValueError("range exchange requires one ordered AO quartet")
        rows = self.aos[list(indices)]
        owners = [int(row[0]) for row in rows]
        angular = tuple(
            sum(self.expansions[index][0][0].count(axis) for axis in "xyz")
            for index in indices
        )
        spec = ShellClassSpec("".join("spdf"[value] for value in angular), angular)
        ranges = [
            range(int(row[1]), int(row[1] + row[2]))
            for row in rows
        ]
        shell_primitives = tuple(
            tuple(tuple(map(float, self.primitives[p])) for p in primitive_range)
            for primitive_range in ranges
        )
        result = np.zeros((4, 3))
        for terms in product(*(self.expansions[index] for index in indices)):
            components = tuple(component for component, _ in terms)
            try:
                component_index = spec.components.index(components)
            except ValueError as error:
                raise ValueError(
                    "public AO component is outside the generated shell class"
                ) from error
            plan = self._prepared(primitive, angular, component_index)
            execution = plan.raw(
                shell_primitives,
                self.centers[owners],
                (component_index,),
            )
            self.records += int(execution.diagnostics["records"])
            scale = float(weight) * math.prod(
                coefficient for _, coefficient in terms
            )
            with np.errstate(over="raise", invalid="raise"):
                result += scale * execution.values[0, 1:].reshape(4, 3)
        return owners, result

    def close(self) -> None:
        for plan in self._plans.values():
            plan.close()
        self._plans.clear()
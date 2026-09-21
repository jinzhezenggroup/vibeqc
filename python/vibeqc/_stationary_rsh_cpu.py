"""CPU binding for generated range-separated exchange first derivatives."""

import typing
from pathlib import Path

import numpy as np
from vibeqc_compiler.integral.ir import four_center_eri_operator
from vibeqc_compiler.integral.range_separation import CoulombKernel
from vibeqc_compiler.integral.shell_spec import cartesian_components
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir
from vibeqc_compiler.integral.weighted_eri_execute import (
    PreparedWeightedEri,
    compile_weighted_eri,
)
from vibeqc_compiler.method.spec import RangeSeparatedExchangePrimitive


class RangeExchangeExecutor:
    """Bind generated SR/LR ERI derivatives to native AO tuples.

    This stationary-consumer binding admits only s/p public AOs. For l < 2
    Cartesian and real-spherical public functions are identical, so each packed
    NativeAO record maps one-to-one to a normalized Cartesian component.
    """

    def __init__(
        self,
        basis: typing.Any,
        cache: typing.Any,
        primitive_tile: typing.Any,
        compiler: typing.Any,
    ) -> None:
        if any(shell.angular_momentum > 1 for shell in basis.shells):
            raise NotImplementedError(
                "CPU RSH stationary gradients currently support s/p bases only"
            )
        start = 3 * basis.natom
        self.primitives = basis.packed[start : start + 2 * basis.nprimitive].reshape(
            -1, 2
        )
        self.aos = basis.packed[start + 2 * basis.nprimitive :].reshape(-1, 16)
        self.centers = basis.packed[:start].reshape(-1, 3)
        if any(int(row[3]) != 1 for row in self.aos):
            raise NotImplementedError(
                "CPU RSH stationary gradients require one Cartesian component "
                "per public AO"
            )
        self.cache = Path(cache)
        self.primitive_tile = primitive_tile
        self.compiler = compiler
        self._plans: dict[tuple[typing.Any, ...], PreparedWeightedEri] = {}
        self.records = 0

    @staticmethod
    def _kernel(primitive: typing.Any) -> CoulombKernel:
        if not isinstance(primitive, RangeSeparatedExchangePrimitive):
            raise TypeError(
                "range exchange execution requires a MethodIR range primitive"
            )
        family = {
            "short-range": "short_range",
            "long-range": "long_range",
        }.get(primitive.operator)
        if family is None:
            raise ValueError("unsupported range-exchange operator")
        return CoulombKernel(family, float(primitive.omega))

    @staticmethod
    def _component(row: typing.Any) -> tuple[int, int]:
        powers = tuple(int(value) for value in row[4:7])
        angular = sum(powers)
        label = "".join(axis * power for axis, power in zip("xyz", powers, strict=True))
        return angular, cartesian_components(angular).index(label)

    def _primitive_shell(self, row: typing.Any) -> tuple[tuple[float, float], ...]:
        begin, count = int(row[1]), int(row[2])
        return tuple(
            (float(exponent), float(coefficient))
            for exponent, coefficient in self.primitives[begin : begin + count]
        )

    def integral(
        self,
        primitive: typing.Any,
        indices: typing.Any,
        weight: typing.Any,
    ) -> typing.Any:
        rows = self.aos[list(indices)]
        angular_and_component = tuple(self._component(row) for row in rows)
        angular = tuple(item[0] for item in angular_and_component)
        shape = tuple(len(cartesian_components(value)) for value in angular)
        component = int(
            np.ravel_multi_index(
                tuple(item[1] for item in angular_and_component), shape
            )
        )
        radial = self._kernel(primitive)
        key = (radial.family.value, radial.omega, angular, component)
        plan = self._plans.get(key)
        if plan is None:
            integral = build_weighted_eri_ir(
                angular, operator=four_center_eri_operator(radial)
            )
            artifact = compile_weighted_eri(
                integral,
                self.compiler,
                self.cache,
                component_indices=(component,),
            )
            plan = PreparedWeightedEri(
                artifact,
                record_capacity=self.primitive_tile,
                tile_capacity=1,
            )
            self._plans[key] = plan
        owners = [int(row[0]) for row in rows]
        primitives = tuple(self._primitive_shell(row) for row in rows)
        centers = tuple(
            tuple(float(value) for value in self.centers[owner]) for owner in owners
        )
        result = plan.raw(primitives, centers, (component,))
        self.records += int(result.diagnostics["records"])
        return owners, float(weight) * result.values[0, 1:].reshape(4, 3)

    def close(self) -> None:
        for plan in self._plans.values():
            plan.close()
        self._plans.clear()

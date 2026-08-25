"""Declarative shell-class metadata for static integral code generation.

The specification deliberately contains no runtime policy.  It describes the
Cartesian component space and compile-time recurrence bounds that emitters use
to produce a fully specialized kernel.  Keeping this information in Python
lets new shell classes reuse the symbolic compiler without adding component
tables or index arithmetic by hand to CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, reduce
from itertools import product
from operator import mul
from typing import Sequence


AXES = ("x", "y", "z")
_AXIS_INDEX = {axis: index for index, axis in enumerate(AXES)}


def cartesian_components(angular_momentum: int) -> tuple[str, ...]:
    """Return Cartesian components in QCE's CCA ordering.

    The ordering is descending in the x exponent and then descending in the y
    exponent.  This produces ``x, y, z`` for p shells and
    ``xx, xy, xz, yy, yz, zz`` for d shells, matching the AO layout consumed
    by the production CUDA kernels.
    """

    if isinstance(angular_momentum, bool) or not isinstance(
        angular_momentum, int
    ):
        raise TypeError("angular momentum must be an integer")
    if angular_momentum < 0:
        raise ValueError("angular momentum must be a non-negative integer")
    if angular_momentum == 0:
        return ("",)
    return tuple(
        "x" * x_order + "y" * y_order + "z" * z_order
        for x_order in range(angular_momentum, -1, -1)
        for y_order in range(angular_momentum - x_order, -1, -1)
        for z_order in (angular_momentum - x_order - y_order,)
    )


@dataclass(frozen=True)
class ShellClassSpec:
    """Compile-time description of one canonical four-center shell class."""

    name: str
    angular: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not self.name or not self.name.isalpha() or not self.name.islower():
            raise ValueError("shell-class name must contain lowercase letters")
        if len(self.angular) != 4:
            raise ValueError("a shell class must contain exactly four centers")
        for value in self.angular:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "shell angular momenta must be non-negative integers"
                )

    @cached_property
    def center_components(self) -> tuple[tuple[str, ...], ...]:
        """Return the CCA-ordered Cartesian labels for every center."""

        return tuple(cartesian_components(order) for order in self.angular)

    @cached_property
    def components(self) -> tuple[tuple[str, str, str, str], ...]:
        """Return the complete shell-class Cartesian product schedule."""

        return tuple(product(*self.center_components))

    @property
    def component_count(self) -> int:
        """Return the number of Cartesian AO quartets in this shell class."""

        return reduce(mul, map(len, self.center_components), 1)

    @cached_property
    def component_strides(self) -> tuple[int, int, int, int]:
        """Return row-major strides used to decode one component lane."""

        counts = tuple(map(len, self.center_components))
        return tuple(reduce(mul, counts[index + 1 :], 1) for index in range(4))

    @property
    def pair_orders(self) -> tuple[int, int]:
        """Return angular orders of the bra and ket Gaussian-product pairs."""

        return (sum(self.angular[:2]), sum(self.angular[2:]))

    @property
    def maximum_force_coulomb_order(self) -> int:
        """Return the largest Coulomb derivative needed by first forces."""

        return sum(self.angular) + 1

    def validate_component(
        self, component: Sequence[str]
    ) -> tuple[str, str, str, str]:
        """Validate and normalize one Cartesian component tuple."""

        normalized = tuple(component)
        if len(normalized) != 4:
            raise ValueError(f"{self.name} requires exactly four components")
        for center, (label, allowed) in enumerate(
            zip(normalized, self.center_components, strict=True)
        ):
            if label not in allowed:
                raise ValueError(
                    f"unsupported center-{center} component {label!r} "
                    f"for {self.name}"
                )
        return normalized

    def component_quantums(
        self, component: Sequence[str]
    ) -> tuple[tuple[int, int], ...]:
        """Return ``(center, axis)`` for every angular quantum in a component."""

        normalized = self.validate_component(component)
        return tuple(
            (center, _AXIS_INDEX[axis])
            for center, label in enumerate(normalized)
            for axis in label
        )

    def component_index(self, component: Sequence[str]) -> int:
        """Encode one component tuple into its cooperative lane index."""

        normalized = self.validate_component(component)
        return sum(
            allowed.index(label) * stride
            for label, allowed, stride in zip(
                normalized,
                self.center_components,
                self.component_strides,
                strict=True,
            )
        )

    def component_from_index(self, index: int) -> tuple[str, str, str, str]:
        """Decode a cooperative lane index into its Cartesian component."""

        if not 0 <= index < self.component_count:
            raise IndexError(
                f"component index {index} is outside {self.name} schedule"
            )
        return tuple(
            allowed[(index // stride) % len(allowed)]
            for allowed, stride in zip(
                self.center_components, self.component_strides, strict=True
            )
        )


DPPP_SPEC = ShellClassSpec("dppp", (2, 1, 1, 1))
DPDS_SPEC = ShellClassSpec("dpds", (2, 1, 2, 0))
DDPS_SPEC = ShellClassSpec("ddps", (2, 2, 1, 0))

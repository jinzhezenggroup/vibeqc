"""Backend-independent mathematical integral representation.

``IntegralIR`` contains only scientific intent.  Accelerator execution
geometry is added later by a backend lowering module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .shell_spec import ShellClassSpec


class KernelConsumer(str, Enum):
    """Observable contracted by a generated shell kernel."""

    FOCK = "fock"
    FORCE = "force"


@dataclass(frozen=True, slots=True)
class IntegralIR:
    """Mathematical definition shared by Fock and force consumers.

    The fourth force center is intentionally absent from
    ``independent_force_centers``: its derivative is reconstructed from exact
    translation invariance.  This is both a scientific invariant and an
    important register-pressure reduction in the generated kernels.
    """

    spec: ShellClassSpec
    consumers: frozenset[KernelConsumer]
    independent_force_centers: tuple[int, ...] = (0, 1, 2)
    recurrence: str = "subset_wick"

    def __post_init__(self) -> None:
        if not self.consumers:
            raise ValueError("an integral IR requires at least one consumer")
        if not self.consumers <= frozenset(KernelConsumer):
            raise ValueError("integral IR contains an unsupported consumer")
        if self.recurrence not in ("subset_wick", "rys3"):
            raise ValueError(f"unsupported integral recurrence {self.recurrence!r}")
        if self.recurrence == "rys3" and (
            self.spec.name != "ppps"
            or self.consumers != frozenset((KernelConsumer.FORCE,))
        ):
            raise ValueError(
                "the direct three-root recurrence currently supports only "
                "force-only ppps kernels"
            )
        if (
            KernelConsumer.FORCE in self.consumers
            and self.independent_force_centers != (0, 1, 2)
        ):
            raise ValueError(
                "first-force lowering requires centers 0/1/2 and "
                "translation recovery for center 3"
            )

    @property
    def value_coulomb_order(self) -> int:
        """Largest Cartesian Coulomb derivative needed for ERI values."""

        return sum(self.spec.angular)

    @property
    def maximum_coulomb_order(self) -> int:
        """Largest derivative required by every requested consumer."""

        force_increment = int(KernelConsumer.FORCE in self.consumers)
        return self.value_coulomb_order + force_increment


def build_integral_ir(
    spec: ShellClassSpec,
    consumers: tuple[KernelConsumer | str, ...] = (KernelConsumer.FORCE,),
    *,
    recurrence: str = "subset_wick",
) -> IntegralIR:
    """Normalize requested observables into one mathematical integral IR."""

    return IntegralIR(
        spec=spec,
        consumers=frozenset(KernelConsumer(item) for item in consumers),
        recurrence=recurrence,
    )

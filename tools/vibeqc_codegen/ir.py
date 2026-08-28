"""Backend-independent mathematical integral representation.

``IntegralIR`` contains only scientific intent.  Accelerator execution
geometry is added later by a backend lowering module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .shell_spec import ShellClassSpec

_COOPERATIVE_RYS3_SHELLS = frozenset(("dpps", "dsps", "pppp"))


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
        if self.recurrence not in ("subset_wick", "rys3", "rys4"):
            raise ValueError(f"unsupported integral recurrence {self.recurrence!r}")
        if self.recurrence == "rys3":
            ppps_force_only = (
                self.spec.name == "ppps"
                and self.consumers == frozenset((KernelConsumer.FORCE,))
            )
            cooperative_force = (
                self.spec.name in _COOPERATIVE_RYS3_SHELLS
                and KernelConsumer.FORCE in self.consumers
            )
            if not ppps_force_only and not cooperative_force:
                raise ValueError(
                    "the direct three-root recurrence requires force-only "
                    "ppps or a supported cooperative force shell class"
                )
        if self.recurrence == "rys4" and (
            self.spec.name != "dppp"
            or KernelConsumer.FORCE not in self.consumers
        ):
            raise ValueError(
                "the direct four-root recurrence requires a dppp force "
                "consumer; its Fock consumer retains the value recurrence"
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

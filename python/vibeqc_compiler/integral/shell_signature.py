"""Operator-independent basis slots and their eventual physical atom bindings.

Position variables belong to mathematical centers, not atoms. In particular,
two shells on the same atom still have distinct centers until the chain rule
has accumulated their derivatives. Auxiliary shells retain their basis role.
"""

from dataclasses import dataclass
from enum import Enum
from math import prod

from .shell_spec import ShellClassSpec

MAX_INDEX = (1 << 63) - 1


def checked_index(value: int, name: str, *, minimum: int = 0) -> int:
    """Validate sizes against a signed 64-bit host/backend descriptor boundary."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be {'positive' if minimum else 'non-negative'}")
    if value > MAX_INDEX:
        raise OverflowError(f"{name} overflow: exceeds signed 64-bit indexing")
    return value


class ShellRole(str, Enum):
    """Basis spaces are distinct even when their angular momenta coincide."""

    ORBITAL = "orbital"
    AUXILIARY = "auxiliary"


class BasisConvention(str, Enum):
    """AO component conventions; neither changes nuclear xyz ordering."""

    CARTESIAN = "cartesian"
    REAL_SPHERICAL = "real_spherical"


@dataclass(frozen=True, slots=True)
class BasisShell:
    """One ordered basis slot with an independent position parameter."""

    slot: int
    center: int
    angular: int
    role: ShellRole | str = ShellRole.ORBITAL
    convention: BasisConvention | str = BasisConvention.CARTESIAN

    def __post_init__(self) -> None:
        checked_index(self.slot, "basis slot")
        checked_index(self.center, "basis center")
        checked_index(self.angular, "angular momentum")
        object.__setattr__(self, "role", ShellRole(self.role))
        object.__setattr__(self, "convention", BasisConvention(self.convention))
        checked_index(self.component_count, "shell component count", minimum=1)

    @property
    def component_count(self) -> int:
        """Count AOs without constructing a potentially huge component table."""
        if self.convention == BasisConvention.REAL_SPHERICAL:
            return 2 * self.angular + 1
        return (self.angular + 1) * (self.angular + 2) // 2


@dataclass(frozen=True, slots=True)
class CenterBinding:
    """Map a position parameter to an atom, or leave it for task-time binding.

    ``None`` is used by the legacy class adapter: its atom indices come from
    ``task.atom[center]``, and must never be inferred from the quartet slot.
    Bounded host requests must resolve every such binding before execution.
    """

    center: int
    atom_index: int | None = None

    def __post_init__(self) -> None:
        checked_index(self.center, "center")
        if self.atom_index is not None:
            checked_index(self.atom_index, "atom index")


@dataclass(frozen=True, slots=True)
class ShellSignature:
    """Two, three, or four basis shells, independent of operator and consumer.

    Bindings also include any external operator centers. The enclosing IR
    validates the complete inventory, since a shell signature alone cannot
    know whether an attraction nucleus is present.
    """

    shells: tuple[BasisShell, ...]
    center_bindings: tuple[CenterBinding, ...]
    legacy_class: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shells", tuple(self.shells))
        object.__setattr__(self, "center_bindings", tuple(self.center_bindings))
        if len(self.shells) not in (2, 3, 4):
            raise ValueError(
                "a shell signature requires two, three, or four basis shells"
            )
        if tuple(s.slot for s in self.shells) != tuple(range(len(self.shells))):
            raise ValueError("basis slots must be unique and ordered from zero")
        centers = tuple(s.center for s in self.shells)
        if len(set(centers)) != len(centers):
            raise ValueError("basis shells require independent mathematical centers")
        bindings = tuple(b.center for b in self.center_bindings)
        if len(set(bindings)) != len(bindings) or not set(centers) <= set(bindings):
            raise ValueError(
                "center bindings must be unique and include every basis center"
            )
        if len({s.convention for s in self.shells}) != 1:
            raise ValueError("mixed spherical/Cartesian conventions are not supported")
        checked_index(self.component_count, "signature component count", minimum=1)
        if self.legacy_class is not None:
            self.to_shell_class()

    @property
    def angular(self) -> tuple[int, ...]:
        """Return the angular order in tensor-slot order."""
        return tuple(s.angular for s in self.shells)

    @property
    def component_shape(self) -> tuple[int, ...]:
        """Return full shell-local AO extents, never global molecule extents."""
        return tuple(s.component_count for s in self.shells)

    @property
    def component_count(self) -> int:
        """Return the shell-block element count without allocating the block."""
        return prod(self.component_shape)

    @property
    def tensor_indices(self) -> tuple[str, ...]:
        """Name tensor axes by basis slot in chemists' integral order."""
        return tuple(f"shell_{s.slot}" for s in self.shells)

    @property
    def atom_indices(self) -> tuple[int | None, ...]:
        """Return atom bindings in declared center order, retaining duplicates."""
        return tuple(b.atom_index for b in self.center_bindings)

    @classmethod
    def from_shell_class(
        cls, spec: ShellClassSpec, atom_indices: tuple[int, ...] | None = None
    ) -> "ShellSignature":
        """Adapt the existing catalog without changing names or canonical IDs."""
        atoms = (None,) * 4 if atom_indices is None else tuple(atom_indices)
        if len(atoms) != 4:
            raise ValueError("legacy shell classes require four atom bindings")
        return cls(
            tuple(BasisShell(i, i, angular) for i, angular in enumerate(spec.angular)),
            tuple(CenterBinding(i, atom) for i, atom in enumerate(atoms)),
            spec.name,
        )

    def to_shell_class(self) -> ShellClassSpec:
        """Recover only a tagged Cartesian orbital quartet, preserving its name.

        Atom bindings are task metadata and are not carried by ShellClassSpec.
        Callers must retain this signature when preparing runtime tasks.
        """
        if (
            self.legacy_class is None
            or len(self.shells) != 4
            or tuple(s.center for s in self.shells) != (0, 1, 2, 3)
            or tuple(b.center for b in self.center_bindings) != (0, 1, 2, 3)
            or any(
                s.role != ShellRole.ORBITAL or s.convention != BasisConvention.CARTESIAN
                for s in self.shells
            )
        ):
            raise ValueError(
                "legacy adaptation requires a tagged Cartesian orbital quartet"
            )
        # A known name must never be silently paired with different angular
        # orders: production dispatch uses that name's stable catalog identity.
        from .shell_spec import FUSED_SHELL_SPEC_BY_NAME

        spec = ShellClassSpec(self.legacy_class, self.angular)
        known = FUSED_SHELL_SPEC_BY_NAME.get(spec.name)
        if known is not None and known != spec:
            raise ValueError("legacy class name disagrees with catalog angular orders")
        return spec

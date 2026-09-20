"""Exact basis-data bindings carried by composite MethodIR manifests."""

from __future__ import annotations

import re
import typing
from dataclasses import dataclass

from vibeqc_compiler.common.provenance import canonical_hash

BASIS_BINDING_VERSION = "basis-binding-v1"


@dataclass(frozen=True)
class BasisBinding:
    """Immutable identity of the orbital basis defining a named method."""

    name: str
    basis_identity: str
    source_sha256: str
    source: str
    source_revision: str
    source_version: str
    supported_atomic_numbers: tuple[int, ...]
    representation: str = "spherical"
    ecp_core_electrons: int = 0
    version: str = BASIS_BINDING_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("basis binding requires a nonempty name")
        for field in ("basis_identity", "source_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field} requires a SHA-256 digest")
        if not all(
            isinstance(v, str) and v
            for v in (self.source, self.source_revision, self.source_version)
        ):
            raise ValueError("basis binding requires source revision metadata")
        if self.representation not in ("cartesian", "spherical"):
            raise ValueError("basis binding representation is invalid")
        if (
            not isinstance(self.supported_atomic_numbers, tuple)
            or not self.supported_atomic_numbers
        ):
            raise ValueError("basis binding requires an immutable element domain")
        if (
            tuple(sorted(set(self.supported_atomic_numbers)))
            != self.supported_atomic_numbers
        ):
            raise ValueError("basis element domain must be sorted and unique")
        if any(
            type(z) is not int or not 1 <= z <= 118
            for z in self.supported_atomic_numbers
        ):
            raise ValueError("basis element domain contains an invalid atomic number")
        if type(self.ecp_core_electrons) is not int or self.ecp_core_electrons < 0:
            raise ValueError("basis binding ECP core count must be nonnegative")
        if self.version != BASIS_BINDING_VERSION:
            raise ValueError("unsupported basis binding version")

    def semantic_payload(self) -> typing.Any:
        return {
            "version": self.version,
            "name": self.name,
            "basis_identity": self.basis_identity,
            "source_sha256": self.source_sha256,
            "source": self.source,
            "source_revision": self.source_revision,
            "source_version": self.source_version,
            "supported_atomic_numbers": list(self.supported_atomic_numbers),
            "representation": self.representation,
            "ecp_core_electrons": self.ecp_core_electrons,
        }

    def to_payload(self) -> typing.Any:
        return self.semantic_payload()

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(self.semantic_payload())

    def require_atomic_numbers(self, atomic_numbers: typing.Any) -> None:
        values = tuple(atomic_numbers)
        if any(type(z) is not int for z in values):
            raise TypeError("atomic numbers must be integers")
        unsupported = tuple(sorted(set(values) - set(self.supported_atomic_numbers)))
        if unsupported:
            raise ValueError(
                f"basis {self.name} does not support atomic numbers {unsupported} "
                f"in this canonical method domain"
            )


def r2scan3c_def2_mtzvpp_h_ar() -> typing.Any:
    """Pinned H-Ar all-electron basis binding for the first r2SCAN-3c domain."""

    return BasisBinding(
        name="def2-mTZVPP",
        basis_identity="11c25a32851c1fd54ca318162d46a1c436280c53c75df931e51b2520153a2c2f",
        source_sha256="b769650c71373053790db59191e68f35e20e7cfab85e011776aea1450d1a8d6f",
        source="MolSSI Basis Set Exchange",
        source_revision="4adaf1372c7101620ca1a9f3130be9ae97fb8f30",
        source_version="1",
        supported_atomic_numbers=tuple(range(1, 19)),
        representation="spherical",
    )


def validate_basis_snapshot(
    binding: typing.Any, basis: typing.Any, *, atomic_numbers: typing.Any = ()
) -> typing.Any:
    """Fail closed if a loaded #169 BasisSet differs from the method binding."""

    if not isinstance(binding, BasisBinding):
        raise TypeError("basis validation requires BasisBinding")
    if getattr(basis, "name", None) != binding.name:
        raise ValueError(
            f"expected basis {binding.name!r}, got {getattr(basis, 'name', None)!r}"
        )
    if getattr(basis, "identity", None) != binding.basis_identity:
        raise ValueError(
            "loaded basis data identity does not match the method manifest"
        )
    provenance = getattr(basis, "provenance", None)
    if getattr(provenance, "checksum", None) != binding.source_sha256:
        raise ValueError(
            "loaded basis source checksum does not match the method manifest"
        )
    if getattr(basis, "representation", None) != binding.representation:
        raise ValueError(
            "loaded basis representation does not match the method manifest"
        )
    binding.require_atomic_numbers(tuple(atomic_numbers))
    return basis

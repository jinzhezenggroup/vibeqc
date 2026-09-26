"""Native-free basis and scientific-model resolution helpers.

This module owns the data-only part of model identity.  It intentionally does
not load ``libvibeqc`` or create a native context, so callers can validate
basis/model contracts independently from execution setup.
"""

from __future__ import annotations

import json
import os
import typing
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from . import _generated_methods as _method_manifest
from ._api_types import Atom, Primitive, Shell
from .basis import BasisProvenance, BasisSet, BasisShell, ElementBasis, load_basis
from .elements import checked_integer
from .profiles import canonical_hash


@lru_cache(maxsize=1)
def _basis_pack() -> dict[str, object]:
    """Load the generated, data-only Basis Set Exchange subset once."""

    path = resources.files("vibeqc").joinpath("data/basis_pack.json")
    with path.open("r", encoding="utf-8") as handle:
        pack = json.load(handle)
    if pack.get("schema_version") != 1:
        raise RuntimeError("unsupported bundled basis-pack schema")
    return pack


def _named_basis_shells(name: str, atoms: typing.Sequence[Atom]) -> tuple[Shell, ...]:
    """Expand a bundled basis into immutable shell records for ``atoms``."""

    canonical_name = name.lower().replace("_", "-")
    bases = _basis_pack()["bases"]
    assert isinstance(bases, dict)
    try:
        basis = bases[canonical_name]
    except KeyError as error:
        supported = ", ".join(sorted(str(key) for key in bases))
        raise NotImplementedError(
            f"bundled basis {name!r} is unavailable; choose one of: {supported}"
        ) from error
    assert isinstance(basis, dict)
    elements = basis["elements"]
    assert isinstance(elements, dict)
    shells: list[Shell] = []
    for atom_index, atom in enumerate(atoms):
        try:
            element_shells = elements[str(atom.atomic_number)]
        except KeyError as error:
            raise NotImplementedError(
                f"{canonical_name} is not bundled for atomic number {atom.atomic_number}"
            ) from error
        assert isinstance(element_shells, list)
        for packed_shell in element_shells:
            primitives = tuple(
                Primitive(float(exponent), float(coefficient))
                for exponent, coefficient in zip(
                    packed_shell["exponents"],
                    packed_shell["coefficients"],
                    strict=True,
                )
            )
            shells.append(
                Shell(atom_index, int(packed_shell["angular_momentum"]), primitives)
            )
    return tuple(shells)


@lru_cache(maxsize=8)
def _named_basis_record(name: typing.Any, representation: typing.Any) -> typing.Any:
    """Wrap bundled decimal tables in the immutable public basis record."""

    canonical = name.lower().replace("_", "-")
    pack = _basis_pack()
    if canonical not in pack["bases"]:
        raise NotImplementedError(f"bundled basis {name!r} is unavailable")
    elements = []
    for z, shells in pack["bases"][canonical]["elements"].items():
        elements.append(
            ElementBasis(
                int(z),
                tuple(
                    BasisShell(
                        s["angular_momentum"],
                        s["exponents"],
                        (s["coefficients"],),
                        i,
                    )
                    for i, s in enumerate(shells)
                ),
            )
        )
    return BasisSet(
        canonical,
        tuple(elements),
        BasisProvenance(
            "vibeqc/data/basis_pack.json; Basis Set Exchange",
            str(pack["source"]),
            "BSD-3-Clause",
            canonical_hash(pack),
        ),
        representation,
    )


def snapshot_basis(
    basis: typing.Any, representation: typing.Any = None
) -> typing.Any:
    """Detach caller-owned basis storage and return a stable basis snapshot."""

    if isinstance(basis, os.PathLike) or (
        isinstance(basis, str) and basis.endswith(".json")
    ):
        basis = load_basis(Path(basis))
    if isinstance(basis, BasisSet):
        if representation is not None and basis.representation != representation:
            raise ValueError(
                "basis representation conflicts with the loaded record; create an explicitly reinterpreted record"
            )
        return basis
    if isinstance(basis, str):
        return _named_basis_record(basis, representation or "cartesian")
    return tuple(
        Shell(
            checked_integer(s.atom_index, "shell atom index", high=2**32 - 1),
            checked_integer(
                s.angular_momentum, "shell angular momentum", high=2**32 - 1
            ),
            tuple(
                Primitive(float(p.exponent), float(p.coefficient))
                for p in s.primitives
            ),
        )
        for s in basis
    )


@dataclass(frozen=True)
class ModelResolutionInput:
    """Native-free inputs needed to construct a :class:`ResolvedModel`."""

    method_id: int
    representation: str
    basis_metadata: dict[str, typing.Any]
    geometry: tuple[Atom, ...]
    charge: int
    multiplicity: int
    density_fitting: bool
    density_fitting_relative_threshold: float | None


def resolve_model_identity(request: ModelResolutionInput) -> typing.Any:
    """Build the mathematical model identity without native execution setup."""

    if request.method_id not in (*_method_manifest.HF_METHOD_IDS, _method_manifest.METHOD_MP2):
        raise NotImplementedError("accuracy model is unavailable for this method")
    orbital = request.basis_metadata["orbital"]
    fitted = request.density_fitting
    auxiliary = request.basis_metadata.get("auxiliary", orbital) if fitted else None
    try:
        resolved_method = _method_manifest.METHOD_ID_TO_NAME[request.method_id]
    except KeyError as error:
        raise NotImplementedError("accuracy model is unavailable for this method") from error
    from .accuracy import ResolvedModel

    return ResolvedModel(
        method=resolved_method,
        geometry_hash=canonical_hash(
            [
                (
                    atom.atomic_number,
                    tuple(float(value).hex() if value else "0x0.0p+0" for value in atom.position),
                )
                for atom in request.geometry
            ]
        ),
        basis_hash=orbital["mathematical_identity"],
        electron_count=orbital["electrons"]["electron_count"],
        charge=request.charge,
        multiplicity=request.multiplicity,
        representation=(
            "real_spherical" if request.representation == "spherical" else "cartesian"
        ),
        approximation="density_fitting" if fitted else "conventional",
        auxiliary_basis_hash=auxiliary["mathematical_identity"] if auxiliary else None,
        metric_relative_threshold=(
            request.density_fitting_relative_threshold if fitted else None
        ),
    )


__all__ = [
    "ModelResolutionInput",
    "_basis_pack",
    "_named_basis_record",
    "_named_basis_shells",
    "resolve_model_identity",
    "snapshot_basis",
]

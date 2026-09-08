"""Explicit offline import of MolSSI Basis Set Exchange complete JSON 0.1."""

import json

from .basis import (
    BasisProvenance,
    BasisSet,
    BasisShell,
    ElementBasis,
    read_local_json,
)
from .elements import checked_integer


def import_bse(path, *, source, source_version, license, representation=None):
    """Import a supplied file; declare its provenance/license, never query a service.

    BSE single-l general contractions remain matrices. Combined shells require
    one coefficient row per angular momentum and retain their source group.
    Ambiguous combined/general layouts are rejected instead of guessed.
    """
    if not isinstance(source_version, str) or not source_version.strip():
        raise ValueError("source version must be explicitly supplied")
    data, checksum = read_local_json(path)
    if data.get("molssi_bse_schema") != {
        "schema_type": "complete",
        "schema_version": "0.1",
    }:
        raise ValueError("import requires BSE complete JSON schema 0.1")
    if not isinstance(data.get("elements"), dict) or not data["elements"]:
        raise ValueError("BSE input requires element records")
    declared_kinds = data.get("function_types")
    if not isinstance(declared_kinds, list) or any(
        not isinstance(kind, str) for kind in declared_kinds
    ):
        raise TypeError("BSE function_types must be an array of strings")
    kinds = set(declared_kinds)
    if kinds - {"gto", "gto_cartesian", "gto_spherical", "scalar_ecp"}:
        raise NotImplementedError(
            "only Gaussian shells and scalar ECP metadata can be imported"
        )
    if representation is None:
        if {"gto_cartesian", "gto_spherical"} <= kinds:
            raise ValueError(
                "mixed Cartesian/spherical BSE input requires an explicit representation"
            )
        representation = "spherical" if "gto_spherical" in kinds else "cartesian"
    elements = []
    for key, element in data["elements"].items():
        if not key.isdecimal() or str(int(key)) != key:
            raise ValueError("BSE element keys must be canonical atomic numbers")
        z = checked_integer(int(key), "BSE atomic number", low=1, high=118)
        if not isinstance(element, dict) or not isinstance(
            element.get("electron_shells"), list
        ):
            raise TypeError("BSE element records require electron_shells arrays")
        shells = []
        for group, shell in enumerate(element.get("electron_shells", [])):
            if not isinstance(shell, dict):
                raise TypeError("BSE shell records must be objects")
            if shell.get("function_type") not in (
                "gto",
                "gto_cartesian",
                "gto_spherical",
            ):
                raise NotImplementedError(
                    f"element Z={z} shell {group}: unsupported function type"
                )
            if shell["function_type"] not in kinds:
                raise ValueError(
                    "BSE shell function type is missing from function_types"
                )
            angular, rows, exponents = (
                shell[k] for k in ("angular_momentum", "coefficients", "exponents")
            )
            if (
                not isinstance(angular, list)
                or not angular
                or not isinstance(rows, list)
            ):
                raise ValueError("malformed BSE angular/contraction arrays")
            if len(angular) == 1:
                shells.append(BasisShell(angular[0], exponents, rows, group))
            else:
                if len(rows) != len(angular) or len(set(angular)) != len(angular):
                    raise ValueError("ambiguous combined/general BSE shell")
                shells.extend(
                    BasisShell(l, exponents, (row,), group)
                    for l, row in zip(angular, rows, strict=True)
                )
        core = element.get("ecp_electrons", 0)
        potentials = element.get("ecp_potentials")
        elements.append(
            ElementBasis(
                z,
                tuple(shells),
                ecp_core_electrons=core,
                ecp_data=json.dumps(potentials, sort_keys=True, separators=(",", ":"))
                if potentials
                else None,
            )
        )
    version = data.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("BSE source requires an explicit basis-data version")
    return BasisSet(
        data.get("name", ""),
        tuple(elements),
        BasisProvenance(
            source, f"{source_version}; basis-data={version}", license, checksum
        ),
        representation,
    )

"""Offline canonical basis records with immutable provenance and exact decimals.

Loading data is independent of integral/ECP execution. Native shell expansion
occurs only after an explicit per-operator capability preflight.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from functools import cached_property
from pathlib import Path

from .elements import checked_integer
from .profiles import canonical_hash

SCHEMA = "vibeqc.basis"
VERSION = 1
NORMALIZATION = "normalized-primitives; native-normalized-contractions"
ORDERING = "CCA-cartesian; libcint-real-spherical"
MAX_FILE_BYTES = 64 << 20


def decimal_text(value, name, *, positive=False):
    """Preserve decimal input; reject nonfinite and nonzero-to-zero FP64 casts."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"{name} must be a decimal string or real number")
    try:
        number = Decimal(str(value))
        numeric = float(number)
    except (ValueError, InvalidOperation, OverflowError) as error:
        raise ValueError(f"invalid {name}") from error
    if (
        not number.is_finite()
        or not math.isfinite(numeric)
        or (number != 0 and numeric == 0)
    ):
        raise ValueError(f"{name} is not representable as a finite nonzero FP64 value")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    # Decimal.normalize() obeys the process context precision; using it here
    # would silently round long source coefficients. Strip zeros textually.
    sign, digits, exponent = number.as_tuple()
    digits = list(digits)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    if not any(digits):
        return "0"
    return ("-" if sign else "") + "".join(map(str, digits)) + "E" + str(exponent)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON record/key {key!r}")
        result[key] = value
    return result


def read_local_json(path):
    """Bound a local interchange file and reject duplicate keys/nonfinite JSON."""
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("basis file exceeds the 64 MiB input limit")
    raw = path.read_bytes()
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("basis file exceeds the 64 MiB input limit")
    data = json.loads(
        raw,
        object_pairs_hook=_pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant {x}")
        ),
    )
    if not isinstance(data, dict):
        raise TypeError("basis file requires an object")
    return data, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class BasisProvenance:
    """A supplied source and redistribution declaration; no implicit license guess."""

    source: str
    version: str
    license: str
    checksum: str

    def __post_init__(self):
        if not all(
            isinstance(v, str) and v.strip()
            for v in (self.source, self.version, self.license)
        ):
            raise ValueError("basis provenance requires source, version and license")
        if not isinstance(self.checksum, str) or not re.fullmatch(
            "[0-9a-f]{64}", self.checksum
        ):
            raise ValueError("basis provenance requires a SHA-256 source checksum")


@dataclass(frozen=True)
class BasisShell:
    """One angular momentum with a contraction-by-primitive coefficient matrix.

    ``source_group`` preserves membership when a combined SP source shell is
    expanded. All coefficients, including exact zeros, survive serialization.
    """

    angular_momentum: int
    exponents: tuple[str, ...]
    coefficients: tuple[tuple[str, ...], ...]
    source_group: int = 0

    def __post_init__(self):
        object.__setattr__(
            self,
            "angular_momentum",
            checked_integer(self.angular_momentum, "shell angular momentum", high=32),
        )
        object.__setattr__(
            self,
            "source_group",
            checked_integer(self.source_group, "source shell group"),
        )
        if (
            not isinstance(self.exponents, (list, tuple))
            or not isinstance(self.coefficients, (list, tuple))
            or any(not isinstance(row, (list, tuple)) for row in self.coefficients)
        ):
            raise TypeError("exponents and contraction rows must be arrays")
        if not self.exponents or not self.coefficients:
            raise ValueError("basis shell requires exponents and contractions")
        checked_integer(len(self.exponents), "primitive count", low=1, high=2**32 - 1)
        checked_integer(
            len(self.coefficients), "contraction count", low=1, high=2**32 - 1
        )
        exponents = tuple(
            decimal_text(v, "exponent", positive=True) for v in self.exponents
        )
        coefficients = tuple(
            tuple(decimal_text(v, "coefficient") for v in row)
            for row in self.coefficients
        )
        if any(len(row) != len(exponents) for row in coefficients):
            raise ValueError(
                "coefficient matrix columns must match primitive exponents"
            )
        if any(all(float(v) == 0 for v in row) for row in coefficients):
            raise ValueError("an identically zero contraction has no normalizable AO")
        object.__setattr__(self, "exponents", exponents)
        object.__setattr__(self, "coefficients", coefficients)


def validate_ecp_data(data):
    """Validate retained BSE scalar-ECP radial arrays without implementing them."""
    if not isinstance(data, list) or not data:
        raise ValueError("ECP data requires a nonempty potential list")
    for potential in data:
        if not isinstance(potential, dict) or potential.get("ecp_type") != "scalar_ecp":
            raise ValueError("ECP metadata requires scalar_ecp potential records")
        angular = potential.get("angular_momentum")
        if not isinstance(angular, list) or len(angular) != 1:
            raise ValueError("ECP potential needs one angular channel")
        checked_integer(angular[0], "ECP angular channel", high=32)
        exponents = potential.get("gaussian_exponents", [])
        powers = potential.get("r_exponents", [])
        rows = potential.get("coefficients", [])
        if (
            not isinstance(exponents, list)
            or not exponents
            or not isinstance(powers, list)
            or len(powers) != len(exponents)
            or not isinstance(rows, list)
            or not rows
        ):
            raise ValueError("malformed ECP radial arrays")
        for exponent in exponents:
            decimal_text(exponent, "ECP Gaussian exponent", positive=True)
        for power in powers:
            checked_integer(power, "ECP radial power", high=32)
        for row in rows:
            if not isinstance(row, list) or len(row) != len(exponents):
                raise ValueError("ECP coefficient/exponent dimensions differ")
            for coefficient in row:
                decimal_text(coefficient, "ECP coefficient")
    canonical_hash(data)


@dataclass(frozen=True)
class ElementBasis:
    """Element/nuclear/ECP identity plus all shells, including unsupported l."""

    atomic_number: int
    shells: tuple[BasisShell, ...]
    nuclear_charge: int | None = None
    ecp_core_electrons: int = 0
    ecp_data: str | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "atomic_number",
            checked_integer(
                self.atomic_number, "element atomic number", low=1, high=118
            ),
        )
        nucleus = (
            self.atomic_number if self.nuclear_charge is None else self.nuclear_charge
        )
        nucleus = checked_integer(nucleus, "nuclear charge", low=1, high=118)
        object.__setattr__(
            self,
            "ecp_core_electrons",
            checked_integer(
                self.ecp_core_electrons, "ECP core electrons", high=nucleus
            ),
        )
        if not self.shells or any(not isinstance(s, BasisShell) for s in self.shells):
            raise ValueError("element requires canonical basis shells")
        if bool(self.ecp_core_electrons) != bool(self.ecp_data):
            raise ValueError(
                "ECP core count and potential metadata must appear together"
            )
        if self.ecp_data is not None:
            if not isinstance(self.ecp_data, str):
                raise ValueError("ECP metadata must use immutable serialized storage")
            data = json.loads(self.ecp_data, object_pairs_hook=_pairs)
            validate_ecp_data(data)
        object.__setattr__(self, "shells", tuple(self.shells))
        object.__setattr__(self, "nuclear_charge", nucleus)


@dataclass(frozen=True)
class BasisSet:
    """Owned local basis snapshot. A changed file requires an explicit new load."""

    name: str
    elements: tuple[ElementBasis, ...]
    provenance: BasisProvenance
    representation: str = "cartesian"
    normalization: str = NORMALIZATION
    exponent_units: str = "bohr^-2"
    ordering: str = ORDERING

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("basis requires a nonempty name")
        if not isinstance(self.provenance, BasisProvenance):
            raise TypeError("basis requires source provenance")
        if self.representation not in ("cartesian", "spherical"):
            raise ValueError("basis representation must be Cartesian or real spherical")
        if (self.normalization, self.exponent_units, self.ordering) != (
            NORMALIZATION,
            "bohr^-2",
            ORDERING,
        ):
            raise ValueError(
                "unsupported units/ordering/normalization; coefficients must multiply normalized primitives and must not contain radial normalization factors"
            )
        if not self.elements or any(
            not isinstance(e, ElementBasis) for e in self.elements
        ):
            raise ValueError("basis requires element records")
        elements = tuple(sorted(self.elements, key=lambda e: e.atomic_number))
        if len({e.atomic_number for e in elements}) != len(elements):
            raise ValueError("duplicate element basis records")
        object.__setattr__(self, "elements", elements)

    @property
    def by_element(self):
        """Return detached lookup storage; the owned records remain immutable."""
        return {e.atomic_number: e for e in self.elements}

    def to_payload(self):
        """Canonical record and checksum, independent of its filesystem location."""
        payload = {"schema": SCHEMA, "schema_version": VERSION, **asdict(self)}
        payload["checksum"] = canonical_hash(payload)
        return payload

    @cached_property
    def identity(self):
        """Full data/provenance identity for result reproducibility."""
        return self.to_payload()["checksum"]

    def write(self, path):
        """Export locally; no service lookup or runtime download is performed."""
        Path(path).write_text(
            json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n"
        )

    def shells_for(self, atoms):
        """Expand general contractions without normalization or dropping zeros."""
        from .basis_capabilities import require_basis
        from .calculator import Primitive, Shell

        require_basis(self, atoms)
        result = []
        for index, atom in enumerate(atoms):
            try:
                element = self.by_element[atom.atomic_number]
            except KeyError as error:
                raise NotImplementedError(
                    f"basis {self.name!r} has no element Z={atom.atomic_number}"
                ) from error
            for shell in element.shells:
                for row in shell.coefficients:
                    result.append(
                        Shell(
                            index,
                            shell.angular_momentum,
                            tuple(
                                Primitive(float(e), float(c))
                                for e, c in zip(shell.exponents, row, strict=True)
                            ),
                        )
                    )
        return tuple(result)


def load_basis(path):
    """Read a checked canonical record; BSE conversion is a separate explicit step."""
    payload, _ = read_local_json(path)
    if (
        payload.get("schema") != SCHEMA
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != VERSION
    ):
        raise ValueError("unsupported canonical basis schema")
    expected = payload.pop("checksum", None)
    if expected != canonical_hash(payload):
        raise ValueError("canonical basis checksum mismatch")
    allowed = {
        "schema",
        "schema_version",
        "name",
        "elements",
        "provenance",
        "representation",
        "normalization",
        "exponent_units",
        "ordering",
    }
    if set(payload) != allowed:
        raise ValueError("canonical basis has missing or unknown fields")
    if not isinstance(payload["elements"], list) or not isinstance(
        payload["provenance"], dict
    ):
        raise TypeError("canonical basis requires element arrays and provenance object")
    elements = []
    for item in payload["elements"]:
        if not isinstance(item, dict) or not isinstance(item.get("shells"), list):
            raise TypeError("element records require shell arrays")
        if any(not isinstance(shell, dict) for shell in item["shells"]):
            raise TypeError("shell records must be objects")
        shells = tuple(BasisShell(**s) for s in item["shells"])
        elements.append(ElementBasis(**{**item, "shells": shells}))
    return BasisSet(
        **{
            k: v
            for k, v in payload.items()
            if k not in ("schema", "schema_version", "elements", "provenance")
        },
        elements=tuple(elements),
        provenance=BasisProvenance(**payload["provenance"]),
    )

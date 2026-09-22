"""Backend-neutral GFN-family compiler method IR (#503).

The IR in this module is representation-only.  It describes the scientific
requirements of an audited GFN method while deliberately keeping SCC iteration,
mixing history, convergence thresholds, occupations, and eigensolver/library
policy outside compiler ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import ClassVar

from vibeqc_compiler.common.provenance import canonical_hash

XTB_METHOD_IR_VERSION = "xtb-method-ir-v1"
XTB_METHOD_CATALOG_VERSION = "xtb-method-catalog-v1"
XTB_PARAMETER_SET_VERSION = "xtb-parameter-set-v1"

_SUPPORTED_MODELS = ("gfn2", "gfn1")
_SCHEMA_EXTENSION_MODELS = _SUPPORTED_MODELS
_REFERENCES = ("restricted", "unrestricted")
_PRODUCT_ORDER = ("energy", "nuclear-gradient")
_STATE_ORDER = ("shell_charge", "charge", "dipole", "quadrupole", "magnetization")
_PARAMETER_DOMAINS = ("basis", "orbital", "correction", "spin")
_DERIVATIVE_CAPABILITIES = (
    "energy",
    "hamiltonian",
    "integral-adjoint",
    "nuclear-gradient",
)
_PRIMITIVE_ORDER = (
    "basis_parameters",
    "coordination_number",
    "overlap_integrals",
    "multipole_integrals",
    "h0",
    "scc_electrostatics",
    "spin_polarization",
    "hamiltonian",
    "repulsion",
    "halogen_correction",
    "dispersion",
)

_GFN2_BASIS_TABLES = (
    "atomic-basis-shell-layout",
    "slater-exponents",
)
_GFN2_ORBITAL_TABLES = (
    "atomic-levels",
    "coordination-dependent-h0",
    "hubbard-and-multipole",
    "third-order-shell",
)
_GFN2_CORRECTION_TABLES = (
    "coordination-number",
    "d4-reference",
    "repulsion",
)
_GFN2_SPIN_TABLES = ("spin-polarization",)
_GFN2_SUPPORTED_ATOMIC_NUMBERS = tuple(range(1, 87))

_GFN1_BASIS_TABLES = (
    "atomic-basis-shell-layout",
    "slater-exponents",
)
_GFN1_ORBITAL_TABLES = (
    "atomic-levels",
    "coordination-dependent-h0",
    "hubbard-shell",
    "third-order-atom",
)
_GFN1_CORRECTION_TABLES = (
    "coordination-number",
    "d3-reference-sha256-9ff932ea598f690c1fb599a67762060ba1907102d5ec132164f2a7e8886cd22e",
    "halogen-correction",
    "repulsion",
)
_GFN1_SPIN_TABLES = ("spin-polarization",)
_GFN1_SUPPORTED_ATOMIC_NUMBERS = tuple(range(1, 87))
_GFN1_PARAMETER_JSON_SHA256 = (
    "0ecdc3f5f12990c5a7e0f0bd7e6fe931ecf72d7630e6a6a3cc396c51766a40a0"
)


class UnsupportedXtbMethod(ValueError):
    """Requested xTB composition is not audited or representable by this schema."""


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise UnsupportedXtbMethod(f"{label} requires a non-empty canonical string")
    return value


def _canonical_strings(values: object, label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise UnsupportedXtbMethod(f"{label} must be an immutable tuple")
    checked = tuple(_require_text(value, label) for value in values)
    if len(checked) != len(set(checked)):
        raise UnsupportedXtbMethod(f"{label} contains duplicate entries")
    return tuple(sorted(checked))


def _canonical_products(values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise UnsupportedXtbMethod("compiler products require a non-empty tuple")
    if len(values) != len(set(values)):
        raise UnsupportedXtbMethod("compiler products contain duplicate entries")
    unknown = set(values) - set(_PRODUCT_ORDER)
    if unknown:
        raise UnsupportedXtbMethod(
            f"unsupported compiler products: {sorted(unknown)!r}"
        )
    if "nuclear-gradient" in values and "energy" not in values:
        raise UnsupportedXtbMethod("nuclear-gradient requests must also request energy")
    return tuple(name for name in _PRODUCT_ORDER if name in values)


@dataclass(frozen=True)
class XtbParameterSet:
    """Audited parameter-table manifest rather than numerical parameter storage.

    Numerical tables are intentionally not copied into this IR layer.  Later
    lowering slices bind the declared table families to versioned data while
    this object supplies the semantic identity and supported-element contract.
    """

    identifier: str
    revision: str
    source: str
    supported_atomic_numbers: tuple[int, ...]
    basis_tables: tuple[str, ...]
    orbital_tables: tuple[str, ...]
    correction_tables: tuple[str, ...]
    spin_tables: tuple[str, ...] = ()
    version: str = XTB_PARAMETER_SET_VERSION

    def __post_init__(self) -> None:
        _require_text(self.identifier, "parameter-set identifier")
        _require_text(self.revision, "parameter-set revision")
        _require_text(self.source, "parameter-set source")
        if self.version != XTB_PARAMETER_SET_VERSION:
            raise UnsupportedXtbMethod("unsupported xTB parameter-set schema version")
        if not isinstance(self.supported_atomic_numbers, tuple):
            raise UnsupportedXtbMethod(
                "supported atomic numbers must be an immutable tuple"
            )
        numbers = self.supported_atomic_numbers
        if not numbers:
            raise UnsupportedXtbMethod(
                "parameter set must support at least one element"
            )
        if any(
            isinstance(number, bool)
            or not isinstance(number, int)
            or not 1 <= number <= 118
            for number in numbers
        ):
            raise UnsupportedXtbMethod("supported atomic numbers must lie in [1, 118]")
        if len(numbers) != len(set(numbers)):
            raise UnsupportedXtbMethod("supported atomic numbers contain duplicates")
        object.__setattr__(self, "supported_atomic_numbers", tuple(sorted(numbers)))

        for field in (
            "basis_tables",
            "orbital_tables",
            "correction_tables",
            "spin_tables",
        ):
            object.__setattr__(
                self,
                field,
                _canonical_strings(getattr(self, field), field.replace("_", " ")),
            )
        if (
            not self.basis_tables
            or not self.orbital_tables
            or not self.correction_tables
        ):
            raise UnsupportedXtbMethod(
                "basis, orbital, and correction table requirements must be explicit"
            )

    def semantic_payload(self) -> dict:
        return {
            "version": self.version,
            "identifier": self.identifier,
            "revision": self.revision,
            "source": self.source,
            "supported_atomic_numbers": self.supported_atomic_numbers,
            "tables": {
                "basis": self.basis_tables,
                "orbital": self.orbital_tables,
                "correction": self.correction_tables,
                "spin": self.spin_tables,
            },
        }

    def to_payload(self) -> dict:
        return self.semantic_payload()

    @property
    def identity(self) -> str:
        return canonical_hash(self.semantic_payload())


@dataclass(frozen=True)
class XtbPrimitive:
    """One canonical compiler-owned node in an xTB method graph."""

    kind: str
    model: str
    requires: tuple[str, ...] = ()
    state_requirements: tuple[str, ...] = ()
    parameter_domains: tuple[str, ...] = ()
    derivative_capabilities: tuple[str, ...] = ()
    self_consistent: bool = False
    version: str = "xtb-primitive-v1"

    def __post_init__(self) -> None:
        if self.version != "xtb-primitive-v1":
            raise UnsupportedXtbMethod("unsupported xTB primitive version")
        if self.kind not in _PRIMITIVE_ORDER:
            raise UnsupportedXtbMethod(f"unsupported xTB primitive kind {self.kind!r}")
        _require_text(self.model, f"{self.kind} model")
        if not isinstance(self.self_consistent, bool):
            raise TypeError("self_consistent must be boolean")

        requires = _canonical_strings(self.requires, f"{self.kind} dependencies")
        if any(requirement not in _PRIMITIVE_ORDER for requirement in requires):
            raise UnsupportedXtbMethod(
                f"{self.kind} has an unknown primitive dependency"
            )
        object.__setattr__(
            self,
            "requires",
            tuple(name for name in _PRIMITIVE_ORDER if name in requires),
        )

        states = _canonical_strings(
            self.state_requirements, f"{self.kind} state requirements"
        )
        if any(state not in _STATE_ORDER for state in states):
            raise UnsupportedXtbMethod(f"{self.kind} has an unknown state requirement")
        object.__setattr__(
            self,
            "state_requirements",
            tuple(name for name in _STATE_ORDER if name in states),
        )

        domains = _canonical_strings(
            self.parameter_domains, f"{self.kind} parameter domains"
        )
        if any(domain not in _PARAMETER_DOMAINS for domain in domains):
            raise UnsupportedXtbMethod(f"{self.kind} has an unknown parameter domain")
        object.__setattr__(
            self,
            "parameter_domains",
            tuple(name for name in _PARAMETER_DOMAINS if name in domains),
        )

        derivatives = _canonical_strings(
            self.derivative_capabilities, f"{self.kind} derivative capabilities"
        )
        if any(
            capability not in _DERIVATIVE_CAPABILITIES for capability in derivatives
        ):
            raise UnsupportedXtbMethod(
                f"{self.kind} has an unknown derivative capability"
            )
        object.__setattr__(
            self,
            "derivative_capabilities",
            tuple(name for name in _DERIVATIVE_CAPABILITIES if name in derivatives),
        )

    def semantic_payload(self) -> dict:
        return {
            "kind": self.kind,
            "model": self.model,
            "requires": self.requires,
            "state_requirements": self.state_requirements,
            "parameter_domains": self.parameter_domains,
            "derivative_capabilities": self.derivative_capabilities,
            "self_consistent": self.self_consistent,
            "version": self.version,
        }

    def to_payload(self) -> dict:
        return self.semantic_payload()


@dataclass(frozen=True)
class XtbMethodSpec:
    """Declarative xTB method request before canonical graph construction."""

    identifier: str
    model_flavor: str
    parameter_set: XtbParameterSet
    reference: str = "restricted"
    requested_products: tuple[str, ...] = ("energy",)
    version: str = XTB_METHOD_CATALOG_VERSION

    def __post_init__(self) -> None:
        _require_text(self.identifier, "xTB method identifier")
        if self.version != XTB_METHOD_CATALOG_VERSION:
            raise UnsupportedXtbMethod("unsupported xTB method manifest version")
        if self.model_flavor not in _SCHEMA_EXTENSION_MODELS:
            raise UnsupportedXtbMethod(
                f"unknown xTB model flavor {self.model_flavor!r}"
            )
        if not isinstance(self.parameter_set, XtbParameterSet):
            raise TypeError("xTB method requires an XtbParameterSet")
        if self.reference not in _REFERENCES:
            raise UnsupportedXtbMethod(f"unsupported xTB reference {self.reference!r}")
        object.__setattr__(
            self, "requested_products", _canonical_products(self.requested_products)
        )

    def to_payload(self) -> dict:
        return {
            "identifier": self.identifier,
            "version": self.version,
            "model_flavor": self.model_flavor,
            "parameter_set": self.parameter_set.to_payload(),
            "reference": self.reference,
            "requested_products": self.requested_products,
        }


@dataclass(frozen=True)
class XtbMethodIR:
    """Canonical backend-neutral xTB compiler graph."""

    identifier: str
    model_flavor: str
    parameter_set: XtbParameterSet
    reference: str
    primitives: tuple[XtbPrimitive, ...]
    requested_products: tuple[str, ...]
    version: str = XTB_METHOD_IR_VERSION
    kind: ClassVar[str] = "xtb_method"

    def __post_init__(self) -> None:
        _require_text(self.identifier, "XtbMethodIR identifier")
        if self.version != XTB_METHOD_IR_VERSION:
            raise UnsupportedXtbMethod("unsupported XtbMethodIR version")
        if self.model_flavor not in _SUPPORTED_MODELS:
            raise UnsupportedXtbMethod(
                f"unsupported xTB model flavor {self.model_flavor!r}"
            )
        if not isinstance(self.parameter_set, XtbParameterSet):
            raise TypeError("XtbMethodIR requires an XtbParameterSet")
        _validate_parameter_set(self.model_flavor, self.parameter_set)
        if self.reference not in _REFERENCES:
            raise UnsupportedXtbMethod(f"unsupported xTB reference {self.reference!r}")
        object.__setattr__(
            self, "requested_products", _canonical_products(self.requested_products)
        )

        if not isinstance(self.primitives, tuple):
            raise UnsupportedXtbMethod("xTB primitives must be an immutable tuple")
        if not all(
            isinstance(primitive, XtbPrimitive) for primitive in self.primitives
        ):
            raise UnsupportedXtbMethod("XtbMethodIR contains an unsupported primitive")
        kinds = tuple(primitive.kind for primitive in self.primitives)
        expected_primitives = _primitives_for_model(self.model_flavor)
        expected_kinds = tuple(primitive.kind for primitive in expected_primitives)
        model_name = self.model_flavor.upper()
        if kinds != expected_kinds:
            raise UnsupportedXtbMethod(
                f"{model_name}-xTB graph must contain the complete canonical primitive sequence"
            )
        if self.primitives != expected_primitives:
            raise UnsupportedXtbMethod(
                f"{model_name}-xTB graph must preserve audited primitive semantics"
            )
        present = set(kinds)
        for primitive in self.primitives:
            if any(requirement not in present for requirement in primitive.requires):
                raise UnsupportedXtbMethod(
                    f"{primitive.kind} depends on a missing primitive"
                )

    @property
    def compiler_requirements(self) -> dict:
        states = {
            state
            for primitive in self.primitives
            for state in primitive.state_requirements
        }
        kinds = {primitive.kind for primitive in self.primitives}
        integral_operators = []
        if "overlap_integrals" in kinds:
            integral_operators.append("overlap")
        if "multipole_integrals" in kinds:
            integral_operators.extend(("dipole", "quadrupole"))
        correction_kinds = {
            "coordination_number",
            "repulsion",
            "halogen_correction",
            "dispersion",
        }
        return {
            "parameter_set_identity": self.parameter_set.identity,
            "supported_atomic_numbers": self.parameter_set.supported_atomic_numbers,
            "parameter_tables": {
                "basis": self.parameter_set.basis_tables,
                "orbital": self.parameter_set.orbital_tables,
                "correction": self.parameter_set.correction_tables,
                "spin": self.parameter_set.spin_tables,
            },
            "integral_operators": tuple(integral_operators),
            "state_requirements": tuple(
                name for name in _STATE_ORDER if name in states
            ),
            "correction_primitives": tuple(
                name
                for name in _PRIMITIVE_ORDER
                if name in kinds and name in correction_kinds
            ),
            "requested_products": self.requested_products,
        }

    @property
    def runtime_requirements(self) -> dict:
        return {
            "scc_fixed_point": True,
            "generalized_eigensolution": True,
            "occupations": True,
            "policy_in_compiler_ir": False,
        }

    @property
    def capability(self) -> dict:
        return {
            "compiler_representable": True,
            "lowering_available": False,
            "runtime_executable": False,
        }

    def semantic_payload(self) -> dict:
        return {
            "kind": self.kind,
            "version": self.version,
            "model_flavor": self.model_flavor,
            "reference": self.reference,
            "parameter_set": self.parameter_set.semantic_payload(),
            "primitives": [
                primitive.semantic_payload() for primitive in self.primitives
            ],
            "requested_products": self.requested_products,
        }

    def to_payload(self) -> dict:
        return {
            "identifier": self.identifier,
            **self.semantic_payload(),
            "compiler_requirements": self.compiler_requirements,
            "runtime_requirements": self.runtime_requirements,
            "capability": self.capability,
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.semantic_payload())

    @property
    def manifest_identity(self) -> str:
        return canonical_hash(
            {"identifier": self.identifier, **self.semantic_payload()}
        )


GFN2_PARAMETER_SET = XtbParameterSet(
    identifier="gfn2-xtb",
    revision="bannwarth-ehlert-grimme-2019",
    source="doi:10.1021/acs.jctc.8b01176",
    supported_atomic_numbers=_GFN2_SUPPORTED_ATOMIC_NUMBERS,
    basis_tables=_GFN2_BASIS_TABLES,
    orbital_tables=_GFN2_ORBITAL_TABLES,
    correction_tables=_GFN2_CORRECTION_TABLES,
    spin_tables=_GFN2_SPIN_TABLES,
)


GFN1_PARAMETER_SET = XtbParameterSet(
    identifier="gfn1-xtb",
    revision=f"sha256:{_GFN1_PARAMETER_JSON_SHA256}",
    source="upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1.json",
    supported_atomic_numbers=_GFN1_SUPPORTED_ATOMIC_NUMBERS,
    basis_tables=_GFN1_BASIS_TABLES,
    orbital_tables=_GFN1_ORBITAL_TABLES,
    correction_tables=_GFN1_CORRECTION_TABLES,
    spin_tables=_GFN1_SPIN_TABLES,
)


def _gfn2_primitives() -> tuple[XtbPrimitive, ...]:
    return (
        XtbPrimitive(
            "basis_parameters",
            "gfn2-minimal-valence-sto-mg",
            parameter_domains=("basis", "orbital"),
        ),
        XtbPrimitive(
            "coordination_number",
            "gfn2-covalent-cn",
            parameter_domains=("correction", "orbital"),
            derivative_capabilities=("nuclear-gradient",),
        ),
        XtbPrimitive(
            "overlap_integrals",
            "sto-mg-overlap",
            requires=("basis_parameters",),
            parameter_domains=("basis",),
            derivative_capabilities=("integral-adjoint",),
        ),
        XtbPrimitive(
            "multipole_integrals",
            "gfn2-cumulative-multipoles",
            requires=("basis_parameters", "overlap_integrals"),
            parameter_domains=("basis", "orbital"),
            derivative_capabilities=("integral-adjoint",),
        ),
        XtbPrimitive(
            "h0",
            "gfn2-extended-huckel-cn",
            requires=(
                "basis_parameters",
                "coordination_number",
                "overlap_integrals",
            ),
            parameter_domains=("orbital",),
            derivative_capabilities=("energy", "hamiltonian", "integral-adjoint"),
        ),
        XtbPrimitive(
            "scc_electrostatics",
            "gfn2-es2-es3-aes2",
            requires=(
                "coordination_number",
                "multipole_integrals",
                "overlap_integrals",
            ),
            state_requirements=("charge", "dipole", "quadrupole"),
            parameter_domains=("orbital",),
            derivative_capabilities=("energy", "hamiltonian", "integral-adjoint"),
            self_consistent=True,
        ),
        XtbPrimitive(
            "spin_polarization",
            "gfn2-shell-spin-polarization",
            requires=("basis_parameters",),
            state_requirements=("magnetization",),
            parameter_domains=("spin",),
            derivative_capabilities=("energy", "hamiltonian"),
            self_consistent=True,
        ),
        XtbPrimitive(
            "hamiltonian",
            "gfn2-fixed-state-hamiltonian",
            requires=("h0", "scc_electrostatics", "spin_polarization"),
            state_requirements=("charge", "dipole", "quadrupole"),
            derivative_capabilities=("energy", "hamiltonian", "integral-adjoint"),
        ),
        XtbPrimitive(
            "repulsion",
            "gfn2-repulsion",
            requires=("coordination_number",),
            parameter_domains=("correction",),
            derivative_capabilities=("energy", "nuclear-gradient"),
        ),
        XtbPrimitive(
            "dispersion",
            "gfn2-self-consistent-d4",
            requires=("coordination_number", "scc_electrostatics"),
            state_requirements=("charge",),
            parameter_domains=("correction",),
            derivative_capabilities=("energy", "hamiltonian", "nuclear-gradient"),
            self_consistent=True,
        ),
    )


def _gfn1_primitives() -> tuple[XtbPrimitive, ...]:
    return (
        XtbPrimitive(
            "basis_parameters",
            "gfn1-sto-mg-shell-basis",
            parameter_domains=("basis", "orbital"),
        ),
        XtbPrimitive(
            "coordination_number",
            "gfn1-exp-covalent-cn",
            parameter_domains=("correction", "orbital"),
            derivative_capabilities=("nuclear-gradient",),
        ),
        XtbPrimitive(
            "overlap_integrals",
            "sto-mg-overlap",
            requires=("basis_parameters",),
            parameter_domains=("basis",),
            derivative_capabilities=("integral-adjoint",),
        ),
        XtbPrimitive(
            "h0",
            "gfn1-extended-huckel-cn",
            requires=(
                "basis_parameters",
                "coordination_number",
                "overlap_integrals",
            ),
            parameter_domains=("orbital",),
            derivative_capabilities=("energy", "hamiltonian", "integral-adjoint"),
        ),
        XtbPrimitive(
            "scc_electrostatics",
            "gfn1-shell-es2-atom-es3",
            requires=("basis_parameters", "overlap_integrals"),
            state_requirements=("shell_charge", "charge"),
            parameter_domains=("orbital",),
            derivative_capabilities=("energy", "hamiltonian", "integral-adjoint"),
            self_consistent=True,
        ),
        XtbPrimitive(
            "spin_polarization",
            "gfn1-shell-spin-polarization",
            requires=("basis_parameters",),
            state_requirements=("magnetization",),
            parameter_domains=("spin",),
            derivative_capabilities=("energy", "hamiltonian"),
            self_consistent=True,
        ),
        XtbPrimitive(
            "hamiltonian",
            "gfn1-fixed-state-hamiltonian",
            requires=("h0", "scc_electrostatics", "spin_polarization"),
            state_requirements=("shell_charge", "charge"),
            derivative_capabilities=("energy", "hamiltonian", "integral-adjoint"),
        ),
        XtbPrimitive(
            "repulsion",
            "gfn1-repulsion",
            parameter_domains=("correction",),
            derivative_capabilities=("energy", "nuclear-gradient"),
        ),
        XtbPrimitive(
            "halogen_correction",
            "gfn1-halogen-correction",
            parameter_domains=("correction",),
            derivative_capabilities=("energy", "nuclear-gradient"),
        ),
        XtbPrimitive(
            "dispersion",
            "gfn1-d3-bj-two-body",
            requires=("coordination_number",),
            parameter_domains=("correction",),
            derivative_capabilities=("energy", "nuclear-gradient"),
        ),
    )


def _validate_gfn2_parameter_set(parameter_set: XtbParameterSet) -> None:
    if parameter_set.identifier != "gfn2-xtb":
        raise UnsupportedXtbMethod(
            "GFN2-xTB requires an explicitly identified gfn2-xtb parameter set"
        )
    if parameter_set.supported_atomic_numbers != _GFN2_SUPPORTED_ATOMIC_NUMBERS:
        raise UnsupportedXtbMethod(
            "GFN2-xTB parameter scope must explicitly cover atomic numbers 1..86"
        )
    required_tables = {
        "basis_tables": _GFN2_BASIS_TABLES,
        "orbital_tables": _GFN2_ORBITAL_TABLES,
        "correction_tables": _GFN2_CORRECTION_TABLES,
        "spin_tables": _GFN2_SPIN_TABLES,
    }
    for field, expected in required_tables.items():
        if getattr(parameter_set, field) != expected:
            raise UnsupportedXtbMethod(
                f"GFN2-xTB {field.replace('_', ' ')} are not an audited manifest"
            )


def _validate_gfn1_parameter_set(parameter_set: XtbParameterSet) -> None:
    if parameter_set.identifier != "gfn1-xtb":
        raise UnsupportedXtbMethod(
            "GFN1-xTB requires an explicitly identified gfn1-xtb parameter set"
        )
    if parameter_set.revision != f"sha256:{_GFN1_PARAMETER_JSON_SHA256}":
        raise UnsupportedXtbMethod(
            "GFN1-xTB parameter revision must match the canonical normalized JSON"
        )
    if (
        parameter_set.source
        != "upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1.json"
    ):
        raise UnsupportedXtbMethod(
            "GFN1-xTB parameter source must use the canonical repository snapshot"
        )
    if parameter_set.supported_atomic_numbers != _GFN1_SUPPORTED_ATOMIC_NUMBERS:
        raise UnsupportedXtbMethod(
            "GFN1-xTB parameter scope must explicitly cover atomic numbers 1..86"
        )
    required_tables = {
        "basis_tables": _GFN1_BASIS_TABLES,
        "orbital_tables": _GFN1_ORBITAL_TABLES,
        "correction_tables": _GFN1_CORRECTION_TABLES,
        "spin_tables": _GFN1_SPIN_TABLES,
    }
    for field, expected in required_tables.items():
        if getattr(parameter_set, field) != expected:
            raise UnsupportedXtbMethod(
                f"GFN1-xTB {field.replace('_', ' ')} are not an audited manifest"
            )


def _validate_parameter_set(model_flavor: str, parameter_set: XtbParameterSet) -> None:
    if model_flavor == "gfn2":
        _validate_gfn2_parameter_set(parameter_set)
        return
    if model_flavor == "gfn1":
        _validate_gfn1_parameter_set(parameter_set)
        return
    raise UnsupportedXtbMethod(f"unsupported xTB model flavor {model_flavor!r}")


def _primitives_for_model(model_flavor: str) -> tuple[XtbPrimitive, ...]:
    if model_flavor == "gfn2":
        return _gfn2_primitives()
    if model_flavor == "gfn1":
        return _gfn1_primitives()
    raise UnsupportedXtbMethod(f"unsupported xTB model flavor {model_flavor!r}")


XTB_METHOD_CATALOG = MappingProxyType(
    {
        "GFN1-xTB": XtbMethodSpec(
            identifier="GFN1-xTB",
            model_flavor="gfn1",
            parameter_set=GFN1_PARAMETER_SET,
            requested_products=("energy",),
        ),
        "GFN2-xTB": XtbMethodSpec(
            identifier="GFN2-xTB",
            model_flavor="gfn2",
            parameter_set=GFN2_PARAMETER_SET,
            requested_products=("energy",),
        ),
    }
)


def resolve_xtb_method(
    method: str | XtbMethodSpec,
    *,
    reference: str | None = None,
    requested_products: tuple[str, ...] | None = None,
) -> XtbMethodIR:
    """Resolve an audited GFN manifest into a canonical compiler-only graph."""

    if isinstance(method, str):
        try:
            spec = XTB_METHOD_CATALOG[method]
        except KeyError as error:
            raise UnsupportedXtbMethod(f"unknown xTB method {method!r}") from error
    elif isinstance(method, XtbMethodSpec):
        spec = method
    else:
        raise TypeError("method must be a catalog name or XtbMethodSpec")

    if reference is not None:
        spec = replace(spec, reference=reference)
    if requested_products is not None:
        spec = replace(spec, requested_products=requested_products)

    if spec.model_flavor not in _SUPPORTED_MODELS:
        raise UnsupportedXtbMethod(
            f"unsupported xTB model flavor {spec.model_flavor!r}"
        )

    _validate_parameter_set(spec.model_flavor, spec.parameter_set)
    return XtbMethodIR(
        identifier=spec.identifier,
        model_flavor=spec.model_flavor,
        parameter_set=spec.parameter_set,
        reference=spec.reference,
        primitives=_primitives_for_model(spec.model_flavor),
        requested_products=spec.requested_products,
    )

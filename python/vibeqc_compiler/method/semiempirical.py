"""Family-neutral semiempirical method composition IR (#875).

This module owns only immutable scientific composition and identity. Family
resolvers remain responsible for constructing audited graphs; runtime SCC/SCF
iteration, mixing, eigensolvers, batching, and capability publication stay
outside this representation.
"""

from __future__ import annotations

from dataclasses import dataclass

from vibeqc_compiler.common.provenance import canonical_hash

SEMIEMPIRICAL_METHOD_IR_VERSION = "semiempirical-method-ir-v1"
SEMIEMPIRICAL_PARAMETER_SET_VERSION = "semiempirical-parameter-set-v1"
SEMIEMPIRICAL_PRIMITIVE_VERSION = "semiempirical-primitive-v1"
SEMIEMPIRICAL_STATE_FIELD_VERSION = "semiempirical-state-field-v1"
SEMIEMPIRICAL_PRODUCT_VERSION = "semiempirical-product-v1"

_ALLOWED_CATEGORIES = (
    "basis",
    "integral",
    "hamiltonian",
    "interaction",
    "state_contraction",
    "spin",
    "geometry_energy",
    "dispersion",
    "correction",
)
_ALLOWED_DTYPES = ("float64",)
_ALLOWED_SPIN_SEMANTICS = ("shared", "spin-resolved")
_GFN_STATE_LAYOUT = {
    "shell_charge": ("shell", 1),
    "charge": ("atom", 1),
    "dipole": ("atom", 3),
    "quadrupole": ("atom", 6),
    "magnetization": ("shell", 1),
}
_GFN_CATEGORY = {
    "basis_parameters": "basis",
    "coordination_number": "geometry_energy",
    "overlap_integrals": "integral",
    "multipole_integrals": "integral",
    "h0": "hamiltonian",
    "scc_electrostatics": "interaction",
    "spin_polarization": "spin",
    "hamiltonian": "hamiltonian",
    "repulsion": "geometry_energy",
    "halogen_correction": "correction",
    "dispersion": "dispersion",
}


class InvalidSemiempiricalMethod(ValueError):
    """The method graph is incomplete, ambiguous, or not canonical."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InvalidSemiempiricalMethod(
            f"{label} requires a non-empty canonical string"
        )
    return value


def _strings(values: object, label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise InvalidSemiempiricalMethod(f"{label} must be an immutable tuple")
    checked = tuple(_text(value, label) for value in values)
    if len(checked) != len(set(checked)):
        raise InvalidSemiempiricalMethod(f"{label} contains duplicate entries")
    return tuple(sorted(checked))


@dataclass(frozen=True, order=True)
class ParameterResource:
    """One immutable named scientific parameter resource."""

    role: str
    identity: str

    def __post_init__(self) -> None:
        _text(self.role, "parameter resource role")
        _text(self.identity, "parameter resource identity")

    def to_payload(self) -> dict:
        return {"role": self.role, "identity": self.identity}


@dataclass(frozen=True, order=True)
class StateField:
    """One fixed-point state field with explicit shape semantics."""

    name: str
    scope: str
    components: int = 1
    dtype: str = "float64"
    spin_semantics: str = "shared"
    version: str = SEMIEMPIRICAL_STATE_FIELD_VERSION

    def __post_init__(self) -> None:
        _text(self.name, "state field name")
        _text(self.scope, f"{self.name} state scope")
        if (
            isinstance(self.components, bool)
            or not isinstance(self.components, int)
            or self.components <= 0
        ):
            raise InvalidSemiempiricalMethod(
                "state components must be a positive integer"
            )
        if self.dtype not in _ALLOWED_DTYPES:
            raise InvalidSemiempiricalMethod(f"unsupported state dtype {self.dtype!r}")
        if self.spin_semantics not in _ALLOWED_SPIN_SEMANTICS:
            raise InvalidSemiempiricalMethod(
                f"unsupported state spin semantics {self.spin_semantics!r}"
            )
        if self.version != SEMIEMPIRICAL_STATE_FIELD_VERSION:
            raise InvalidSemiempiricalMethod(
                "unsupported semiempirical state schema version"
            )

    def to_payload(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "scope": self.scope,
            "components": self.components,
            "dtype": self.dtype,
            "spin_semantics": self.spin_semantics,
        }


@dataclass(frozen=True, order=True)
class ProductSpec:
    """Requested scientific product without runtime execution policy."""

    name: str
    derivative_order: int
    version: str = SEMIEMPIRICAL_PRODUCT_VERSION

    def __post_init__(self) -> None:
        _text(self.name, "product name")
        if (
            isinstance(self.derivative_order, bool)
            or not isinstance(self.derivative_order, int)
            or self.derivative_order < 0
        ):
            raise InvalidSemiempiricalMethod(
                "product derivative order must be non-negative"
            )
        if self.version != SEMIEMPIRICAL_PRODUCT_VERSION:
            raise InvalidSemiempiricalMethod(
                "unsupported semiempirical product version"
            )

    def to_payload(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "derivative_order": self.derivative_order,
        }


@dataclass(frozen=True)
class ParameterSetRef:
    """Audited parameter product identity shared by semiempirical families."""

    identifier: str
    schema: str
    source_identity: str
    supported_atomic_numbers: tuple[int, ...]
    resources: tuple[ParameterResource, ...]
    version: str = SEMIEMPIRICAL_PARAMETER_SET_VERSION

    def __post_init__(self) -> None:
        _text(self.identifier, "parameter-set identifier")
        _text(self.schema, "parameter-set schema")
        _text(self.source_identity, "parameter-set source identity")
        if self.version != SEMIEMPIRICAL_PARAMETER_SET_VERSION:
            raise InvalidSemiempiricalMethod(
                "unsupported semiempirical parameter-set version"
            )
        if not isinstance(self.supported_atomic_numbers, tuple):
            raise InvalidSemiempiricalMethod(
                "supported atomic numbers must be an immutable tuple"
            )
        numbers = self.supported_atomic_numbers
        if (
            not numbers
            or any(
                isinstance(number, bool)
                or not isinstance(number, int)
                or not 1 <= number <= 118
                for number in numbers
            )
            or len(numbers) != len(set(numbers))
        ):
            raise InvalidSemiempiricalMethod(
                "supported atomic numbers must be unique integers in [1, 118]"
            )
        object.__setattr__(self, "supported_atomic_numbers", tuple(sorted(numbers)))
        if not isinstance(self.resources, tuple) or not self.resources:
            raise InvalidSemiempiricalMethod(
                "parameter set requires immutable named resources"
            )
        if not all(
            isinstance(resource, ParameterResource) for resource in self.resources
        ):
            raise InvalidSemiempiricalMethod("unsupported parameter resource")
        roles = tuple(resource.role for resource in self.resources)
        if len(roles) != len(set(roles)):
            raise InvalidSemiempiricalMethod("parameter resource roles must be unique")
        object.__setattr__(self, "resources", tuple(sorted(self.resources)))

    def to_payload(self) -> dict:
        return {
            "version": self.version,
            "identifier": self.identifier,
            "schema": self.schema,
            "source_identity": self.source_identity,
            "supported_atomic_numbers": self.supported_atomic_numbers,
            "resources": tuple(resource.to_payload() for resource in self.resources),
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())


@dataclass(frozen=True)
class PrimitiveNode:
    """One family-owned scientific primitive in the common composition DAG."""

    node_id: str
    category: str
    equation_identity: str
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    state_reads: tuple[str, ...] = ()
    state_writes: tuple[str, ...] = ()
    parameter_resources: tuple[str, ...] = ()
    derivative_capabilities: tuple[str, ...] = ()
    fixed_point: bool = False
    version: str = SEMIEMPIRICAL_PRIMITIVE_VERSION

    def __post_init__(self) -> None:
        _text(self.node_id, "primitive node id")
        if self.category not in _ALLOWED_CATEGORIES:
            raise InvalidSemiempiricalMethod(
                f"unsupported primitive category {self.category!r}"
            )
        _text(self.equation_identity, f"{self.node_id} equation identity")
        if not isinstance(self.fixed_point, bool):
            raise TypeError("fixed_point must be boolean")
        if self.version != SEMIEMPIRICAL_PRIMITIVE_VERSION:
            raise InvalidSemiempiricalMethod(
                "unsupported semiempirical primitive version"
            )
        for field in (
            "requires",
            "produces",
            "state_reads",
            "state_writes",
            "parameter_resources",
            "derivative_capabilities",
        ):
            object.__setattr__(
                self,
                field,
                _strings(
                    getattr(self, field), f"{self.node_id} {field.replace('_', ' ')}"
                ),
            )

    def to_payload(self) -> dict:
        return {
            "version": self.version,
            "node_id": self.node_id,
            "category": self.category,
            "equation_identity": self.equation_identity,
            "requires": self.requires,
            "produces": self.produces,
            "state_reads": self.state_reads,
            "state_writes": self.state_writes,
            "parameter_resources": self.parameter_resources,
            "derivative_capabilities": self.derivative_capabilities,
            "fixed_point": self.fixed_point,
        }


def _canonical_nodes(nodes: tuple[PrimitiveNode, ...]) -> tuple[PrimitiveNode, ...]:
    if not isinstance(nodes, tuple) or not nodes:
        raise InvalidSemiempiricalMethod(
            "semiempirical graph requires immutable primitives"
        )
    if not all(isinstance(node, PrimitiveNode) for node in nodes):
        raise InvalidSemiempiricalMethod(
            "semiempirical graph contains an unsupported primitive"
        )
    by_id = {node.node_id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise InvalidSemiempiricalMethod(
            "semiempirical graph has duplicate primitive ids"
        )
    identifiers = set(by_id)
    for node in nodes:
        missing = set(node.requires) - identifiers
        if missing:
            raise InvalidSemiempiricalMethod(
                f"{node.node_id} depends on missing primitives {sorted(missing)!r}"
            )

    incoming = {node_id: set(node.requires) for node_id, node in by_id.items()}
    ordered: list[PrimitiveNode] = []
    ready = sorted(
        node_id for node_id, dependencies in incoming.items() if not dependencies
    )
    while ready:
        node_id = ready.pop(0)
        ordered.append(by_id[node_id])
        ordered_ids = {node.node_id for node in ordered}
        for candidate in sorted(incoming):
            if node_id not in incoming[candidate]:
                continue
            incoming[candidate].remove(node_id)
            if (
                not incoming[candidate]
                and candidate not in ordered_ids
                and candidate not in ready
            ):
                ready.append(candidate)
                ready.sort()
    if len(ordered) != len(nodes):
        raise InvalidSemiempiricalMethod(
            "semiempirical primitive graph contains a cycle"
        )
    return tuple(ordered)


@dataclass(frozen=True)
class SemiempiricalMethodIR:
    """Canonical family-neutral composition graph, not a runtime capability."""

    family: str
    model: str
    reference: str
    parameter_set: ParameterSetRef
    state_fields: tuple[StateField, ...]
    primitives: tuple[PrimitiveNode, ...]
    requested_products: tuple[ProductSpec, ...]
    version: str = SEMIEMPIRICAL_METHOD_IR_VERSION

    def __post_init__(self) -> None:
        _text(self.family, "semiempirical family")
        _text(self.model, "semiempirical model")
        _text(self.reference, "semiempirical reference")
        if self.version != SEMIEMPIRICAL_METHOD_IR_VERSION:
            raise InvalidSemiempiricalMethod(
                "unsupported SemiempiricalMethodIR version"
            )
        if not isinstance(self.parameter_set, ParameterSetRef):
            raise TypeError("SemiempiricalMethodIR requires a ParameterSetRef")

        if not isinstance(self.state_fields, tuple):
            raise InvalidSemiempiricalMethod("state fields must be an immutable tuple")
        if not all(isinstance(field, StateField) for field in self.state_fields):
            raise InvalidSemiempiricalMethod("unsupported state field")
        state_names = tuple(field.name for field in self.state_fields)
        if len(state_names) != len(set(state_names)):
            raise InvalidSemiempiricalMethod("state field names must be unique")
        object.__setattr__(self, "state_fields", tuple(sorted(self.state_fields)))

        nodes = _canonical_nodes(self.primitives)
        state_names_set = set(state_names)
        resource_roles = {resource.role for resource in self.parameter_set.resources}
        produced: dict[str, str] = {}
        for node in nodes:
            missing_states = (
                set(node.state_reads) | set(node.state_writes)
            ) - state_names_set
            if missing_states:
                raise InvalidSemiempiricalMethod(
                    f"{node.node_id} references unknown states {sorted(missing_states)!r}"
                )
            missing_resources = set(node.parameter_resources) - resource_roles
            if missing_resources:
                raise InvalidSemiempiricalMethod(
                    f"{node.node_id} references unknown parameter resources "
                    f"{sorted(missing_resources)!r}"
                )
            for output in node.produces:
                if output in produced:
                    owner = produced[output]
                    raise InvalidSemiempiricalMethod(
                        f"ambiguous producer for {output!r}: {owner!r} and {node.node_id!r}"
                    )
                produced[output] = node.node_id
        object.__setattr__(self, "primitives", nodes)

        if (
            not isinstance(self.requested_products, tuple)
            or not self.requested_products
        ):
            raise InvalidSemiempiricalMethod(
                "requested products must be a non-empty immutable tuple"
            )
        if not all(
            isinstance(product, ProductSpec) for product in self.requested_products
        ):
            raise InvalidSemiempiricalMethod("unsupported requested product")
        product_keys = tuple(
            (product.name, product.derivative_order)
            for product in self.requested_products
        )
        if len(product_keys) != len(set(product_keys)):
            raise InvalidSemiempiricalMethod("requested products contain duplicates")
        object.__setattr__(
            self, "requested_products", tuple(sorted(self.requested_products))
        )

    def to_payload(self) -> dict:
        return {
            "version": self.version,
            "family": self.family,
            "model": self.model,
            "reference": self.reference,
            "parameter_set": self.parameter_set.to_payload(),
            "state_fields": tuple(field.to_payload() for field in self.state_fields),
            "primitives": tuple(node.to_payload() for node in self.primitives),
            "requested_products": tuple(
                product.to_payload() for product in self.requested_products
            ),
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())


def _gfn_parameter_resources(parameter_set: object) -> tuple[ParameterResource, ...]:
    resources: list[ParameterResource] = []
    parameter_identity = parameter_set.identity
    for domain in ("basis", "orbital", "correction", "spin"):
        tables = getattr(parameter_set, f"{domain}_tables")
        for table in tables:
            role = f"{domain}:{table}"
            identity = canonical_hash(
                {
                    "xtb_parameter_set_identity": parameter_identity,
                    "domain": domain,
                    "table": table,
                }
            )
            resources.append(ParameterResource(role=role, identity=identity))
    return tuple(resources)


def semiempirical_from_xtb(method: object) -> SemiempiricalMethodIR:
    """Convert an audited GFN ``XtbMethodIR`` without changing xTB behavior.

    This is migration scaffolding only: ``resolve_xtb_method`` remains the
    authoritative GFN resolver and public/runtime capability is unchanged.
    """

    from vibeqc_compiler.method.xtb import XtbMethodIR

    if not isinstance(method, XtbMethodIR):
        raise TypeError("GFN adapter requires an audited XtbMethodIR")

    parameter_set = method.parameter_set
    common_parameters = ParameterSetRef(
        identifier=parameter_set.identifier,
        schema=parameter_set.version,
        source_identity=canonical_hash(
            {
                "source": parameter_set.source,
                "revision": parameter_set.revision,
                "legacy_identity": parameter_set.identity,
            }
        ),
        supported_atomic_numbers=parameter_set.supported_atomic_numbers,
        resources=_gfn_parameter_resources(parameter_set),
    )
    states = sorted(
        {
            state
            for primitive in method.primitives
            for state in primitive.state_requirements
        }
    )
    state_fields = tuple(
        StateField(
            name=state,
            scope=_GFN_STATE_LAYOUT[state][0],
            components=_GFN_STATE_LAYOUT[state][1],
        )
        for state in states
    )

    resources_by_domain = {
        domain: tuple(
            resource.role
            for resource in common_parameters.resources
            if resource.role.startswith(f"{domain}:")
        )
        for domain in ("basis", "orbital", "correction", "spin")
    }
    nodes = []
    for primitive in method.primitives:
        if primitive.kind not in _GFN_CATEGORY:
            raise InvalidSemiempiricalMethod(
                f"GFN adapter has no category for primitive {primitive.kind!r}"
            )
        parameter_resources = tuple(
            role
            for domain in primitive.parameter_domains
            for role in resources_by_domain[domain]
        )
        nodes.append(
            PrimitiveNode(
                node_id=primitive.kind,
                category=_GFN_CATEGORY[primitive.kind],
                equation_identity=canonical_hash(primitive.semantic_payload()),
                requires=primitive.requires,
                produces=(f"gfn:{primitive.kind}",),
                state_reads=primitive.state_requirements,
                parameter_resources=parameter_resources,
                derivative_capabilities=primitive.derivative_capabilities,
                fixed_point=primitive.self_consistent,
            )
        )

    products = tuple(
        ProductSpec(
            name=product,
            derivative_order=1 if product == "nuclear-gradient" else 0,
        )
        for product in method.requested_products
    )
    return SemiempiricalMethodIR(
        family="gfn-xtb",
        model=method.model_flavor,
        reference=method.reference,
        parameter_set=common_parameters,
        state_fields=state_fields,
        primitives=tuple(nodes),
        requested_products=products,
    )

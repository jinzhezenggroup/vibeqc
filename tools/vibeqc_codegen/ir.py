"""Backend-independent mathematical integral representation.

``IntegralIR`` contains scientific intent only. Operator center semantics,
requested derivatives, exact invariants, and consumer-directed contractions
are represented explicitly; accelerator execution geometry is added later by
the backend schedule IR.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .blocks import RawBlock, WeightedDerivative
from .shell_signature import ShellSignature, checked_index
from .shell_spec import ShellClassSpec


class KernelConsumer(str, Enum):
    """Compatibility names for observables emitted by a shell kernel."""

    FOCK = "fock"
    FORCE = "force"


class OperatorFamily(str, Enum):
    """Backend-neutral integral operator families known to the compiler."""

    OVERLAP = "overlap"
    KINETIC = "kinetic"
    NUCLEAR_ATTRACTION = "nuclear_attraction"
    COULOMB_METRIC = "coulomb_metric"
    THREE_CENTER_ERI = "three_center_eri"
    FOUR_CENTER_ERI = "four_center_eri"


class DensityModel(str, Enum):
    """Density layouts accepted by a direct contraction."""

    RHF = "rhf"
    UHF = "uhf"


class ContractionConsumer(str, Enum):
    """Mathematical contraction requested from the integral values."""

    DIRECT_FOCK = "direct_fock"
    DIRECT_FORCE = "direct_force"
    RAW_BLOCK = "raw_block"
    WEIGHTED_DERIVATIVE = "weighted_derivative"


class ContractionOutput(str, Enum):
    """Logical output produced by a contracted integral kernel."""

    FOCK_MATRIX = "fock_matrix"
    ATOMIC_FORCE = "atomic_force"
    NUCLEAR_DERIVATIVE = "nuclear_derivative"


@dataclass(frozen=True, slots=True)
class NuclearCoordinates:
    """Nuclear-coordinate parameters selected for differentiation.

    ``"all"`` is resolved against the center inventory declared by the
    operator. Explicit center tuples are useful for one-electron attraction
    and future density-fitting operators, whose parameter sets differ from a
    four-center ERI.
    """

    centers: tuple[int, ...] | str = "all"

    def __post_init__(self) -> None:
        if self.centers == "all":
            return
        if not isinstance(self.centers, tuple) or not self.centers:
            raise ValueError("nuclear coordinates require 'all' or a center tuple")
        for center in self.centers:
            checked_index(center, "nuclear-coordinate center")
        if len(set(self.centers)) != len(self.centers):
            raise ValueError("nuclear-coordinate centers must be unique")

    def resolve(self, operator_centers: tuple[int, ...]) -> tuple[int, ...]:
        """Resolve this parameter selection against an operator inventory."""

        selected = operator_centers if self.centers == "all" else self.centers
        if not set(selected) <= set(operator_centers):
            raise ValueError("derivative parameters are not centers of the operator")
        return tuple(selected)


_ALL_NUCLEAR_COORDINATES = NuclearCoordinates()


@dataclass(frozen=True, slots=True)
class TranslationInvariant:
    """Declare that simultaneous translation leaves an operator unchanged.

    The dependent center is lowering policy attached to the invariant rather
    than an implicit FORCE convention. When it is omitted, the compiler uses
    the final invariant center, preserving a deterministic compact basis while
    allowing other operators to declare different center inventories.
    """

    parameters: NuclearCoordinates = _ALL_NUCLEAR_COORDINATES
    dependent_center: int | None = None

    def __post_init__(self) -> None:
        if self.dependent_center is not None:
            checked_index(self.dependent_center, "translation dependent center")

    def recovered_center(
        self,
        requested_centers: tuple[int, ...],
        operator_centers: tuple[int, ...],
    ) -> int | None:
        """Return the derivative center recoverable from this invariant."""

        invariant_centers = self.parameters.resolve(operator_centers)
        if not set(invariant_centers) <= set(requested_centers):
            return None
        dependent = (
            invariant_centers[-1]
            if self.dependent_center is None
            else self.dependent_center
        )
        if dependent not in invariant_centers:
            raise ValueError("translation dependent center is outside the invariant")
        return dependent


@dataclass(frozen=True, slots=True)
class NuclearCenter:
    """External attraction position with nuclear charge, never a Gaussian shell.

    The electronic operator is ``-charge / |r - R_center|`` in atomic units;
    this physical minus sign is part of the integral, not a weight convention.
    """

    center: int
    charge: float

    def __post_init__(self) -> None:
        checked_index(self.center, "nuclear center")
        if (
            isinstance(self.charge, bool)
            or not isfinite(self.charge)
            or self.charge <= 0
        ):
            raise ValueError("nuclear charge must be finite and positive")
        object.__setattr__(self, "charge", float(self.charge))


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Operator family, mathematical centers, and exact invariants."""

    family: OperatorFamily | str
    centers: tuple[int, ...]
    invariants: tuple[TranslationInvariant, ...] = ()
    external_centers: tuple[NuclearCenter, ...] = ()
    permutations: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", OperatorFamily(self.family))
        object.__setattr__(self, "centers", tuple(self.centers))
        object.__setattr__(self, "invariants", tuple(self.invariants))
        object.__setattr__(self, "external_centers", tuple(self.external_centers))
        object.__setattr__(
            self, "permutations", tuple(tuple(p) for p in self.permutations)
        )
        if not self.centers or len(set(self.centers)) != len(self.centers):
            raise ValueError("operator centers must be non-empty and unique")
        for center in self.centers:
            checked_index(center, "operator center")
        attraction = self.family == OperatorFamily.NUCLEAR_ATTRACTION
        if len(self.centers) != len(self.basis_roles) + int(attraction):
            raise ValueError("operator center inventory has the wrong center count")
        if attraction:
            if (
                len(self.external_centers) != 1
                or self.external_centers[0].center not in self.centers
            ):
                raise ValueError(
                    "attraction requires one external nuclear center in its inventory"
                )
        elif self.external_centers:
            raise ValueError("this operator does not accept external nuclear centers")
        if len(self.invariants) > 1:
            # All families here have one simultaneous-translation relation.
            # Recovering two centers from it would create a circular system.
            raise ValueError(
                "duplicate translation relations cannot recover multiple centers"
            )
        for invariant in self.invariants:
            if set(invariant.parameters.resolve(self.centers)) != set(self.centers):
                raise ValueError("translation must include all operator centers")
            if invariant.dependent_center is not None:
                invariant.recovered_center(self.centers, self.centers)
        if len(set(self.permutations)) != len(self.permutations):
            raise ValueError("operator contains duplicate permutations")
        n = len(self.basis_roles)
        allowed = {tuple(range(n)), (1, 0, *range(2, n))}
        if self.family == OperatorFamily.FOUR_CENTER_ERI:
            allowed = {
                (0, 1, 2, 3),
                (1, 0, 2, 3),
                (0, 1, 3, 2),
                (1, 0, 3, 2),
                (2, 3, 0, 1),
                (3, 2, 0, 1),
                (2, 3, 1, 0),
                (3, 2, 1, 0),
            }
        for permutation in self.permutations:
            for slot in permutation:
                checked_index(slot, "permutation slot")
            if permutation not in allowed:
                raise ValueError(
                    "illegal basis-slot permutation for operator symmetry/auxiliary roles"
                )

    @property
    def basis_roles(self) -> tuple[str, ...]:
        """Declare orbital/auxiliary tensor slots independently of atom identity."""
        if self.family == OperatorFamily.COULOMB_METRIC:
            return ("auxiliary", "auxiliary")
        if self.family == OperatorFamily.THREE_CENTER_ERI:
            return ("orbital", "orbital", "auxiliary")
        if self.family == OperatorFamily.FOUR_CENTER_ERI:
            return ("orbital",) * 4
        return ("orbital", "orbital")

    def nuclear_derivative(
        self,
        *,
        order: int = 1,
        parameters: NuclearCoordinates = _ALL_NUCLEAR_COORDINATES,
    ) -> DerivativeSpec:
        """Construct a derivative using invariants declared by this operator."""

        parameters.resolve(self.centers)
        return DerivativeSpec(
            order=order,
            parameters=parameters,
            invariants=self.invariants,
        )


@dataclass(frozen=True, slots=True)
class DerivativeSpec:
    """Derivative order, parameters, and exact recovery relations."""

    order: int
    parameters: NuclearCoordinates
    invariants: tuple[TranslationInvariant, ...] = ()

    def __post_init__(self) -> None:
        checked_index(self.order, "derivative order", minimum=1)
        object.__setattr__(self, "invariants", tuple(self.invariants))
        if len(self.invariants) > 1:
            raise ValueError("duplicate derivative translation recovery relations")

    def requested_centers(self, operator: OperatorSpec) -> tuple[int, ...]:
        """Return every logical nuclear center requested by the derivative."""

        return self.parameters.resolve(operator.centers)

    def recovered_centers(self, operator: OperatorSpec) -> tuple[int, ...]:
        """Return centers reconstructed exactly instead of differentiated."""

        requested = self.requested_centers(operator)
        if self.order != 1:
            raise ValueError("translation recovery exposes only order-one derivatives")
        recovered: list[int] = []
        for invariant in self.invariants:
            center = invariant.recovered_center(requested, operator.centers)
            if center is not None and center not in recovered:
                recovered.append(center)
        return tuple(recovered)

    def independent_centers(self, operator: OperatorSpec) -> tuple[int, ...]:
        """Return the minimal center set that lowering must differentiate."""

        recovered = set(self.recovered_centers(operator))
        return tuple(
            center
            for center in self.requested_centers(operator)
            if center not in recovered
        )


@dataclass(frozen=True, slots=True)
class ContractionSpec:
    """Consumer-directed density contraction and its logical output."""

    consumer: ContractionConsumer | str
    density: frozenset[DensityModel | str] | tuple[DensityModel | str, ...] | str
    output: ContractionOutput | str

    def __post_init__(self) -> None:
        consumer = ContractionConsumer(self.consumer)
        if consumer not in (
            ContractionConsumer.DIRECT_FOCK,
            ContractionConsumer.DIRECT_FORCE,
        ):
            raise ValueError("use RawBlock or WeightedDerivative for non-HF consumers")
        output = ContractionOutput(self.output)
        density_items = (
            self.density.split("|") if isinstance(self.density, str) else self.density
        )
        density = frozenset(DensityModel(item) for item in density_items)
        object.__setattr__(self, "consumer", consumer)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "output", output)
        if not density:
            raise ValueError("a contraction requires at least one density model")
        if consumer == ContractionConsumer.DIRECT_FOCK:
            valid_outputs = (ContractionOutput.FOCK_MATRIX,)
        else:
            # The derivative order is carried by ``DerivativeSpec`` on the
            # surrounding IntegralIR, so the contraction only declares the
            # tensor family here. IntegralIR performs the order-aware check.
            valid_outputs = (
                ContractionOutput.ATOMIC_FORCE,
                ContractionOutput.NUCLEAR_DERIVATIVE,
            )
        if output not in valid_outputs:
            raise ValueError(
                f"{consumer.value} contraction requires one of "
                f"{', '.join(item.value for item in valid_outputs)} outputs"
            )

    @property
    def kernel_consumer(self) -> KernelConsumer:
        """Return the established generated-kernel registry category."""

        if self.consumer == ContractionConsumer.DIRECT_FOCK:
            return KernelConsumer.FOCK
        return KernelConsumer.FORCE


FOUR_CENTER_ERI_OPERATOR = OperatorSpec(
    family=OperatorFamily.FOUR_CENTER_ERI,
    centers=(0, 1, 2, 3),
    invariants=(TranslationInvariant(),),
)

_DENSITY_MODELS = frozenset(DensityModel)
_CONTRACTION_BY_CONSUMER = {
    KernelConsumer.FOCK: ContractionSpec(
        consumer=ContractionConsumer.DIRECT_FOCK,
        density=_DENSITY_MODELS,
        output=ContractionOutput.FOCK_MATRIX,
    ),
    KernelConsumer.FORCE: ContractionSpec(
        consumer=ContractionConsumer.DIRECT_FORCE,
        density=_DENSITY_MODELS,
        output=ContractionOutput.ATOMIC_FORCE,
    ),
}


@dataclass(frozen=True, slots=True)
class IntegralIR:
    """Operator-independent scientific intent with explicit consumer contracts.

    The mathematical IR carries explicit nuclear derivative orders. The CUDA
    backend currently lowers only first derivatives; that backend limitation is
    validated at the CUDA kernel boundary without discarding higher-order
    intent or treating FORCE as an implicit derivative specification.
    """

    spec: ShellClassSpec | ShellSignature
    operator: OperatorSpec
    derivative: DerivativeSpec | None
    contractions: tuple[ContractionSpec | RawBlock | WeightedDerivative, ...]
    recurrence: str = "subset_wick"

    def __post_init__(self) -> None:
        if not isinstance(self.spec, (ShellClassSpec, ShellSignature)):
            raise TypeError(
                "integral spec must be a shell signature or legacy shell class"
            )
        if any(
            not isinstance(c, (ContractionSpec, RawBlock, WeightedDerivative))
            for c in self.contractions
        ):
            raise TypeError(
                "integral consumers must use a declared contraction contract"
            )
        signature = self.signature
        if len(signature.shells) != len(self.operator.basis_roles):
            raise ValueError("operator shell count does not match the shell signature")
        if tuple(s.role for s in signature.shells) != self.operator.basis_roles:
            raise ValueError(
                "operator orbital/auxiliary roles do not match basis slots"
            )
        inventory = tuple(s.center for s in signature.shells) + tuple(
            c.center for c in self.operator.external_centers
        )
        if len(set(inventory)) != len(inventory) or set(inventory) != set(
            self.operator.centers
        ):
            raise ValueError(
                "operator center inventory does not match basis/external centers"
            )
        if tuple(b.center for b in signature.center_bindings) != self.operator.centers:
            raise ValueError(
                "center inventory bindings must follow operator center order"
            )
        if not self.contractions:
            raise ValueError("an integral IR requires at least one contraction")
        object.__setattr__(self, "contractions", tuple(self.contractions))
        consumers = tuple(item.consumer for item in self.contractions)
        if len(set(consumers)) != len(consumers):
            raise ValueError("integral IR contains duplicate contraction consumers")
        direct = tuple(
            item for item in self.contractions if isinstance(item, ContractionSpec)
        )
        if direct and self.operator.family != OperatorFamily.FOUR_CENTER_ERI:
            raise ValueError("direct HF consumers require four-center ERIs")
        if self.recurrence not in (
            "hermite",
            "subset_wick",
            "rys1",
            "rys2",
            "rys3",
            "rys4",
            "rys5",
        ):
            raise ValueError(f"unsupported integral recurrence {self.recurrence!r}")
        if self.recurrence == "hermite" and self.operator.family not in (
            OperatorFamily.OVERLAP,
            OperatorFamily.KINETIC,
            OperatorFamily.NUCLEAR_ATTRACTION,
        ):
            # The one-electron lowering owns this recurrence identity. Do not
            # let a new label imply support for an unimplemented ERI/DF route.
            raise ValueError(
                "Hermite recurrence currently requires a one-electron operator"
            )

        force_requested = any(
            item.kernel_consumer == KernelConsumer.FORCE for item in direct
        )
        weighted_requested = any(
            isinstance(item, WeightedDerivative) for item in self.contractions
        )
        if (force_requested or weighted_requested) and self.derivative is None:
            raise ValueError(
                "direct-force or weighted-derivative contraction requires a derivative spec"
            )
        if self.derivative is not None:
            requested = self.derivative.requested_centers(self.operator)
            if not set(self.derivative.invariants) <= set(self.operator.invariants):
                raise ValueError(
                    "derivative recovery invariants must be declared by the operator"
                )
            if self.derivative.order == 1:
                self.derivative.recovered_centers(self.operator)
            if force_requested:
                if requested != self.operator.centers:
                    raise ValueError(
                        "current direct-force output requires every operator center"
                    )
                expected_output = (
                    ContractionOutput.ATOMIC_FORCE
                    if self.derivative.order == 1
                    else ContractionOutput.NUCLEAR_DERIVATIVE
                )
                for contraction in direct:
                    if (
                        contraction.kernel_consumer == KernelConsumer.FORCE
                        and contraction.output != expected_output
                    ):
                        raise ValueError(
                            f"derivative order {self.derivative.order} requires {expected_output.value} output"
                        )
            elif not weighted_requested and not any(
                isinstance(item, RawBlock) for item in self.contractions
            ):
                raise ValueError("a derivative spec requires a derivative contraction")
        for contraction in self.contractions:
            if isinstance(contraction, (RawBlock, WeightedDerivative)):
                self._validate_block_consumer(contraction)

        if self.recurrence.startswith("rys"):
            if self.operator.family in (OperatorFamily.OVERLAP, OperatorFamily.KINETIC):
                raise ValueError("a Rys recurrence requires a Coulomb operator")
            required = self.required_rys_roots
            selected = int(self.recurrence.removeprefix("rys"))
            if selected != required:
                raise ValueError(
                    f"{getattr(self.spec, 'name', self.operator.family.value)} lowering requires rys{required}, "
                    f"not {self.recurrence}"
                )

    @property
    def signature(self) -> ShellSignature:
        """Expose legacy quartet classes through the operator-independent adapter."""
        return (
            ShellSignature.from_shell_class(self.spec)
            if isinstance(self.spec, ShellClassSpec)
            else self.spec
        )

    def _validate_block_consumer(self, consumer: RawBlock | WeightedDerivative) -> None:
        """Check scientific tensor axes; per-tile budgets are checked on requests."""
        signature = self.signature
        derivative_axes = ("center", "xyz") if self.derivative is not None else ()
        if isinstance(consumer, RawBlock):
            layout = consumer.layout
            expected = derivative_axes + signature.tensor_indices
            if self.derivative is not None and layout.shape[:2] != (
                len(self.requested_derivative_centers),
                3,
            ):
                raise ValueError(
                    "raw derivative tensor index layout requires requested center and xyz extents"
                )
        else:
            layout = consumer.weights.layout
            expected = signature.tensor_indices
            output_indices = (
                ("atom", "xyz")
                if consumer.output == ContractionOutput.ATOMIC_FORCE
                else ("center", "xyz")
            )
            if (
                consumer.output_layout.indices != output_indices
                or consumer.output_layout.shape[-1] != 3
            ):
                raise ValueError(
                    "weighted output tensor index order must be center/atom then xyz"
                )
            if (
                consumer.output == ContractionOutput.NUCLEAR_DERIVATIVE
                and consumer.output_layout.shape[0]
                != len(self.requested_derivative_centers)
            ):
                raise ValueError(
                    "weighted output layout requires every requested center"
                )
        if layout.indices != expected:
            raise ValueError(f"tensor index order must be {expected}")
        if any(
            n > full
            for n, full in zip(
                layout.shape[-len(signature.shells) :], signature.component_shape
            )
        ):
            raise ValueError("consumer tensor shape exceeds shell bounds")

    @property
    def consumers(self) -> frozenset[KernelConsumer]:
        """Return compatibility registry categories for existing call sites."""

        return frozenset(
            item.kernel_consumer
            for item in self.contractions
            if item.kernel_consumer is not None
        )

    @property
    def independent_derivative_centers(self) -> tuple[int, ...]:
        """Return centers that must be evaluated by derivative lowering."""

        if self.derivative is None:
            return ()
        return self.derivative.independent_centers(self.operator)

    @property
    def requested_derivative_centers(self) -> tuple[int, ...]:
        """Return every logical derivative center before invariant recovery."""

        if self.derivative is None:
            return ()
        return self.derivative.requested_centers(self.operator)

    @property
    def recovered_derivative_centers(self) -> tuple[int, ...]:
        """Return centers reconstructed from declared exact invariants."""

        if self.derivative is None:
            return ()
        return self.derivative.recovered_centers(self.operator)

    @property
    def independent_force_centers(self) -> tuple[int, ...]:
        """Compatibility alias for first-force lowering call sites."""

        return self.independent_derivative_centers

    @property
    def value_coulomb_order(self) -> int:
        """Largest Cartesian Coulomb derivative for a Coulomb-family operator."""

        if self.operator.family in (OperatorFamily.OVERLAP, OperatorFamily.KINETIC):
            raise ValueError("Coulomb recurrence bounds require a Coulomb operator")
        return sum(self.spec.angular)

    @property
    def maximum_coulomb_order(self) -> int:
        """Largest derivative required by every requested contraction."""

        derivative_order = 0 if self.derivative is None else self.derivative.order
        return self.value_coulomb_order + derivative_order

    @property
    def required_rys_roots(self) -> int:
        """Return the fixed-root count implied by value/derivative order."""

        return self.maximum_coulomb_order // 2 + 1


def build_integral_ir(
    spec: ShellClassSpec | ShellSignature,
    consumers: tuple[KernelConsumer | str, ...] | None = None,
    *,
    operator: OperatorSpec = FOUR_CENTER_ERI_OPERATOR,
    derivative: DerivativeSpec | None = None,
    contractions: tuple[ContractionSpec | RawBlock | WeightedDerivative, ...]
    | None = None,
    recurrence: str = "subset_wick",
) -> IntegralIR:
    """Normalize legacy consumers or explicit contractions into one IR.

    Existing callers may continue to request ``KernelConsumer`` values. New
    compiler stages can supply explicit derivative and contraction records;
    both paths produce the same backend-neutral representation.
    """

    if contractions is not None and consumers is not None:
        normalized = frozenset(KernelConsumer(item) for item in consumers)
        explicit = frozenset(item.kernel_consumer for item in contractions)
        if normalized != explicit:
            raise ValueError("consumer and contraction specifications disagree")
    force_requested = (
        any(item.kernel_consumer == KernelConsumer.FORCE for item in contractions)
        if contractions is not None
        else any(
            KernelConsumer(item) == KernelConsumer.FORCE
            for item in ((KernelConsumer.FORCE,) if consumers is None else consumers)
        )
    )
    selected_derivative = derivative
    if force_requested and selected_derivative is None:
        selected_derivative = operator.nuclear_derivative()

    if contractions is None:
        requested_consumers = (
            (KernelConsumer.FORCE,) if consumers is None else consumers
        )
        normalized_consumers = tuple(
            KernelConsumer(item) for item in requested_consumers
        )
        contractions = tuple(
            (
                ContractionSpec(
                    consumer=ContractionConsumer.DIRECT_FORCE,
                    density=_DENSITY_MODELS,
                    output=(
                        ContractionOutput.ATOMIC_FORCE
                        if selected_derivative is None or selected_derivative.order == 1
                        else ContractionOutput.NUCLEAR_DERIVATIVE
                    ),
                )
                if item == KernelConsumer.FORCE
                else _CONTRACTION_BY_CONSUMER[item]
            )
            for item in normalized_consumers
        )

    return IntegralIR(
        spec=spec,
        operator=operator,
        derivative=selected_derivative,
        contractions=contractions,
        recurrence=recurrence,
    )

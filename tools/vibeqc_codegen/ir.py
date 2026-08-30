"""Backend-independent mathematical integral representation.

``IntegralIR`` contains scientific intent only. Operator center semantics,
requested derivatives, exact invariants, and consumer-directed contractions
are represented explicitly; accelerator execution geometry is added later by
the backend schedule IR.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
    FOUR_CENTER_ERI = "four_center_eri"


class DensityModel(str, Enum):
    """Density layouts accepted by a direct contraction."""

    RHF = "rhf"
    UHF = "uhf"


class ContractionConsumer(str, Enum):
    """Mathematical contraction requested from the integral values."""

    DIRECT_FOCK = "direct_fock"
    DIRECT_FORCE = "direct_force"


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
        if any(not isinstance(center, int) or center < 0 for center in self.centers):
            raise ValueError("nuclear-coordinate centers must be non-negative integers")
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
class OperatorSpec:
    """Operator family, mathematical centers, and exact invariants."""

    family: OperatorFamily | str
    centers: tuple[int, ...]
    invariants: tuple[TranslationInvariant, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", OperatorFamily(self.family))
        if not self.centers or len(set(self.centers)) != len(self.centers):
            raise ValueError("operator centers must be non-empty and unique")
        if any(center < 0 for center in self.centers):
            raise ValueError("operator centers must be non-negative")
        for invariant in self.invariants:
            invariant.parameters.resolve(self.centers)
            if invariant.dependent_center is not None:
                invariant.recovered_center(self.centers, self.centers)

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
        if self.order < 1:
            raise ValueError("derivative order must be positive")

    def requested_centers(self, operator: OperatorSpec) -> tuple[int, ...]:
        """Return every logical nuclear center requested by the derivative."""

        return self.parameters.resolve(operator.centers)

    def recovered_centers(self, operator: OperatorSpec) -> tuple[int, ...]:
        """Return centers reconstructed exactly instead of differentiated."""

        requested = self.requested_centers(operator)
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
    """Mathematical definition shared by Fock and force contractions.

    The mathematical IR carries explicit nuclear derivative orders. The CUDA
    backend currently lowers only first derivatives; that backend limitation is
    validated at the CUDA kernel boundary without discarding higher-order
    intent or treating FORCE as an implicit derivative specification.
    """

    spec: ShellClassSpec
    operator: OperatorSpec
    derivative: DerivativeSpec | None
    contractions: tuple[ContractionSpec, ...]
    recurrence: str = "subset_wick"

    def __post_init__(self) -> None:
        if self.operator.family != OperatorFamily.FOUR_CENTER_ERI:
            raise ValueError(
                "shell-quartet lowering currently requires a four-center ERI"
            )
        if len(self.operator.centers) != len(self.spec.angular):
            raise ValueError(
                "operator center inventory does not match the shell quartet"
            )
        if not self.contractions:
            raise ValueError("an integral IR requires at least one contraction")
        consumers = tuple(item.kernel_consumer for item in self.contractions)
        if len(set(consumers)) != len(consumers):
            raise ValueError("integral IR contains duplicate contraction consumers")
        if self.recurrence not in (
            "subset_wick",
            "rys2",
            "rys3",
            "rys4",
            "rys5",
        ):
            raise ValueError(f"unsupported integral recurrence {self.recurrence!r}")

        force_requested = KernelConsumer.FORCE in consumers
        if force_requested:
            if self.derivative is None:
                raise ValueError("direct-force contraction requires a derivative spec")
            requested = self.derivative.requested_centers(self.operator)
            if requested != self.operator.centers:
                raise ValueError(
                    "current direct-force output requires every operator center"
                )
            if not set(self.derivative.invariants) <= set(self.operator.invariants):
                raise ValueError(
                    "derivative recovery invariants must be declared by the operator"
                )
            self.derivative.recovered_centers(self.operator)
            expected_output = (
                ContractionOutput.ATOMIC_FORCE
                if self.derivative.order == 1
                else ContractionOutput.NUCLEAR_DERIVATIVE
            )
            for contraction in self.contractions:
                if (
                    contraction.kernel_consumer == KernelConsumer.FORCE
                    and contraction.output != expected_output
                ):
                    raise ValueError(
                        f"derivative order {self.derivative.order} requires "
                        f"{expected_output.value} output"
                    )
        elif self.derivative is not None:
            raise ValueError("a derivative spec requires a derivative contraction")

        if self.recurrence.startswith("rys"):
            if not force_requested:
                raise ValueError(
                    "direct Rys lowering currently requires a force contraction"
                )
            required = self.required_rys_roots
            selected = int(self.recurrence.removeprefix("rys"))
            if selected != required:
                raise ValueError(
                    f"{self.spec.name} first-derivative lowering requires rys{required}, "
                    f"not {self.recurrence}"
                )

    @property
    def consumers(self) -> frozenset[KernelConsumer]:
        """Return compatibility registry categories for existing call sites."""

        return frozenset(item.kernel_consumer for item in self.contractions)

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
        """Largest Cartesian Coulomb derivative needed for ERI values."""

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
    spec: ShellClassSpec,
    consumers: tuple[KernelConsumer | str, ...] | None = None,
    *,
    operator: OperatorSpec = FOUR_CENTER_ERI_OPERATOR,
    derivative: DerivativeSpec | None = None,
    contractions: tuple[ContractionSpec, ...] | None = None,
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
    force_requested = any(
        item.kernel_consumer == KernelConsumer.FORCE
        for item in contractions
    ) if contractions is not None else any(
        KernelConsumer(item) == KernelConsumer.FORCE
        for item in ((KernelConsumer.FORCE,) if consumers is None else consumers)
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
                        if selected_derivative is None
                        or selected_derivative.order == 1
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

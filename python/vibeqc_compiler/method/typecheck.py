"""Static execution-type checking for MethodIR before backend lowering.

MethodIR remains the mathematical identity. This module validates the selected
execution contract (dtype, spin, logical feature shape, derivative order and
backend capability) without promoting backend availability into method
representability.
"""

from __future__ import annotations

from dataclasses import dataclass

from vibeqc_compiler.common.provenance import canonical_hash

from .spec import MethodIR, UnsupportedMethod

_DTYPES = ("float32", "float64")
_SPINS = ("unpolarized", "polarized")
_INGREDIENT_COMPONENTS = {
    "unpolarized": {"rho": 1, "sigma": 1, "tau": 1},
    "polarized": {"rho": 2, "sigma": 3, "tau": 2},
}


class MethodTypeError(UnsupportedMethod):
    """A representable MethodIR is illegal for the requested execution type."""


def _unique_tuple(values, name, *, allow_empty=False):
    if not isinstance(values, tuple) or (not values and not allow_empty):
        qualifier = (
            "an immutable tuple" if allow_empty else "a nonempty immutable tuple"
        )
        raise TypeError(f"{name} must be {qualifier}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


@dataclass(frozen=True)
class FeatureType:
    """Logical per-point feature type presented to a semilocal primitive.

    ``shape`` is the component shape at one grid point, not a runtime batch
    extent. For example polarized sigma has logical shape ``(3,)``.
    """

    ingredient: str
    dtype: str
    shape: tuple[int, ...]
    spin: str

    def __post_init__(self):
        if self.ingredient not in ("rho", "sigma", "tau"):
            raise MethodTypeError(f"unknown MethodIR ingredient {self.ingredient!r}")
        if self.dtype not in _DTYPES:
            raise MethodTypeError(f"unsupported MethodIR dtype {self.dtype!r}")
        if self.spin not in _SPINS:
            raise MethodTypeError(f"unsupported MethodIR spin {self.spin!r}")
        if (
            not isinstance(self.shape, tuple)
            or not self.shape
            or any(type(n) is not int or n <= 0 for n in self.shape)
        ):
            raise MethodTypeError(
                "feature shape must be a nonempty positive integer tuple"
            )

    def to_payload(self):
        return {
            "ingredient": self.ingredient,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "spin": self.spin,
        }


@dataclass(frozen=True)
class BackendCapability:
    """Finite lowering capability supplied by the selected backend owner."""

    backend: str
    dtypes: tuple[str, ...]
    spins: tuple[str, ...]
    derivative_orders: tuple[int, ...]
    ingredients: tuple[str, ...]
    operators: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend capability requires a nonempty name")
        for field in (
            "dtypes",
            "spins",
            "derivative_orders",
            "ingredients",
            "operators",
        ):
            _unique_tuple(
                getattr(self, field),
                field,
                allow_empty=field in ("ingredients", "operators"),
            )
        if any(dtype not in _DTYPES for dtype in self.dtypes):
            raise ValueError("backend capability contains an unsupported dtype")
        if any(spin not in _SPINS for spin in self.spins):
            raise ValueError("backend capability contains an unsupported spin")
        if any(
            type(order) is not int or order < 0 or order > 2
            for order in self.derivative_orders
        ):
            raise ValueError("backend derivative orders must be integers in [0, 2]")
        if any(name not in ("rho", "sigma", "tau") for name in self.ingredients):
            raise ValueError("backend capability contains an unknown ingredient")
        if any(not isinstance(name, str) or not name for name in self.operators):
            raise ValueError("backend capability contains an invalid operator")


@dataclass(frozen=True)
class TypedMethodIR:
    """MethodIR paired with a validated execution type.

    ``identity`` is an execution/schedule identity. It deliberately does not
    replace ``method.identity``, which remains the mathematical method identity.
    """

    method: MethodIR
    backend: str
    dtype: str
    derivative_order: int
    features: tuple[FeatureType, ...]

    def __post_init__(self):
        if not isinstance(self.method, MethodIR):
            raise TypeError("typed method requires MethodIR")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("typed method requires a backend")
        if self.dtype not in _DTYPES:
            raise MethodTypeError(f"unsupported MethodIR dtype {self.dtype!r}")
        if (
            type(self.derivative_order) is not int
            or not 0 <= self.derivative_order <= 2
        ):
            raise MethodTypeError("derivative order must be an integer in [0, 2]")
        if not isinstance(self.features, tuple):
            raise TypeError("typed method features must be an immutable tuple")
        if not all(isinstance(feature, FeatureType) for feature in self.features):
            raise TypeError("typed method features require FeatureType entries")
        expected = {
            name: (_INGREDIENT_COMPONENTS[self.method.spin][name],)
            for name in self.method.requirements["ingredients"]
        }
        actual = {}
        for feature in self.features:
            if feature.ingredient in actual:
                raise MethodTypeError(f"duplicate feature type {feature.ingredient!r}")
            actual[feature.ingredient] = feature
        if set(actual) != set(expected):
            missing = tuple(sorted(set(expected) - set(actual)))
            extra = tuple(sorted(set(actual) - set(expected)))
            raise MethodTypeError(
                f"feature binding mismatch: missing={missing}, extra={extra}"
            )
        for name, shape in expected.items():
            feature = actual[name]
            if feature.spin != self.method.spin:
                raise MethodTypeError(
                    f"{name} spin {feature.spin!r} does not match "
                    f"MethodIR spin {self.method.spin!r}"
                )
            if feature.dtype != self.dtype:
                raise MethodTypeError(
                    f"{name} dtype {feature.dtype!r} requires an explicit cast "
                    f"to {self.dtype!r}"
                )
            if feature.shape != shape:
                raise MethodTypeError(
                    f"{name} shape {feature.shape!r} does not match "
                    f"logical shape {shape!r}"
                )

    def to_payload(self):
        return {
            "method_identity": self.method.identity,
            "backend": self.backend,
            "dtype": self.dtype,
            "derivative_order": self.derivative_order,
            "features": [feature.to_payload() for feature in self.features],
        }

    @property
    def identity(self):
        return canonical_hash(self.to_payload())


def infer_feature_types(method: MethodIR, *, dtype: str) -> tuple[FeatureType, ...]:
    """Infer the exact logical feature types required by a MethodIR."""

    if not isinstance(method, MethodIR):
        raise TypeError("feature inference requires MethodIR")
    if dtype not in _DTYPES:
        raise MethodTypeError(f"unsupported MethodIR dtype {dtype!r}")
    component_counts = _INGREDIENT_COMPONENTS[method.spin]
    return tuple(
        FeatureType(name, dtype, (component_counts[name],), method.spin)
        for name in method.requirements["ingredients"]
    )


def verify_method_ir(
    method: MethodIR,
    *,
    capability: BackendCapability,
    dtype: str = "float64",
    derivative_order: int = 0,
    features: tuple[FeatureType, ...] | None = None,
) -> TypedMethodIR:
    """Fail closed before lowering when a MethodIR execution type is illegal."""

    if not isinstance(method, MethodIR):
        raise TypeError("MethodIR type checking requires MethodIR")
    if not isinstance(capability, BackendCapability):
        raise TypeError("MethodIR type checking requires BackendCapability")
    inferred = infer_feature_types(method, dtype=dtype)
    features = inferred if features is None else features
    typed = TypedMethodIR(
        method,
        capability.backend,
        dtype,
        derivative_order,
        tuple(features),
    )

    if method.spin not in capability.spins:
        raise MethodTypeError(
            f"{capability.backend} does not support MethodIR spin {method.spin!r}"
        )
    if dtype not in capability.dtypes:
        raise MethodTypeError(
            f"{capability.backend} does not support MethodIR dtype {dtype!r}"
        )
    if derivative_order not in capability.derivative_orders:
        raise MethodTypeError(
            f"{capability.backend} does not support derivative order {derivative_order}"
        )

    requirements = method.requirements
    missing_ingredients = set(requirements["ingredients"]) - set(capability.ingredients)
    if missing_ingredients:
        raise MethodTypeError(
            f"{capability.backend} lacks ingredients "
            f"{tuple(sorted(missing_ingredients))}"
        )
    missing_operators = set(requirements["operators"]) - set(capability.operators)
    if missing_operators:
        raise MethodTypeError(
            f"{capability.backend} lacks operators {tuple(sorted(missing_operators))}"
        )

    return typed

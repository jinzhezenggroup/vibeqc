"""Static MethodIR execution-type checks."""

from fractions import Fraction

import pytest
from vibeqc_compiler.method import (
    BackendCapability,
    FeatureType,
    MethodSpec,
    MethodTypeError,
    infer_feature_types,
    resolve_method,
    verify_method_ir,
)


def _semilocal_capability(
    *,
    backend="test-cpu",
    dtypes=("float64",),
    spins=("unpolarized", "polarized"),
    derivative_orders=(0, 1),
    ingredients=("rho", "sigma"),
    operators=("semilocal-xc",),
):
    return BackendCapability(
        backend,
        dtypes,
        spins,
        derivative_orders,
        ingredients,
        operators,
    )


@pytest.mark.parametrize(
    ("spin", "expected"),
    [
        ("unpolarized", (("rho", (1,)), ("sigma", (1,)))),
        ("polarized", (("rho", (2,)), ("sigma", (3,)))),
    ],
)
def test_feature_type_inference_tracks_spin_component_shapes(spin, expected):
    method = resolve_method("PBE", spin=spin)
    features = infer_feature_types(method, dtype="float64")
    assert tuple((item.ingredient, item.shape) for item in features) == expected
    assert all(item.spin == spin and item.dtype == "float64" for item in features)


def test_verify_method_ir_returns_separate_execution_identity():
    method = resolve_method("PBE")
    fp64 = verify_method_ir(
        method,
        capability=_semilocal_capability(dtypes=("float32", "float64")),
        dtype="float64",
    )
    fp32 = verify_method_ir(
        method,
        capability=_semilocal_capability(dtypes=("float32", "float64")),
        dtype="float32",
    )
    assert fp64.method.identity == fp32.method.identity == method.identity
    assert fp64.identity != fp32.identity
    assert fp64.to_payload()["method_identity"] == method.identity


def test_backend_gates_dtype_spin_derivative_and_ingredients():
    polarized = resolve_method("PBE", spin="polarized")
    with pytest.raises(MethodTypeError, match="spin"):
        verify_method_ir(
            polarized,
            capability=_semilocal_capability(spins=("unpolarized",)),
        )

    unpolarized = resolve_method("PBE")
    with pytest.raises(MethodTypeError, match="dtype"):
        verify_method_ir(
            unpolarized,
            capability=_semilocal_capability(),
            dtype="float32",
        )
    with pytest.raises(MethodTypeError, match="derivative order 2"):
        verify_method_ir(
            unpolarized,
            capability=_semilocal_capability(derivative_orders=(0, 1)),
            derivative_order=2,
        )
    with pytest.raises(MethodTypeError, match="lacks ingredients"):
        verify_method_ir(
            unpolarized,
            capability=_semilocal_capability(ingredients=("rho",)),
        )


def test_backend_operator_gate_rejects_hybrid_without_exchange_provider():
    method = resolve_method("PBE0")
    with pytest.raises(MethodTypeError, match="full-range-exchange"):
        verify_method_ir(method, capability=_semilocal_capability())

    typed = verify_method_ir(
        method,
        capability=_semilocal_capability(
            operators=("semilocal-xc", "full-range-exchange")
        ),
    )
    assert typed.method is method


def test_feature_bindings_fail_closed_on_missing_extra_duplicate_and_shape():
    method = resolve_method("PBE", spin="polarized")
    capability = _semilocal_capability()
    inferred = infer_feature_types(method, dtype="float64")
    rho, sigma = inferred

    with pytest.raises(MethodTypeError, match="missing=.*sigma"):
        verify_method_ir(method, capability=capability, features=(rho,))
    extra = FeatureType("tau", "float64", (2,), "polarized")
    with pytest.raises(MethodTypeError, match="extra=.*tau"):
        verify_method_ir(
            method,
            capability=capability,
            features=(rho, sigma, extra),
        )
    with pytest.raises(MethodTypeError, match="duplicate feature"):
        verify_method_ir(
            method,
            capability=capability,
            features=(rho, sigma, sigma),
        )
    wrong_shape = FeatureType("sigma", "float64", (2,), "polarized")
    with pytest.raises(MethodTypeError, match="sigma shape"):
        verify_method_ir(
            method,
            capability=capability,
            features=(rho, wrong_shape),
        )


def test_feature_bindings_require_explicit_cast_and_matching_spin():
    method = resolve_method("PBE", spin="polarized")
    capability = _semilocal_capability(dtypes=("float32", "float64"))
    rho, sigma = infer_feature_types(method, dtype="float64")

    mixed = FeatureType("sigma", "float32", (3,), "polarized")
    with pytest.raises(MethodTypeError, match="explicit cast"):
        verify_method_ir(
            method,
            capability=capability,
            dtype="float64",
            features=(rho, mixed),
        )

    wrong_spin = FeatureType("rho", "float64", (1,), "unpolarized")
    with pytest.raises(MethodTypeError, match="does not match MethodIR spin"):
        verify_method_ir(
            method,
            capability=capability,
            features=(wrong_spin, sigma),
        )


def test_tau_logical_type_is_reserved_for_meta_gga_extension():
    # Current audited catalog has no meta-GGA component yet. The checker still
    # owns tau's spin-dependent logical shape so #164 can add it data-only.
    assert FeatureType("tau", "float64", (1,), "unpolarized").shape == (1,)
    assert FeatureType("tau", "float64", (2,), "polarized").shape == (2,)


def test_capability_and_feature_contracts_are_strict():
    with pytest.raises(ValueError, match="duplicates"):
        _semilocal_capability(dtypes=("float64", "float64"))
    with pytest.raises(ValueError, match="derivative orders"):
        _semilocal_capability(derivative_orders=(3,))
    with pytest.raises(MethodTypeError, match="feature shape"):
        FeatureType("rho", "float64", (), "unpolarized")


def test_exact_exchange_only_graph_needs_no_semilocal_features():
    method = resolve_method(
        MethodSpec(
            "exchange-only",
            (("LDA_X", Fraction(1)), ("LDA_X", Fraction(-1))),
            exact_exchange=Fraction(1, 2),
        )
    )
    capability = BackendCapability(
        "exchange",
        ("float64",),
        ("unpolarized",),
        (0,),
        (),
        ("full-range-exchange",),
    )
    typed = verify_method_ir(method, capability=capability)
    assert typed.features == ()

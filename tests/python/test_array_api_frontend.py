"""Issue #633 bounded Array API frontend and TensorIR-equivalence tests."""

from fractions import Fraction

import numpy as np
import pytest
from vibeqc_compiler.array_api import (
    VibeArray,
    capabilities,
    input_array,
    trace,
)
from vibeqc_compiler.array_api import (
    namespace as xp,
)
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    execute,
    input_tensor,
    linearize,
    multiply,
    reduce_sum,
)


def _matrix_spec(
    space: IndexSpace,
    *,
    role: str = "input",
    differentiable: bool = False,
) -> TensorSpec:
    return TensorSpec(
        (Index("p", space), Index("q", space)),
        role=role,
        differentiable=differentiable,
    )


def test_namespace_capture_matches_handwritten_tensorir_hash_and_values() -> None:
    ao = IndexSpace("ao", "ao", 3)
    spec = _matrix_spec(ao)
    captured = trace(
        lambda x, y: {"out": xp.sum(x * y, axis=1)},
        {"x": spec, "y": spec},
    )

    x_node = input_tensor("x", spec)
    y_node = input_tensor("y", spec)
    manual = Program({"out": reduce_sum(multiply(x_node, y_node), axes=(1,))})

    assert captured.logical_hash == manual.logical_hash
    rng = np.random.default_rng(633)
    feeds = {"x": rng.normal(size=(3, 3)), "y": rng.normal(size=(3, 3))}
    np.testing.assert_array_equal(
        execute(captured, feeds).outputs["out"],
        execute(manual, feeds).outputs["out"],
    )


def test_exact_scalar_capture_reuses_tensorir_ad_without_frontend_rules() -> None:
    ao = IndexSpace("ao", "ao", 2)
    spec = _matrix_spec(ao, role="parameter", differentiable=True)
    captured = trace(
        lambda x: {"out": xp.sum(Fraction(1, 2) * x * x)},
        {"x": spec},
    )
    derivative = linearize(captured, ["x"]).program
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    dx = np.array([[0.5, -1.0], [2.0, 0.25]])
    result = execute(derivative, {"x": x, "d_x": dx}).outputs["d_out"]
    assert result == pytest.approx(float(np.sum(x * dx)))


def test_equal_shape_different_scientific_domains_fail_closed() -> None:
    ao = IndexSpace("ao", "ao", 2)
    occupied = IndexSpace("occupied", "occupied", 2)
    ao_spec = TensorSpec((Index("p", ao),), role="input")
    occupied_spec = TensorSpec((Index("i", occupied),), role="input")

    with pytest.raises(ValueError, match="identical index domains"):
        trace(
            lambda x, y: {"out": x + y},
            {"x": ao_spec, "y": occupied_spec},
        )


def test_float_scalar_and_python_control_flow_are_rejected() -> None:
    ao = IndexSpace("ao", "ao", 2)
    spec = TensorSpec((Index("p", ao),), role="input")

    with pytest.raises(TypeError, match="floating-point scalar spelling"):
        trace(lambda x: {"out": x * 0.5}, {"x": spec})

    value = input_array("x", spec)
    with pytest.raises(TypeError, match="Python control flow"):
        bool(value)


def test_array_namespace_is_internal_and_version_request_fails_closed() -> None:
    ao = IndexSpace("ao", "ao", 1)
    value = input_array("x", TensorSpec((Index("p", ao),), role="input"))
    assert isinstance(value, VibeArray)
    assert value.__array_namespace__() is xp
    with pytest.raises(ValueError, match="versioned Array API conformance"):
        value.__array_namespace__(api_version="2025.12")


def test_scf_density_expression_has_same_tensorir_identity() -> None:
    batch = IndexSpace("batch", "batch", 2)
    spin = IndexSpace("spin", "spin", 1)
    ao = IndexSpace("ao", "ao", 3)
    orbital = IndexSpace("orbital", "orbital", 3)
    b, s, p, i = (
        Index("b", batch),
        Index("s", spin),
        Index("p", ao),
        Index("i", orbital),
    )
    coefficient_spec = TensorSpec((b, s, p, i), role="input")
    occupation_spec = TensorSpec((b, s, i), role="input")
    captured = trace(
        lambda coefficients, occupations: {
            "density": xp.einsum(
                "bspi,bsi,bsqi->bspq",
                coefficients,
                occupations,
                coefficients,
            )
        },
        {
            "coefficients": coefficient_spec,
            "occupations": occupation_spec,
        },
    )
    coefficients = input_tensor("coefficients", coefficient_spec)
    occupations = input_tensor("occupations", occupation_spec)
    manual = Program(
        {
            "density": einsum(
                "bspi,bsi,bsqi->bspq",
                coefficients,
                occupations,
                coefficients,
            )
        }
    )
    assert captured.logical_hash == manual.logical_hash


def test_capability_report_does_not_claim_full_conformance() -> None:
    report = capabilities()
    assert report["array_api_conformance"] == "bounded-internal-subset"
    assert report["implicit_broadcast"] is False
    assert report["dtype_promotion"] is False
    assert "einsum_extension" in report["functions"]

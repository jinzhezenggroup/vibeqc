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


def test_preview_does_not_advertise_array_namespace_conformance() -> None:
    ao = IndexSpace("ao", "ao", 1)
    value = input_array("x", TensorSpec((Index("p", ao),), role="input"))
    assert isinstance(value, VibeArray)
    assert not hasattr(value, "__array_namespace__")


def test_declared_preview_surfaces_execute_through_tensorir() -> None:
    ao = IndexSpace("ao", "ao", 2)
    spec = _matrix_spec(ao)
    program = trace(
        lambda x, y: {
            "add": x + y,
            "subtract": x - y,
            "divide": x / y,
            "scale": x / Fraction(2),
            "negative": -x,
            "power": xp.pow(x, 2),
            "exp": xp.exp(x),
            "log": xp.log(x),
            "sqrt": xp.sqrt(x),
            "permute": xp.permute_dims(x, (1, 0)),
            "matmul": xp.matmul(x, y),
        },
        {"x": spec, "y": spec},
    )
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[2.0, 1.0], [1.0, 2.0]])
    outputs = execute(program, {"x": x, "y": y}).outputs

    np.testing.assert_allclose(outputs["add"], x + y)
    np.testing.assert_allclose(outputs["subtract"], x - y)
    np.testing.assert_allclose(outputs["divide"], x / y)
    np.testing.assert_allclose(outputs["scale"], x / 2)
    np.testing.assert_allclose(outputs["negative"], -x)
    np.testing.assert_allclose(outputs["power"], x**2)
    np.testing.assert_allclose(outputs["exp"], np.exp(x))
    np.testing.assert_allclose(outputs["log"], np.log(x))
    np.testing.assert_allclose(outputs["sqrt"], np.sqrt(x))
    np.testing.assert_allclose(outputs["permute"], x.T)
    np.testing.assert_allclose(outputs["matmul"], x @ y)


def test_preview_unsupported_conveniences_fail_closed() -> None:
    ao = IndexSpace("ao", "ao", 2)
    vector_spec = TensorSpec((Index("p", ao),), role="input")
    matrix_spec = _matrix_spec(ao)
    vector = input_array("vector", vector_spec)
    matrix = input_array("matrix", matrix_spec)

    with pytest.raises(TypeError, match="scalar / VibeArray"):
        xp.divide(1, vector)
    with pytest.raises(ValueError, match="dtype conversions"):
        xp.sum(vector, dtype="float32")
    with pytest.raises(ValueError, match="keepdims=False"):
        xp.sum(vector, keepdims=True)
    with pytest.raises(TypeError, match="axis must be"):
        xp.sum(vector, axis=[0])
    with pytest.raises(ValueError, match="rank-2"):
        xp.matmul(vector, vector)
    with pytest.raises(TypeError, match="add left operand"):
        xp.add(1, matrix)


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
    assert report["surface"] == "array-api-shaped-internal-preview"
    assert report["array_api_version"] is None
    assert report["array_namespace_protocol"] is False
    assert report["implicit_broadcast"] is False
    assert report["dtype_promotion"] is False
    assert "einsum_extension" in report["functions"]

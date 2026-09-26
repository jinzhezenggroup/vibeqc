"""ECP scientific identities, derivative rules, and strict lowering boundaries."""

import math
import subprocess
import sys
import typing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.integral.ecp import build_ecp_ir, gaussian_roots
from vibeqc_compiler.integral.ecp_grid import harmonic_roots, radial_map_roots
from vibeqc_compiler.integral.ecp_projector import (
    emit_ecp_quadrature_cpp,
    pair_roots,
    radial_roots,
)
from vibeqc_compiler.integral.ir import EcpRadialTerm, OperatorSpec
from vibeqc_compiler.integral.ir_serialization import (
    integral_from_payload,
    integral_to_payload,
)
from vibeqc_compiler.integral.shell_spec import cartesian_components


def test_ecp_codegen_without_site_packages(tmp_path: typing.Any) -> None:
    """CPU builds generate this header before installing Python dependencies."""
    generator = Path(__file__).resolve().parents[2] / "tools/generate_ecp_kernels.py"
    output = tmp_path / "generated" / "generated_ecp_ao.cuh"
    subprocess.run(
        [sys.executable, "-I", "-S", str(generator), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert output.read_text() == emit_ecp_quadrature_cpp()


def test_generated_grid_radial_map_and_jacobian() -> None:
    graph, (radius, weight) = radial_map_roots()
    for z in (-0.999, -0.7, 0.0, 0.4, 0.999):
        values = {"z": z, "weight": 0.37}
        # Independent rational form and its analytic Jacobian.
        assert graph.evaluate(radius, values) == pytest.approx((1 + z) / (1 - z))
        assert graph.evaluate(weight, values) == pytest.approx(0.74 / (1 - z) ** 2)
        derivative = graph.differentiate(radius, graph.variable("z"))
        assert graph.evaluate(weight, values) == pytest.approx(
            0.37 * graph.evaluate(derivative, values)
        )


def test_generated_harmonic_channel_addition_theorem() -> None:
    graph, roots = harmonic_roots()
    rng = np.random.default_rng(17103)
    for _ in range(30):
        a, b = rng.normal(size=(2, 3))
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        ya, yb = (
            np.array(
                [graph.evaluate(r, dict(zip("xyz", v, strict=True))) for r in roots]
            )
            for v in (a, b)
        )
        dot = a @ b
        for angular_momentum, polynomial in enumerate(
            (1.0, dot, (3 * dot * dot - 1) / 2, (5 * dot**3 - 3 * dot) / 2)
        ):
            selected = slice(
                angular_momentum * angular_momentum, (angular_momentum + 1) ** 2
            )
            assert ya[selected] @ yb[selected] == pytest.approx(
                (2 * angular_momentum + 1) / (4 * math.pi) * polynomial, abs=3e-15
            )


@pytest.mark.parametrize("weighted", [False, True])
def test_ecp_center_and_versioned_roundtrip(weighted: typing.Any) -> None:
    ir = build_ecp_ir(
        (1, 2),
        (EcpRadialTerm(-1, 2, 0.7, -3), EcpRadialTerm(3, 0, 1.5, 2)),
        derivatives=True,
        weighted=weighted,
    )
    assert len(ir.signature.shells) == 2
    assert ir.operator.centers == (0, 1, 2)
    assert ir.derivative.recovered_centers(ir.operator) == (2,)
    payload = integral_to_payload(ir)
    assert payload["schema_version"] == 4
    assert integral_from_payload(payload) == ir
    old = {**payload, "schema_version": 1}
    with pytest.raises(ValueError, match="schema version 4"):
        integral_from_payload(old)
    changed = replace(
        ir.operator.external_centers[0], terms=(EcpRadialTerm(-1, 2, 0.8, -3),)
    )
    assert (
        integral_to_payload(
            replace(ir, operator=replace(ir.operator, external_centers=(changed,)))
        )
        != payload
    )
    with pytest.raises(ValueError):
        OperatorSpec("nuclear_attraction", (0, 1, 2), external_centers=(changed,))


def test_generated_center_derivatives_independent_finite_difference() -> None:
    rng = np.random.default_rng(171)
    for l in range(4):
        for component in cartesian_components(l):
            graph, roots = gaussian_roots(component)
            for xyz in (np.zeros(3), rng.normal(size=3)):
                alpha = 0.67
                values = dict(zip("xyz", xyz, strict=True), alpha=alpha)
                actual = np.array([graph.evaluate(root, values) for root in roots])

                def reference(
                    x: typing.Any,
                    alpha: typing.Any = alpha,
                    component: typing.Any = component,
                ) -> typing.Any:
                    return math.exp(-alpha * (x @ x)) * math.prod(
                        x[i] ** component.count(a) for i, a in enumerate("xyz")
                    )

                assert actual[0] == pytest.approx(reference(xyz), abs=1e-14)
                for axis in range(3):
                    delta = np.eye(3)[axis] * 1e-5
                    # A-center derivative is minus the electronic-coordinate derivative.
                    expected = (reference(xyz - delta) - reference(xyz + delta)) / 2e-5
                    assert actual[axis + 1] == pytest.approx(expected, abs=2e-9)


def test_invalid_ecp_lowerings_fail_closed() -> None:
    term = EcpRadialTerm(-1, 2, 1.0, 1.0)
    for angular in ((0,), (0, 4)):
        with pytest.raises(ValueError):
            build_ecp_ir(angular, (term,))
    with pytest.raises(ValueError):
        build_ecp_ir((0, 0), (term,), weighted=True)
    for kwargs in (
        {"power": -1},
        {"channel": 4},
        {"exponent": 0},
        {"coefficient": float("nan")},
    ):
        with pytest.raises(ValueError):
            replace(term, **kwargs)


@pytest.mark.parametrize("angular", [(0, 3), (3, 2), (3, 3)])
@pytest.mark.parametrize("weighted", [False, True])
def test_f_orbital_ir_derivative_contract(
    angular: typing.Any, weighted: typing.Any
) -> None:
    ir = build_ecp_ir(
        angular,
        (EcpRadialTerm(2, 4, 0.8, -1.2),),
        derivatives=True,
        weighted=weighted,
    )
    assert ir.derivative.recovered_centers(ir.operator) == (2,)
    assert integral_from_payload(integral_to_payload(ir)) == ir


@pytest.mark.parametrize("power", range(5))
def test_radial_measure_and_operator_contract(power: typing.Any) -> None:
    graph, (root,) = radial_roots(power)
    for r in (0.0, 1e-8, 0.37, 1.8, 17.0):
        values = {"r": r, "alpha": 0.63, "coefficient": -1.23, "weight": 0.41}
        expected = -1.23 * 0.41 * r**power * math.exp(-0.63 * r**2)
        assert graph.evaluate(root, values) == pytest.approx(expected, abs=1e-15)
    for bad in (-1, 5, True, 1.5):
        with pytest.raises(ValueError):
            radial_roots(bad)


def test_projector_pair_jets_against_displaced_bilinear() -> None:
    graph, roots = pair_roots()
    rng = np.random.default_rng(17102)
    for _ in range(20):
        a, b = rng.normal(size=(2, 4))
        weight = rng.normal()
        values = {
            f"{ab}{d}": jet[d] for ab, jet in (("a", a), ("b", b)) for d in range(4)
        }
        actual = np.array(
            [graph.evaluate(root, dict(values, weight=weight)) for root in roots]
        )
        assert actual[0] == pytest.approx(weight * a[0] * b[0])
        for d in range(1, 4):
            step = 1e-5
            da = (
                weight * (a[0] + step * a[d]) * b[0]
                - weight * (a[0] - step * a[d]) * b[0]
            ) / (2 * step)
            db = (
                weight * a[0] * (b[0] + step * b[d])
                - weight * a[0] * (b[0] - step * b[d])
            ) / (2 * step)
            np.testing.assert_allclose(
                actual[[d, d + 3]], [da, db], atol=2e-10, rtol=1e-9
            )

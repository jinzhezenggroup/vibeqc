"""ECP scientific identities, derivative rules, and strict lowering boundaries."""

import math
from dataclasses import replace

import numpy as np
import pytest
from vibeqc_compiler.integral.ecp import build_ecp_ir, gaussian_roots
from vibeqc_compiler.integral.ir import EcpRadialTerm, OperatorSpec
from vibeqc_compiler.integral.ir_serialization import (
    integral_from_payload,
    integral_to_payload,
)
from vibeqc_compiler.integral.shell_spec import cartesian_components


@pytest.mark.parametrize("weighted", [False, True])
def test_ecp_center_and_versioned_roundtrip(weighted):
    ir = build_ecp_ir(
        (1, 2),
        (EcpRadialTerm(-1, 2, 0.7, -3), EcpRadialTerm(0, 0, 1.5, 2)),
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


def test_generated_center_derivatives_independent_finite_difference():
    rng = np.random.default_rng(171)
    for l in range(3):
        for component in cartesian_components(l):
            graph, roots = gaussian_roots(component)
            for xyz in (np.zeros(3), rng.normal(size=3)):
                alpha = 0.67
                values = dict(zip("xyz", xyz, strict=True), alpha=alpha)
                actual = np.array([graph.evaluate(root, values) for root in roots])

                def reference(x, alpha=alpha, component=component):
                    return math.exp(-alpha * (x @ x)) * math.prod(
                        x[i] ** component.count(a) for i, a in enumerate("xyz")
                    )

                assert actual[0] == pytest.approx(reference(xyz), abs=1e-14)
                for axis in range(3):
                    delta = np.eye(3)[axis] * 1e-5
                    # A-center derivative is minus the electronic-coordinate derivative.
                    expected = (reference(xyz - delta) - reference(xyz + delta)) / 2e-5
                    assert actual[axis + 1] == pytest.approx(expected, abs=2e-9)


def test_invalid_ecp_lowerings_fail_closed():
    term = EcpRadialTerm(-1, 2, 1.0, 1.0)
    for angular in ((0,), (0, 3)):
        with pytest.raises(ValueError):
            build_ecp_ir(angular, (term,))
    with pytest.raises(ValueError):
        build_ecp_ir((0, 0), (term,), weighted=True)
    for kwargs in (
        {"power": -1},
        {"channel": 3},
        {"exponent": 0},
        {"coefficient": float("nan")},
    ):
        with pytest.raises(ValueError):
            replace(term, **kwargs)

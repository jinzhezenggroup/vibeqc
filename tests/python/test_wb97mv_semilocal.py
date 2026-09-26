"""omegaB97M-V semilocal expression and MethodIR composition gates."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.common.provenance import file_hash
from vibeqc_compiler.method import (
    MethodSpec,
    NonlocalCorrelationPrimitive,
    RangeSeparatedExchangePrimitive,
    SemilocalXCPrimitive,
    resolve_method,
)
from vibeqc_compiler.xc.cuda_emit import XCSchedule, emit_cuda
from vibeqc_compiler.xc.program import build_program
from vibeqc_compiler.xc.semilocal_codegen import build_roots
from vibeqc_compiler.xc.spec import FunctionalSpec, UnsupportedXC
from vibeqc_compiler.xc.wb97mv_maple import (
    energy_expression as production_energy_expression,
)


def test_wb97mv_manifest_resolves_exact_semilocal_rsh_and_vv10_definition() -> None:
    method = resolve_method("WB97M-V", spin="polarized")
    assert method.requirements["ingredients"] == ("rho", "sigma", "tau")
    assert method.requirements["operators"] == (
        "semilocal-xc",
        "short-range-exchange",
        "long-range-exchange",
        "nonlocal-correlation",
    )

    semilocal, short, long, nonlocal_correlation = method.primitives
    assert isinstance(semilocal, SemilocalXCPrimitive)
    assert semilocal.functional.components == (
        ("MGGA_C_WB97M_V", Fraction(1)),
        ("MGGA_X_WB97M_V", Fraction(1)),
    )
    assert semilocal.functional.range_omega == Fraction(3, 10)

    assert isinstance(short, RangeSeparatedExchangePrimitive)
    assert (short.operator, short.coefficient, short.omega) == (
        "short-range",
        Fraction(3, 20),
        Fraction(3, 10),
    )
    assert isinstance(long, RangeSeparatedExchangePrimitive)
    assert (long.operator, long.coefficient, long.omega) == (
        "long-range",
        Fraction(1),
        Fraction(3, 10),
    )

    assert isinstance(nonlocal_correlation, NonlocalCorrelationPrimitive)
    assert nonlocal_correlation.spec.variant == "vv10"
    assert nonlocal_correlation.spec.b == Fraction(6)
    assert nonlocal_correlation.spec.c == Fraction(1, 100)


def test_wb97mv_unpolarized_energy_vxc_fxc_match_libxc_7_pinned_oracle() -> None:
    spec = resolve_method("WB97M-V").primitives[0].functional
    point = np.array([[0.5, 0.031, 0.14]]).T
    result = build_program(spec, order=2).unpack(
        build_program(spec, order=2).evaluate(point)
    )

    # Independently pinned with PySCF 2.14.0 / Libxc 7.0.0
    # HYB_MGGA_XC_WB97M_V. Libxc's hybrid/NLC coefficients are separate
    # metadata, so eval_xc here is exactly the semilocal contribution.
    expected_energy = -0.20794318755308802
    expected_gradient = np.array(
        [
            -0.584943261595045,
            -0.0057485048406733,
            -0.04806556699716644,
        ]
    )
    expected_hessian = np.array(
        [
            [-0.4519292851770841, 0.01463966616204928, -0.20181739805554566],
            [0.01463966616204928, -0.01737790146793691, 0.00094734584691088],
            [-0.20181739805554566, 0.00094734584691088, 0.6669726284210433],
        ]
    )
    np.testing.assert_allclose(result["energy_density"][0], expected_energy, rtol=2e-13)
    np.testing.assert_allclose(
        result["gradient"][:, 0], expected_gradient, rtol=3e-12, atol=2e-13
    )
    np.testing.assert_allclose(
        result["hessian"][:, :, 0], expected_hessian, rtol=8e-11, atol=2e-12
    )


def test_wb97mv_polarized_energy_vxc_fxc_match_libxc_7_pinned_oracle() -> None:
    spec = resolve_method("WB97M-V", spin="polarized").primitives[0].functional
    point = np.array([[0.3, 0.2, 0.015, 0.003, 0.010, 0.08, 0.05]]).T
    program = build_program(spec, order=2)
    result = program.unpack(program.evaluate(point))

    expected_energy = -0.20814586702136345
    expected_gradient = np.array(
        [
            -0.5948515352856814,
            -0.5697576268765359,
            -0.01082457361721382,
            0.0,
            -0.01425699937015483,
            -0.05540206273456351,
            -0.05804981432341172,
        ]
    )
    expected_hessian = np.array(
        [
            [
                -0.6632284662958083,
                -0.23375989368154276,
                0.03991423114493268,
                0.0,
                0.02206254040240553,
                0.04995652637246642,
                -0.5915952733366672,
            ],
            [
                -0.23375989368154276,
                -0.701408486079739,
                0.01245379922721197,
                0.0,
                0.05009914610193475,
                -0.48052050548127057,
                0.2143245890867168,
            ],
            [
                0.03991423114493268,
                0.01245379922721197,
                -0.08719639941693026,
                0.0,
                -0.00176539099134805,
                0.01138841320580272,
                -0.00075558141135182,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.02206254040240553,
                0.05009914610193475,
                -0.00176539099134805,
                0.0,
                -0.3843831191022294,
                -0.00171042258402394,
                0.10144789143005019,
            ],
            [
                0.04995652637246642,
                -0.48052050548127057,
                0.01138841320580272,
                0.0,
                -0.00171042258402394,
                -0.09126638204883764,
                1.8185344269533137,
            ],
            [
                -0.5915952733366672,
                0.2143245890867168,
                -0.00075558141135182,
                0.0,
                0.10144789143005019,
                1.8185344269533137,
                -0.5858794275754127,
            ],
        ]
    )
    np.testing.assert_allclose(result["energy_density"][0], expected_energy, rtol=2e-13)
    np.testing.assert_allclose(
        result["gradient"][:, 0], expected_gradient, rtol=3e-12, atol=2e-13
    )
    np.testing.assert_allclose(
        result["hessian"][:, :, 0], expected_hessian, rtol=8e-11, atol=2e-12
    )


def _production_first_derivatives(
    spec: FunctionalSpec, point: np.ndarray
) -> tuple[np.ndarray, object]:
    graph, energy, variables = production_energy_expression(spec)
    roots = (energy, *(graph.differentiate(energy, variable) for variable in variables))
    values = evaluate_array_graph(
        graph,
        roots,
        dict(zip(spec.features, np.asarray(point, dtype=float), strict=True)),
    )
    return np.asarray([float(np.asarray(value)) for value in values]), graph


def test_wb97mv_production_maple_preserves_interior_and_large_a_tail() -> None:
    spec = resolve_method("WB97M-V", spin="polarized").primitives[0].functional
    interior = np.array((0.3, 0.2, 0.015, 0.003, 0.010, 0.08, 0.05))
    actual, graph = _production_first_derivatives(spec, interior)
    reference = build_program(spec, order=1).evaluate(interior[:, None])[:, 0]
    np.testing.assert_allclose(actual, reference, rtol=3e-11, atol=3e-12)

    # Each spin remains above Libxc's 1e-13 density screen while omega/(2*kF)
    # is far beyond the direct attenuation branch. Production must take the
    # pinned order-16 smooth-LR series rather than reject or clip this point.
    tail = np.array((5e-11, 5e-11, 2.5e-31, 2.5e-31, 2.5e-31, 5e-13, 5e-13))
    tail_value, _ = _production_first_derivatives(spec, tail)
    assert np.all(np.isfinite(tail_value))
    assert any(node.operation == "select_le" for node in graph.nodes)


@pytest.mark.parametrize("swap_spins", (False, True))
def test_wb97mv_empty_spin_work_point_matches_independent_quad_oracle(
    swap_spins: bool,
) -> None:
    """Guard the Stoll cancellation at a captured, floored CUDA work point.

    Reference values come from the original pinned Libxc 7.0.0 Maple C
    evaluated in libquadmath (tools/qualify_wb97mv_tail.py, point 11113).
    """
    spec = resolve_method("WB97M-V", spin="polarized").primitives[0].functional
    point = (
        0.03283928360431155,
        1e-13,
        0.0036369242335227894,
        0.0,
        2.1544346900318932e-35,
        0.02265827832448588,
        1e-20,
    )
    if swap_spins:
        point = (point[1], point[0], point[4], point[3], point[2], point[6], point[5])
    minority = 0 if swap_spins else 1
    kinetic = 5 if swap_spins else 6
    graph, roots, _ = build_roots(spec, ((), (minority,), (kinetic,)), production=True)
    variables = dict(zip(spec.features, point, strict=True))
    actual = tuple(graph.evaluate(root, variables) for root in roots)
    expected = (-0.005195313331621749, -0.14552535052309215, -42743.40113431417)
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=3e-11, atol=3e-12)
    np.testing.assert_allclose(actual[2], expected[2], rtol=2e-12)


def test_wb97mv_generated_cuda_uses_same_tau_expression_graph() -> None:
    spec = resolve_method("WB97M-V").primitives[0].functional
    outputs = ((), (0,), (1,), (2,), (0, 2), (1, 2), (2, 2))
    program = build_program(spec, order=2, outputs=outputs)
    schedule = XCSchedule("split", group_size=4)
    source, contract, _ = emit_cuda(program, schedule)
    assert emit_cuda(program, schedule)[0] == source
    assert contract["functional"]["ingredients"] == ("rho", "sigma", "tau")
    assert contract["outputs"] == outputs
    assert "erf(" in source
    assert "expm1(" in source
    assert "input[2 * npoint + point]" in source


def test_wb97mv_equal_spin_recovers_unpolarized_semilocal_value() -> None:
    rho, sigma, tau = 0.6, 0.024, 0.16
    restricted = resolve_method("WB97M-V").primitives[0].functional
    polarized = resolve_method("WB97M-V", spin="polarized").primitives[0].functional
    e_r = build_program(restricted, order=0).evaluate(np.array([[rho, sigma, tau]]).T)[
        0, 0
    ]
    e_u = build_program(polarized, order=0).evaluate(
        np.array(
            [
                [
                    rho / 2,
                    rho / 2,
                    sigma / 4,
                    sigma / 4,
                    sigma / 4,
                    tau / 2,
                    tau / 2,
                ]
            ]
        ).T
    )[0, 0]
    np.testing.assert_allclose(e_u, e_r, rtol=2e-13, atol=2e-14)


def test_wb97mv_range_parameter_changes_semilocal_and_method_identity() -> None:
    def custom(identifier: str, omega: Fraction) -> MethodSpec:
        return MethodSpec(
            identifier,
            (
                ("MGGA_X_WB97M_V", Fraction(1)),
                ("MGGA_C_WB97M_V", Fraction(1)),
            ),
            short_range_exchange=Fraction(3, 20),
            long_range_exchange=Fraction(1),
            range_omega=omega,
        )

    low = resolve_method(custom("WB97M-test-low", Fraction(1, 4)))
    high = resolve_method(custom("WB97M-test-high", Fraction(2, 5)))
    assert low.identity != high.identity
    point = np.array([[0.5, 0.031, 0.14]]).T
    e_low = build_program(low.primitives[0].functional, order=0).evaluate(point)[0, 0]
    e_high = build_program(high.primitives[0].functional, order=0).evaluate(point)[0, 0]
    assert e_low != pytest.approx(e_high, rel=0.0, abs=1e-12)


def test_wb97mv_family_mixing_and_unqualified_attenuation_fail_closed() -> None:
    mixed = FunctionalSpec(
        "bad-mix",
        (
            ("MGGA_X_WB97M_V", Fraction(1)),
            ("MGGA_C_WB97M_V", Fraction(1)),
            ("GGA_C_PBE", Fraction(1)),
        ),
        range_omega=Fraction(3, 10),
    )
    with pytest.raises(UnsupportedXC, match="cannot be mixed"):
        build_program(mixed, order=0)

    spec = resolve_method("WB97M-V").primitives[0].functional
    # Very small density makes a=omega/(2*kF) leave the audited direct
    # Libxc attenuation branch. This slice rejects it rather than clipping.
    with pytest.raises(UnsupportedXC, match="attenuation"):
        build_program(spec, order=1).evaluate(np.array([[1e-12, 1e-30, 1e-18]]).T)


def test_wb97mv_libxc_source_manifest_is_pinned() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "upstream/libxc/7.0.0"
    manifest_root = root / "manifests/libxc/7.0.0"
    manifest = json.loads((manifest_root / "wb97mv-manifest.json").read_text())
    assert manifest["version"] == "7.0.0"
    for name, item in manifest["files"].items():
        assert file_hash(source / name) == item["sha256"]

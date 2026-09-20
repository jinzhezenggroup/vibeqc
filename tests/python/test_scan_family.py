"""SCAN/SCAN0 MethodIR, Libxc scalar and alpha-cutoff gates."""

from fractions import Fraction

import numpy as np
from vibeqc_compiler.method import (
    ExactExchangePrimitive,
    SemilocalXCPrimitive,
    resolve_method,
)
from vibeqc_compiler.xc.program import build_program

RA, RB, AA, AB, BB = 0.3, 0.2, 0.015, 0.003, 0.01
TA = AA / (8 * RA) + 0.8 * RA ** (5 / 3)
TB = BB / (8 * RB) + 0.8 * RB ** (5 / 3)
POLARIZED = np.array([[RA, RB, AA, AB, BB, TA, TB]]).T
N, SIGMA = 0.5, 0.031
TAU = SIGMA / (8 * N) + 0.8 * N ** (5 / 3)
UNPOLARIZED = np.array([[N, SIGMA, TAU]]).T


def test_scan_and_scan0_resolve_typed_tau_graphs():
    scan = resolve_method("SCAN")
    assert len(scan.primitives) == 1
    assert isinstance(scan.primitives[0], SemilocalXCPrimitive)
    assert dict(scan.primitives[0].functional.components) == {
        "MGGA_X_SCAN": Fraction(1),
        "MGGA_C_SCAN": Fraction(1),
    }
    assert scan.requirements["ingredients"] == ("rho", "sigma", "tau")

    scan0 = resolve_method("SCAN0")
    semilocal, exact = scan0.primitives
    assert dict(semilocal.functional.components) == {
        "MGGA_X_SCAN": Fraction(3, 4),
        "MGGA_C_SCAN": Fraction(1),
    }
    assert isinstance(exact, ExactExchangePrimitive)
    assert exact.coefficient == Fraction(1, 4)


ORACLES = {
    "SCAN": (
        [
            -0.3551059573279283,
            -0.9975207375849945,
            -0.8863490162412595,
            -0.01340322882820632,
            0.012274881217852467,
            -0.0272381214171394,
            0.02584448556839901,
            0.03197969793818749,
        ],
        [
            -0.34942360204036843,
            -0.9506250315017777,
            -0.007192458001472722,
            0.03307417335447151,
        ],
    ),
    "SCAN0": (
        [
            -0.26996146256031844,
            -0.7520953734416457,
            -0.6721825603170811,
            -0.008518061468923182,
            0.012274881217852467,
            -0.01889423091062299,
            0.01486924038938113,
            0.019470649666722486,
        ],
        [
            -0.26621419878740993,
            -0.7175193825466477,
            -0.0034299688450157694,
            0.01940330748501263,
        ],
    ),
}


def test_scan_scalar_values_match_pyscf_libxc_oracle():
    for name, (expected_p, expected_u) in ORACLES.items():
        p_spec = resolve_method(name, spin="polarized").primitives[0].functional
        u_spec = resolve_method(name, spin="unpolarized").primitives[0].functional
        actual_p = build_program(p_spec, order=1).evaluate(POLARIZED).ravel()
        actual_u = build_program(u_spec, order=1).evaluate(UNPOLARIZED).ravel()
        np.testing.assert_allclose(actual_p, expected_p, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(actual_u, expected_u, rtol=2e-12, atol=2e-13)


def _feature_at_alpha(alpha):
    k_factor = 3 / 10 * (6 * np.pi**2) ** (2 / 3)
    tau = SIGMA / (8 * N) + alpha * k_factor * 2 ** (-2 / 3) * N ** (5 / 3)
    return np.array([[N, SIGMA, tau]]).T


def test_scan_alpha_one_cutoff_matches_pinned_libxc_values():
    expected = {
        0.99: [
            -0.3257599186396406,
            -0.855904697450041,
            0.0037549963801816197,
            0.000154608930983691,
        ],
        1.0: [
            -0.32575893974375536,
            -0.8556314415341646,
            0.0037956091828432554,
            6.185228517857435e-05,
        ],
        1.03: [
            -0.32576103544623236,
            -0.8547930906946155,
            0.003920513779184946,
            -0.00021608380480091533,
        ],
    }
    program = build_program(resolve_method("SCAN").primitives[0].functional, order=1)
    for alpha, oracle in expected.items():
        actual = program.evaluate(_feature_at_alpha(alpha)).ravel()
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, oracle, rtol=2e-12, atol=2e-13)

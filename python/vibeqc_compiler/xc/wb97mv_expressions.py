# Copyright (C) 2017 M.A.L. Marques
# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See upstream/libxc/7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Audited Libxc 7.0.0 omegaB97M-V semilocal scalar expression.

This module represents only the density-dependent short-range exchange and
B97M correlation terms. Exact SR/LR exchange and VV10 are separate MethodIR
primitives; keeping those operators separate is required for capability,
cache, derivative, and provider identity.
"""

from __future__ import annotations

import math
import typing
from fractions import Fraction as F

from vibeqc_compiler.integral.expr import Graph

_COMPONENTS = frozenset(("MGGA_X_WB97M_V", "MGGA_C_WB97M_V"))
_K_FACTOR = 3 / 10 * (6 * math.pi**2) ** (2 / 3)
_CX = F(3, 8) * (3 / math.pi) ** (1 / 3) * 4 ** (2 / 3)

# Libxc 7.0.0 src/hyb_mgga_xc_wb97mv.c and
# maple/mgga_exc/hyb_mgga_xc_wb97mv.mpl.
_X_COEFFICIENTS = (F("0.85"), F("1.007"), F("0.259"))
_SS_COEFFICIENTS = (
    F("0.443"),
    F("-1.437"),
    F("-4.535"),
    F("-3.39"),
    F("4.278"),
)
_OS_COEFFICIENTS = (
    F("1.0"),
    F("1.358"),
    F("2.924"),
    F("-8.812"),
    F("-1.39"),
    F("9.142"),
)

# Modified PW92 parameters selected by Libxc's b97mv.mpl.
_PW_A = (F("0.0310907"), F("0.01554535"), F("0.0168869"))
_PW_ALPHA = (F("0.21370"), F("0.20548"), F("0.11125"))
_PW_B1 = (F("7.5957"), F("14.1189"), F("10.357"))
_PW_B2 = (F("3.5876"), F("6.1977"), F("3.6231"))
_PW_B3 = (F("1.6382"), F("3.3662"), F("0.88026"))
_PW_B4 = (F("0.49294"), F("0.62517"), F("0.49671"))
_PW_FZ20 = F("1.709920934161365617563962776245")


def _pw_epsilon(
    graph: typing.Any,
    density: typing.Any,
    zeta: typing.Any,
    up: typing.Any,
    down: typing.Any,
) -> typing.Any:
    """Modified PW92 correlation energy per electron."""
    rs = (3 / (4 * math.pi)) ** (1 / 3) * density.pow(-1 / 3)
    values = []
    for a, alpha, b1, b2, b3, b4 in zip(
        _PW_A, _PW_ALPHA, _PW_B1, _PW_B2, _PW_B3, _PW_B4, strict=True
    ):
        aux = b1 * rs.pow(0.5) + b2 * rs + b3 * rs.pow(1.5) + b4 * rs.pow(2)
        u = 1 / (2 * a * aux)
        values.append(-2 * a * (1 + alpha * rs) * graph.stable_unary("log1p", u))
    fz = (up.pow(4 / 3) + down.pow(4 / 3) - 2) / (2 ** (4 / 3) - 2)
    g0, g1, gm = values
    return g0 + zeta.pow(4) * fz * (g1 - g0 + gm / _PW_FZ20) - fz * gm / _PW_FZ20


def _attenuation_erf(graph: typing.Any, a: typing.Any) -> typing.Any:
    """Direct Libxc/Tawada short-range attenuation branch (a < 1.35)."""
    inverse_2a = 1 / (2 * a)
    aux1 = math.sqrt(math.pi) * graph.transcendental_unary("erf", inverse_2a)
    aux2 = graph.stable_unary("expm1", -1 / (4 * a.pow(2)))
    aux3 = 2 * a.pow(2) * aux2 + F(1, 2)
    return 1 - F(8, 3) * a * (aux1 + 2 * a * (aux2 - aux3))


def _u(gamma: typing.Any, x2: typing.Any) -> typing.Any:
    value = gamma * x2
    return value / (1 + value)


def _w_same(t: typing.Any) -> typing.Any:
    return (_K_FACTOR - t) / (_K_FACTOR + t)


def _w_opposite(ta: typing.Any, tb: typing.Any) -> typing.Any:
    product = 2 * ta * tb
    total = _K_FACTOR * (ta + tb)
    return (total - product) / (total + product)


def energy_expression(spec: typing.Any) -> typing.Any:
    """Return the omegaB97M-V semilocal energy-density DAG and variables."""
    active = {name for name, coefficient in spec.components if coefficient}
    if not active or not active <= _COMPONENTS:
        raise ValueError("omegaB97M-V expression received unsupported components")
    if "MGGA_X_WB97M_V" in active and spec.range_omega <= 0:
        raise ValueError("omegaB97M-V exchange requires positive range_omega")

    graph = Graph()
    variables = tuple(graph.variable(name) for name in spec.features)
    if spec.spin == "polarized":
        ra, rb, saa, _sab, sbb, ta, tb = variables
        n = ra + rb
        up, down = 2 * ra / n, 2 * rb / n
        zeta = (ra - rb) / n
    else:
        rho, sigma, tau = variables
        ra = rb = rho / 2
        saa = sbb = sigma / 4
        ta = tb = tau / 2
        n = rho
        up = down = graph.constant(1)
        zeta = graph.constant(0)

    xa2 = saa * ra.pow(-8 / 3)
    xb2 = sbb * rb.pow(-8 / 3)
    tsa = ta * ra.pow(-5 / 3)
    tsb = tb * rb.pow(-5 / 3)

    def exchange() -> typing.Any:
        terms = []
        gamma = F("0.004")
        c0, c_u, c_w = _X_COEFFICIENTS
        omega = spec.range_omega
        for density, x2, ts in ((ra, xa2, tsa), (rb, xb2, tsb)):
            a = omega / (2 * (6 * math.pi**2) ** (1 / 3) * density.pow(1 / 3))
            g = c0 + c_u * _u(gamma, x2) + c_w * _w_same(ts)
            terms.append(-_CX * density.pow(4 / 3) * _attenuation_erf(graph, a) * g)
        return graph.sum(terms)

    def correlation() -> typing.Any:
        total_epsilon = _pw_epsilon(graph, n, zeta, up, down)
        one = graph.constant(1)
        zero = graph.constant(0)
        two = graph.constant(2)
        ss_a = ra * _pw_epsilon(graph, ra, one, two, zero)
        ss_b = rb * _pw_epsilon(graph, rb, one, two, zero)
        opposite = n * total_epsilon - ss_a - ss_b

        ua = _u(F("0.2"), xa2)
        ub = _u(F("0.2"), xb2)
        wa, wb = _w_same(tsa), _w_same(tsb)
        css0, css_u4, css_w1, css_w2, css_w4u3 = _SS_COEFFICIENTS
        gssa = (
            css0
            + css_u4 * ua.pow(4)
            + css_w1 * wa
            + css_w2 * wa.pow(2)
            + css_w4u3 * wa.pow(4) * ua.pow(3)
        )
        gssb = (
            css0
            + css_u4 * ub.pow(4)
            + css_w1 * wb
            + css_w2 * wb.pow(2)
            + css_w4u3 * wb.pow(4) * ub.pow(3)
        )

        uos = _u(F("0.006"), (xa2 + xb2) / 2)
        wos = _w_opposite(tsa, tsb)
        cos0, cos_w1, cos_w2, cos_w2u1, cos_w6, cos_w6u1 = _OS_COEFFICIENTS
        gos = (
            cos0
            + cos_w1 * wos
            + cos_w2 * wos.pow(2)
            + cos_w2u1 * wos.pow(2) * uos
            + cos_w6 * wos.pow(6)
            + cos_w6u1 * wos.pow(6) * uos
        )
        return ss_a * gssa + ss_b * gssb + opposite * gos

    builders = {
        "MGGA_X_WB97M_V": exchange,
        "MGGA_C_WB97M_V": correlation,
    }
    return (
        graph,
        graph.sum(
            coefficient * builders[name]()
            for name, coefficient in spec.components
            if coefficient
        ),
        variables,
    )

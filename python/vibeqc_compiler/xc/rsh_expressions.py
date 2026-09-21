# Copyright (C) 2017 M.A.L. Marques
# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See external/libxc-7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Audited Libxc 7.0 extended GGA/hybrid scalar expressions."""

from __future__ import annotations

import math
import typing
from fractions import Fraction as F

from vibeqc_compiler.integral.expr import Expr, Graph

from .p86_pz_maple import p86_correlation, pz_correlation
from .rsh_maple import lyp_correlation


def energy_expression(spec: typing.Any) -> typing.Any:
    """Return the range-separated semilocal energy DAG and feature variables."""
    graph = Graph()
    variables = tuple(graph.variable(name) for name in spec.features)
    if spec.spin == "polarized":
        ra, rb, saa, sab, sbb, _, _ = variables
    else:
        rho, sigma, _ = variables
        ra = rb = rho / 2
        saa = sab = sbb = sigma / 4
    n = ra + rb
    if spec.spin == "polarized":
        up, down = 2 * ra / n, 2 * rb / n
        z = (ra - rb) / n
    else:
        up = down = graph.constant(1)
        z = graph.constant(0)
    rs = (3 / (4 * math.pi)) ** (1 / 3) * n.pow(-1 / 3)
    cx = F(3, 8) * (3 / math.pi) ** (1 / 3) * 4 ** (2 / 3)

    def lda_exchange() -> Expr:
        return graph.sum(-cx * density.pow(4 / 3) for density in (ra, rb))

    def pw91_exchange() -> Expr:
        x2s = 1 / (2 * (6 * math.pi**2) ** (1 / 3))
        a = F("0.19645")
        b = F("7.7956")
        c = F("0.2743")
        d = F("-0.1508")
        f = F("0.004")
        alpha = F(100)
        terms = []
        for density, sigma in ((ra, saa), (rb, sbb)):
            s2 = x2s**2 * sigma * density.pow(-8 / 3)
            s = s2.pow(0.5)
            s4 = s2.pow(2)
            numerator = (c + d * graph.exponential(-alpha * s2)) * s2 - f * s4
            denominator = (
                1 + a * s * graph.transcendental_unary("asinh", b * s) + f * s4
            )
            enhancement = 1 + numerator / denominator
            terms.append(-cx * density.pow(4 / 3) * enhancement)
        return graph.sum(terms)

    def pw92_epsilon() -> Expr:
        parameters = {
            "a": ("0.031091", "0.015545", "0.016887"),
            "alpha": ("0.21370", "0.20548", "0.11125"),
            "b1": ("7.5957", "14.1189", "10.357"),
            "b2": ("3.5876", "6.1977", "3.6231"),
            "b3": ("1.6382", "3.3662", "0.88026"),
            "b4": ("0.49294", "0.62517", "0.49671"),
        }
        values = []
        for i in range(3):
            aux = (
                F(parameters["b1"][i]) * rs.pow(0.5)
                + F(parameters["b2"][i]) * rs
                + F(parameters["b3"][i]) * rs.pow(1.5)
                + F(parameters["b4"][i]) * rs.pow(2)
            )
            u = 1 / (2 * F(parameters["a"][i]) * aux)
            values.append(
                -2
                * F(parameters["a"][i])
                * (1 + F(parameters["alpha"][i]) * rs)
                * graph.stable_unary("log1p", u)
            )
        fz = (up.pow(4 / 3) + down.pow(4 / 3) - 2) / (2 ** (4 / 3) - 2)
        fz20 = F("1.709921")
        g0, g1, gm = values
        return g0 + z.pow(4) * fz * (g1 - g0 + gm / fz20) - fz * gm / fz20

    def pw92_correlation() -> Expr:
        return n * pw92_epsilon()

    def pw91_correlation() -> Expr:
        epsilon = pw92_epsilon()
        phi = (up.pow(2 / 3) + down.pow(2 / 3)) / 2
        phi3 = phi.pow(3)
        total_sigma = saa + 2 * sab + sbb
        t2 = total_sigma * n.pow(-8 / 3) / (16 * 2 ** (2 / 3) * phi.pow(2) * rs)
        alpha = F("0.09")
        c0 = F("0.004235")
        nu = 16 / math.pi * (3 * math.pi**2) ** (1 / 3)
        beta = nu * c0
        c1 = beta**2 / (2 * alpha)
        c2 = 2 * alpha / beta
        a_term = c2 / graph.stable_unary(
            "expm1", -2 * alpha * epsilon / (phi3 * beta**2)
        )
        h0 = (
            c1
            * phi3
            * graph.stable_unary(
                "log1p",
                c2
                * (t2 + a_term * t2.pow(2))
                / (1 + a_term * t2 + a_term.pow(2) * t2.pow(2)),
            )
        )
        rg_c_xc = (F("2.568") + F("23.266") * rs + F("0.007389") * rs.pow(2)) / (
            1000 * (1 + F("8.723") * rs + F("0.472") * rs.pow(2))
        )
        c_xc0 = F("0.002568")
        c_x = F("-0.001667")
        h_a1 = -100 * 4 / math.pi * (4 / (9 * math.pi)) ** (1 / 3)
        h1 = (
            nu
            * (rg_c_xc - c_xc0 - 3 * c_x / 7)
            * phi3
            * t2
            * graph.exponential(h_a1 * rs * phi.pow(4) * t2)
        )
        return n * (epsilon + h0 + h1)

    def b88_enhancement(density: typing.Any, sigma: typing.Any) -> Expr:
        beta_b88 = F("0.0042")
        gamma_b88 = F(6)
        x2 = sigma * density.pow(-8 / 3)
        x = x2.pow(0.5)
        return 1 + beta_b88 / cx * x2 / (
            1 + gamma_b88 * beta_b88 * x * graph.transcendental_unary("asinh", x)
        )

    def b88_exchange(short_range: typing.Any = False) -> Expr:
        terms = []
        omega = spec.range_omega
        for density, sigma in ((ra, saa), (rb, sbb)):
            enhancement = b88_enhancement(density, sigma)
            if short_range:
                # Iikura-Tsuneda-Yanai-Hirao short-range B88 attenuation.
                k_gga = (9 * math.pi / (2 * cx * enhancement)).pow(0.5) * density.pow(
                    1 / 3
                )
                a = omega / (2 * k_gga)
                inverse_2a = 1 / (2 * a)
                aux1 = math.sqrt(math.pi) * graph.transcendental_unary(
                    "erf", inverse_2a
                )
                aux2 = graph.stable_unary("expm1", -1 / (4 * a.pow(2)))
                aux3 = 2 * a.pow(2) * aux2 + F(1, 2)
                attenuation = 1 - F(8, 3) * a * (aux1 + 2 * a * (aux2 - aux3))
                enhancement = enhancement * attenuation
            terms.append(-cx * density.pow(4 / 3) * enhancement)
        return graph.sum(terms)

    def vwn_correlation() -> Expr:
        # VWN5 parameters in Hartree, matching Libxc 7.0 lda_c_vwn.mpl.
        av = (F("0.0310907"), F("0.01554535"), -1 / (6 * math.pi**2))
        bv = (F("3.72744"), F("7.06042"), F("1.13107"))
        cv = (F("12.9352"), F("18.0578"), F("13.0045"))
        x0v = (F("-0.10498"), F("-0.32500"), F("-0.0047584"))

        def aux(index: typing.Any) -> Expr:
            aa, bb, cc, x0 = av[index], bv[index], cv[index], x0v[index]
            q = math.sqrt(float(4 * cc - bb * bb))
            root = rs.pow(0.5)
            fx = rs + bb * root + cc
            f1 = 2 * bb / q
            f2 = bb * x0 / (x0 * x0 + bb * x0 + cc)
            f3 = 2 * (2 * x0 + bb) / q
            return aa * (
                graph.stable_unary("log", rs / fx)
                + (f1 - f2 * f3)
                * graph.transcendental_unary("atan", q / (2 * root + bb))
                - f2 * graph.stable_unary("log", (root - x0).pow(2) / fx)
            )

        fz = (up.pow(4 / 3) + down.pow(4 / 3) - 2) / (2 ** (4 / 3) - 2)
        fpp = 4 / (9 * (2 ** (1 / 3) - 1))
        g0, g1, gm = aux(0), aux(1), aux(2)
        epsilon = g0 + gm * fz * (1 - z.pow(4)) / fpp + (g1 - g0) * fz * z.pow(4)
        return n * epsilon

    def vwn_rpa_correlation() -> Expr:
        # Libxc 7.0 LDA_C_VWN_RPA (ID 8): the RPA parameterization and
        # two-endpoint spin interpolation used by its canonical B3LYP.
        av = (F("0.0310907"), F("0.01554535"))
        bv = (F("13.0720"), F("20.1231"))
        cv = (F("42.7198"), F("101.578"))
        x0v = (F("-0.409286"), F("-0.743294"))

        def aux(index: int) -> Expr:
            aa, bb, cc, x0 = av[index], bv[index], cv[index], x0v[index]
            q = math.sqrt(float(4 * cc - bb * bb))
            root = rs.pow(0.5)
            fx = rs + bb * root + cc
            f1 = 2 * bb / q
            f2 = bb * x0 / (x0 * x0 + bb * x0 + cc)
            f3 = 2 * (2 * x0 + bb) / q
            return aa * (
                graph.stable_unary("log", rs / fx)
                + (f1 - f2 * f3)
                * graph.transcendental_unary("atan", q / (2 * root + bb))
                - f2 * graph.stable_unary("log", (root - x0).pow(2) / fx)
            )

        fz = (up.pow(4 / 3) + down.pow(4 / 3) - 2) / (2 ** (4 / 3) - 2)
        epsilon = aux(0) * (1 - fz) + aux(1) * fz
        return n * epsilon

    builders = {
        "LDA_X": lda_exchange,
        "GGA_X_B88": lambda: b88_exchange(False),
        "GGA_X_ITYH": lambda: b88_exchange(True),
        "GGA_X_PW91": pw91_exchange,
        "LDA_C_PW": pw92_correlation,
        "GGA_C_PW91": pw91_correlation,
        "LDA_C_PZ": lambda: pz_correlation(graph, spec, variables),
        "GGA_C_P86": lambda: p86_correlation(graph, spec, variables),
        "LDA_C_VWN": vwn_correlation,
        "LDA_C_VWN_RPA": vwn_rpa_correlation,
        "GGA_C_LYP": lambda: lyp_correlation(graph, spec, variables),
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

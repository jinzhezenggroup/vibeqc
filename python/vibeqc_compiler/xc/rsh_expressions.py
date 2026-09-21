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

from .b88_vwn_maple import b88_exchange as maple_b88_exchange
from .b88_vwn_maple import vwn_correlation as maple_vwn_correlation
from .p86_pz_maple import p86_correlation, pz_correlation
from .pw91_maple import pw91_correlation as imported_pw91_correlation
from .pw91_maple import pw91_exchange as imported_pw91_exchange
from .rsh_maple import lyp_correlation


def energy_expression(spec: typing.Any, *, production: bool = False) -> typing.Any:
    """Return the extended-GGA semilocal energy DAG and feature variables.

    ``production`` enables only algebraically exact B3-family endpoint
    continuations plus the declared numerical vacuum cutoff.  The interior
    expression remains available as an independent oracle/compatibility path.
    """
    if type(production) is not bool:
        raise TypeError("production must be bool")
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
        x_asinh_x = x * graph.transcendental_unary("asinh", x)
        if production:
            # x*asinh(x) is analytic in y=x^2 at y=0.  Writing its
            # low-y branch as a polynomial gives the exact derivative limit
            # needed by B88 at zero density gradient; the old sqrt derivative
            # produced NaNs even though the physical v_sigma is finite.
            y = x2
            series = y * (
                1 + y * (-F(1, 6) + y * (F(3, 40) + y * (-F(5, 112) + y * F(35, 1152))))
            )
            x_asinh_x = graph.select_le(y, F(1, 100000000), series, x_asinh_x)
        return 1 + beta_b88 / cx * x2 / (1 + gamma_b88 * beta_b88 * x_asinh_x)

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
            term = -cx * density.pow(4 / 3) * enhancement
            if production and not short_range:
                # Physical zero-spin density carries zero same-spin gradient.
                # Keep the exact zero-density exchange limit lazy so inverse
                # density powers in the inactive B88 branch are never formed.
                term = graph.select_le(density, 0, 0, term)
            terms.append(term)
        return graph.sum(terms)

    def b3lyp_lyp_tail_continuation() -> Expr:
        a_lyp = F("0.04918")
        b_lyp = F("0.132")
        c_lyp = F("0.2533")
        d_lyp = F("0.349")
        cf = F(3, 10) * (3 * math.pi**2) ** (2 / 3)
        rr = n.pow(-1 / 3)
        omega_lyp = b_lyp * graph.exponential(-c_lyp * rr) / (1 + d_lyp * rr)
        delta = (c_lyp + d_lyp / (1 + d_lyp * rr)) * rr
        one_minus_z2 = 1 - z.pow(2)
        n_m83 = n.pow(-8 / 3)
        xt2 = (saa + 2 * sab + sbb) * n_m83
        up8 = up.pow(8 / 3)
        down8 = down.pow(8 / 3)
        t1 = -one_minus_z2 / (1 + d_lyp * rr)
        t2 = -xt2 * (one_minus_z2 * (47 - 7 * delta) / 72 - F(2, 3))
        t3 = -cf / 2 * one_minus_z2 * (up8 + down8)
        aux6 = 1 / 2 ** (8 / 3)
        aux4 = aux6 / 4
        aux5 = aux4 / 18
        if production:
            # Cancel the apparent rho_spin^(-8/3) factors analytically before
            # differentiation.  These identities are exact for positive spin
            # density and provide the correct fully polarized limit at zero.
            two83 = 2 ** (8 / 3)
            xs0_up8 = saa * two83 * n_m83
            xs1_down8 = sbb * two83 * n_m83
            xs0_up11 = saa * 2 ** (11 / 3) * ra * n.pow(-11 / 3)
            xs1_down11 = sbb * 2 ** (11 / 3) * rb * n.pow(-11 / 3)
        else:
            xs02 = saa * ra.pow(-8 / 3)
            xs12 = sbb * rb.pow(-8 / 3)
            xs0_up8 = xs02 * up8
            xs1_down8 = xs12 * down8
            xs0_up11 = xs02 * up.pow(11 / 3)
            xs1_down11 = xs12 * down.pow(11 / 3)
        t4 = aux4 * one_minus_z2 * (F(5, 2) - delta / 18) * (xs0_up8 + xs1_down8)
        t5 = aux5 * one_minus_z2 * (delta - 11) * (xs0_up11 + xs1_down11)
        t6 = -aux6 * (
            F(2, 3) * (xs0_up8 + xs1_down8)
            - up.pow(2) * xs1_down8 / 4
            - down.pow(2) * xs0_up8 / 4
        )
        return n * a_lyp * (t1 + omega_lyp * (t2 + t3 + t4 + t5 + t6))

    builders = {
        "LDA_X": lda_exchange,
        "GGA_X_B88": lambda: (
            b88_exchange(False)
            if production
            else maple_b88_exchange(graph, spec, variables)
        ),
        "GGA_X_ITYH": lambda: b88_exchange(True),
        "GGA_X_PW91": lambda: imported_pw91_exchange(graph, spec, variables),
        "LDA_C_PW": pw92_correlation,
        "GGA_C_PW91": lambda: imported_pw91_correlation(graph, spec, variables),
        "LDA_C_PZ": lambda: pz_correlation(graph, spec, variables),
        "GGA_C_P86": lambda: p86_correlation(graph, spec, variables),
        "LDA_C_VWN": lambda: maple_vwn_correlation(graph, spec, variables, "LDA_C_VWN"),
        "LDA_C_VWN_RPA": lambda: maple_vwn_correlation(
            graph, spec, variables, "LDA_C_VWN_RPA"
        ),
        "GGA_C_LYP": lambda: (
            b3lyp_lyp_tail_continuation()
            if production
            else lyp_correlation(graph, spec, variables)
        ),
    }
    energy = graph.sum(
        coefficient * builders[name]()
        for name, coefficient in spec.components
        if coefficient
    )
    if production:
        supported = {"LDA_X", "GGA_X_B88", "LDA_C_VWN_RPA", "GGA_C_LYP"}
        active = {name for name, coefficient in spec.components if coefficient}
        if not active <= supported:
            raise ValueError(
                "production tail is qualified only for the canonical B3LYP semilocal family"
            )
        # The finite quadrature policy treats <=1e-18 total density as vacuum.
        # This branch is lazy in generated C, so singular positive-density
        # algebra is never evaluated in the discarded numerical tail.
        energy = graph.select_le(n, F(1, 10**18), 0, energy)
    return graph, energy, variables

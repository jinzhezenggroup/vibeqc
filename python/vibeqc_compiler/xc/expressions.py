# Copyright (C) 2017 M.A.L. Marques
# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See external/libxc-7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Audited Libxc 7.0.0 expressions, translated to the existing scalar DAG.

The exact upstream files/hashes are in external/libxc-7.0.0/manifest.json.
Screening branches are deliberately excluded: the separately versioned domain
contract rejects their inputs. Energy is per volume throughout this module.
"""

import math
from fractions import Fraction as F

from vibeqc_compiler.integral.expr import Graph

_PW_PARAMETERS = {
    False: {
        "a": ("0.031091", "0.015545", "0.016887"),
        "alpha": ("0.21370", "0.20548", "0.11125"),
        "b1": ("7.5957", "14.1189", "10.357"),
        "b2": ("3.5876", "6.1977", "3.6231"),
        "b3": ("1.6382", "3.3662", "0.88026"),
        "b4": ("0.49294", "0.62517", "0.49671"),
    },
    True: {
        "a": ("0.0310907", "0.01554535", "0.0168869"),
        "alpha": ("0.21370", "0.20548", "0.11125"),
        "b1": ("7.5957", "14.1189", "10.357"),
        "b2": ("3.5876", "6.1977", "3.6231"),
        "b3": ("1.6382", "3.3662", "0.88026"),
        "b4": ("0.49294", "0.62517", "0.49671"),
    },
}


def lda_xc_pw_unpolarized_tail_expression():
    """Return a positive-density LDA DAG without inverse-density overflow.

    With ``x=rho^(1/6)``, the PW92 low-density intermediates become bounded
    polynomials in ``x``. The returned derivative is still exactly dE/d(rho);
    zero density remains a caller-owned analytic limit rather than a branch in
    the expression graph.
    """

    graph = Graph()
    x = graph.variable("rho_sixth_root")
    parameters = _PW_PARAMETERS[False]
    a = F(parameters["a"][0])
    alpha = F(parameters["alpha"][0])
    b1 = F(parameters["b1"][0])
    b2 = F(parameters["b2"][0])
    b3 = F(parameters["b3"][0])
    b4 = F(parameters["b4"][0])
    c = (3 / (4 * math.pi)) ** (1 / 3)
    sqrt_c = math.sqrt(c)
    d = alpha * c
    q = b1 * sqrt_c * x.pow(3) + b2 * c * x.pow(2) + b3 * c**1.5 * x + b4 * c**2
    u = x.pow(4) / (2 * a * q)
    log_term = graph.stable_unary("log1p", u)
    correlation = -2 * a * x.pow(4) * (x.pow(2) + d) * log_term

    cx = F(3, 8) * (3 / math.pi) ** (1 / 3) * 4 ** (2 / 3)
    exchange_coefficient = -2 * cx / 2 ** (4 / 3)
    exchange = exchange_coefficient * x.pow(8)

    q_derivative = 3 * b1 * sqrt_c * x.pow(2) + 2 * b2 * c * x + b3 * c**1.5
    correlation_derivative = (
        -2
        * a
        * (
            (1 + 2 * d / (3 * x.pow(2))) * log_term
            + (x + d / x) / 6 * (u / (1 + u)) * (4 / x - q_derivative / q)
        )
    )
    exchange_derivative = F(4, 3) * exchange_coefficient * x.pow(2)
    return (
        graph,
        exchange + correlation,
        exchange_derivative + correlation_derivative,
        x,
    )


def energy_expression(spec):
    """Return the uninterpreted energy DAG and its ordered feature variables."""
    graph = Graph()
    variables = tuple(graph.variable(name) for name in spec.features)
    if spec.spin == "polarized":
        ra, rb, saa, sab, sbb, ta, tb = variables
    else:
        rho, sigma, tau = variables
        ra = rb = rho / 2
        saa = sab = sbb = sigma / 4
        ta = tb = tau / 2
    n = ra + rb
    if spec.spin == "polarized":
        # Ratios avoid cancellation in 1 +/- z near complete spin polarization.
        up, down = 2 * ra / n, 2 * rb / n
        z = (ra - rb) / n
    else:
        # The declared unpolarized contract has ra=rb=n/2 identically. Keep
        # those exact constants out of the generated graph so the rho->0+
        # limit never evaluates a numerically meaningless rho/rho quotient.
        up = down = graph.constant(1)
        z = graph.constant(0)
    rs = (3 / (4 * math.pi)) ** (1 / 3) * n.pow(-1 / 3)
    beta = F("0.06672455060314922")
    gamma = (1 - math.log(2)) / math.pi**2
    kappa = F("0.8040")
    mu = beta * graph.constant(math.pi**2) / 3
    cx = F(3, 8) * (3 / math.pi) ** (1 / 3) * 4 ** (2 / 3)
    x2s = 1 / (2 * (6 * math.pi**2) ** (1 / 3))
    x2s2 = x2s**2
    k_factor = 3 / 10 * (6 * math.pi**2) ** (2 / 3)
    fz = (up.pow(4 / 3) + down.pow(4 / 3) - 2) / (2 ** (4 / 3) - 2)

    def pw(modified, *, with_rs_derivative=False):
        parameters = _PW_PARAMETERS[modified]
        a = parameters["a"]
        alpha = parameters["alpha"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]
        b3 = parameters["b3"]
        b4 = parameters["b4"]
        values = []
        derivatives = []
        for i in range(3):
            aux = (
                F(b1[i]) * rs.pow(0.5)
                + F(b2[i]) * rs
                + F(b3[i]) * rs.pow(1.5)
                + F(b4[i]) * rs.pow(2)
            )
            u = 1 / (2 * F(a[i]) * aux)
            log_term = graph.stable_unary("log1p", u)
            values.append(-2 * F(a[i]) * (1 + F(alpha[i]) * rs) * log_term)
            if with_rs_derivative:
                aux_prime = (
                    F(b1[i]) / 2 * rs.pow(-0.5)
                    + F(b2[i])
                    + F(3, 2) * F(b3[i]) * rs.pow(0.5)
                    + 2 * F(b4[i]) * rs
                )
                derivatives.append(
                    -2
                    * F(a[i])
                    * (
                        F(alpha[i]) * log_term
                        - (1 + F(alpha[i]) * rs) * (u / (1 + u)) * aux_prime / aux
                    )
                )
        fz20 = F("1.709920934161365617563962776245" if modified else "1.709921")

        def combine(items):
            g0, g1, gm = items
            return g0 + z.pow(4) * fz * (g1 - g0 + gm / fz20) - fz * gm / fz20

        value = combine(values)
        return (value, combine(derivatives)) if with_rs_derivative else value

    def exchange(gga):
        terms = []
        for density, sigma in ((ra, saa), (rb, sbb)):
            enhancement = 1
            if gga:
                s2 = x2s2 * sigma * density.pow(-8 / 3)
                enhancement = 1 + kappa * (1 - kappa / (kappa + mu * s2))
            terms.append(-cx * density.pow(4 / 3) * enhancement)
        return graph.sum(terms)

    def correlation(gga, modified):
        eps = pw(modified)
        if gga:
            phi = (up.pow(2 / 3) + down.pow(2 / 3)) / 2
            phi3 = phi.pow(3)
            # Use squared reduced gradient directly: derivatives remain finite
            # at sigma=0, unlike differentiating an intermediate sqrt(sigma).
            t2 = (
                (saa + 2 * sab + sbb)
                * n.pow(-8 / 3)
                / (16 * 2 ** (2 / 3) * phi.pow(2) * rs)
            )
            a = beta / (gamma * graph.stable_unary("expm1", -eps / (gamma * phi3)))
            f1 = t2 + a * t2.pow(2)
            eps = eps + gamma * phi3 * graph.stable_unary(
                "log1p", beta * f1 / (gamma * (1 + a * f1))
            )
        return n * eps

    def r2_switch(alpha, coefficients, c1, c2, d):
        polynomial = graph.sum(
            coefficient * alpha.pow(power)
            for power, coefficient in enumerate(coefficients)
        )
        negative = graph.exponential(-c1 * alpha / (1 - alpha))
        large = -d * graph.exponential(c2 / (1 - alpha))
        return graph.select_le(
            alpha,
            0,
            negative,
            graph.select_le(alpha, F(5, 2), polynomial, large),
        )

    r2_x_coefficients = tuple(
        F(value)
        for value in (
            "1",
            "-0.667",
            "-0.4445555",
            "-0.663086601049",
            "1.451297044490",
            "-0.887998041597",
            "0.234528941479",
            "-0.023185843322",
        )
    )
    r2_c_coefficients = tuple(
        F(value)
        for value in (
            "1",
            "-0.64",
            "-0.4352",
            "-1.535685604549",
            "3.061560252175",
            "-1.915710236206",
            "0.516884468372",
            "-0.051848879792",
        )
    )

    def scan_switch(alpha, c1, c2, d):
        epsilon = 2.220446049250313e-16
        log_epsilon = -math.log(epsilon)
        left_cutoff = log_epsilon / (log_epsilon + float(c1))
        right_log = -math.log(epsilon / float(d))
        right_cutoff = (right_log + float(c2)) / right_log
        left = graph.select_le(
            alpha,
            left_cutoff,
            graph.exponential(-c1 * alpha / (1 - alpha)),
            0,
        )
        right = graph.select_le(
            alpha,
            right_cutoff,
            0,
            -d * graph.exponential(c2 / (1 - alpha)),
        )
        return graph.select_le(alpha, 1, left, right)

    def scan_exchange():
        k1 = F("0.065")
        h0 = F("1.174")
        c1 = F("0.667")
        c2 = F("0.8")
        d = F("1.24")
        a1 = F("4.9479")
        mu_ge = F(10, 81)
        b2 = math.sqrt(5913 / 405000)
        b1 = F(511, 13500) / (2 * b2)
        b3 = F(1, 2)
        b4 = mu_ge**2 / k1 - F(1606, 18225) - b1**2

        terms = []
        for density, sigma, tau in ((ra, saa, ta), (rb, sbb, tb)):
            x2 = sigma * density.pow(-8 / 3)
            p = x2s2 * x2
            alpha = (tau * density.pow(-5 / 3) - x2 / 8) / k_factor
            y = (
                mu_ge * p
                + b4 * p.pow(2) * graph.exponential(-b4 * p / mu_ge)
                + (
                    b1 * p
                    + b2 * (1 - alpha) * graph.exponential(-b3 * (1 - alpha).pow(2))
                ).pow(2)
            )
            h1 = 1 + k1 * (1 - k1 / (k1 + y))
            f_alpha = scan_switch(alpha, c1, c2, d)
            gx = graph.select_le(
                x2,
                0,
                1,
                1 - graph.exponential(-a1 / (math.sqrt(x2s) * x2.pow(0.25))),
            )
            enhancement = (h1 + f_alpha * (h0 - h1)) * gx
            terms.append(-cx * density.pow(4 / 3) * enhancement)
        return graph.sum(terms)

    def scan_correlation():
        phi = (up.pow(2 / 3) + down.pow(2 / 3)) / 2
        spin_fifth = (up.pow(5 / 3) + down.pow(5 / 3)) / 2
        total_sigma = saa + 2 * sab + sbb
        xt2 = total_sigma * n.pow(-8 / 3)
        normalized_tau = (ta + tb) * n.pow(-5 / 3)
        alpha = (normalized_tau - xt2 / 8) / (k_factor * 2 ** (-2 / 3) * spin_fifth)
        f_alpha = scan_switch(alpha, F("0.64"), F("1.5"), F("0.7"))

        pw_value = pw(True)
        phi3 = phi.pow(3)
        w1 = graph.stable_unary("expm1", -pw_value / (gamma * phi3))
        beta_rs = (
            F("0.066724550603149220") * (1 + F("0.1") * rs) / (1 + F("0.1778") * rs)
        )
        t2 = xt2 / (16 * 2 ** (2 / 3) * phi.pow(2) * rs)
        y = beta_rs * t2 / (gamma * w1)
        g = (1 + 4 * y).pow(-0.25)
        h = gamma * phi3 * graph.stable_unary("log1p", w1 * (1 - g))
        ec1 = pw_value + h

        s2 = x2s2 * 2 ** (2 / 3) * xt2
        b1c = F("0.0285764")
        b2c = F("0.0889")
        b3c = F("0.125541")
        eclda0 = -b1c / (1 + b2c * rs.pow(0.5) + b3c * rs)
        gc = (1 - F("2.363") * (2 ** (1 / 3) - 1) * fz) * (1 - z.pow(12))
        chi_infinity = F("0.12802585262625815")
        g_infinity = (1 + 4 * chi_infinity * s2).pow(-0.25)
        h0 = b1c * graph.stable_unary(
            "log1p",
            graph.stable_unary("expm1", -eclda0 / b1c) * (1 - g_infinity),
        )
        ec0 = (eclda0 + h0) * gc
        return n * (ec1 + f_alpha * (ec0 - ec1))

    def r2scan_exchange():
        eta = F("0.001")
        dp2 = F("0.361")
        k1 = F("0.065")
        h0 = F("1.174")
        c1 = F("0.667")
        c2 = F("0.8")
        d = F("1.24")
        a1 = F("4.9479")
        mu_ge = F(10, 81)
        cn = F(20, 27) + eta * F(5, 3)
        c2_coefficient = -sum(
            (power + 1) * coefficient
            for power, coefficient in enumerate(r2_x_coefficients)
        ) * (1 - h0)

        terms = []
        for density, sigma, tau in ((ra, saa, ta), (rb, sbb, tb)):
            x2 = sigma * density.pow(-8 / 3)
            p = x2s2 * x2
            alpha = (tau * density.pow(-5 / 3) - x2 / 8) / (k_factor + eta * x2 / 8)
            x_r2 = (
                cn * c2_coefficient * graph.exponential(-p.pow(2) / dp2**4) + mu_ge
            ) * p
            h1 = 1 + k1 * (1 - k1 / (k1 + x_r2))
            f_alpha = r2_switch(alpha, r2_x_coefficients, c1, c2, d)
            gx = graph.select_le(
                x2,
                0,
                1,
                1 - graph.exponential(-a1 / (math.sqrt(x2s) * x2.pow(0.25))),
            )
            enhancement = (h1 + f_alpha * (h0 - h1)) * gx
            terms.append(-cx * density.pow(4 / 3) * enhancement)
        return graph.sum(terms)

    def r2scan_correlation():
        eta = F("0.001")
        dp2 = F("0.361")
        phi = (up.pow(2 / 3) + down.pow(2 / 3)) / 2
        phi3 = phi.pow(3)
        spin_fifth = (up.pow(5 / 3) + down.pow(5 / 3)) / 2
        total_sigma = saa + 2 * sab + sbb
        xt2 = total_sigma * n.pow(-8 / 3)
        normalized_tau = (ta + tb) * n.pow(-5 / 3)
        alpha = (normalized_tau - xt2 / 8) / (
            k_factor * 2 ** (-2 / 3) * spin_fifth + eta * xt2 / 8
        )
        f_alpha = r2_switch(
            alpha,
            r2_c_coefficients,
            F("0.64"),
            F("1.5"),
            F("0.7"),
        )

        pw_value, pw_rs_derivative = pw(True, with_rs_derivative=True)
        w1 = graph.stable_unary("expm1", -pw_value / (gamma * phi3))
        s2 = x2s2 * 2 ** (2 / 3) * xt2
        t2 = xt2 / (16 * 2 ** (2 / 3) * phi.pow(2) * rs)

        b1c = F("0.0285764")
        b2c = F("0.0889")
        b3c = F("0.125541")
        e0_denominator = 1 + b2c * rs.pow(0.5) + b3c * rs
        eclda0 = -b1c / e0_denominator
        eclda0_rs_derivative = (
            b1c * (b2c / (2 * rs.pow(0.5)) + b3c) / e0_denominator.pow(2)
        )
        gc = (1 - F("2.363") * (2 ** (1 / 3) - 1) * fz) * (1 - z.pow(12))
        elsda0 = eclda0 * gc
        delsda0 = eclda0_rs_derivative * gc
        beta_rs = (
            F("0.066724550603149220") * (1 + F("0.1") * rs) / (1 + F("0.1778") * rs)
        )
        dfc2 = sum(
            power * coefficient
            for power, coefficient in enumerate(r2_c_coefficients)
            if power
        )
        dy = (
            dfc2
            / (27 * gamma * spin_fifth * phi3 * w1)
            * (20 * rs * (delsda0 - pw_rs_derivative) - 45 * eta * (elsda0 - pw_value))
            * s2
            * graph.exponential(-s2.pow(2) / dp2**4)
        )
        y = beta_rs * t2 / (gamma * w1)
        g = (1 + 4 * (y - dy)).pow(-0.25)
        h = gamma * phi3 * graph.stable_unary("log1p", w1 * (1 - g))
        ec1 = pw_value + h

        chi_infinity = F("0.12802585262625815")
        g_infinity = (1 + 4 * chi_infinity * s2).pow(-0.25)
        h0 = b1c * graph.stable_unary(
            "log1p",
            graph.stable_unary("expm1", -eclda0 / b1c) * (1 - g_infinity),
        )
        ec0 = (eclda0 + h0) * gc
        return n * (ec1 + f_alpha * (ec0 - ec1))

    builders = {
        "LDA_X": lambda: exchange(False),
        "GGA_X_PBE": lambda: exchange(True),
        "LDA_C_PW": lambda: correlation(False, False),
        "LDA_C_PW_MOD": lambda: correlation(False, True),
        "GGA_C_PBE": lambda: correlation(True, True),
        "MGGA_X_SCAN": scan_exchange,
        "MGGA_C_SCAN": scan_correlation,
        "MGGA_X_R2SCAN": r2scan_exchange,
        "MGGA_C_R2SCAN": r2scan_correlation,
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

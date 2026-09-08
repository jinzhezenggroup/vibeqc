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

from tools.vibeqc_codegen.expr import Graph


def energy_expression(spec):
    """Return the uninterpreted energy DAG and its ordered feature variables."""
    graph = Graph()
    variables = tuple(graph.variable(name) for name in spec.features)
    if spec.spin == "polarized":
        ra, rb, saa, sab, sbb, _, _ = variables
    else:
        rho, sigma, _ = variables
        ra = rb = rho / 2
        saa = sab = sbb = sigma / 4
    n = ra + rb
    # Ratios avoid cancellation in 1 +/- z near complete spin polarization.
    up, down = 2 * ra / n, 2 * rb / n
    z = (ra - rb) / n
    rs = (3 / (4 * math.pi)) ** (1 / 3) * n.pow(-1 / 3)
    beta = F("0.06672455060314922")
    gamma = (1 - math.log(2)) / math.pi**2
    kappa = F("0.8040")
    mu = beta * graph.constant(math.pi**2) / 3
    cx = F(3, 8) * (3 / math.pi) ** (1 / 3) * 4 ** (2 / 3)
    x2s2 = 1 / (4 * (6 * math.pi**2) ** (2 / 3))

    def pw(modified):
        a = (
            ("0.0310907", "0.01554535", "0.0168869")
            if modified
            else ("0.031091", "0.015545", "0.016887")
        )
        alpha = ("0.21370", "0.20548", "0.11125")
        b1 = ("7.5957", "14.1189", "10.357")
        b2 = ("3.5876", "6.1977", "3.6231")
        b3 = ("1.6382", "3.3662", "0.88026")
        b4 = ("0.49294", "0.62517", "0.49671")
        values = []
        for i in range(3):
            aux = (
                F(b1[i]) * rs.pow(0.5)
                + F(b2[i]) * rs
                + F(b3[i]) * rs.pow(1.5)
                + F(b4[i]) * rs.pow(2)
            )
            values.append(
                -2
                * F(a[i])
                * (1 + F(alpha[i]) * rs)
                * graph.stable_unary("log1p", 1 / (2 * F(a[i]) * aux))
            )
        fz = (up.pow(4 / 3) + down.pow(4 / 3) - 2) / (2 ** (4 / 3) - 2)
        fz20 = F("1.709920934161365617563962776245" if modified else "1.709921")
        g0, g1, gm = values  # gm parameterizes minus the spin stiffness.
        return g0 + z.pow(4) * fz * (g1 - g0 + gm / fz20) - fz * gm / fz20

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

    builders = {
        "LDA_X": lambda: exchange(False),
        "GGA_X_PBE": lambda: exchange(True),
        "LDA_C_PW": lambda: correlation(False, False),
        "LDA_C_PW_MOD": lambda: correlation(False, True),
        "GGA_C_PBE": lambda: correlation(True, True),
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

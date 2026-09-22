# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See upstream/libxc/7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""B3LYP production-domain numerical policy retained after Libxc cutover."""

from __future__ import annotations

import math
import typing
from fractions import Fraction as F

if typing.TYPE_CHECKING:
    from vibeqc_compiler.integral.expr import Expr, Graph


def _spin_channels(
    graph: Graph, spec: typing.Any, variables: tuple[typing.Any, ...]
) -> tuple[typing.Any, ...]:
    if spec.spin == "polarized":
        ra, rb, saa, sab, sbb, _, _ = variables
        n = ra + rb
        up, down = 2 * ra / n, 2 * rb / n
        z = (ra - rb) / n
    elif spec.spin == "unpolarized":
        rho, sigma, _ = variables
        ra = rb = rho / 2
        saa = sab = sbb = sigma / 4
        n = rho
        up = down = graph.constant(1)
        z = graph.constant(0)
    else:
        raise ValueError(f"unsupported B3LYP spin layout {spec.spin!r}")
    return ra, rb, saa, sab, sbb, n, up, down, z


def b88_exchange(
    graph: Graph, spec: typing.Any, variables: tuple[typing.Any, ...]
) -> Expr:
    """Return the full-range B88 production branch with stable endpoint limits."""
    ra, rb, saa, _, sbb, _, _, _, _ = _spin_channels(graph, spec, variables)
    cx = F(3, 8) * (3 / math.pi) ** (1 / 3) * 4 ** (2 / 3)
    beta_b88 = F("0.0042")
    gamma_b88 = F(6)
    terms = []
    for density, sigma in ((ra, saa), (rb, sbb)):
        y = sigma * density.pow(-8 / 3)
        x = y.pow(0.5)
        x_asinh_x = x * graph.transcendental_unary("asinh", x)
        # x*asinh(x) is analytic in y=x^2. This branch keeps the
        # sigma derivative finite at zero gradient before differentiation.
        series = y * (
            1 + y * (-F(1, 6) + y * (F(3, 40) + y * (-F(5, 112) + y * F(35, 1152))))
        )
        x_asinh_x = graph.select_le(y, F(1, 100000000), series, x_asinh_x)
        enhancement = 1 + beta_b88 / cx * y / (1 + gamma_b88 * beta_b88 * x_asinh_x)
        term = -cx * density.pow(4 / 3) * enhancement
        # The lazy branch avoids evaluating inverse powers of a zero spin
        # density; physical zero density carries zero same-spin gradient.
        term = graph.select_le(density, 0, 0, term)
        terms.append(term)
    return graph.sum(terms)


def lyp_correlation(
    graph: Graph, spec: typing.Any, variables: tuple[typing.Any, ...]
) -> Expr:
    """Return the B3LYP LYP production continuation at spin-density endpoints."""
    ra, rb, saa, sab, sbb, n, up, down, z = _spin_channels(graph, spec, variables)
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

    # These are the exact positive-density identities from the qualified
    # production path, with the apparent rho_spin^(-8/3) factors cancelled
    # before differentiation to expose the fully polarized limit.
    two83 = 2 ** (8 / 3)
    xs0_up8 = saa * two83 * n_m83
    xs1_down8 = sbb * two83 * n_m83
    xs0_up11 = saa * 2 ** (11 / 3) * ra * n.pow(-11 / 3)
    xs1_down11 = sbb * 2 ** (11 / 3) * rb * n.pow(-11 / 3)

    t4 = aux4 * one_minus_z2 * (F(5, 2) - delta / 18) * (xs0_up8 + xs1_down8)
    t5 = aux5 * one_minus_z2 * (delta - 11) * (xs0_up11 + xs1_down11)
    t6 = -aux6 * (
        F(2, 3) * (xs0_up8 + xs1_down8)
        - up.pow(2) * xs1_down8 / 4
        - down.pow(2) * xs0_up8 / 4
    )
    return n * a_lyp * (t1 + omega_lyp * (t2 + t3 + t4 + t5 + t6))

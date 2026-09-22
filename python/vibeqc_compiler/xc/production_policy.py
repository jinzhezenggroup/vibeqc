# Copyright (C) 2017 M.A.L. Marques
# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See upstream/libxc/7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""VibeQC production-domain numerical policies for semilocal XC.

These algebraic continuations preserve the qualified functional while keeping
finite floating-point evaluation stable outside the importer-owned interior DAG.
They are policy, not a second Libxc mathematical source of truth.
"""

from __future__ import annotations

import math
import typing
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


def lda_xc_pw_unpolarized_tail_expression() -> typing.Any:
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


def lda_xc_pw_polarized_tail_expression() -> typing.Any:
    """Return exact polarized LDA E/vxc in stable scaled density coordinates.

    The caller supplies rho_scale = rho_a + rho_b, normalized spin fractions,
    and rho_scale_sixth_root. Keeping the tiny physical scale out of spin
    interpolation and PW92 radial algebra prevents inverse-density overflow,
    while the returned derivatives remain physical dE/d(rho_s).

    This is a compiler-owned form of the existing production numerical policy,
    not a density cutoff or modified functional.
    """

    graph = Graph()
    a = graph.variable("normalized_rho_a")
    b = graph.variable("normalized_rho_b")
    scale = graph.variable("rho_scale")
    scale_sixth_root = graph.variable("rho_scale_sixth_root")
    n = a + b
    up, down = 2 * a / n, 2 * b / n
    z = (a - b) / n
    x = scale_sixth_root * n.pow(1.0 / 6.0)

    c = (3 / (4 * math.pi)) ** (1 / 3)
    sqrt_c = math.sqrt(c)
    parameters = _PW_PARAMETERS[False]

    def log1p_over_x(value: typing.Any) -> typing.Any:
        series = 1 + value * (
            F(-1, 2)
            + value
            * (F(1, 3) + value * (F(-1, 4) + value * (F(1, 5) - value * F(1, 6))))
        )
        return graph.select_le(
            value,
            F("1e-4"),
            series,
            graph.stable_unary("log1p", value) / value,
        )

    def pw_channel(index: int) -> typing.Any:
        aa = F(parameters["a"][index])
        alpha = F(parameters["alpha"][index])
        b1 = F(parameters["b1"][index])
        b2 = F(parameters["b2"][index])
        b3 = F(parameters["b3"][index])
        b4 = F(parameters["b4"][index])
        x2 = x * x
        q = b1 * sqrt_c * x2 * x + b2 * c * x2 + b3 * c**1.5 * x + b4 * c**2
        u = x2 * x2 / (2 * aa * q)
        return -(x2 + alpha * c) * x2 / q * log1p_over_x(u)

    e0, e1, em = (pw_channel(i) for i in range(3))
    fz20 = F("1.709921")
    fz = (up.pow(4.0 / 3.0) + down.pow(4.0 / 3.0) - 2) / (2 ** (4 / 3) - 2)
    eps = e0 + z.pow(4) * fz * (e1 - e0 + em / fz20) - fz * em / fz20
    correlation = n * eps
    correlation_a = graph.differentiate(correlation, a)
    correlation_b = graph.differentiate(correlation, b)

    cx = F(3, 8) * (3 / math.pi) ** (1 / 3) * 4 ** (2 / 3)
    exchange_shape = -(a.pow(4.0 / 3.0) + b.pow(4.0 / 3.0)) * cx
    exchange_energy = scale_sixth_root.pow(8) * exchange_shape
    exchange_factor = -F(4, 3) * cx * scale_sixth_root.pow(2)
    exchange_a = exchange_factor * a.pow(1.0 / 3.0)
    exchange_b = exchange_factor * b.pow(1.0 / 3.0)

    energy = scale * correlation + exchange_energy
    return (
        graph,
        energy,
        correlation_a + exchange_a,
        correlation_b + exchange_b,
        (a, b, scale, scale_sixth_root),
    )


def pbe_correlation_scaled_expression(*, gradient_correction: bool) -> typing.Any:
    """Return tail-stable polarized PBE correlation E/vxc in scaled coordinates.

    rho_scale and gradient_ratio are fixed numerical scales selected by the
    production lowering. Differentiation is therefore only with respect to
    normalized spin densities and bounded total-gradient components, exactly as
    documented by semilocal-scaled-v1/pbe-spin-c2-1e-18.
    """

    graph = Graph()
    a = graph.variable("normalized_rho_a")
    b = graph.variable("normalized_rho_b")
    scale = graph.variable("rho_scale")
    scale_sixth_root = graph.variable("rho_scale_sixth_root")
    scale_cuberoot = graph.variable("rho_scale_cuberoot")
    ratio = graph.variable("gradient_ratio")
    g = tuple(graph.variable(f"normalized_gradient_{axis}") for axis in range(3))
    n = a + b
    up, down = 2 * a / n, 2 * b / n
    z = (a - b) / n
    x = scale_sixth_root * n.pow(1.0 / 6.0)

    def log1p_over_x(value: typing.Any) -> typing.Any:
        series = 1 + value * (
            F(-1, 2)
            + value
            * (F(1, 3) + value * (F(-1, 4) + value * (F(1, 5) - value * F(1, 6))))
        )
        return graph.select_le(
            value,
            F("1e-4"),
            series,
            graph.stable_unary("log1p", value) / value,
        )

    def pw_channel(index: int) -> typing.Any:
        parameters = _PW_PARAMETERS[True]
        aa = F(parameters["a"][index])
        alpha = F(parameters["alpha"][index])
        b1 = F(parameters["b1"][index])
        b2 = F(parameters["b2"][index])
        b3 = F(parameters["b3"][index])
        b4 = F(parameters["b4"][index])
        c = (3 / (4 * math.pi)) ** (1 / 3)
        x2 = x * x
        q = b1 * math.sqrt(c) * x2 * x + b2 * c * x2 + b3 * c**1.5 * x + b4 * c**2
        u = x2 * x2 / (2 * aa * q)
        return -(x2 + alpha * c) * x2 / q * log1p_over_x(u)

    e0, e1, em = (pw_channel(index) for index in range(3))
    fz20 = F("1.709920934161365617563962776245")
    fz = (up.pow(4.0 / 3.0) + down.pow(4.0 / 3.0) - 2) / (2 ** (4 / 3) - 2)
    eps = e0 + z.pow(4) * fz * (e1 - e0 + em / fz20) - fz * em / fz20

    if gradient_correction:
        cutoff = F("1e-18")

        def spin_two_thirds(value: typing.Any) -> typing.Any:
            t = value / cutoff
            extension = F("1e-12") * t * (F(14, 9) + t * (F(-7, 9) + t * F(2, 9)))
            return graph.select_le(value, cutoff, extension, value.pow(2.0 / 3.0))

        beta = F("0.06672455060314922")
        gamma = (1 - math.log(2)) / math.pi**2
        phi = (spin_two_thirds(up) + spin_two_thirds(down)) / 2
        phi3 = phi.pow(3)
        g2 = graph.sum(component * component for component in g)
        d = (
            16
            * 2 ** (2 / 3)
            * (3 / (4 * math.pi)) ** (1 / 3)
            * (ratio * scale_cuberoot)
            * ratio
            * n.pow(7.0 / 3.0)
            * phi
            * phi
        )
        aa = beta / (gamma * graph.stable_unary("expm1", -eps / (gamma * phi3)))
        denominator = d + aa * g2
        v = d / denominator
        shape = 1 - v + v * v
        ordinary = eps + gamma * phi3 * graph.stable_unary(
            "log1p", (beta / gamma) * g2 / (denominator * shape)
        )
        q = -graph.stable_unary("expm1", eps / (gamma * phi3))
        tail = gamma * phi3 * graph.stable_unary("log1p", -q * v * v / shape)
        eps = graph.select_le(F(1, 2), v, ordinary, tail)

    correlation = n * eps
    rho_a = graph.differentiate(correlation, a)
    rho_b = graph.differentiate(correlation, b)
    gradient = tuple(
        ratio * graph.differentiate(correlation, component) for component in g
    )
    return (
        graph,
        (scale * correlation, rho_a, rho_b, *gradient),
        (a, b, scale, scale_sixth_root, scale_cuberoot, ratio, *g),
    )


def pbe_exchange_direct_expression() -> typing.Any:
    """Return the bounded direct-reduced-gradient PBE exchange branch."""

    graph = Graph()
    rho_cuberoot = graph.variable("rho_cuberoot")
    rho_four_thirds = graph.variable("rho_four_thirds")
    u = tuple(graph.variable(f"reduced_gradient_{axis}") for axis in range(3))
    beta = F("0.06672455060314922")
    kappa = F("0.804")
    pi = math.pi
    cx = F(3, 8) * (3 / pi) ** (1 / 3) * 4 ** (2 / 3)
    mu = beta * pi * pi / (12 * (6 * pi * pi) ** (2 / 3))
    u2 = graph.sum(component * component for component in u)
    denominator = kappa + mu * u2
    response = mu * kappa * kappa / (denominator * denominator)
    enhancement = 1 + kappa * mu * u2 / denominator
    radial_response = response * u2
    energy = -cx * rho_four_thirds * enhancement
    rho = -cx * F(4, 3) * rho_cuberoot * (enhancement - 2 * radial_response)
    gradient = tuple(-2 * cx * response * component for component in u)
    return graph, (energy, rho, *gradient), (rho_cuberoot, rho_four_thirds, *u)


def pbe_exchange_reciprocal_expression() -> typing.Any:
    """Return the bounded reciprocal-reduced-gradient PBE exchange branch."""

    graph = Graph()
    rho_cuberoot = graph.variable("rho_cuberoot")
    rho_four_thirds = graph.variable("rho_four_thirds")
    reciprocal_reduced = graph.variable("reciprocal_reduced_gradient")
    direction = tuple(graph.variable(f"gradient_direction_{axis}") for axis in range(3))
    beta = F("0.06672455060314922")
    kappa = F("0.804")
    pi = math.pi
    cx = F(3, 8) * (3 / pi) ** (1 / 3) * 4 ** (2 / 3)
    mu = beta * pi * pi / (12 * (6 * pi * pi) ** (2 / 3))
    t2 = reciprocal_reduced * reciprocal_reduced
    denominator = kappa * t2 + mu
    response = mu * kappa * kappa / (denominator * denominator)
    enhancement = 1 + kappa - kappa * kappa * t2 / denominator
    radial_response = response * t2
    energy = -cx * rho_four_thirds * enhancement
    rho = -cx * F(4, 3) * rho_cuberoot * (enhancement - 2 * radial_response)
    gradient = tuple(
        -2 * cx * response * t2 * reciprocal_reduced * component
        for component in direction
    )
    return (
        graph,
        (energy, rho, *gradient),
        (rho_cuberoot, rho_four_thirds, reciprocal_reduced, *direction),
    )

# Copyright (C) 2017 M.A.L. Marques
# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See upstream/libxc/7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Audited Libxc 7.0 extended GGA/hybrid scalar expressions."""

from __future__ import annotations

import math
import typing
from fractions import Fraction as F

from vibeqc_compiler.integral.expr import Expr, Graph

from . import ityh_maple
from .b3lyp_production_policy import b88_exchange as production_b88_exchange
from .b3lyp_production_policy import lyp_correlation as production_lyp_correlation
from .b88_vwn_maple import b88_exchange as maple_b88_exchange
from .b88_vwn_maple import vwn_correlation as maple_vwn_correlation
from .p86_pz_maple import p86_correlation, pz_correlation
from .pw91_maple import pw91_correlation as imported_pw91_correlation
from .pw91_maple import pw91_exchange as imported_pw91_exchange
from .pw_maple import pw_correlation
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
        ra, rb = variables[:2]
    else:
        rho = variables[0]
        ra = rb = rho / 2
    n = ra + rb
    cx = F(3, 8) * (3 / math.pi) ** (1 / 3) * 4 ** (2 / 3)

    def lda_exchange() -> Expr:
        return graph.sum(-cx * density.pow(4 / 3) for density in (ra, rb))

    builders = {
        "LDA_X": lda_exchange,
        "GGA_X_B88": lambda: (
            production_b88_exchange(graph, spec, variables)
            if production
            else maple_b88_exchange(graph, spec, variables)
        ),
        "GGA_X_ITYH": lambda: ityh_maple.ityh_exchange(graph, spec, variables),
        "GGA_X_PW91": lambda: imported_pw91_exchange(graph, spec, variables),
        "LDA_C_PW": lambda: pw_correlation(graph, spec, variables, modified=False),
        "GGA_C_PW91": lambda: imported_pw91_correlation(graph, spec, variables),
        "LDA_C_PZ": lambda: pz_correlation(graph, spec, variables),
        "GGA_C_P86": lambda: p86_correlation(graph, spec, variables),
        "LDA_C_VWN": lambda: maple_vwn_correlation(graph, spec, variables, "LDA_C_VWN"),
        "LDA_C_VWN_RPA": lambda: maple_vwn_correlation(
            graph, spec, variables, "LDA_C_VWN_RPA"
        ),
        "GGA_C_LYP": lambda: (
            production_lyp_correlation(graph, spec, variables)
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

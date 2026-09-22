# Copyright (C) 2017 M.A.L. Marques
# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See upstream/libxc/7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Audited Libxc 7.0.0 expressions, translated to the existing scalar DAG.

The exact upstream files/hashes are in manifests/libxc/7.0.0/manifest.json.
Screening branches are deliberately excluded: the separately versioned domain
contract rejects their inputs. Energy is per volume throughout this module.
"""

from __future__ import annotations

import math
import typing
from fractions import Fraction as F

from vibeqc_compiler.integral.expr import Graph

from .pbe_maple import pbe_correlation, pbe_exchange
from .pw_maple import pw_correlation
from .scan_maple import scan_component


def energy_expression(spec: typing.Any, *, production: bool = False) -> typing.Any:
    """Return the energy DAG and ordered feature variables.

    PBE and SCAN/r2SCAN mathematics are lowered from pinned Libxc Maple sources. Production
    mode independently selects versioned SCF endpoint continuations for
    families that still require them.
    """
    graph = Graph()
    variables = tuple(graph.variable(name) for name in spec.features)
    if spec.spin == "polarized":
        ra, rb = variables[:2]
    else:
        rho = variables[0]
        ra = rb = rho / 2
    cx = F(3, 8) * (3 / math.pi) ** (1 / 3) * 4 ** (2 / 3)

    def lda_exchange() -> typing.Any:
        return graph.sum(-cx * density.pow(4 / 3) for density in (ra, rb))

    builders = {
        "LDA_X": lda_exchange,
        "GGA_X_PBE": lambda: pbe_exchange(graph, spec, variables),
        "LDA_C_PW": lambda: pw_correlation(graph, spec, variables, modified=False),
        "LDA_C_PW_MOD": lambda: pw_correlation(graph, spec, variables, modified=True),
        "GGA_C_PBE": lambda: pbe_correlation(graph, spec, variables),
        "MGGA_X_SCAN": lambda: scan_component(graph, spec, variables, "MGGA_X_SCAN"),
        "MGGA_C_SCAN": lambda: scan_component(graph, spec, variables, "MGGA_C_SCAN"),
        "MGGA_X_R2SCAN": lambda: scan_component(
            graph, spec, variables, "MGGA_X_R2SCAN"
        ),
        "MGGA_C_R2SCAN": lambda: scan_component(
            graph, spec, variables, "MGGA_C_R2SCAN"
        ),
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

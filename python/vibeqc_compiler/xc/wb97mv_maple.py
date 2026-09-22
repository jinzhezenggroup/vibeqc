# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See upstream/libxc/7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Production omegaB97M-V semilocal XC lowered from pinned Libxc Maple."""

from __future__ import annotations

import math
import typing
from functools import cache
from pathlib import Path

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import file_hash
from vibeqc_compiler.integral.expr import Graph

from . import libxc_maple
from .libxc_maple import IMPORTER_SEMANTICS, MapleModule, import_maple_file

WB97MV_COMPONENTS = ("MGGA_X_WB97M_V", "MGGA_C_WB97M_V")
_ZETA_THRESHOLD = "2.220446049250313e-16"
_BASE_BINDINGS = {
    "p_a_zeta_threshold": _ZETA_THRESHOLD,
    "p_a_dens_threshold": "1e-13",
    "params_a_c_x": ("0.85", "1.007", "0.259"),
    "params_a_c_ss": ("0.443", "-1.437", "-4.535", "-3.39", "4.278"),
    "params_a_c_os": ("1.0", "1.358", "2.924", "-8.812", "-1.39", "9.142"),
}


def _libxc_root() -> typing.Any:
    return asset_path("upstream/libxc/7.0.0")


@cache
def _wb97mv_module(range_omega: typing.Any) -> MapleModule:
    bindings = {**_BASE_BINDINGS, "p_a_cam_omega": range_omega}
    return import_maple_file(
        _libxc_root(),
        "hyb_mgga_xc_wb97mv.mpl",
        bindings=bindings,
        support_files=("hyb_mgga_xc_wb97mv.c", "util.mpl"),
    )


def _coordinates(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> tuple[typing.Any, ...]:
    if spec.spin == "polarized":
        rho_a, rho_b, sigma_aa, _sigma_ab, sigma_bb, tau_a, tau_b = variables
        density = rho_a + rho_b
        zeta = (rho_a - rho_b) / density
    elif spec.spin == "unpolarized":
        density, sigma, tau = variables
        rho_a = rho_b = density / 2
        sigma_aa = sigma_bb = sigma / 4
        tau_a = tau_b = tau / 2
        zeta = graph.constant(0)
    else:
        raise ValueError(f"unsupported omegaB97M-V spin layout {spec.spin!r}")

    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    xs_a = sigma_aa.pow(0.5) * rho_a.pow(-4.0 / 3.0)
    xs_b = sigma_bb.pow(0.5) * rho_b.pow(-4.0 / 3.0)
    ts_a = tau_a * rho_a.pow(-5.0 / 3.0)
    ts_b = tau_b * rho_b.pow(-5.0 / 3.0)
    return density, zeta, rs, xs_a, xs_b, ts_a, ts_b


def energy_expression(spec: typing.Any) -> typing.Any:
    """Return the semilocal omegaB97M-V DAG from pinned Libxc Maple."""
    active = {name for name, coefficient in spec.components if coefficient}
    if not active or not active <= set(WB97MV_COMPONENTS):
        raise ValueError("omegaB97M-V expression received unsupported components")
    if "MGGA_X_WB97M_V" in active and spec.range_omega <= 0:
        raise ValueError("omegaB97M-V exchange requires positive range_omega")

    graph = Graph()
    variables = tuple(graph.variable(name) for name in spec.features)
    density, zeta, rs, xs_a, xs_b, ts_a, ts_b = _coordinates(graph, spec, variables)
    module = _wb97mv_module(spec.range_omega)
    builders = {
        "MGGA_X_WB97M_V": lambda: (
            density * module.call(graph, "wb97mv_f", rs, zeta, xs_a, xs_b, ts_a, ts_b)
        ),
        "MGGA_C_WB97M_V": lambda: (
            density * module.call(graph, "b97mv_f", rs, zeta, xs_a, xs_b, ts_a, ts_b)
        ),
    }
    total = graph.sum(
        coefficient * builders[name]()
        for name, coefficient in spec.components
        if coefficient
    )
    return graph, total, variables


def wb97mv_maple_provenance(
    components: tuple[tuple[str, typing.Any], ...],
    range_omega: typing.Any,
) -> dict[str, typing.Any] | None:
    """Return stable importer/source identity for active omegaB97M-V semilocal XC."""
    active = {name for name, coefficient in components if coefficient}
    if not (active & set(WB97MV_COMPONENTS)):
        return None
    module = _wb97mv_module(range_omega)
    return {
        "kind": "libxc-maple",
        "importer_semantics": IMPORTER_SEMANTICS,
        "adapter_sha256": file_hash(Path(__file__)),
        "importer_sha256": file_hash(Path(libxc_maple.__file__)),
        "components": {
            name: {
                "entry": "hyb_mgga_xc_wb97mv.mpl",
                "source_sha256": module.source_sha256,
                "transitive_sha256": module.transitive_sha256,
            }
            for name in sorted(active & set(WB97MV_COMPONENTS))
        },
    }

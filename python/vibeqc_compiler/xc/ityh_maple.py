# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See external/libxc-7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Production ITYH short-range exchange lowered from pinned Libxc Maple."""

from __future__ import annotations

import math
import typing
from functools import cache
from pathlib import Path

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import file_hash

from . import libxc_maple
from .libxc_maple import IMPORTER_SEMANTICS, MapleModule, import_maple_file

_ZETA_THRESHOLD = "2.220446049250313e-16"


def _libxc_root() -> typing.Any:
    try:
        return asset_path("upstream/libxc/7.0.0")
    except FileNotFoundError:
        return asset_path("external/libxc-7.0.0")


@cache
def _ityh_module(range_omega: typing.Any) -> MapleModule:
    return import_maple_file(
        _libxc_root(),
        "gga_x_ityh.mpl",
        bindings={
            "p_a_zeta_threshold": _ZETA_THRESHOLD,
            "p_a_dens_threshold": "1e-14",
            "p_a_cam_omega": range_omega,
        },
        support_files=("util.mpl",),
    )


def _coordinates(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> tuple[typing.Any, ...]:
    if spec.spin == "polarized":
        rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, _, _ = variables
        density = rho_a + rho_b
        zeta = (rho_a - rho_b) / density
    elif spec.spin == "unpolarized":
        density, sigma, _ = variables
        rho_a = rho_b = density / 2
        sigma_aa = sigma_ab = sigma_bb = sigma / 4
        zeta = graph.constant(0)
    else:
        raise ValueError(f"unsupported ITYH spin layout {spec.spin!r}")

    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    total_sigma = sigma_aa + 2 * sigma_ab + sigma_bb
    xt = total_sigma.pow(0.5) * density.pow(-4.0 / 3.0)
    xs_a = sigma_aa.pow(0.5) * rho_a.pow(-4.0 / 3.0)
    xs_b = sigma_bb.pow(0.5) * rho_b.pow(-4.0 / 3.0)
    return density, zeta, rs, xt, xs_a, xs_b


def ityh_exchange(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> typing.Any:
    """Lower GGA_X_ITYH from the pinned Libxc Maple source."""
    density, zeta, rs, xt, xs_a, xs_b = _coordinates(graph, spec, variables)
    module = _ityh_module(spec.range_omega)
    return density * module.call(graph, "f", rs, zeta, xt, xs_a, xs_b)


def ityh_maple_provenance(
    components: tuple[tuple[str, typing.Any], ...],
    range_omega: typing.Any,
) -> dict[str, typing.Any] | None:
    """Return stable importer/source identity for active ITYH exchange."""
    active = {name for name, coefficient in components if coefficient}
    if "GGA_X_ITYH" not in active:
        return None
    module = _ityh_module(range_omega)
    return {
        "kind": "libxc-maple",
        "importer_semantics": IMPORTER_SEMANTICS,
        "adapter_sha256": file_hash(Path(__file__)),
        "importer_sha256": file_hash(Path(libxc_maple.__file__)),
        "components": {
            "GGA_X_ITYH": {
                "entry": "gga_x_ityh.mpl",
                "source_sha256": module.source_sha256,
                "transitive_sha256": module.transitive_sha256,
            }
        },
    }

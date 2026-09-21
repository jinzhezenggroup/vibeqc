# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See external/libxc-7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Production extended-GGA expressions lowered from pinned Libxc Maple sources."""

from __future__ import annotations

import typing
from functools import cache
from pathlib import Path

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import file_hash

from . import libxc_maple
from .libxc_maple import IMPORTER_SEMANTICS, MapleModule, import_maple_file

_LYP_BINDINGS = {
    "params_a_a": "0.04918",
    "params_a_b": "0.132",
    "params_a_c": "0.2533",
    "params_a_d": "0.349",
}


def _libxc_root() -> typing.Any:
    """Resolve packaged/repository Libxc assets lazily."""

    try:
        return asset_path("upstream/libxc/7.0.0")
    except FileNotFoundError:
        return asset_path("external/libxc-7.0.0")


@cache
def _lyp_module() -> MapleModule:
    """Return the qualified pinned LYP Maple module and parameter owner."""

    return import_maple_file(
        _libxc_root(),
        "gga_c_lyp.mpl",
        bindings=_LYP_BINDINGS,
        support_files=("gga_c_lyp.c",),
    )


def _spin_channels(
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
        raise ValueError(f"unsupported extended-GGA spin layout {spec.spin!r}")
    return rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, density, zeta


def lyp_correlation(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> typing.Any:
    """Lower GGA_C_LYP from the pinned Libxc Maple source into the caller Graph."""

    rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, density, zeta = _spin_channels(
        graph, spec, variables
    )
    rr = density.pow(-1.0 / 3.0)
    total_sigma = sigma_aa + 2 * sigma_ab + sigma_bb
    xt = total_sigma.pow(0.5) * density.pow(-4.0 / 3.0)
    xs0 = sigma_aa.pow(0.5) * rho_a.pow(-4.0 / 3.0)
    xs1 = sigma_bb.pow(0.5) * rho_b.pow(-4.0 / 3.0)
    return density * _lyp_module().call(graph, "f_lyp_rr", rr, zeta, xt, xs0, xs1)


def rsh_maple_provenance(
    components: tuple[tuple[str, typing.Any], ...],
) -> dict[str, typing.Any] | None:
    """Return stable imported-source identity for migrated extended-GGA families."""

    active = {name for name, coefficient in components if coefficient}
    selected: dict[str, typing.Any] = {}
    if "GGA_C_LYP" in active:
        module = _lyp_module()
        selected["GGA_C_LYP"] = {
            "entry": "gga_c_lyp.mpl",
            "source_sha256": module.source_sha256,
            "transitive_sha256": module.transitive_sha256,
        }
    if not selected:
        return None
    return {
        "kind": "libxc-maple",
        "importer_semantics": IMPORTER_SEMANTICS,
        "adapter_sha256": file_hash(Path(__file__)),
        "importer_sha256": file_hash(Path(libxc_maple.__file__)),
        "components": selected,
    }

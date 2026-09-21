# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See external/libxc-7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Production B88 and VWN expressions lowered from pinned Libxc Maple sources."""

from __future__ import annotations

import math
import typing
from functools import cache
from pathlib import Path

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import file_hash

from . import libxc_maple
from .libxc_maple import IMPORTER_SEMANTICS, MapleModule, import_maple_file

_VWN_ENTRIES = {
    "LDA_C_VWN": "lda_c_vwn.mpl",
    "LDA_C_VWN_RPA": "lda_c_vwn_rpa.mpl",
}


def _libxc_root() -> typing.Any:
    try:
        return asset_path("upstream/libxc/7.0.0")
    except FileNotFoundError:
        return asset_path("external/libxc-7.0.0")


@cache
def _b88_module() -> MapleModule:
    return import_maple_file(
        _libxc_root(),
        "gga_x_b88.mpl",
        defines={"gga_x_b88_params"},
    )


@cache
def _vwn_module(name: str) -> MapleModule:
    try:
        entry = _VWN_ENTRIES[name]
    except KeyError as error:
        raise ValueError(f"unsupported VWN component {name!r}") from error
    return import_maple_file(_libxc_root(), entry)


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


def b88_exchange(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> typing.Any:
    """Lower canonical full-range GGA_X_B88 into the caller Graph."""

    rho_a, rho_b, sigma_aa, _, sigma_bb, _, _ = _spin_channels(graph, spec, variables)
    module = _b88_module()
    cx = graph.approximate_constant(
        3.0 / 8.0 * (3.0 / math.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)
    )
    return graph.sum(
        -cx
        * density.pow(4.0 / 3.0)
        * module.call(
            graph,
            "b88_f",
            sigma.pow(0.5) * density.pow(-4.0 / 3.0),
        )
        for density, sigma in ((rho_a, sigma_aa), (rho_b, sigma_bb))
    )


def vwn_correlation(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
    name: str,
) -> typing.Any:
    """Lower one pinned VWN correlation component into the caller Graph."""

    _, _, _, _, _, density, zeta = _spin_channels(graph, spec, variables)
    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    return density * _vwn_module(name).call(graph, "f", rs, zeta)


def b88_vwn_maple_provenance(
    components: tuple[tuple[str, typing.Any], ...],
) -> dict[str, typing.Any] | None:
    """Return stable source/importer identity for migrated B88/VWN components."""

    active = {name for name, coefficient in components if coefficient}
    selected: dict[str, typing.Any] = {}
    if "GGA_X_B88" in active:
        module = _b88_module()
        selected["GGA_X_B88"] = {
            "entry": "gga_x_b88.mpl",
            "source_sha256": module.source_sha256,
            "transitive_sha256": module.transitive_sha256,
        }
    for name, entry in _VWN_ENTRIES.items():
        if name in active:
            module = _vwn_module(name)
            selected[name] = {
                "entry": entry,
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

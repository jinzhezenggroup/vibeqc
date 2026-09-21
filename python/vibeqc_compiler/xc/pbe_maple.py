# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See external/libxc-7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Production PBE expressions lowered from the pinned Libxc Maple source."""

from __future__ import annotations

import math
import typing
from functools import cache
from pathlib import Path

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import file_hash

from . import libxc_maple
from .libxc_maple import (
    IMPORTER_SEMANTICS,
    MapleModule,
    import_maple_file,
    import_maple_source,
)


def _libxc_root() -> typing.Any:
    """Resolve packaged/repository assets lazily instead of at module import."""

    try:
        return asset_path("upstream/libxc/7.0.0")
    except FileNotFoundError:
        return asset_path("external/libxc-7.0.0")


@cache
def _pbe_x_module() -> MapleModule:
    root = _libxc_root()
    source = (root / "gga_x_pbe.mpl").read_text(encoding="utf-8")
    return import_maple_source(source, defines={"gga_x_pbe_params"})


@cache
def _pbe_c_module() -> MapleModule:
    return import_maple_file(
        _libxc_root(),
        "gga_c_pbe.mpl",
        defines={"gga_c_pbe_params"},
        support_files=("util.mpl",),
    )


def _spin_channels(
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    if spec.spin == "polarized":
        rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, _, _ = variables
    elif spec.spin == "unpolarized":
        rho, sigma, _ = variables
        rho_a = rho_b = rho / 2
        sigma_aa = sigma_ab = sigma_bb = sigma / 4
    else:
        raise ValueError(f"unsupported PBE spin layout {spec.spin!r}")
    return rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, rho_a + rho_b


def pbe_exchange(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> typing.Any:
    """Lower GGA_X_PBE from gga_x_pbe.mpl into the caller's scalar Graph."""

    rho_a, rho_b, sigma_aa, _, sigma_bb, _ = _spin_channels(spec, variables)
    module = _pbe_x_module()
    cx = graph.approximate_constant(
        3.0 / 8.0 * (3.0 / math.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)
    )
    return graph.sum(
        -cx
        * density.pow(4.0 / 3.0)
        * module.call(
            graph,
            "pbe_f",
            sigma.pow(0.5) * density.pow(-4.0 / 3.0),
        )
        for density, sigma in ((rho_a, sigma_aa), (rho_b, sigma_bb))
    )


def pbe_correlation(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> typing.Any:
    """Lower GGA_C_PBE and its pinned PW include graph into the scalar Graph."""

    rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, density = _spin_channels(
        spec, variables
    )
    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    if spec.spin == "polarized":
        zeta = (rho_a - rho_b) / density
    else:
        zeta = graph.constant(0)
    total_sigma = sigma_aa + 2 * sigma_ab + sigma_bb
    xt = total_sigma.pow(0.5) * density.pow(-4.0 / 3.0)
    return density * _pbe_c_module().call(graph, "f", rs, zeta, xt, 0, 0)


def pbe_maple_provenance(
    components: tuple[tuple[str, typing.Any], ...],
) -> dict[str, typing.Any] | None:
    """Return stable source/importer identity only for active PBE components."""

    active = {name for name, coefficient in components if coefficient}
    selected: dict[str, typing.Any] = {}
    if "GGA_X_PBE" in active:
        module = _pbe_x_module()
        selected["GGA_X_PBE"] = {
            "entry": "gga_x_pbe.mpl",
            "source_sha256": module.source_sha256,
            "transitive_sha256": module.transitive_sha256,
        }
    if "GGA_C_PBE" in active:
        module = _pbe_c_module()
        selected["GGA_C_PBE"] = {
            "entry": "gga_c_pbe.mpl",
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

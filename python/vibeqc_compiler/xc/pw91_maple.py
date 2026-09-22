# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See upstream/libxc/7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Production PW91 exchange/correlation lowered from pinned Libxc Maple."""

from __future__ import annotations

import math
import typing
from functools import cache
from pathlib import Path

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import file_hash

from . import libxc_maple
from .libxc_maple import IMPORTER_SEMANTICS, MapleModule, import_maple_file


def _libxc_root() -> typing.Any:
    """Resolve packaged/repository Libxc assets lazily."""

    return asset_path("upstream/libxc/7.0.0")


@cache
def _pw91_x_module() -> MapleModule:
    return import_maple_file(
        _libxc_root(),
        "gga_x_pw91.mpl",
        defines={"gga_x_pw91_params"},
        support_files=("gga_x_pw91.c",),
    )


@cache
def _pw91_c_module() -> MapleModule:
    return import_maple_file(
        _libxc_root(),
        "gga_c_pw91.mpl",
        support_files=("gga_c_pw91.c", "util.mpl"),
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
        raise ValueError(f"unsupported PW91 spin layout {spec.spin!r}")

    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    total_sigma = sigma_aa + 2 * sigma_ab + sigma_bb
    xt = total_sigma.pow(0.5) * density.pow(-4.0 / 3.0)
    xs_a = sigma_aa.pow(0.5) * rho_a.pow(-4.0 / 3.0)
    xs_b = sigma_bb.pow(0.5) * rho_b.pow(-4.0 / 3.0)
    return rho_a, rho_b, density, zeta, rs, xt, xs_a, xs_b


def pw91_exchange(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> typing.Any:
    """Lower GGA_X_PW91 from the pinned Libxc Maple source."""

    rho_a, rho_b, _, _, _, _, xs_a, xs_b = _coordinates(graph, spec, variables)
    module = _pw91_x_module()
    cx = graph.approximate_constant(
        3.0 / 8.0 * (3.0 / math.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)
    )
    return graph.sum(
        (
            -cx * rho_a.pow(4.0 / 3.0) * module.call(graph, "pw91_f", xs_a),
            -cx * rho_b.pow(4.0 / 3.0) * module.call(graph, "pw91_f", xs_b),
        )
    )


def pw91_correlation(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> typing.Any:
    """Lower GGA_C_PW91 from the pinned Libxc Maple source."""

    _, _, density, zeta, rs, xt, _, _ = _coordinates(graph, spec, variables)
    return density * _pw91_c_module().call(graph, "f", rs, zeta, xt, 0, 0)


def pw91_component(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
    component: str,
) -> typing.Any:
    if component == "GGA_X_PW91":
        return pw91_exchange(graph, spec, variables)
    if component == "GGA_C_PW91":
        return pw91_correlation(graph, spec, variables)
    raise ValueError(f"unsupported PW91 component {component!r}")


def pw91_maple_provenance(
    components: tuple[tuple[str, typing.Any], ...],
) -> dict[str, typing.Any] | None:
    """Return stable importer/source identity for active PW91 components."""

    active = {name for name, coefficient in components if coefficient}
    selected: dict[str, typing.Any] = {}
    for name, entry, loader in (
        ("GGA_X_PW91", "gga_x_pw91.mpl", _pw91_x_module),
        ("GGA_C_PW91", "gga_c_pw91.mpl", _pw91_c_module),
    ):
        if name in active:
            module = loader()
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

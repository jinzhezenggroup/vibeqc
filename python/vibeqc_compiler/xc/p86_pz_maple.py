# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See upstream/libxc/7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Production P86/PZ expressions lowered from pinned Libxc Maple sources."""

from __future__ import annotations

import math
import typing
from functools import cache
from pathlib import Path

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import file_hash

from . import libxc_maple
from .libxc_maple import IMPORTER_SEMANTICS, MapleModule, import_maple_file

_RS_FACTOR = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
_P86_BINDINGS = {
    "params_a_malpha": "0.023266",
    "params_a_mbeta": "0.000007389",
    "params_a_mgamma": "8.723",
    "params_a_mdelta": "0.472",
    "params_a_aa": "0.001667",
    "params_a_bb": "0.002568",
    "params_a_ftilde": "0.19195",
    "RS_FACTOR": repr(_RS_FACTOR),
}


def _libxc_root() -> typing.Any:
    return asset_path("upstream/libxc/7.0.0")


@cache
def _pz_module() -> MapleModule:
    return import_maple_file(
        _libxc_root(),
        "lda_c_pz.mpl",
        defines=("lda_c_pz_params",),
        support_files=("lda_c_pz.c", "util.mpl"),
    )


@cache
def _p86_module() -> MapleModule:
    return import_maple_file(
        _libxc_root(),
        "gga_c_p86.mpl",
        bindings=_P86_BINDINGS,
        support_files=("gga_c_p86.c", "lda_c_pz.c", "util.mpl"),
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
        total_sigma = sigma_aa + 2 * sigma_ab + sigma_bb
    elif spec.spin == "unpolarized":
        density, total_sigma, _ = variables
        zeta = graph.constant(0)
    else:
        raise ValueError(f"unsupported P86/PZ spin layout {spec.spin!r}")
    rs = graph.approximate_constant(_RS_FACTOR) * density.pow(-1.0 / 3.0)
    xt = total_sigma.pow(0.5) * density.pow(-4.0 / 3.0)
    return density, zeta, rs, xt


def pz_correlation(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> typing.Any:
    density, zeta, rs, _ = _coordinates(graph, spec, variables)
    return density * _pz_module().call(graph, "f", rs, zeta)


def p86_correlation(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> typing.Any:
    density, zeta, rs, xt = _coordinates(graph, spec, variables)
    return density * _p86_module().call(graph, "f", rs, zeta, xt, 0, 0)


def p86_pz_maple_provenance(
    components: tuple[tuple[str, typing.Any], ...],
) -> dict[str, typing.Any] | None:
    active = {name for name, coefficient in components if coefficient}
    selected: dict[str, typing.Any] = {}
    for name, entry, loader in (
        ("LDA_C_PZ", "lda_c_pz.mpl", _pz_module),
        ("GGA_C_P86", "gga_c_p86.mpl", _p86_module),
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

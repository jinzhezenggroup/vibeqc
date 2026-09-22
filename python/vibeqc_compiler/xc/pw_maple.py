# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See upstream/libxc/7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Production PW/PW-mod LDA correlation lowered from pinned Libxc Maple."""

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
    return asset_path("upstream/libxc/7.0.0")


@cache
def _pw_module(modified: bool) -> MapleModule:
    defines = {"lda_c_pw_params"}
    if modified:
        defines.add("lda_c_pw_modified_params")
    return import_maple_file(
        _libxc_root(),
        "lda_c_pw.mpl",
        defines=defines,
        support_files=("util.mpl",),
    )


def _coordinates(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> tuple[typing.Any, typing.Any, typing.Any]:
    if spec.spin == "polarized":
        rho_a, rho_b = variables[:2]
        density = rho_a + rho_b
        zeta = (rho_a - rho_b) / density
    elif spec.spin == "unpolarized":
        density = variables[0]
        zeta = graph.constant(0)
    else:
        raise ValueError(f"unsupported PW spin layout {spec.spin!r}")
    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    return density, zeta, rs


def pw_correlation(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
    *,
    modified: bool,
) -> typing.Any:
    """Lower LDA_C_PW or LDA_C_PW_MOD into the caller Graph."""
    density, zeta, rs = _coordinates(graph, spec, variables)
    return density * _pw_module(modified).call(graph, "f", rs, zeta)


def pw_maple_provenance(
    components: tuple[tuple[str, typing.Any], ...],
) -> dict[str, typing.Any] | None:
    """Return stable importer/source identity for active PW LDA components."""
    active = {name for name, coefficient in components if coefficient}
    selected: dict[str, typing.Any] = {}
    for name, modified in (("LDA_C_PW", False), ("LDA_C_PW_MOD", True)):
        if name in active:
            module = _pw_module(modified)
            selected[name] = {
                "entry": "lda_c_pw.mpl",
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

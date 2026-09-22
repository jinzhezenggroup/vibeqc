# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See upstream/libxc/7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Production-ready SCAN/r2SCAN adapters lowered from pinned Libxc Maple."""

from __future__ import annotations

import math
import typing
from functools import cache
from pathlib import Path

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import file_hash

from . import libxc_maple
from .libxc_maple import IMPORTER_SEMANTICS, MapleModule, import_maple_file

_SCAN_BINDINGS = {
    "params_a_c1": "0.667",
    "params_a_c2": "0.8",
    "params_a_d": "1.24",
    "params_a_k1": "0.065",
}
_R2SCAN_BINDINGS = {
    **_SCAN_BINDINGS,
    "params_a_eta": "0.001",
    "params_a_dp2": "0.361",
}
_EXCHANGE_SPECS = {
    "MGGA_X_SCAN": (
        "mgga_x_scan.mpl",
        "mgga_x_scan.c",
        _SCAN_BINDINGS,
        False,
        "scan_f",
    ),
    "MGGA_X_R2SCAN": (
        "mgga_x_r2scan.mpl",
        "mgga_x_r2scan.c",
        _R2SCAN_BINDINGS,
        True,
        "r2scan_f",
    ),
}
_CORRELATION_SPECS = {
    "MGGA_C_SCAN": ("mgga_c_scan.mpl", "mgga_c_scan.c", {}, False, "scan_f"),
    "MGGA_C_R2SCAN": (
        "mgga_c_r2scan.mpl",
        "mgga_c_r2scan.c",
        {"params_a_eta": "0.001"},
        True,
        "r2scan_f",
    ),
}
SCAN_COMPONENTS = tuple(_EXCHANGE_SPECS) + tuple(_CORRELATION_SPECS)


def _libxc_root() -> typing.Any:
    """Resolve packaged/repository Libxc assets lazily."""

    return asset_path("upstream/libxc/7.0.0")


@cache
def _exchange_module(name: str) -> MapleModule:
    entry, parameter_source, bindings, allow_duplicates, _ = _EXCHANGE_SPECS[name]
    return import_maple_file(
        _libxc_root(),
        entry,
        bindings=bindings,
        support_files=("util.mpl", parameter_source),
        allow_duplicate_includes=allow_duplicates,
    )


@cache
def _correlation_module(name: str) -> MapleModule:
    entry, parameter_source, bindings, allow_duplicates, _ = _CORRELATION_SPECS[name]
    return import_maple_file(
        _libxc_root(),
        entry,
        bindings=bindings,
        support_files=("util.mpl", parameter_source),
        allow_duplicate_includes=allow_duplicates,
    )


def _spin_layout(
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
) -> tuple[typing.Any, ...]:
    if spec.spin == "polarized":
        rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b = variables
    elif spec.spin == "unpolarized":
        rho, sigma, tau = variables
        rho_a = rho_b = rho / 2
        sigma_aa = sigma_ab = sigma_bb = sigma / 4
        tau_a = tau_b = tau / 2
    else:
        raise ValueError(f"unsupported SCAN-family spin layout {spec.spin!r}")
    return rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b


def scan_exchange(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
    component: str,
) -> typing.Any:
    """Lower one SCAN-family exchange component into the caller's Graph."""

    if component not in _EXCHANGE_SPECS:
        raise ValueError(f"unsupported SCAN-family exchange component {component!r}")
    rho_a, rho_b, sigma_aa, _, sigma_bb, tau_a, tau_b = _spin_layout(spec, variables)
    module = _exchange_module(component)
    function_name = _EXCHANGE_SPECS[component][4]
    cx = graph.approximate_constant(
        3.0 / 8.0 * (3.0 / math.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)
    )

    terms = []
    for density, sigma, tau in (
        (rho_a, sigma_aa, tau_a),
        (rho_b, sigma_bb, tau_b),
    ):
        xs = sigma.pow(0.5) * density.pow(-4.0 / 3.0)
        ts = tau * density.pow(-5.0 / 3.0)
        terms.append(
            -cx * density.pow(4.0 / 3.0) * module.call(graph, function_name, xs, 0, ts)
        )
    return graph.sum(terms)


def scan_correlation(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
    component: str,
) -> typing.Any:
    """Lower one SCAN-family correlation component into the caller's Graph."""

    if component not in _CORRELATION_SPECS:
        raise ValueError(f"unsupported SCAN-family correlation component {component!r}")
    rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b = _spin_layout(
        spec, variables
    )
    density = rho_a + rho_b
    zeta = (rho_a - rho_b) / density if spec.spin == "polarized" else graph.constant(0)
    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    total_sigma = sigma_aa + 2 * sigma_ab + sigma_bb
    xt = total_sigma.pow(0.5) * density.pow(-4.0 / 3.0)
    ts_a = tau_a * rho_a.pow(-5.0 / 3.0)
    ts_b = tau_b * rho_b.pow(-5.0 / 3.0)
    module = _correlation_module(component)
    function_name = _CORRELATION_SPECS[component][4]
    epsilon = module.call(graph, function_name, rs, zeta, xt, 0, 0, ts_a, ts_b)
    return density * epsilon


def scan_component(
    graph: typing.Any,
    spec: typing.Any,
    variables: tuple[typing.Any, ...],
    component: str,
) -> typing.Any:
    """Lower one supported SCAN/r2SCAN X/C component."""

    if component in _EXCHANGE_SPECS:
        return scan_exchange(graph, spec, variables, component)
    if component in _CORRELATION_SPECS:
        return scan_correlation(graph, spec, variables, component)
    raise ValueError(f"unsupported SCAN-family component {component!r}")


def scan_maple_provenance(
    components: tuple[tuple[str, typing.Any], ...],
) -> dict[str, typing.Any] | None:
    """Return stable importer/source identity for active SCAN/r2SCAN components."""

    active = {name for name, coefficient in components if coefficient}
    selected: dict[str, typing.Any] = {}
    for name in SCAN_COMPONENTS:
        if name not in active:
            continue
        if name in _EXCHANGE_SPECS:
            module = _exchange_module(name)
            entry = _EXCHANGE_SPECS[name][0]
        else:
            module = _correlation_module(name)
            entry = _CORRELATION_SPECS[name][0]
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

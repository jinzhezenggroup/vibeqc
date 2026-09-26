# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See upstream/libxc/7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Production omegaB97M-V semilocal XC lowered from pinned Libxc Maple."""

from __future__ import annotations

import math
import typing
from fractions import Fraction
from functools import cache
from pathlib import Path

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import file_hash
from vibeqc_compiler.integral.expr import Expr, Graph

from . import libxc_maple
from .libxc_maple import IMPORTER_SEMANTICS, MapleModule, import_maple_file

WB97MV_COMPONENTS = ("MGGA_X_WB97M_V", "MGGA_C_WB97M_V")
DENSITY_THRESHOLD = 1.0e-13
SIGMA_THRESHOLD = DENSITY_THRESHOLD ** (4.0 / 3.0)
TAU_THRESHOLD = 1.0e-20
SMOOTH_LR_CUTOFF = 1.35
SMOOTH_LR_ORDER = 16
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
def _validate_runtime_policy_sources() -> tuple[Path, Path, Path]:
    "Audit the exact Libxc work-driver semantics reproduced by native XC."
    root = _libxc_root()
    functional = root / "hyb_mgga_xc_wb97mv.c"
    initialization = root / "functionals.c"
    work = root / "work_mgga_inc.c"
    required = {
        functional: ("XC_FLAGS_NEEDS_TAU | XC_FLAGS_VV10", "1e-13,"),
        initialization: (
            "func->sigma_threshold = pow(func->info->dens_threshold, 4.0/3.0);",
            "func->tau_threshold   = 1e-20;",
        ),
        work: (
            "if(dens < p->dens_threshold)",
            "my_rho[0] = m_max(p->dens_threshold, VAR(rho, ip, 0));",
            "my_sigma[0] = m_max(p->sigma_threshold * p->sigma_threshold, VAR(sigma, ip, 0));",
            "my_tau[0] = m_max(p->tau_threshold, VAR(tau, ip, 0));",
        ),
    }
    for path, snippets in required.items():
        text = path.read_text()
        if any(snippet not in text for snippet in snippets):
            raise ValueError(
                f"Libxc WB97M-V runtime policy source changed: {path.name}"
            )
    return functional, initialization, work


@cache
def _wb97mv_module(range_omega: typing.Any) -> MapleModule:
    _validate_runtime_policy_sources()
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


def _restore_work_spin_density(
    graph: Graph, energy: Expr, variables: tuple[Expr, ...], zeta: Expr
) -> Expr:
    """Screen on the work spin density, not a rounded reconstruction from rs/zeta.

    At the pinned Libxc density floor the Maple n_spin(rs,zeta) expansion may
    round to either side of the screen. The work driver compares rho directly.
    Identify the two polarized screens structurally and fail closed if the
    pinned Maple import changes rather than rewriting an unrelated predicate.
    """
    rho_a, rho_b = variables[:2]
    plus, minus = 1 + zeta, 1 - zeta
    threshold = graph.constant(Fraction(str(DENSITY_THRESHOLD)))
    substitutions: dict[int, tuple[Expr, Expr]] = {}
    for identifier in graph.topological_order((energy,)):
        node = graph.nodes[identifier]
        if node.operation != "select_le" or node.arguments[1] != threshold.identifier:
            continue
        if node.arguments[0] in (rho_a.identifier, rho_b.identifier):
            continue
        reconstructed = Expr(graph, node.arguments[0])
        dependencies = set(graph.topological_order((reconstructed,)))
        alpha, beta = plus.identifier in dependencies, minus.identifier in dependencies
        if alpha == beta:
            raise ValueError("omegaB97M-V work spin screen changed")
        substitutions[reconstructed.identifier] = (
            reconstructed,
            rho_a if alpha else rho_b,
        )
    if len(substitutions) != 2:
        raise ValueError("omegaB97M-V work spin screens changed")
    return graph.replace_subexpressions((energy,), dict(substitutions.values()))[0]


def _stable_pw_parallel_difference(graph: Graph, rs: Expr, fraction: Expr) -> Expr:
    """Evaluate PW g(2,rs)-g(2,rs*(1-fraction)^(-1/3)) without subtraction.

    The pinned modified PW parameters below come from lda_c_pw.mpl. The
    logarithmic and radial increments are O(fraction), even at full spin
    polarization where both individual PW energies remain O(1).
    """
    amplitude = Fraction("0.01554535")
    alpha = Fraction("0.20548")
    beta = tuple(map(Fraction, ("14.1189", "6.1977", "3.3662", "0.62517")))
    log_radius = -graph.stable_unary("log1p", -fraction) / 3
    radial_increment = rs * graph.stable_unary("expm1", log_radius)
    sqrt_rs = rs.pow(0.5)
    power_three_halves = rs.pow(1.5)
    square_rs = rs.pow(2)
    auxiliary = (
        beta[0] * sqrt_rs
        + beta[1] * rs
        + beta[2] * power_three_halves
        + beta[3] * square_rs
    )
    auxiliary_increment = (
        beta[0] * sqrt_rs * graph.stable_unary("expm1", log_radius / 2)
        + beta[1] * radial_increment
        + beta[2] * power_three_halves * graph.stable_unary("expm1", 3 * log_radius / 2)
        + beta[3] * square_rs * graph.stable_unary("expm1", 2 * log_radius)
    )
    denominator = 2 * amplitude * auxiliary
    increment = 2 * amplitude * auxiliary_increment
    new_log = graph.stable_unary("log1p", 1 / (denominator + increment))
    log_difference = graph.stable_unary(
        "log1p", increment / (denominator + 1)
    ) - graph.stable_unary("log1p", increment / denominator)
    return (
        2
        * amplitude
        * (alpha * radial_increment * new_log + (1 + alpha * rs) * log_difference)
    )


def _stable_pw_stoll_perp(
    graph: Graph, module: MapleModule, rs: Expr, major: Expr, minor: Expr
) -> Expr:
    """Evaluate the pinned Stoll antiparallel term near full polarization.

    Analytically separate terms that vanish at the pure-spin limit before
    evaluating them, rather than subtracting three O(1) PW correlation terms.
    """
    fraction = minor / (major + minor)
    eta = 2 * fraction
    two_power = graph.approximate_constant(2.0 ** (4.0 / 3.0))
    one_minus_f = (
        -two_power
        * graph.stable_unary(
            "expm1", Fraction(4, 3) * graph.stable_unary("log1p", -fraction)
        )
        - eta.pow(4.0 / 3.0)
    ) / (two_power - 2)
    spin_factor = 1 - one_minus_f
    one_minus_zeta_fourth = eta * (4 - eta * (6 - eta * (4 - eta)))
    correlation_zero = module.call(graph, "g", Fraction(1), rs)
    correlation_one = module.call(graph, "g", Fraction(2), rs)
    correlation_spin = module.call(graph, "g", Fraction(3), rs)
    major_rs = rs * (1 - fraction).pow(-1.0 / 3.0)
    minor_rs = rs * fraction.pow(-1.0 / 3.0)
    # opz_pow_n(-1, 4/3) uses the zeta floor even for the nominally pure-spin
    # PW parallel terms. Keep its small contribution separate: adding it to
    # g(2) first would round it away before the antiparallel subtraction.
    pure_spin_offset = graph.constant(Fraction(_ZETA_THRESHOLD)).pow(4.0 / 3.0) / (
        two_power - 2
    )
    major_parallel_correction = (
        (1 - fraction)
        * pure_spin_offset
        * (
            module.call(graph, "g", Fraction(2), major_rs)
            - module.call(graph, "g", Fraction(1), major_rs)
        )
    )
    parallel_minor = graph.select_le(
        minor,
        graph.constant(Fraction(str(DENSITY_THRESHOLD))),
        0,
        fraction
        * (
            module.call(graph, "g", Fraction(2), minor_rs)
            + pure_spin_offset
            * (
                module.call(graph, "g", Fraction(2), minor_rs)
                - module.call(graph, "g", Fraction(1), minor_rs)
            )
        ),
    )
    return (
        (one_minus_f + spin_factor * one_minus_zeta_fourth)
        * (correlation_zero - correlation_one)
        + _stable_pw_parallel_difference(graph, rs, fraction)
        + fraction * module.call(graph, "g", Fraction(2), major_rs)
        - major_parallel_correction
        - parallel_minor
        - spin_factor
        * one_minus_zeta_fourth
        * correlation_spin
        / Fraction("1.709920934161365617563962776245")
    )


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
    if spec.spin == "polarized":
        if "MGGA_C_WB97M_V" in active:
            os_term = module.call(graph, "b97mv_fos", rs, zeta, xs_a, xs_b, ts_a, ts_b)
            os_node = graph.node(os_term)
            if os_node.operation != "multiply" or len(os_node.arguments) != 2:
                raise ValueError("omegaB97M-V Stoll decomposition changed")
            original = Expr(graph, os_node.arguments[0])
            if original.identifier not in graph.topological_order((total,)):
                raise ValueError("omegaB97M-V Stoll decomposition is not shared")
            rho_a, rho_b = variables[:2]
            alpha_major = _stable_pw_stoll_perp(graph, module, rs, rho_a, rho_b)
            beta_major = _stable_pw_stoll_perp(graph, module, rs, rho_b, rho_a)
            zeta_floor = graph.constant(Fraction(_ZETA_THRESHOLD))
            alpha_major = graph.select_le(1 - zeta, zeta_floor, original, alpha_major)
            beta_major = graph.select_le(1 + zeta, zeta_floor, original, beta_major)
            stabilized = graph.select_le(
                1000 * rho_b,
                rho_a,
                alpha_major,
                graph.select_le(1000 * rho_a, rho_b, beta_major, original),
            )
            total = graph.replace_subexpressions((total,), {original: stabilized})[0]
        total = _restore_work_spin_density(graph, total, variables, zeta)
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
    runtime_sources = _validate_runtime_policy_sources()
    return {
        "kind": "libxc-maple",
        "importer_semantics": IMPORTER_SEMANTICS,
        "adapter_sha256": file_hash(Path(__file__)),
        "importer_sha256": file_hash(Path(libxc_maple.__file__)),
        "runtime_policy": {
            "density_threshold": DENSITY_THRESHOLD,
            "sigma_threshold": SIGMA_THRESHOLD,
            "tau_threshold": TAU_THRESHOLD,
            "smooth_lr_cutoff": SMOOTH_LR_CUTOFF,
            "smooth_lr_order": SMOOTH_LR_ORDER,
            "source_sha256": {path.name: file_hash(path) for path in runtime_sources},
        },
        "components": {
            name: {
                "entry": "hyb_mgga_xc_wb97mv.mpl",
                "source_sha256": module.source_sha256,
                "transitive_sha256": module.transitive_sha256,
            }
            for name in sorted(active & set(WB97MV_COMPONENTS))
        },
    }

"""Split-hybrid MGGA spin-screen repair at the floored Libxc work boundary."""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.integral.expr import Expr
from vibeqc_compiler.xc.libxc_maple import MapleImportError

if TYPE_CHECKING:
    from typing import Any

    from vibeqc_compiler.integral.expr import Graph
    from vibeqc_compiler.xc.libxc_bulk import BulkProgram
    from vibeqc_compiler.xc.libxc_maple import MapleModule


_PW_PARAMETERS = {
    "params_a_a": "[0.0310907, 0.01554535, 0.0168869]",
    "params_a_alpha1": "[0.21370, 0.20548, 0.11125]",
    "params_a_beta1": "[7.5957, 14.1189, 10.357]",
    "params_a_beta2": "[3.5876, 6.1977, 3.6231]",
    "params_a_beta3": "[1.6382, 3.3662, 0.88026]",
    "params_a_beta4": "[0.49294, 0.62517, 0.49671]",
    "params_a_fz20": "1.709920934161365617563962776245",
}


def _pw_parallel_difference(graph: Graph, rs: Expr, fraction: Expr) -> Expr:
    """Evaluate the pinned PW radial increment without subtracting O(1) terms."""
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


def _stable_pw_perpendicular(
    graph: Graph,
    module: MapleModule,
    rs: Expr,
    major: Expr,
    minor: Expr,
    density_threshold: Fraction,
    zeta_threshold: Fraction,
) -> Expr:
    """Analytically cancel the O(1) Stoll terms before evaluating minority vxc."""
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
    pure_spin_offset = graph.constant(zeta_threshold).pow(4.0 / 3.0) / (two_power - 2)
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
        graph.constant(density_threshold),
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
        + _pw_parallel_difference(graph, rs, fraction)
        + fraction * module.call(graph, "g", Fraction(2), major_rs)
        - major_parallel_correction
        - parallel_minor
        - spin_factor
        * one_minus_zeta_fourth
        * correlation_spin
        / Fraction(_PW_PARAMETERS["params_a_fz20"])
    )


def stabilize_m06_2x_stoll(
    program: BulkProgram, module: MapleModule, record: dict[str, Any]
) -> BulkProgram:
    """Replace only the source-identical M06-2X Stoll perpendicular subgraph.

    The original PW total minus two parallel energies subtracts O(1) terms
    to obtain an O(minority/total) result. That cancellation amplifies one-ULP
    density perturbations in the minority sigma derivative. Preserve the
    source's PW and work parameters; fail closed if either source term changes.
    """
    if program.name != "MGGA_C_M06_2X" or program.spin != "polarized":
        raise MapleImportError(
            "M06-2X Stoll stabilization requires polarized correlation"
        )
    assignments = dict(module.assignments)
    if any(assignments.get(name) != value for name, value in _PW_PARAMETERS.items()):
        raise MapleImportError("M06-2X modified PW parameter policy changed")
    density_threshold = Fraction(record["bindings"]["p_a_dens_threshold"])
    zeta_threshold = Fraction(record["bindings"]["p_a_zeta_threshold"])
    graph = program.graph
    rho_a, rho_b, sigma_aa, _, sigma_bb, _, _, tau_a, tau_b = program.variables
    density = rho_a + rho_b
    zeta = (rho_a - rho_b) / density
    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    xs_a = sigma_aa.pow(0.5) * rho_a.pow(-4.0 / 3.0)
    xs_b = sigma_bb.pow(0.5) * rho_b.pow(-4.0 / 3.0)
    ts_a = tau_a * rho_a.pow(-5.0 / 3.0)
    ts_b = tau_b * rho_b.pow(-5.0 / 3.0)
    terms = (
        module.call(graph, "m05_fperp", rs, zeta, xs_a, xs_b, ts_a, ts_b),
        module.call(graph, "vsxc_fperp", rs, zeta, xs_a, xs_b, ts_a, ts_b),
    )
    reachable = set(graph.topological_order((program.energy,)))
    if any(term.identifier not in reachable for term in terms):
        raise MapleImportError("M06-2X perpendicular PW terms changed")
    nodes = tuple(graph.node(term) for term in terms)
    if any(node.operation != "multiply" for node in nodes):
        raise MapleImportError("M06-2X perpendicular PW shape changed")
    shared = set(nodes[0].arguments) & set(nodes[1].arguments)
    if len(shared) != 1:
        raise MapleImportError("M06-2X perpendicular PW source is not shared")
    original = next(iter(shared))
    if graph.nodes[original].operation != "add":
        raise MapleImportError("M06-2X perpendicular PW structure changed")
    alpha_major = _stable_pw_perpendicular(
        graph, module, rs, rho_a, rho_b, density_threshold, zeta_threshold
    )
    beta_major = _stable_pw_perpendicular(
        graph, module, rs, rho_b, rho_a, density_threshold, zeta_threshold
    )
    stable = graph.select_le(rho_a, rho_b, beta_major, alpha_major)
    # When both floored spins hit the inclusive source screen, neither
    # parallel PW term exists. Keep the original source branch for that case.
    major = graph.select_le(rho_a, rho_b, rho_b, rho_a)
    stable = graph.select_le(major, density_threshold, Expr(graph, original), stable)
    (energy,) = graph.replace_subexpressions(
        (program.energy,), {Expr(graph, original): stable}
    )
    return replace(
        program,
        energy=energy,
        identity=canonical_hash(
            {
                "source_identity": program.identity,
                "stoll_semantics": "pinned-pw-small-minority/v1",
                "adapter_sha256": file_hash(Path(__file__)),
            }
        ),
    )


def exact_spin_density_screens(
    program: BulkProgram, record: dict[str, Any]
) -> BulkProgram:
    """Preserve source screening at floored rho without rs/zeta round trips.

    At a Libxc work floor, reconstructing a spin density from rs and zeta can
    round just above the source's inclusive screen. Replace only the exact
    source-derived spin-density nodes before taking point derivatives.
    """
    if program.spin != "polarized" or program.family != "mgga":
        raise MapleImportError("split-hybrid screening requires polarized MGGA")
    graph = program.graph
    rho_a, rho_b = program.variables[:2]
    density = rho_a + rho_b
    zeta = (rho_a - rho_b) / density
    rs_factor = graph.approximate_constant((3.0 / (4.0 * math.pi)) ** (1.0 / 3.0))
    rs = rs_factor * density.pow(-1.0 / 3.0)
    reconstructed = (rs_factor / rs).pow(3.0)
    spin_a = (1 + zeta) * reconstructed / 2
    spin_b = (1 - zeta) * reconstructed / 2
    reachable = set(graph.topological_order((program.energy,)))
    if spin_a.identifier not in reachable and spin_b.identifier not in reachable:
        return program
    if not {spin_a.identifier, spin_b.identifier} <= reachable:
        raise MapleImportError(
            f"split-hybrid {program.name} has no matching source spin screens"
        )
    (energy,) = graph.replace_subexpressions(
        (program.energy,), {spin_a: rho_a, spin_b: rho_b}
    )
    return replace(
        program,
        energy=energy,
        identity=canonical_hash(
            {
                "source_identity": program.identity,
                "spin_screen": "libxc-work-spin-density/v1",
                "density_threshold": record["bindings"]["p_a_dens_threshold"],
                "adapter_sha256": file_hash(Path(__file__)),
            }
        ),
    )

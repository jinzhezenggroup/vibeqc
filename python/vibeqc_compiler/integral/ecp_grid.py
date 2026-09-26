"""Compiler-owned bounded ECP host quadrature and real s/p/d/f harmonics.

The root iteration and grid traversal are finite compiler schedules. Scalar
recurrences, coordinate transforms and normalization use the common DAG.
The independent CPU ECP oracle deliberately retains its own implementation.
"""

import math
import typing

from .expr import Graph
from .ir import ECP_MAX_ORBITAL_ANGULAR, ECP_MAX_PROJECTOR_ANGULAR
from .scalar_c import ScalarCEmitter
from .shell_spec import cartesian_components


def harmonic_roots() -> typing.Any:
    """Real orthonormal harmonics in the existing projector slot order."""
    graph = Graph()
    x, y, z = (graph.variable(a) for a in "xyz")
    return graph, (
        graph.constant(1 / math.sqrt(4 * math.pi)),
        math.sqrt(3 / (4 * math.pi)) * x,
        math.sqrt(3 / (4 * math.pi)) * y,
        math.sqrt(3 / (4 * math.pi)) * z,
        math.sqrt(15 / (4 * math.pi)) * x * y,
        math.sqrt(15 / (4 * math.pi)) * y * z,
        math.sqrt(5 / (16 * math.pi)) * (3 * z * z - 1),
        math.sqrt(15 / (4 * math.pi)) * x * z,
        math.sqrt(15 / (16 * math.pi)) * (x * x - y * y),
        math.sqrt(35 / (32 * math.pi)) * y * (3 * x * x - y * y),
        math.sqrt(105 / (4 * math.pi)) * x * y * z,
        math.sqrt(21 / (32 * math.pi)) * y * (5 * z * z - 1),
        math.sqrt(7 / (16 * math.pi)) * z * (5 * z * z - 3),
        math.sqrt(21 / (32 * math.pi)) * x * (5 * z * z - 1),
        math.sqrt(105 / (16 * math.pi)) * z * (x * x - y * y),
        math.sqrt(35 / (32 * math.pi)) * x * (x * x - 3 * y * y),
    )


def radial_map_roots() -> typing.Any:
    """Map a Legendre abscissa to r=t/(1-t), including dr/dz."""
    graph = Graph()
    z, weight = (graph.variable(a) for a in ("z", "weight"))
    t = (z + 1) / 2
    return graph, (t / (1 - t), weight / (2 * (1 - t) * (1 - t)))


def _assign(graph: typing.Any, roots: typing.Any, targets: typing.Any) -> typing.Any:
    emitter = ScalarCEmitter(graph, {})
    emitter.emit(roots)
    return emitter.lines + [
        f"  {target} = {emitter.reference(root)};"
        for target, root in zip(targets, roots, strict=True)
    ]


def _scalar(
    name: typing.Any, arguments: typing.Any, expression: typing.Any
) -> typing.Any:
    graph = Graph()
    root = expression(*(graph.variable(a) for a in arguments))
    emitter = ScalarCEmitter(graph, {})
    emitter.emit((root,))
    return [
        f"inline double {name}(" + ", ".join(f"double {a}" for a in arguments) + ") {",
        *emitter.lines,
        f"  return {emitter.reference(root)};",
        "}",
    ]


def emit_ecp_grid_cpp() -> typing.Any:
    """Host-only helpers; grid sizes and Newton stopping policy are unchanged."""
    lines = _scalar(
        "ecp_legendre_next",
        ("k", "z", "p", "previous"),
        lambda k, z, p, previous: ((2 * k - 1) * z * p - (k - 1) * previous) / k,
    )
    lines += _scalar(
        "ecp_legendre_derivative",
        ("n", "z", "p", "previous"),
        lambda n, z, p, previous: n * (z * p - previous) / (z * z - 1),
    )
    lines += _scalar(
        "ecp_legendre_weight",
        ("z", "derivative"),
        lambda z, d: 2 / ((1 - z * z) * d * d),
    )
    lines += _scalar(
        "ecp_root_angle",
        ("i", "n"),
        lambda i, n: math.pi * (i + 0.75) / (n + 0.5),
    )
    lines += _scalar("ecp_root_delta", ("p", "derivative"), lambda p, d: p / d)
    lines += _scalar("ecp_root_update", ("z", "delta"), lambda z, d: z - d)
    lines += [
        "inline std::vector<std::array<double, 2>> ecp_legendre(unsigned n) {",
        '  if (n < 8 || n > 512) throw std::invalid_argument("invalid ECP Legendre order");',
        "  std::vector<std::array<double, 2>> result(n);",
        "  for (unsigned i=0; i<(n+1)/2; ++i) {",
        "    double z = std::cos(ecp_root_angle(i, n)), derivative = 0;",
        "    for (unsigned iteration=0; iteration<64; ++iteration) {",
        "      double p=1, previous=0;",
        "      for (unsigned k=1; k<=n; ++k) {",
        "        const double next = ecp_legendre_next(k, z, p, previous);",
        "        previous=p; p=next;",
        "      }",
        "      derivative=ecp_legendre_derivative(n, z, p, previous);",
        "      const double delta=ecp_root_delta(p, derivative);",
        "      z=ecp_root_update(z, delta);",
        "      if (std::abs(delta)<2e-15) break;",
        '      if (iteration==63) throw std::runtime_error("ECP quadrature root did not converge");',
        "    }",
        "    const double weight=ecp_legendre_weight(z, derivative);",
        "    result[i]={-z, weight}; result[n-i-1]={z, weight};",
        "  }",
        "  return result;",
        "}",
        "inline void ecp_harmonics(double x, double y, double z, double* out) {",
    ]
    graph, roots = harmonic_roots()
    lines += _assign(
        graph, roots, [f"out[{i}]" for i in range((ECP_MAX_PROJECTOR_ANGULAR + 1) ** 2)]
    ) + ["}"]
    lines += [
        "template<class Radial>",
        "inline Radial ecp_radial_node(double z, double weight) {",
        "  Radial out{};",
    ]
    graph, roots = radial_map_roots()
    lines += _assign(graph, roots, ("out.r", "out.weight")) + ["  return out;", "}"]
    # Trigonometric calls lower directly to the host standard library. The DAG
    # consumes their values; no second symbolic trigonometric algebra is added.
    lines += _scalar(
        "ecp_azimuth",
        ("k", "nphi"),
        lambda k, n: 2 * math.pi * k / n,
    )
    lines += [
        "template<class Point>",
        "inline Point ecp_sphere_node(double z, double weight, unsigned k, unsigned nphi) {",
        "  const double phi=ecp_azimuth(k, nphi);",
        "  const double cosine=std::cos(phi), sine=std::sin(phi);",
        "  Point out{};",
    ]
    graph = Graph()
    z, weight, nphi, cosine, sine = (
        graph.variable(a) for a in ("z", "weight", "nphi", "cosine", "sine")
    )
    s = graph.power(1 - z * z, 0.5)
    lines += _assign(
        graph,
        (s * cosine, s * sine, z, weight * 2 * math.pi / nphi),
        ("out.x", "out.y", "out.z", "out.weight"),
    )
    lines += [
        "  ecp_harmonics(out.x, out.y, out.z, out.harmonics);",
        "  return out;",
        "}",
        "template<class Radial, class Point>",
        "inline void ecp_make_grid(unsigned radial, unsigned angular,",
        "    std::vector<Radial>& radii, std::vector<Point>& sphere) {",
        "  if (radial<16 || radial>512 || angular<8 || angular>96)",
        '    throw std::invalid_argument("invalid ECP quadrature grid");',
        "  for (const auto& tw: ecp_legendre(radial))",
        "    radii.push_back(ecp_radial_node<Radial>(tw[0], tw[1]));",
        "  const unsigned nphi=2*angular;",
        "  for (const auto& zw: ecp_legendre(angular))",
        "    for (unsigned k=0; k<nphi; ++k)",
        "      sphere.push_back(ecp_sphere_node<Point>(zw[0], zw[1], k, nphi));",
        "}",
        "inline double ecp_component_coefficient(unsigned x, unsigned y, unsigned z, double coefficient) {",
    ]
    for angular in range(ECP_MAX_ORBITAL_ANGULAR + 1):
        for component in cartesian_components(angular):
            powers = tuple(component.count(a) for a in "xyz")
            denominator = math.prod(math.prod(range(1, 2 * p, 2)) for p in powers)
            graph = Graph()
            coefficient = graph.variable("coefficient")
            root = coefficient * (1 / math.sqrt(denominator))
            emitter = ScalarCEmitter(graph, {})
            emitter.emit((root,))
            x, y, z = powers
            lines += [
                f"  if (x=={x} && y=={y} && z=={z}) {{",
                *emitter.lines,
                f"    return {emitter.reference(root)};",
                "  }",
            ]
    lines += [
        '  throw std::invalid_argument("ECP component exceeds validated s/p/d/f domain");',
        "}",
    ]
    return lines

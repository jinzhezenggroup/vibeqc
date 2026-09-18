"""Bounded scalar ECP quadrature lowering shared by host tests and CUDA.

The radial measure cancels the r**-2 in EcpRadialTerm. Projected AO jets
carry derivatives of the two basis centers at fixed ECP-centered nodes;
translation invariance recovers the third center after each reduction.
The native adapter owns scheduling, storage and physical-atom scatter.
"""

from .ecp import emit_ecp_ao_cuda
from .ecp_grid import emit_ecp_grid_cpp
from .expr import Graph
from .ir import ECP_MAX_PROJECTOR_ANGULAR, EcpRadialTerm
from .scalar_c import ScalarCEmitter


def radial_roots(power):
    """Weighted residual radial integrand, including the volume measure."""
    EcpRadialTerm(-1, power, 1.0, 1.0)  # Reuse the operator domain contract.
    graph = Graph()
    r, alpha, coefficient, weight = (
        graph.variable(name) for name in ("r", "alpha", "coefficient", "weight")
    )
    polynomial = graph.constant(1)
    for _ in range(power):
        polynomial = polynomial * r
    return graph, (
        weight * coefficient * polynomial * graph.exponential(-alpha * r * r),
    )


def pair_roots():
    """Value and A/B center jets from the same bilinear expression."""
    graph = Graph()
    a, b, weight = (graph.variable(name) for name in ("a0", "b0", "weight"))
    value = weight * a * b
    da, db = graph.differentiate(value, a), graph.differentiate(value, b)
    return graph, (
        value,
        *(da * graph.variable(f"a{d}") for d in range(1, 4)),
        *(db * graph.variable(f"b{d}") for d in range(1, 4)),
    )


def _emit(graph, roots, variables=None, *, accumulate=False):
    emitter = ScalarCEmitter(graph, variables or {})
    emitter.emit(roots)
    op = "+=" if accumulate else "="
    return emitter.lines + [
        f"  out[{i}] {op} {emitter.reference(root)};" for i, root in enumerate(roots)
    ]


def _scalar_function(name, arguments, expression):
    """Lower small consumer expressions through the shared scalar algebra."""
    graph = Graph()
    root = expression(*(graph.variable(arg) for arg in arguments))
    emitter = ScalarCEmitter(graph, {})
    emitter.emit((root,))
    return [
        f"VIBEQC_ECP_INLINE double {name}("
        + ", ".join(f"double {arg}" for arg in arguments)
        + ") {",
        *emitter.lines,
        f"  return {emitter.reference(root)};",
        "}",
    ]


def _emit_ao_consumer():
    """Contract normalized Cartesian components without changing loop order."""
    lines = _scalar_function(
        "ecp_node_displacement",
        ("center", "radius", "direction", "basis_center"),
        lambda c, r, u, a: c + r * u - a,
    )
    lines += _scalar_function(
        "ecp_ao_term",
        ("component", "primitive", "value"),
        lambda c, p, v: c * p * v,
    )
    lines += [
        "template<class Jet, class AO, class Primitive, class Point>",
        "VIBEQC_ECP_INLINE Jet ecp_evaluate_ao(const AO& ao,",
        "    const Primitive* primitives, const Point& point, double radius,",
        "    double cx, double cy, double cz, bool derivatives) {",
        "  const double x = ecp_node_displacement(cx, radius, point.x, ao.x);",
        "  const double y = ecp_node_displacement(cy, radius, point.y, ao.y);",
        "  const double z = ecp_node_displacement(cz, radius, point.z, ao.z);",
        "  Jet out{};",
        "  for (int t=0; t<ao.term_count; ++t) {",
        "    const auto term = ao.components[t];",
        "    for (int k=0; k<ao.primitive_count; ++k) {",
        "      const auto p = primitives[ao.primitive_offset+k];",
        "      double roots[4];",
        "      ecp_ao(term.x, term.y, term.z, x, y, z, p.exponent, roots);",
        "      for (int d=0; d<(derivatives ? 4 : 1); ++d)",
        "        out.v[d] += ecp_ao_term(term.coefficient, p.coefficient, roots[d]);",
        "    }",
        "  }",
        "  return out;",
        "}",
    ]
    return lines


def _emit_weighted_consumer():
    """Full AO contraction: arbitrary real weights, no occupancy multiplier."""
    lines = _scalar_function(
        "ecp_add_operator",
        ("hcore", "local", "nonlocal_value"),
        lambda h, l, n: h + (l + n),
    )
    lines += _scalar_function(
        "ecp_weighted_term",
        ("weight", "local", "nonlocal_value"),
        lambda w, l, n: w * (l + n),
    )
    lines += _scalar_function(
        "ecp_energy_derivative_to_force", ("derivative",), lambda d: -d
    )
    lines += [
        "VIBEQC_ECP_INLINE double ecp_force_component(const double* local,",
        "    const double* nonlocal_value, const double* weights, int size) {",
        "  double sum = 0;",
        "  for (int j=0; j<size; ++j)",
        "    sum += ecp_weighted_term(weights[j], local[j], nonlocal_value[j]);",
        "  return ecp_energy_derivative_to_force(sum);",
        "}",
    ]
    return lines


def emit_ecp_quadrature_cpp():
    """Emit the validated s/p/d/f projector and powers 0..4 domain.

    Type parameters expose only scalar term/node/jet records. They do not
    select a mathematical backend. The same emitted arithmetic is compiled
    as ordinary C++ for independent native tests and as CUDA for production.
    """
    lines = [
        "#pragma once",
        "#include <cmath>",
        "#include <array>",
        "#include <stdexcept>",
        "#include <vector>",
        "#if defined(__CUDACC__)",
        "#define VIBEQC_ECP_INLINE __host__ __device__ inline",
        "#else",
        "#define VIBEQC_ECP_INLINE inline",
        "#endif",
        emit_ecp_ao_cuda()
        .replace("#pragma once\n", "")
        .replace("__device__ inline", "VIBEQC_ECP_INLINE"),
        "namespace vibeqc::generated {",
        f"inline constexpr int ecp_max_projector_angular = {ECP_MAX_PROJECTOR_ANGULAR};",
        f"inline constexpr int ecp_projector_count = {(ECP_MAX_PROJECTOR_ANGULAR + 1) ** 2};",
        *emit_ecp_grid_cpp(),
        *_emit_ao_consumer(),
        *_emit_weighted_consumer(),
        "// Scalar ECP operator contract: local=-1, projectors=0..3, powers=0..4.",
        "VIBEQC_ECP_INLINE double ecp_radial(unsigned power, double r,",
        "    double alpha, double coefficient, double weight) {",
        "  double out[1];",
        "  switch (power) {",
    ]
    for power in range(5):
        graph, roots = radial_roots(power)
        lines += [f"  case {power}: {{", *_emit(graph, roots), "return out[0]; }"]
    lines += [
        "  default: return NAN; // The public resolver rejects unsupported powers.",
        "  }",
        "}",
        "VIBEQC_ECP_INLINE void ecp_pair(const double* a, const double* b,",
        "    double weight, bool derivatives, double* out) {",
    ]
    graph, roots = pair_roots()
    variables = {f"{ab}{d}": f"{ab}[{d}]" for ab in "ab" for d in range(4)}
    lines += _emit(graph, roots[:1], variables, accumulate=True)
    # A separate lexical scope avoids temporary-name collisions and does not
    # read derivative slots on a value-only call.
    lines += ["  if (derivatives) {"]
    emitter = ScalarCEmitter(graph, variables)
    emitter.emit(roots[1:])
    lines += emitter.lines
    lines += [
        f"  out[{i}] += {emitter.reference(root)};"
        for i, root in enumerate(roots[1:], 1)
    ]
    lines += ["  }", "}"]
    graph = Graph()
    weight, harmonic, value = (
        graph.variable(name) for name in ("weight", "harmonic", "value")
    )
    emitter = ScalarCEmitter(graph, {})
    root = weight * harmonic * value
    emitter.emit((root,))
    lines += [
        "VIBEQC_ECP_INLINE double ecp_projection_term(double weight,",
        "    double harmonic, double value) {",
        *emitter.lines,
        f"  return {emitter.reference(root)};",
        "}",
        "template<class Jet, class Point>",
        "VIBEQC_ECP_INLINE Jet ecp_project(const Jet* values, const Point* sphere,",
        "    int nq, int m, bool derivatives) {",
        "  Jet out{};",
        "  for (int q=0; q<nq; ++q) {",
        "    const auto value = values[q];",
        "    for (int d=0; d<(derivatives ? 4 : 1); ++d)",
        "      out.v[d] += ecp_projection_term(sphere[q].weight,",
        "          sphere[q].harmonics[m], value.v[d]);",
        "  }",
        "  return out;",
        "}",
        "template<class Term, class Radial, class Point, class Jet>",
        "VIBEQC_ECP_INLINE void ecp_contract(const Term* terms, int nt,",
        "    const Radial& radial, const Point* sphere, int nq, int center,",
        "    const Jet* va, const Jet* vb, const Jet* pa, const Jet* pb,",
        "    bool derivatives, double (&parts)[2][10]) {",
        f"  double potential[{ECP_MAX_PROJECTOR_ANGULAR + 2}] = {{}};",
        "  for (int part=0; part<2; ++part)",
        "    for (int d=0; d<10; ++d) parts[part][d]=0;",
        "  for (int t=0; t<nt; ++t) {",
        "    const auto term=terms[t];",
        "    if (term.atom_index == static_cast<unsigned>(center))",
        "      potential[term.channel+1] += ecp_radial(term.power, radial.r,",
        "          term.exponent, term.coefficient, radial.weight);",
        "  }",
        "  for (int q=0; q<nq; ++q)",
        "    ecp_pair(va[q].v, vb[q].v, potential[0]*sphere[q].weight,",
        "        derivatives, parts[0]);",
    ]
    # Emit the projector multiplicities from the operator channel convention;
    # each l contains 2*l+1 orthonormal real harmonics, beginning at l*l.
    for channel in range(ECP_MAX_PROJECTOR_ANGULAR + 1):
        EcpRadialTerm(channel, 0, 1.0, 1.0)
        lines += [
            f"  for (int m={channel**2}; m<{(channel + 1) ** 2}; ++m)",
            f"    ecp_pair(pa[m].v, pb[m].v, potential[{channel + 1}],",
            "        derivatives, parts[1]);",
        ]
    graph = Graph()
    a, b = graph.variable("a"), graph.variable("b")
    emitter = ScalarCEmitter(graph, {"a": "parts[part][d+1]", "b": "parts[part][d+4]"})
    root = -(a + b)
    emitter.emit((root,))
    lines += [
        "  if (derivatives)",
        "    for (int part=0; part<2; ++part)",
        "      for (int d=0; d<3; ++d) {",
        *emitter.lines,
        f"        parts[part][d+7] = {emitter.reference(root)};",
        "      }",
        "}",
        "} // namespace vibeqc::generated",
        "#undef VIBEQC_ECP_INLINE",
        "",
    ]
    return "\n".join(lines)

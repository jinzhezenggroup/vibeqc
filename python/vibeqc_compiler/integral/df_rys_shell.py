"""Angular-parameterized full-range first DF derivatives at Rys t² nodes.

One lowering substitutes root-dependent Gaussian means/covariances into the
shared moment IR and applies raised/lowered orbital derivatives at generation
time. All six orbital coordinates contract directly into the existing response
sink; the common finish() recovers the auxiliary center by translation. Root
eligibility is mathematical availability, never automatic production promotion.
"""

import typing
from itertools import product

from .cuda import CudaEmitter
from .df_derivatives_cuda import emit_df_geometry_cuda
from .df_values import build_df_axis_moment
from .expr import Graph, Node
from .shell_spec import cartesian_components

COMPONENT_RYS_SHELL_CLASSES = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (2, 0, 0),
)

# Practical s/p/d orbital bases with f auxiliary shells need only the existing
# independently qualified three/four-root quadrature for these five classes.
# The d-d-f class needs five roots and deliberately retains its polynomial
# fallback until that quadrature meets the same strict DF accuracy contract.
AUXILIARY_F_RYS_SHELL_CLASSES = (
    (0, 0, 3),
    (1, 0, 3),
    (1, 1, 3),
    (2, 0, 3),
    (2, 1, 3),
)

# The measured low-angular winners retain their delivered component lowering.
# Controls 101/110 and all remaining canonical s/p/d classes use shared axes;
# mathematical availability never changes the qualified production manifest.
COOPERATIVE_RYS_SHELL_CLASSES = (
    (1, 0, 1),
    (1, 0, 2),
    (1, 1, 0),
    (1, 1, 1),
    (1, 1, 2),
    (2, 0, 1),
    (2, 0, 2),
    (2, 1, 0),
    (2, 1, 1),
    (2, 1, 2),
    (2, 2, 0),
    (2, 2, 1),
    (2, 2, 2),
) + AUXILIARY_F_RYS_SHELL_CLASSES
RYS_SHELL_CLASSES = tuple(
    sorted(set(COMPONENT_RYS_SHELL_CLASSES + COOPERATIVE_RYS_SHELL_CLASSES))
)


from .df_shell_derivatives import select_shell_classes


def shell_rys_roots(angular: typing.Any) -> typing.Any:
    """Bound eligibility to the declared full-range first-derivative family."""
    if tuple(angular) not in RYS_SHELL_CLASSES:
        raise ValueError("Rys shell class is not generated")
    return (sum(angular) + 1) // 2 + 1


def build_df_rys_component_ir(components: typing.Any) -> typing.Any:
    """Contract one Cartesian component through shared Gaussian-moment IR.

    The node is u=t², not t or u/(1-u). Exponents and external response weights
    are held fixed when differentiating centers. The covariance substitution is
    identical for every angular tuple. State counts describe visited nonconstant
    mathematical moments before CSE, not hardware instructions or elapsed time.
    """
    angular = tuple(len(component) for component in components)
    shell_rys_roots(angular)
    graph = Graph()
    root = graph.variable("root")
    sx, sy, ip, iq = (graph.variable(n) for n in ("sx", "sy", "ip", "iq"))
    state_sets = [set() for _ in range(3)]
    moment_cache = {}

    def moment(axis: typing.Any, powers: typing.Any) -> typing.Any:
        key = (axis, powers)
        if key in moment_cache:
            return moment_cache[key]
        source, value = build_df_axis_moment(
            *powers, internal_derivative=True, states=state_sets[axis]
        )
        dx = graph.variable(f"dx_{axis}")
        replacements = {
            "mean_0": graph.variable(f"pa_{axis}") - dx * sx * root,
            "mean_1": graph.variable(f"pb_{axis}") - dx * sx * root,
            "mean_2": dx * sy * root,
            "variance_x": ip * (1 - sx * root),
            "covariance_xy": ip * sy * root,
            "variance_y": iq * (1 - sy * root),
        }
        cloned = {}
        for identifier in source.topological_order((value,)):
            node = source.nodes[identifier]
            if node.operation == "variable":
                cloned[identifier] = replacements[node.payload]
            elif node.operation == "constant":
                cloned[identifier] = graph.clone_constant(node)
            else:
                cloned[identifier] = graph._intern(
                    Node(
                        node.operation,
                        tuple(cloned[child].identifier for child in node.arguments),
                        node.payload,
                    )
                )
        moment_cache[key] = cloned[value.identifier]
        return moment_cache[key]

    powers = [tuple(c.count(axis) for c in components) for axis in "xyz"]
    base = [moment(axis, p) for axis, p in enumerate(powers)]
    factor = (
        graph.variable("weight")
        * graph.variable("prefactor")
        * graph.variable("root_weight")
    )
    outputs = []
    for center, exponent in enumerate(("alpha", "beta")):
        for axis, p in enumerate(powers):
            raised = list(p)
            raised[center] += 1
            derivative = 2 * graph.variable(exponent) * moment(axis, tuple(raised))
            if p[center]:
                lowered = list(p)
                lowered[center] -= 1
                derivative -= p[center] * moment(axis, tuple(lowered))
                weighted = factor * derivative
            else:
                # Preserve the delivered SSS multiplication order while using
                # the same angular-parameterized path for every component.
                weighted = (
                    factor * 2 * graph.variable(exponent) * moment(axis, tuple(raised))
                )
            outputs.append(weighted * base[(axis + 1) % 3] * base[(axis + 2) % 3])
    return graph, tuple(outputs), sum(map(len, state_sets))


def build_df_rys_sss_ir() -> typing.Any:
    """Compatibility entry point; SSS uses the same parameterized derivative IR."""
    graph, outputs, _ = build_df_rys_component_ir(("", "", ""))
    return graph, outputs


def shell_rys_work_model(angular: typing.Any) -> typing.Any:
    """Report root work and per-active-component recurrence work before CSE."""
    roots = shell_rys_roots(angular)
    components = tuple(product(*(cartesian_components(l) for l in angular)))
    if tuple(angular) in COOPERATIVE_RYS_SHELL_CLASSES:
        _, outputs, states = build_df_rys_shared_axis_ir(angular)
        return {
            "rys_roots": roots,
            "recurrence_states": 3 * roots * states,
            "shared_recurrence_states": 3 * roots * states,
            "component_recurrence_states": [0] * len(components),
            "axis_polynomial_calls": 0,
            "specialized_prepare_axis_calls": 0,
            "cache_coefficient_values": 0,
            "shared_cache_values": 3 * roots * len(outputs),
            "component_convolution_iterations": [0] * len(components),
        }
    states = [roots * build_df_rys_component_ir(c)[2] for c in components]
    return {
        "rys_roots": roots,
        "recurrence_states": sum(states),
        "component_recurrence_states": states,
        "axis_polynomial_calls": 0,
        "specialized_prepare_axis_calls": 0,
        "cache_coefficient_values": 0,
        "component_convolution_iterations": [0] * len(components),
    }


def emit_df_rys_policy_cpp(*, classes: typing.Any = None) -> typing.Any:
    """Expose generated availability without importing CUDA into host policy."""
    selected = select_shell_classes(classes)
    storage = "inline" if classes is None else "static"
    mask = sum(
        1 << (16 * a + 4 * b + c)
        for a, b, c in RYS_SHELL_CLASSES
        if (a, b, c) in selected
    )
    return f"""// Generated derivative-lowering capability; promotion uses the tuning manifest.
#pragma once
#include <cstdint>
namespace vibeqc::scf::generated_df_shell {{
{storage} constexpr std::uint64_t rys_available_mask={mask}ULL;
{storage} constexpr std::uint64_t rys_qualified_mask=0;
template<unsigned A,unsigned B,unsigned C>
{storage} constexpr bool rys_available=(rys_available_mask & (1ULL<<(16*A+4*B+C)))!=0;
}} // namespace vibeqc::scf::generated_df_shell
"""


def emit_df_rys_shell_cuda(
    *,
    classes: typing.Any = None,
    shell_header: typing.Any = "generated_df_shell_derivatives.cuh",
    policy_header: typing.Any = "generated_df_rys_policy.hpp",
) -> typing.Any:
    """Emit the complete bounded family against the unchanged packet/sink ABI.

    Delivered low-angular winners retain pruned component moments in registers.
    Higher classes share one bounded axis program per root/axis. Unsupported
    tuples have no specialization; qualification remains separate from support.
    """
    selected = select_shell_classes(classes)
    geometry = emit_df_geometry_cuda(
        "prepare_geometry_rys",
        moments="if(work) *work={}; (void)total; generated_df_rys::roots<Roots>(rho*distance,g.f,g.f+Roots);",
    ).replace("__device__", "static __device__")
    lines = [
        "// Compiler-owned low-l Rys derivatives from shared Gaussian-moment IR.",
        "#ifndef VIBEQC_GENERATED_DF_RYS_SHELL_CUH",
        "#define VIBEQC_GENERATED_DF_RYS_SHELL_CUH",
        '#include "generated_df_rys.cuh"',
        f'#include "{policy_header}"',
        f'#include "{shell_header}"',
        "namespace vibeqc::scf::generated_df_derivatives {",
        "template<unsigned Roots>",
        geometry,
        "}",
        "namespace vibeqc::scf::generated_df_shell {",
        "template<unsigned A,unsigned B,unsigned C> struct RysShell;",
    ]
    for angular in RYS_SHELL_CLASSES:
        if angular not in selected:
            continue
        if angular in COOPERATIVE_RYS_SHELL_CLASSES:
            lines.extend(_emit_cooperative_shell(angular))
            continue
        parameters = ",".join(map(str, angular))
        roots = shell_rys_roots(angular)
        components = tuple(product(*(cartesian_components(l) for l in angular)))
        programs = [build_df_rys_component_ir(c) for c in components]
        states = [roots * p[2] for p in programs]
        lines += [
            f"template<> struct RysShell<{parameters}> : Shell<{parameters}> {{",
            f"  static constexpr unsigned nroots={roots};",
            "  static constexpr unsigned shared_recurrence_states=0;",
            "  static constexpr unsigned axis_size=1,cache_coefficient_values=0;",
            "  static constexpr unsigned polynomial_calls=0,specialized_axis_calls=0;",
            f"  static constexpr unsigned recurrence_states_per_primitive={sum(states)};",
            "  __device__ static unsigned recurrence_work(unsigned item) {",
            "    switch(item) {",
            *(f"    case {i}: return {count};" for i, count in enumerate(states)),
            "    } return 0;",
            "  }",
            "  __device__ static unsigned convolution_work(unsigned) { return 0; }",
            "  __device__ static void prepare(const scalar::Geometry&,double*,unsigned,unsigned) {}",
            "  __device__ static void accumulate(unsigned item,double alpha,double beta,",
            "      const scalar::Geometry& g,const double*,double weight,double* out) {",
            "    switch(item) {",
        ]
        for item, (graph, outputs, _) in enumerate(programs):
            variables = {name: name for name in ("weight", "alpha", "beta")}
            variables.update(
                {name: f"g.{name}" for name in ("prefactor", "sx", "sy", "ip", "iq")}
            )
            variables.update(
                {
                    f"{name}_{axis}": f"g.{name}[{axis}]"
                    for name in ("pa", "pb", "dx")
                    for axis in range(3)
                }
            )
            variables.update(root="g.f[root]", root_weight=f"g.f[root+{roots}]")
            emitter = CudaEmitter(graph, variables)
            emitter.emit(outputs)
            lines += [
                f"    case {item}: {{",
                "#pragma unroll",
                f"      for(unsigned root=0;root<{roots};++root) {{",
                *emitter.lines,
            ]
            lines += [
                f"        out[{axis}]+={emitter.reference(value)};"
                for axis, value in enumerate(outputs)
            ]
            lines += ["      }", "      return;", "    }"]
        lines += ["    }", "  }", "};"]
    lines += ["} // namespace vibeqc::scf::generated_df_shell", "#endif", ""]
    return "\n".join(lines)


def build_df_rys_shared_axis_ir(angular: typing.Any) -> typing.Any:
    """Share the existing moment IR over one rectangular root/axis cache.

    Raising B follows from raised A plus (A-B) times the base state, so only
    A's bound is extended. Every requested moment and its dependencies are
    interned in one graph; components do no recurrence work of their own.
    The returned count excludes the constant seed and describes mathematical
    recurrence states before CSE, matching the existing work-ledger convention.
    """
    if tuple(angular) not in COOPERATIVE_RYS_SHELL_CLASSES:
        raise ValueError("cooperative Rys requires a declared canonical shell class")
    graph = Graph()
    root = graph.variable("root")
    pa, pb, dx, sx, sy, ip, iq = (
        graph.variable(name) for name in ("pa", "pb", "dx", "sx", "sy", "ip", "iq")
    )
    replacements = {
        "mean_0": pa - dx * sx * root,
        "mean_1": pb - dx * sx * root,
        "mean_2": dx * sy * root,
        "variance_x": ip * (1 - sx * root),
        "covariance_xy": ip * sy * root,
        "variance_y": iq * (1 - sy * root),
    }
    states, outputs = set(), []
    for powers in product(
        range(angular[0] + 2), range(angular[1] + 1), range(angular[2] + 1)
    ):
        source, value = build_df_axis_moment(
            *powers, internal_derivative=True, states=states
        )
        cloned = {}
        for identifier in source.topological_order((value,)):
            node = source.nodes[identifier]
            if node.operation == "variable":
                cloned[identifier] = replacements[node.payload]
            elif node.operation == "constant":
                cloned[identifier] = graph.clone_constant(node)
            else:
                cloned[identifier] = graph._intern(
                    Node(
                        node.operation,
                        tuple(cloned[child].identifier for child in node.arguments),
                        node.payload,
                    )
                )
        outputs.append(cloned[value.identifier])
    return graph, tuple(outputs), len(states)


def _emit_cooperative_shell(angular: typing.Any) -> typing.Any:
    """Lower the shared root/axis graph into the existing shell packet ABI."""
    parameters = ",".join(map(str, angular))
    roots = shell_rys_roots(angular)
    graph, outputs, states = build_df_rys_shared_axis_ir(angular)
    variables = {name: f"g.{name}" for name in ("sx", "sy", "ip", "iq")}
    variables.update({name: f"g.{name}[axis]" for name in ("pa", "pb", "dx")})
    variables["root"] = "g.f[root]"
    emitter = CudaEmitter(graph, variables)
    emitter.emit(outputs)
    a, b, c = angular
    return [
        f"template<> struct RysShell<{parameters}> : Shell<{parameters}> {{",
        f"  using Base=Shell<{parameters}>;",
        f"  static constexpr unsigned nroots={roots},entries={len(outputs)};",
        "  static constexpr bool shared_root_state=true;",
        "  static constexpr unsigned axis_size=nroots*entries,cache_coefficient_values=0;",
        "  static constexpr unsigned polynomial_calls=0,specialized_axis_calls=0;",
        f"  static constexpr unsigned shared_recurrence_states={3 * roots * states};",
        "  static constexpr unsigned recurrence_states_per_primitive=shared_recurrence_states;",
        "  __device__ static unsigned recurrence_work(unsigned) { return 0; }",
        "  __device__ static unsigned convolution_work(unsigned) { return 0; }",
        "  __device__ __forceinline__ static unsigned offset(unsigned a,unsigned b,unsigned c) {",
        f"    return (a*{b + 1}+b)*{c + 1}+c;",
        "  }",
        "  // Independent root/axis owners publish all states before the caller's rendezvous.",
        "  __device__ static void prepare(const scalar::Geometry& g,double* cache,unsigned lane,unsigned lanes) {",
        "    for(unsigned item=lane;item<3*nroots;item+=lanes) {",
        "      const unsigned axis=item/nroots,root=item%nroots;",
        "      double* values=cache+item*entries;",
        *emitter.lines,
        *(
            f"      values[{i}]={emitter.reference(value)};"
            for i, value in enumerate(outputs)
        ),
        "    }",
        "  }",
        "  __device__ static void accumulate(unsigned item,double alpha,double beta,",
        "      const scalar::Geometry& g,const double* cache,double weight,double* out) {",
        f"    const auto a=angular<{a}>(item/Base::nb/Base::nc);",
        f"    const auto b=angular<{b}>(item/Base::nc%Base::nb),c=angular<{c}>(item%Base::nc);",
        "#pragma unroll",
        "    for(unsigned root=0;root<nroots;++root) {",
        "      double base[3],da[3],db[3];",
        "#pragma unroll",
        "      for(unsigned axis=0;axis<3;++axis) {",
        "        const unsigned x=scalar::power(a,axis),y=scalar::power(b,axis),z=scalar::power(c,axis);",
        "        const double* values=cache+axis*axis_size+root*entries;",
        "        base[axis]=values[offset(x,y,z)];",
        "        const double raised=values[offset(x+1,y,z)];",
        "        da[axis]=2*alpha*raised-(x ? x*values[offset(x-1,y,z)] : 0.0);",
        "        const double raised_b=raised+(g.pb[axis]-g.pa[axis])*base[axis];",
        "        db[axis]=2*beta*raised_b-(y ? y*values[offset(x,y-1,z)] : 0.0);",
        "      }",
        "      const double factor=weight*g.prefactor*g.f[nroots+root];",
        "#pragma unroll",
        "      for(unsigned axis=0;axis<3;++axis) {",
        "        const double other=base[(axis+1)%3]*base[(axis+2)%3];",
        "        out[axis]+=factor*da[axis]*other;",
        "        out[axis+3]+=factor*db[axis]*other;",
        "      }",
        "    }",
        "  }",
        "};",
    ]

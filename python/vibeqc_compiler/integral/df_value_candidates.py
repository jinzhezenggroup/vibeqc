"""Class-specialized FP64 three-center value candidates from existing scalar IR.

The generic production value evaluator already uses Rys quadrature. These
candidates expose a pruned Hermite/Boys alternative and a Rys moment DAG without
runtime axis-moment calls for the first useful total-angular-degree <=2 domain.
Higher classes always retain the existing exact generic evaluator.
"""

import typing
from itertools import product

from ..common.cuda_target import compute_capability_from_architecture
from .cuda import CudaEmitter
from .df_derivatives_cuda import emit_df_geometry_cuda
from .df_values import (
    build_df_axis_moment,
    build_df_component_kernel,
    build_df_value_ir,
)
from .expr import Graph, Node
from .shell_spec import cartesian_components

VALUE_CLASSES = tuple(a for a in product(range(3), repeat=3) if sum(a) <= 2)


def build_value_rys_ir(components: typing.Any) -> typing.Any:
    """Substitute root-affine Gaussian means/covariances into the shared moments."""
    graph = Graph()
    root = graph.variable("root")
    values = []
    for axis, name in enumerate("xyz"):
        powers = tuple(component.count(name) for component in components)
        source, value = build_df_axis_moment(*powers)
        sx, sy = graph.variable("sx"), graph.variable("sy")
        ip, iq = graph.variable("ip"), graph.variable("iq")
        dx = graph.variable(f"dx_{axis}")
        variables = {
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
            if node.operation == "constant":
                cloned[identifier] = graph.clone_constant(node)
            elif node.operation == "variable":
                if not isinstance(node.payload, str):
                    raise TypeError("variable node payload must be a string")
                cloned[identifier] = variables[node.payload]
            else:
                cloned[identifier] = graph._intern(
                    Node(
                        node.operation,
                        tuple(cloned[child].identifier for child in node.arguments),
                        node.payload,
                    )
                )
        values.append(cloned[value.identifier])
    return graph, values[0] * values[1] * values[2]


def emit_df_value_candidates_cuda(manifest: typing.Any = None) -> typing.Any:
    """Emit candidates against shared geometry, Boys, Rys and normalization ABIs."""
    from .df_tuning.value_manifest import VALUE_MANIFEST, load_value_manifest

    manifest_payload = load_value_manifest(manifest or VALUE_MANIFEST)
    profiles = manifest_payload["architectures"]

    def architecture_condition(architecture: str) -> str:
        major, minor = compute_capability_from_architecture(architecture)
        macro = major * 100 + minor * 10
        return f"defined(__CUDA_ARCH__) && __CUDA_ARCH__ == {macro}"

    geometry = emit_df_geometry_cuda(
        "prepare_value_rys_geometry",
        moments="""(void)total; (void)work;
  if constexpr(Roots==1) generated_df::df_rys1_roots(rho*distance,g.f);
  else generated_df::df_rys2_roots(rho*distance,g.f,1U);""",
    ).replace("__device__", "static __device__")
    lines = [
        "// Generated experimental three-center value candidates.",
        "#pragma once",
        '#include "df_values.cuh"',
        '#include "generated_df_derivatives.cuh"',
        "namespace vibeqc::scf::generated_df_derivatives {",
        "template<unsigned Roots>",
        geometry,
        "}",
        "namespace vibeqc::scf::generated_df_value_candidates {",
        "namespace scalar=generated_df_derivatives;",
        "using Vec3=generated_df::Vec3; using Angular=generated_df::Angular;",
    ]
    for index, (architecture, profile) in enumerate(sorted(profiles.items())):
        directive = "#if" if index == 0 else "#elif"
        numeric = int(architecture.removeprefix("sm_"))
        condition = (
            f"{directive} defined(VIBEQC_CUDA_PROFILE_ARCHITECTURE)"
            f" && VIBEQC_CUDA_PROFILE_ARCHITECTURE == {numeric}"
        )
        lines += [
            condition,
            f"inline constexpr unsigned candidate_raw_lanes={profile['raw_lanes']};",
        ]
    if profiles:
        lines += ["#else", "inline constexpr unsigned candidate_raw_lanes=1;", "#endif"]
    else:
        lines += ["inline constexpr unsigned candidate_raw_lanes=1;"]
    lines += [
        "template<unsigned A,unsigned B,unsigned C,bool Rys> struct Value;",
    ]
    for angular in VALUE_CLASSES:
        a, b, c = angular
        sizes = [(l + 1) * (l + 2) // 2 for l in angular]
        components = tuple(product(*(cartesian_components(l) for l in angular)))
        for rys in (False, True):
            lines += [
                f"template<> struct Value<{a},{b},{c},{str(rys).lower()}> {{",
                "static __device__ __noinline__ double evaluate(double alpha,Vec3 A,Angular a,",
                "    double beta,Vec3 B,Angular b,double gamma,Vec3 C,Angular c) {",
                "  scalar::Geometry g;",
                "  const scalar::Vec3 ca{A.x,A.y,A.z},cb{B.x,B.y,B.z},cc{C.x,C.y,C.z};",
            ]
            if rys:
                roots = sum(angular) // 2 + 1
                lines += [
                    f"  scalar::prepare_value_rys_geometry<{roots}>(alpha,ca,beta,cb,gamma,cc,{sum(angular)},g);"
                ]
            else:
                lines += [
                    f"  scalar::prepare_geometry(alpha,ca,beta,cb,gamma,cc,{sum(angular)},g);"
                ]
            lines += [
                "  const unsigned ia=(a.y+a.z)*(a.y+a.z+1)/2+a.z;",
                "  const unsigned ib=(b.y+b.z)*(b.y+b.z+1)/2+b.z;",
                "  const unsigned ic=(c.y+c.z)*(c.y+c.z+1)/2+c.z;",
                f"  switch((ia*{sizes[1]}+ib)*{sizes[2]}+ic) {{",
            ]
            for index, component in enumerate(components):
                lines.append(f"    case {index}: {{")
                variables = {
                    f"{field}_{axis}": f"g.{field}[{axis}]"
                    for field in ("pa", "pb", "dx")
                    for axis in range(3)
                }
                if rys:
                    graph, expression = build_value_rys_ir(component)
                    variables.update(
                        {field: f"g.{field}" for field in ("sx", "sy", "ip", "iq")}
                    )
                    variables["root"] = "g.f[2*root]"
                    lines += [
                        "      double result=0;",
                        f"      for(unsigned root=0;root<{roots};++root) {{",
                    ]
                else:
                    integral = build_df_value_ir("three_center_eri", angular)
                    kernel = build_df_component_kernel(integral, component)
                    graph, expression = kernel.graph, kernel.value
                    variables.update(
                        {
                            "inverse_two_p": "g.ip",
                            "inverse_two_q": "g.iq",
                            "rho": "(0.5/(g.ip+g.iq))",
                        }
                    )
                    variables.update(
                        {f"boys_{n}": f"g.f[{n}]" for n in range(sum(angular) + 1)}
                    )
                    variables.update(
                        {
                            f"difference_{name}": f"g.dx[{axis}]"
                            for axis, name in enumerate("xyz")
                        }
                    )
                    variables.update(
                        {
                            f"p{slot}_{name}": f"g.{field}[{axis}]"
                            for slot, field in enumerate(("pa", "pb"))
                            for axis, name in enumerate("xyz")
                        }
                    )
                emitter = CudaEmitter(graph, variables)
                emitter.emit((expression,))
                lines.extend(emitter.lines)
                if rys:
                    lines += [
                        f"        result+=g.f[2*root+1]*{emitter.reference(expression)};",
                        "      }",
                        "      return g.prefactor*result;",
                    ]
                else:
                    lines.append(
                        f"      return g.prefactor*{emitter.reference(expression)};"
                    )
                lines.append("    }")
            lines += ["  }", "  return NAN;", "}", "};"]
    lines += [
        "template<unsigned Math> static __device__ __forceinline__ double three_center(",
        "    double alpha,Vec3 A,Angular a,double beta,Vec3 B,Angular b,double gamma,Vec3 C,Angular c) {",
        "  if constexpr(Math!=0) {",
        "    switch(generated_df::order(a)*16+generated_df::order(b)*4+generated_df::order(c)) {",
    ]
    for a, b, c in VALUE_CLASSES:
        class_name = f"{a}{b}{c}"
        lines += [
            f"      case {16 * a + 4 * b + c}: {{",
            "        if constexpr(Math==3) {",
        ]
        for index, (architecture, profile) in enumerate(sorted(profiles.items())):
            directive = "#if" if index == 0 else "#elif"
            lowering = profile["kernels"][class_name]
            lines.append(f"{directive} {architecture_condition(architecture)}")
            if lowering == "generic":
                lines.append(
                    "          return generated_df::three_center(alpha,A,a,beta,B,b,gamma,C,c);"
                )
            else:
                rys = str(lowering == "rys").lower()
                lines.append(
                    f"          return Value<{a},{b},{c},{rys}>::evaluate(alpha,A,a,beta,B,b,gamma,C,c);"
                )
        if profiles:
            lines += [
                "#else",
                "          return generated_df::three_center(alpha,A,a,beta,B,b,gamma,C,c);",
                "#endif",
            ]
        else:
            lines.append(
                "          return generated_df::three_center(alpha,A,a,beta,B,b,gamma,C,c);"
            )
        lines += [
            "        }",
            f"        return Value<{a},{b},{c},Math==2>::evaluate(alpha,A,a,beta,B,b,gamma,C,c);",
            "      }",
        ]
    lines += [
        "    }",
        "  }",
        "  return generated_df::three_center(alpha,A,a,beta,B,b,gamma,C,c);",
        "}",
        "} // namespace vibeqc::scf::generated_df_value_candidates",
        "",
    ]
    return "\n".join(lines)

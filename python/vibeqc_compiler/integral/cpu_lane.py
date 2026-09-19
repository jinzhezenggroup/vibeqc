"""CPU scalar/SIMD lowering of first-derivative ERI component tiles.

The mathematical recurrence remains the shared ShellClassComponentKernel DAG.
This module changes only target arithmetic, primitive-record lane packing, and
source scheduling. Special functions remain strict scalar libm evaluations per
lane; recurrence arithmetic uses scalar, AVX2, or AVX-512 FP64 values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import product

from vibeqc_compiler.common.cpu_target import CpuTargetInfo
from vibeqc_compiler.common.provenance import canonical_hash

from .expr import Expr, Graph, MaterializationPlan
from .first_derivatives_native import validate_first_components
from .ir import OperatorFamily
from .ir_serialization import integral_to_payload
from .scalar_c import format_constant
from .shell_class import build_shell_class_component_kernel
from .shell_spec import cartesian_components


class CpuLaneEmitter:
    """Emit one scientific DAG over a backend-selected FP64 lane value."""

    def __init__(
        self,
        graph: Graph,
        variables: Mapping[str, str],
        materialization_plan: MaterializationPlan | None = None,
        temporary_prefix: str = "v",
    ) -> None:
        self.graph = graph
        self.variables = dict(variables)
        self.materialization_plan = materialization_plan
        self.names: dict[int, str] = {}
        self.lines: list[str] = []
        self._temporary = 0
        self._temporary_prefix = temporary_prefix
        self._materialized: set[int] = set()
        self._fma_by_add: dict[int, int] = {}

    def emit(self, roots: Sequence[Expr]) -> None:
        roots = tuple(roots)
        order = tuple(self.graph.topological_order(roots))
        if self.materialization_plan is None:
            materialized = {
                identifier
                for identifier in order
                if self.graph.nodes[identifier].operation
                not in ("constant", "variable")
            }
            emission_order = order
        else:
            if tuple(root.identifier for root in roots) != (
                self.materialization_plan.root_identifiers
            ):
                raise ValueError("materialization plan roots do not match lane roots")
            materialized = set(self.materialization_plan.materialized_identifiers)
            emission_order = self.materialization_plan.emission_order
            self._fma_by_add = dict(self.materialization_plan.fma_operations)
            for identifier in order:
                node = self.graph.nodes[identifier]
                if node.operation == "constant":
                    self.names[identifier] = self._constant(node.payload)
                elif node.operation == "variable":
                    name = str(node.payload)
                    self.names[identifier] = self.variables.get(name, name)
        self._materialized = materialized
        for identifier in emission_order:
            if identifier in self.names:
                continue
            node = self.graph.nodes[identifier]
            if node.operation == "constant":
                self.names[identifier] = self._constant(node.payload)
                continue
            if node.operation == "variable":
                name = str(node.payload)
                self.names[identifier] = self.variables.get(name, name)
                continue
            if identifier not in materialized:
                continue
            name = f"{self._temporary_prefix}{self._temporary}"
            self._temporary += 1
            self.names[identifier] = name
            self.lines.append(
                f"  const Vec {name} = {self._operation_code(identifier)};"
            )

    @staticmethod
    def _constant(payload: object) -> str:
        if not isinstance(payload, (int, float, Fraction)):
            raise TypeError("constant lane node requires a numeric payload")
        return f"v_set1({format_constant(payload)})"

    @staticmethod
    def _fold(function: str, arguments: list[str]) -> str:
        if not arguments:
            raise ValueError("lane arithmetic requires at least one operand")
        result = arguments[0]
        for argument in arguments[1:]:
            result = f"{function}({result}, {argument})"
        return result

    def _operation_code(self, identifier: int) -> str:
        node = self.graph.nodes[identifier]
        fused = self._fma_by_add.get(identifier)
        if fused is not None:
            multiply = self.graph.nodes[fused]
            remaining = list(node.arguments)
            remaining.remove(fused)
            accumulator = self._fold(
                "v_add", [self._reference(item) for item in remaining]
            )
            arguments = [self._reference(item) for item in multiply.arguments]
            return f"v_fma({arguments[0]}, {arguments[1]}, {accumulator})"
        arguments = [self._reference(item) for item in node.arguments]
        if node.operation == "add":
            return self._fold("v_add", arguments)
        if node.operation == "multiply":
            return self._fold("v_mul", arguments)
        if node.operation == "reciprocal":
            return f"v_div(v_set1(1.0), {arguments[0]})"
        if node.operation in ("exp", "log", "log1p", "expm1"):
            return f"v_{node.operation}({arguments[0]})"
        if node.operation == "power":
            if node.payload is None:
                raise ValueError("power lane node requires a numeric exponent")
            exponent = float(node.payload)
            if exponent == 0.5:
                return f"v_sqrt({arguments[0]})"
            return f"v_pow({arguments[0]}, {format_constant(exponent)})"
        raise ValueError(f"unsupported CPU lane operation {node.operation!r}")

    def _reference(self, identifier: int) -> str:
        name = self.names.get(identifier)
        if name is not None:
            return name
        node = self.graph.nodes[identifier]
        if node.operation in ("constant", "variable"):
            raise RuntimeError("CPU lane leaf was not initialized before use")
        if identifier in self._materialized:
            raise RuntimeError("CPU lane dependency was not emitted before use")
        code = self._operation_code(identifier)
        if (
            node.operation in ("exp", "power", "log", "log1p", "expm1")
            or identifier in self._fma_by_add
        ):
            return code
        return f"({code})"

    def reference(self, expression: Expr) -> str:
        if expression.graph is not self.graph:
            raise ValueError("CPU lane expression belongs to a different graph")
        return self._reference(expression.identifier)


def _lane_prelude(target: CpuTargetInfo) -> str:
    lanes = target.vector_lanes
    common = """#include <array>
#include <cmath>
#include <cstddef>
"""
    if lanes == 1:
        return (
            common
            + """using Vec = double;
static inline Vec v_load(const double* p) { return *p; }
static inline void v_store(double* p, Vec x) { *p = x; }
static inline Vec v_set1(double x) { return x; }
static inline Vec v_add(Vec a, Vec b) { return a + b; }
static inline Vec v_mul(Vec a, Vec b) { return a * b; }
static inline Vec v_div(Vec a, Vec b) { return a / b; }
static inline Vec v_sqrt(Vec x) { return std::sqrt(x); }
static inline Vec v_exp(Vec x) { return std::exp(x); }
static inline Vec v_log(Vec x) { return std::log(x); }
static inline Vec v_log1p(Vec x) { return std::log1p(x); }
static inline Vec v_expm1(Vec x) { return std::expm1(x); }
static inline Vec v_pow(Vec x, double p) { return std::pow(x, p); }
static inline Vec v_fma(Vec a, Vec b, Vec c) { return std::fma(a, b, c); }
"""
        )
    if lanes not in (4, 8):
        raise ValueError("unsupported CPU vector width")
    intrinsic = "256" if lanes == 4 else "512"
    vector_type = "__m256d" if lanes == 4 else "__m512d"
    alignment = 32 if lanes == 4 else 64
    load = f"_mm{intrinsic}_loadu_pd"
    store = f"_mm{intrinsic}_storeu_pd"
    set1 = f"_mm{intrinsic}_set1_pd"
    add = f"_mm{intrinsic}_add_pd"
    mul = f"_mm{intrinsic}_mul_pd"
    div = f"_mm{intrinsic}_div_pd"
    sqrt = f"_mm{intrinsic}_sqrt_pd"
    fma = f"_mm{intrinsic}_fmadd_pd"
    helpers = []
    for name, function in (
        ("exp", "std::exp"),
        ("log", "std::log"),
        ("log1p", "std::log1p"),
        ("expm1", "std::expm1"),
    ):
        helpers.append(
            f"""static inline Vec v_{name}(Vec x) {{
  alignas({alignment}) double values[{lanes}];
  v_store(values, x);
  for (unsigned lane = 0; lane < {lanes}; ++lane) values[lane] = {function}(values[lane]);
  return v_load(values);
}}"""
        )
    helpers.append(
        f"""static inline Vec v_pow(Vec x, double p) {{
  alignas({alignment}) double values[{lanes}];
  v_store(values, x);
  for (unsigned lane = 0; lane < {lanes}; ++lane) values[lane] = std::pow(values[lane], p);
  return v_load(values);
}}"""
    )
    return (
        common
        + "#include <immintrin.h>\n"
        + f"""using Vec = {vector_type};
static inline Vec v_load(const double* p) {{ return {load}(p); }}
static inline void v_store(double* p, Vec x) {{ {store}(p, x); }}
static inline Vec v_set1(double x) {{ return {set1}(x); }}
static inline Vec v_add(Vec a, Vec b) {{ return {add}(a, b); }}
static inline Vec v_mul(Vec a, Vec b) {{ return {mul}(a, b); }}
static inline Vec v_div(Vec a, Vec b) {{ return {div}(a, b); }}
static inline Vec v_sqrt(Vec x) {{ return {sqrt}(x); }}
static inline Vec v_fma(Vec a, Vec b, Vec c) {{ return {fma}(a, b, c); }}
"""
        + "\n".join(helpers)
        + "\n"
    )


def cpu_lane_component_identity(integral, indices, target, schedule) -> str:
    return canonical_hash(
        {
            "schema": "vibeqc.first-components.cpu-lanes.v1",
            "integral": integral_to_payload(integral),
            "components": tuple(indices),
            "target": target.to_payload(),
            "schedule": schedule.to_payload(),
        }
    )


def _optimized(graph, roots, schedule):
    optimized, roots = graph.apply_algebra_form(
        roots,
        schedule.algebra_form,
        schedule.power_lowering,
    )
    plan = optimized.materialization_plan(
        roots,
        schedule.algebra_placement.policy(),
        schedule.algebra_ordering,
        schedule.algebra_fusion,
    )
    return optimized, roots, plan


def _emit_component(integral, components, target, schedule, name):
    if integral.operator.family != OperatorFamily.FOUR_CENTER_ERI:
        raise ValueError("CPU lane lowering currently supports four-center ERIs")
    if integral.operator.range_separated:
        raise ValueError("CPU lane lowering currently supports full-range ERIs")
    if integral.recurrence != "subset_wick":
        raise ValueError("CPU lane ERIs require subset_wick recurrence")
    kernel = build_shell_class_component_kernel(
        integral.spec, components, integral=integral
    )
    roots = (kernel.value,) + tuple(
        value for axes in kernel.gradients for value in axes
    )
    names = ["alpha", "beta", "gamma", "delta"] + [
        f"{center}_{axis}"
        for center in ("first", "second", "third", "fourth")
        for axis in "xyz"
    ]
    lanes = target.vector_lanes
    variables = {field: field for field in names}
    variables["kPi"] = "v_set1(3.141592653589793238462643383279502884)"
    lines = [
        f"static inline bool {name}(const double* input, double* output) {{",
        *(
            f"  const Vec {field} = v_load(input + {index * lanes});"
            for index, field in enumerate(names)
        ),
    ]
    boys_count = integral.maximum_coulomb_order + 1
    if kernel.boys_argument is not None:
        t_graph, t_roots, t_plan = _optimized(
            kernel.graph, (kernel.boys_argument,), schedule
        )
        t_emitter = CpuLaneEmitter(t_graph, variables, t_plan, temporary_prefix="t")
        t_emitter.emit(t_roots)
        lines += t_emitter.lines
        lines += [
            f"  alignas(64) double boys_t[{lanes}];",
            f"  v_store(boys_t, {t_emitter.reference(t_roots[0])});",
            f"  alignas(64) double boys_scalar[{boys_count}][{lanes}]{{}};",
            f"  for (unsigned lane = 0; lane < {lanes}; ++lane) {{",
            "    const double t = boys_t[lane];",
            f"    if (t < {boys_count + 16}.0) {{",
            f"      for (unsigned n = 0; n < {boys_count}; ++n) {{",
            "        double term = 1.0 / (2.0*n + 1.0), sum = term;",
            "        for (unsigned k = 1; k < 512; ++k) {",
            "          term *= 2.0*t / (2.0*n + 2.0*k + 1.0);",
            "          sum += term;",
            "          if (term <= sum * 2e-16) break;",
            "        }",
            "        boys_scalar[n][lane] = std::exp(-t) * sum;",
            "      }",
            "    } else {",
            "      boys_scalar[0][lane] = 0.5 * std::sqrt(3.14159265358979323846/t) * std::erf(std::sqrt(t));",
            f"      for (unsigned n = 1; n < {boys_count}; ++n)",
            "        boys_scalar[n][lane] = ((2.0*n-1.0)*boys_scalar[n-1][lane] - std::exp(-t))/(2.0*t);",
            "    }",
            "  }",
        ]
        for index in range(boys_count):
            variable = f"boys_{index}"
            variables[variable] = variable
            lines.append(f"  const Vec {variable} = v_load(boys_scalar[{index}]);")
    graph, optimized_roots, plan = _optimized(kernel.graph, roots, schedule)
    emitter = CpuLaneEmitter(graph, variables, plan)
    emitter.emit(optimized_roots)
    lines += emitter.lines
    lines += [
        f"  v_store(output + {index * lanes}, {emitter.reference(root)});"
        for index, root in enumerate(optimized_roots)
    ]
    lines += ["  return true;", "}"]
    return "\n".join(lines)


def emit_first_components_cpu_lanes(integral, indices, target, schedule):
    """Emit one component tile for generic, AVX2, or AVX-512 CPU execution."""

    indices = tuple(indices)
    validate_first_components(integral, indices)
    schedule.validate_for(target)
    if integral.operator.family != OperatorFamily.FOUR_CENTER_ERI:
        raise ValueError("CPU lane component execution currently supports ERIs")
    component_labels = tuple(
        product(*(cartesian_components(l) for l in integral.signature.angular))
    )
    functions = [
        _emit_component(
            integral,
            component_labels[index],
            target,
            schedule,
            f"first_{packed}",
        )
        for packed, index in enumerate(indices)
    ]
    cases = "\n".join(
        f"      case {packed}: return first_{packed}(input, output);"
        for packed in range(len(indices))
    )
    ncenter = len(integral.operator.centers)
    nexponent = len(integral.signature.shells)
    identity = cpu_lane_component_identity(integral, indices, target, schedule)
    return (
        _lane_prelude(target)
        + '#include "integrals/first_component_cpu_lane_runtime.hpp"\n'
        + "\n".join(functions)
        + f"""
namespace {{
struct FirstProgram {{
  static constexpr std::size_t exponents = {nexponent};
  static constexpr std::size_t inputs = {nexponent + 3 * ncenter};
  static constexpr std::size_t outputs = {1 + 3 * ncenter};
  static constexpr std::size_t components = {len(indices)};
  static constexpr std::size_t lanes = {target.vector_lanes};
  static bool evaluate(const double* input, std::size_t component, double* output) {{
    switch (component) {{
{cases}
      default: return false;
    }}
  }}
}};
}}
extern "C" const char* vibeqc_first_cpu_lane_identity_v1() {{ return "{identity}"; }}
extern "C" const char* vibeqc_first_cpu_target_v1() {{ return "{target.name}"; }}
extern "C" int vibeqc_first_sum_cpu_lane_v1(const double* records, std::size_t count,
    std::size_t stride, double* output, std::size_t size) {{
  return vibeqc::integrals::contract_first_components_cpu_lanes<FirstProgram>(
      records, count, stride, output, size);
}}
"""
    )

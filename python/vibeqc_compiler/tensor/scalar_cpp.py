"""Small scalar FP64 C++ lowering for compiler-owned runtime kernels.

This backend is intentionally shape-zero-dimensional: it lowers an ordinary
TensorIR Program whose live values are all scalar float64 values into one
inline C++ function. Runtime loops/topology stay outside the generated
scientific expression.
"""

from __future__ import annotations

import re
import typing
from fractions import Fraction
from math import isfinite

from .ir import TRANSCENDENTALS, Node
from .program import Program

SCALAR_CPP_PRIMITIVES = frozenset(
    {
        "input",
        "constant",
        "add",
        "multiply",
        "divide",
        "scaled_bilinear",
        *TRANSCENDENTALS,
    }
)


def _literal(pair: typing.Any) -> str:
    value = float(Fraction(*pair))
    if not isfinite(value):
        raise ValueError("scalar C++ literal is not finite FP64")
    return value.hex()


def _identifier(name: str, label: str) -> str:
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_]\w*", name) is None:
        raise ValueError(f"{label} must be a C++ identifier")
    return name


def _scalar_constant(node: typing.Any) -> Fraction | None:
    if node.op != "constant":
        return None
    values = node.attrs["values"]
    return Fraction(*values[0]) if len(values) == 1 else None


def _scaled_bilinear_helper(function_name: str) -> str:
    helper = f"{function_name}_scaled_bilinear"
    return f"""inline bool {helper}(
    double a, double b, double c, double d, double e, double f,
    double& out) noexcept {{
  if (e == 0.0 || f == 0.0) return false;
  int ea, eb, ec, ed, ee, ef;
  const double ma = std::frexp(a, &ea), mb = std::frexp(b, &eb);
  const double mc = std::frexp(c, &ec), md = std::frexp(d, &ed);
  const double me = std::frexp(e, &ee), mf = std::frexp(f, &ef);
  double p = ma * mb, q = mc * md;
  double pe = std::fma(ma, mb, -p), qe = std::fma(mc, md, -q);
  const int ep = ea + eb, eq = ec + ed;
  const int exponent = p == 0.0 ? eq : (q == 0.0 ? ep : std::max(ep, eq));
  constexpr int limit = 110;
  const int dp = ep - exponent, dq = eq - exponent;
  if (dp < -limit) {{ p = 0.0; pe = 0.0; }}
  else {{ p = std::scalbn(p, dp); pe = std::scalbn(pe, dp); }}
  if (dq < -limit) {{ q = 0.0; qe = 0.0; }}
  else {{ q = std::scalbn(q, dq); qe = std::scalbn(qe, dq); }}
  const double difference = p - q;
  const double tail = difference - p;
  const double residual = (p - (difference - tail)) - (q + tail);
  const double numerator = difference + ((pe - qe) + residual);
  out = std::scalbn(numerator / (me * mf), exponent - ee - ef);
  return std::isfinite(out);
}}"""


def emit_scalar_cpp(
    program: Program,
    *,
    function_name: str,
    input_order: typing.Iterable[str] | None = None,
    output_order: typing.Iterable[str] | None = None,
    direct_scaled_bilinear: bool = False,
    check_intermediates: bool = True,
    fused_accumulation: bool = False,
) -> str:
    """Lower a scalar FP64 Program to one checked inline C++ function.

    direct_scaled_bilinear is an explicit admission for callers that have
    already proved bounded nonzero denominator domains. check_intermediates
    may likewise be disabled only for a bounded domain; inputs and published
    outputs remain finite-checked. Defaults retain the conservative behavior.
    fused_accumulation explicitly contracts single-use product addends with
    coefficient +/-1 into std::fma, retaining the addend order and finite result
    checks. It is opt-in because fused rounding is part of the caller contract.
    """

    if not isinstance(program, Program):
        raise TypeError("scalar C++ lowering requires a TensorIR Program")
    function_name = _identifier(function_name, "function_name")
    nodes = program.live_nodes
    if any(node.spec.shape != () or node.spec.dtype != "float64" for node in nodes):
        raise ValueError("scalar C++ lowering requires scalar float64 values")
    unsupported = sorted({node.op for node in nodes} - SCALAR_CPP_PRIMITIVES)
    if unsupported:
        raise ValueError(f"unsupported scalar C++ primitives: {unsupported}")

    inputs: dict[str, Node] = {}
    for node in nodes:
        if node.op == "input":
            name = _identifier(node.attrs["name"], "input name")
            inputs.setdefault(name, node)
    ordered_inputs = tuple(sorted(inputs) if input_order is None else input_order)
    if set(ordered_inputs) != set(inputs) or len(ordered_inputs) != len(inputs):
        raise ValueError("input_order must name every scalar input exactly once")

    ordered_outputs = tuple(
        sorted(program.outputs) if output_order is None else output_order
    )
    if set(ordered_outputs) != set(program.outputs) or len(ordered_outputs) != len(
        program.outputs
    ):
        raise ValueError("output_order must name every scalar output exactly once")
    for name in ordered_outputs:
        _identifier(name, "output name")

    uses: dict[Node, list[tuple[Node, int]]] = {}
    for parent in nodes:
        for slot, child in enumerate(parent.inputs):
            uses.setdefault(child, []).append((parent, slot))
    fused_products: set[Node] = set()
    if fused_accumulation:
        for node in nodes:
            consumers = uses.get(node, [])
            if (
                node.op != "multiply"
                or len(consumers) != 1
                or node in program.outputs.values()
            ):
                continue
            parent, slot = consumers[0]
            if parent.op == "add" and Fraction(*parent.attrs["coefficients"][slot]) in (
                -1,
                1,
            ):
                fused_products.add(node)

    index = {node: position for position, node in enumerate(nodes)}

    def ref(node: Node) -> str:
        return f"v{index[node]}"

    # Keep user-visible TensorIR labels out of the generated local namespace.
    input_parameters = {
        name: f"tensor_input_{i}" for i, name in enumerate(ordered_inputs)
    }
    output_parameters = {
        name: f"tensor_output_{i}" for i, name in enumerate(ordered_outputs)
    }
    parameters = [f"double {input_parameters[name]}" for name in ordered_inputs]
    parameters += [f"double& {output_parameters[name]}" for name in ordered_outputs]
    lines: list[str] = []
    if not direct_scaled_bilinear and any(
        node.op == "scaled_bilinear" for node in nodes
    ):
        lines.append(_scaled_bilinear_helper(function_name))
    lines.append(f"inline bool {function_name}({', '.join(parameters)}) noexcept {{")
    if ordered_inputs:
        condition = " || ".join(
            f"!std::isfinite({input_parameters[name]})" for name in ordered_inputs
        )
        lines.append(f"  if ({condition}) return false;")

    for node in nodes:
        if node in fused_products:
            continue
        name = ref(node)
        attrs = node.attrs
        if node.op == "input":
            lines.append(f"  const double {name} = {input_parameters[attrs['name']]};")
            continue
        if node.op == "constant":
            values = attrs["values"]
            if len(values) != 1:
                raise ValueError("scalar constant must contain exactly one value")
            lines.append(f"  const double {name} = {_literal(values[0])};")
            continue
        if node.op == "add":
            terms = list(zip(node.inputs, attrs["coefficients"], strict=True))
            if not check_intermediates:
                terms = [
                    (child, coefficient)
                    for child, coefficient in terms
                    if Fraction(*coefficient) != 0 and _scalar_constant(child) != 0
                ]
            lines.append(f"  double {name} = 0.0;")
            for child, coefficient in terms:
                if child in fused_products:
                    left, right = child.inputs
                    sign = "-" if Fraction(*coefficient) == -1 else ""
                    lines.append(
                        f"  {name} = std::fma({sign}{ref(left)}, {ref(right)}, {name});"
                    )
                    if check_intermediates:
                        lines.append(f"  if (!std::isfinite({name})) return false;")
                else:
                    lines.append(f"  {name} += {_literal(coefficient)} * {ref(child)};")
        elif node.op == "multiply":
            left, right = node.inputs
            if not check_intermediates and (
                _scalar_constant(left) == 0 or _scalar_constant(right) == 0
            ):
                lines.append(f"  const double {name} = 0.0;")
            elif not check_intermediates and _scalar_constant(left) == 1:
                lines.append(f"  const double {name} = {ref(right)};")
            elif not check_intermediates and _scalar_constant(right) == 1:
                lines.append(f"  const double {name} = {ref(left)};")
            else:
                lines.append(f"  const double {name} = {ref(left)} * {ref(right)};")
        elif node.op == "divide":
            numerator, denominator_node = node.inputs
            denominator = ref(denominator_node)
            lines.append(f"  if ({denominator} == 0.0) return false;")
            if not check_intermediates and _scalar_constant(numerator) == 0:
                lines.append(f"  const double {name} = 0.0;")
            elif not check_intermediates and _scalar_constant(denominator_node) == 1:
                lines.append(f"  const double {name} = {ref(numerator)};")
            else:
                lines.append(
                    f"  const double {name} = {ref(numerator)} / {denominator};"
                )
        elif node.op == "scaled_bilinear":
            if direct_scaled_bilinear:
                a_node, b_node, c_node, d_node, _e_node, _f_node = node.inputs
                a, b, c, d, e, f = (ref(child) for child in node.inputs)
                lines.append(f"  if ({e} == 0.0 || {f} == 0.0) return false;")
                left_zero = (
                    _scalar_constant(a_node) == 0 or _scalar_constant(b_node) == 0
                )
                right_zero = (
                    _scalar_constant(c_node) == 0 or _scalar_constant(d_node) == 0
                )
                if not check_intermediates and left_zero and right_zero:
                    expression = "0.0"
                elif not check_intermediates and left_zero:
                    expression = f"-({c} * {d}) / ({e} * {f})"
                elif not check_intermediates and right_zero:
                    expression = f"({a} * {b}) / ({e} * {f})"
                else:
                    expression = f"({a} * {b} - {c} * {d}) / ({e} * {f})"
                lines.append(f"  const double {name} = {expression};")
            else:
                args = ", ".join(ref(child) for child in node.inputs)
                lines.append(f"  double {name} = 0.0;")
                lines.append(
                    f"  if (!{function_name}_scaled_bilinear({args}, {name})) return false;"
                )
        elif node.op in TRANSCENDENTALS:
            child = ref(node.inputs[0])
            if node.op == "sqrt":
                lines.append(f"  if ({child} < 0.0) return false;")
                expression = f"std::sqrt({child})"
            elif node.op == "log":
                lines.append(f"  if (!({child} > 0.0)) return false;")
                expression = f"std::log({child})"
            elif node.op == "power":
                lines.append(f"  if (!({child} > 0.0)) return false;")
                expression = f"std::pow({child}, {_literal(attrs['exponent'])})"
            else:
                expression = f"std::exp({child})"
            lines.append(f"  const double {name} = {expression};")
        else:
            raise AssertionError(node.op)
        if check_intermediates:
            lines.append(f"  if (!std::isfinite({name})) return false;")

    for output_name in ordered_outputs:
        value = ref(program.outputs[output_name])
        lines.append(f"  if (!std::isfinite({value})) return false;")
    # Validate the complete result before modifying any caller-owned reference.
    for output_name in ordered_outputs:
        value = ref(program.outputs[output_name])
        lines.append(f"  {output_parameters[output_name]} = {value};")
    lines.append("  return true;")
    lines.append("}")
    return "\n".join(lines) + "\n"

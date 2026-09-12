"""One arithmetic specification for CG10 Python and native block capacities.

The small, fixed expression language is interpreted without eval and lowered
to checked native integer arithmetic. This keeps a native consumer from
inventing a second budget formula. Topology/identity validation stays in the
respective provider boundary.
"""

import ast

FORMULAS = (
    ("output_elements", "m0*m1*m2*m3"),
    ("tile_elements", "t0*t1*t2*t3"),
    ("coefficient_elements", "nbf*(m0+m1+m2+m3)"),
    (
        "stage_elements",
        "max(tile_elements,t1*t2*t3*m0,t2*t3*m0*m1,t3*m0*m1*m2,output_elements)",
    ),
    (
        "host_bytes",
        "reference_bytes+source_bytes+8388608+8*(4*tile_elements+2*stage_elements+3*output_elements+2*coefficient_elements)",
    ),
    (
        "aligned_numeric",
        "((8*(coefficient_elements+2*stage_elements+output_elements)+255)//256)*256",
    ),
    (
        "allocation_bytes",
        "aligned_numeric+256+4194304 if cuda and output_elements else 0",
    ),
    ("device_bytes", "allocation_bytes+100663296 if allocation_bytes else 0"),
)
LIMIT = (1 << 63) - 1


def _checked(value):
    if type(value) is not int or not 0 <= value <= LIMIT:
        raise ValueError("CG10 capacity must fit nonnegative signed 64-bit bytes")
    return value


def _evaluate(node, values):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return _checked(node.value)
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp):
        a, b = _evaluate(node.left, values), _evaluate(node.right, values)
        if isinstance(node.op, ast.Add):
            return _checked(a + b)
        if isinstance(node.op, ast.Mult):
            return _checked(a * b)
        if isinstance(node.op, ast.FloorDiv) and b:
            return _checked(a // b)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return all(_evaluate(v, values) for v in node.values)
    if isinstance(node, ast.IfExp):
        return _evaluate(
            node.body if _evaluate(node.test, values) else node.orelse, values
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "max"
    ):
        return max(_evaluate(v, values) for v in node.args)
    raise ValueError("unsupported capacity expression")


def numeric_capacity(*, nbf, reference_bytes, source_bytes, shape, tile, cuda):
    values = {
        "nbf": _checked(nbf),
        "reference_bytes": _checked(reference_bytes),
        "source_bytes": _checked(source_bytes),
        "cuda": bool(cuda),
    }
    values.update({f"m{k}": _checked(v) for k, v in enumerate(shape)})
    values.update({f"t{k}": _checked(v) for k, v in enumerate(tile)})
    for name, expression in FORMULAS:
        values[name] = _evaluate(ast.parse(expression, mode="eval").body, values)
    return {name: values[name] for name, _ in FORMULAS}


def _cpp(node):
    if isinstance(node, ast.Constant):
        return f"{_checked(node.value)}ULL"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BinOp):
        a, b = _cpp(node.left), _cpp(node.right)
        if isinstance(node.op, ast.Add):
            return f"checked_add({a},{b})"
        if isinstance(node.op, ast.Mult):
            return f"checked_mul({a},{b})"
        if isinstance(node.op, ast.FloorDiv):
            return f"({a}/{b})"
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return "(" + " && ".join(_cpp(v) for v in node.values) + ")"
    if isinstance(node, ast.IfExp):
        return f"({_cpp(node.test)} ? {_cpp(node.body)} : {_cpp(node.orelse)})"
    if isinstance(node, ast.Call) and node.func.id == "max":
        return "std::max<std::size_t>({" + ",".join(_cpp(v) for v in node.args) + "})"
    raise ValueError("unsupported native capacity expression")


def native_header():
    names = [name for name, _ in FORMULAS]
    lines = [
        "// Generated from tools/vibeqc_posthf/plan_spec.py; do not change formulas here.",
        "#pragma once",
        "#include <algorithm>",
        "#include <array>",
        '#include "posthf/capacity.hpp"',
        "namespace vibeqc::posthf {",
        "struct NumericBlockPlan { "
        + " ".join(f"std::size_t {n};" for n in names)
        + " };",
        "inline NumericBlockPlan numeric_block_plan(std::size_t nbf, std::size_t reference_bytes, std::size_t source_bytes, const std::array<std::size_t,4>& shape, const std::array<std::size_t,4>& tile, bool cuda) {",
        "const auto m0=shape[0],m1=shape[1],m2=shape[2],m3=shape[3];",
        "const auto t0=tile[0],t1=tile[1],t2=tile[2],t3=tile[3];",
    ]
    for name, expression in FORMULAS:
        lines.append(
            f"const std::size_t {name} = {_cpp(ast.parse(expression, mode='eval').body)};"
        )
    lines.append("return {" + ",".join(names) + "};\n}\n}")
    return "\n".join(lines) + "\n"

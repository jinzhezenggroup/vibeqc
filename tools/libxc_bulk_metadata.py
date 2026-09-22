"""Fail-closed extraction of pinned Libxc C registration/default parameters.

This is not a C interpreter. Only homogeneous double records populated by the
upstream ``set_ext_params_cpy`` setter are admitted. Parameter *layout*, not the
human-readable external parameter names, determines Maple binding names.
"""

from __future__ import annotations

import ast
import math
import re
import sys
from typing import Any

C_BINDING_SEMANTICS = "libxc-c-copy-parameters/v1"
_IDENTIFIER = r"[A-Za-z_]\w*"


class CMetadataError(ValueError):
    """A registration needs semantics outside the audited extraction subset."""


def strip_c_comments(text: str) -> str:
    """Remove comments while preserving string literals and line boundaries."""
    return re.sub(
        r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*',
        lambda match: (
            match[0] if match[0].startswith('"') else " " + "\n" * match[0].count("\n")
        ),
        text,
        flags=re.DOTALL,
    )


def split_fields(text: str) -> list[str]:
    """Split a C initializer at top-level commas; reject malformed nesting."""
    fields: list[str] = []
    stack: list[str] = []
    quote = escape = False
    start = 0
    for index, char in enumerate(text):
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                quote = False
        elif char == '"':
            quote = True
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack.pop() != dict(zip(")]}", "([{"))[char]:
                raise CMetadataError("unbalanced C initializer")
        elif char == "," and not stack:
            field = text[start:index].strip()
            if not field:
                raise CMetadataError("empty C initializer field")
            fields.append(field)
            start = index + 1
    if quote or stack:
        raise CMetadataError("unterminated C initializer")
    if text[start:].strip():
        fields.append(text[start:].strip())
    return fields


def brace_body(text: str, start: int) -> tuple[str, int]:
    """Read a balanced brace group, returning its body and following offset."""
    if start >= len(text) or text[start] != "{":
        raise CMetadataError("expected C initializer brace")
    depth = 0
    quote = escape = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                quote = False
        elif char == '"':
            quote = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if not depth:
                return text[start + 1 : index], index + 1
    raise CMetadataError("unterminated C brace group")


def constant_definitions(*sources: str) -> dict[str, str | None]:
    """Keep object macros only; conflicting definitions remain unresolved."""
    definitions: dict[str, str | None] = {}
    for raw_source in sources:
        source = raw_source.replace("\\\n", "")
        for name, raw_expression in re.findall(
            rf"^\s*#\s*define[ \t]+({_IDENTIFIER})[ \t]+([^\n]+)",
            source,
            flags=re.MULTILINE,
        ):
            expression = raw_expression.strip()
            if name in definitions and definitions[name] != expression:
                definitions[name] = None
            else:
                definitions[name] = expression
    return definitions


def constant_value(
    expression: str,
    definitions: dict[str, str | None],
    seen: tuple[str, ...] = (),
) -> int | float:
    """Evaluate bounded numeric C constants without eval or executing C.

    Integer division retains C truncation semantics. Numeric suffixes are
    removed only from decimal tokens, never from hexadecimal digits or names.
    """
    if len(expression) > 4096 or len(seen) > 32:
        raise CMetadataError("C constant exceeds the bounded expression limit")
    expression = re.sub(
        r"(?<![\w.])((?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)[lLfF]\b",
        r"\1",
        expression.strip(),
    )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise CMetadataError(f"unsupported C constant: {expression}") from error
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise CMetadataError("C constant exceeds the bounded AST limit")

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            value = definitions.get(node.id)
            if node.id in seen or value is None:
                raise CMetadataError(
                    f"unknown, ambiguous or cyclic C constant: {node.id}"
                )
            return constant_value(value, definitions, (*seen, node.id))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if isinstance(left, int) and isinstance(right, int):
                    sign = -1 if (left < 0) != (right < 0) else 1
                    return sign * (abs(left) // abs(right))
                return left / right
        raise CMetadataError(f"unsupported C constant: {expression}")

    try:
        result = visit(tree.body)
        if not math.isfinite(result):
            raise CMetadataError("nonfinite C parameter")
    except (OverflowError, ZeroDivisionError) as error:
        raise CMetadataError("nonfinite or singular C parameter") from error
    return result


def array_values(text: str, name: str) -> list[str]:
    if not re.fullmatch(_IDENTIFIER, name):
        raise CMetadataError("parameter table is not a named array")
    matches = list(
        re.finditer(rf"\b{re.escape(name)}\s*(?:\[[^\]]*\])+\s*=\s*\{{", text)
    )
    if len(matches) != 1:
        raise CMetadataError(f"missing or ambiguous parameter array: {name}")
    body, _ = brace_body(text, matches[0].end() - 1)
    return split_fields(body)


def parameter_layout(
    text: str, type_name: str, definitions: dict[str, str | None]
) -> list[tuple[str, int]]:
    """Extract scalar/double-array layout; reject mixed types and partial arrays."""
    layouts: list[list[tuple[str, int]]] = []
    for match in re.finditer(r"\btypedef\s+struct\s*\{", text):
        body, end = brace_body(text, match.end() - 1)
        declaration = re.match(rf"\s*({_IDENTIFIER})\s*;", text[end:])
        if declaration is None or declaration[1] != type_name:
            continue
        fields: list[tuple[str, int]] = []
        for raw_statement in body.split(";"):
            statement = raw_statement.strip()
            if not statement:
                continue
            if not re.match(r"double\s", statement):
                raise CMetadataError("non-double parameter layout")
            for field in split_fields(re.sub(r"^double\s+", "", statement)):
                entry = re.fullmatch(rf"({_IDENTIFIER})\s*((?:\[[^\]]+\])*)", field)
                if entry is None:
                    raise CMetadataError("unsupported parameter field declaration")
                dimensions = [
                    constant_value(size, definitions)
                    for size in re.findall(r"\[([^\]]+)\]", entry[2])
                ]
                if len(dimensions) > 1 or any(
                    type(size) is not int or not 1 <= size <= 32 for size in dimensions
                ):
                    raise CMetadataError("parameter array exceeds supported dimensions")
                fields.append((entry[1], dimensions[0] if dimensions else 0))
        if len({name for name, _ in fields}) != len(fields):
            raise CMetadataError("duplicate parameter field")
        layouts.append(fields)
    if len(layouts) != 1:
        raise CMetadataError(f"missing or ambiguous parameter struct: {type_name}")
    return layouts[0]


def extract_registrations(
    c_source: str, maple_source: str, header_source: str
) -> list[dict[str, Any]]:
    """Extract each registration independently, retaining every rejection."""
    text, header = strip_c_comments(c_source), strip_c_comments(header_source)
    definitions = constant_definitions(header, text)
    prefixes = set(re.findall(rf"\b({_IDENTIFIER})\s*\*\s*params\s*;", maple_source))
    records: list[dict[str, Any]] = []
    for match in re.finditer(
        rf"\bconst\s+xc_func_info_type\s+xc_func_info_({_IDENTIFIER})\s*=\s*\{{", text
    ):
        record: dict[str, Any] = {
            "name": match[1].upper(),
            "binding_semantics": C_BINDING_SEMANTICS,
        }
        try:
            body, _ = brace_body(text, match.end() - 1)
            fields = split_fields(body)
            if len(fields) != 13:
                raise CMetadataError("unsupported xc_func_info_type layout")
            record["id"] = constant_value(fields[0], definitions)
            if type(record["id"]) is not int or record["id"] <= 0:
                raise CMetadataError("functional ID must be a positive integer")
            if fields[3] not in ("XC_FAMILY_LDA", "XC_FAMILY_GGA", "XC_FAMILY_MGGA"):
                raise CMetadataError(
                    "non-semilocal family requires MethodIR composition"
                )
            # Match flag tokens rather than similarly named substrings.
            if "XC_FLAGS_3D" not in re.split(r"\s*\|\s*", fields[5]):
                raise CMetadataError("non-3D functional")
            if fields[1] not in (
                "XC_EXCHANGE",
                "XC_CORRELATION",
                "XC_EXCHANGE_CORRELATION",
            ):
                raise CMetadataError("non-XC functional (e.g. kinetic)")
            family = fields[3].removeprefix("XC_FAMILY_").lower()
            record.update(
                family=family, kind=fields[1].removeprefix("XC_"), flags=fields[5]
            )
            workers = ["NULL", "NULL", "NULL"]
            workers[("lda", "gga", "mgga").index(family)] = f"&work_{family}"
            if fields[10:] != workers:
                raise CMetadataError("custom or absent Libxc worker")
            if not fields[7].startswith("{") or not fields[7].endswith("}"):
                raise CMetadataError(
                    "external parameter contract is not an initializer"
                )
            params = split_fields(fields[7][1:-1])
            if len(params) != 5:
                raise CMetadataError("unsupported external parameter contract")
            count = constant_value(params[0], definitions)
            if type(count) is not int or not 0 <= count <= 256:
                raise CMetadataError("external parameter count exceeds supported bound")
            threshold = float(constant_value(fields[6], definitions))
            if threshold < 0:
                raise CMetadataError("negative density threshold")
            bindings: dict[str, Any] = {
                "p_a_dens_threshold": repr(threshold),
                "p_a_zeta_threshold": repr(sys.float_info.epsilon),
            }
            record["parameter_setter"] = params[4]
            if count:
                if params[4] != "set_ext_params_cpy":
                    raise CMetadataError(f"custom parameter setter: {params[4]}")
                if len(prefixes) != 1:
                    raise CMetadataError("missing or ambiguous Maple parameter prefix")
                names = array_values(text, params[1])
                values = [
                    constant_value(item, definitions)
                    for item in array_values(text, params[3])
                ]
                if len(names) != count or len(values) != count:
                    raise CMetadataError(
                        "parameter array length does not match registration"
                    )
                layout = parameter_layout(
                    text + "\n" + header, next(iter(prefixes)), definitions
                )
                offset = 0
                for field, size in layout:
                    if offset == count:
                        break
                    width = size or 1
                    if offset + width > count:
                        raise CMetadataError("partial parameter array copy")
                    packed = [
                        repr(float(value)) for value in values[offset : offset + width]
                    ]
                    bindings[f"params_a_{field}"] = packed if size else packed[0]
                    if size:
                        # Libxc Maple supports both array and flattened C-field spellings.
                        for index, value in enumerate(packed):
                            bindings[f"params_a_{field}_{index}_"] = value
                    offset += width
                if offset != count:
                    raise CMetadataError("parameter struct is smaller than copy count")
            elif any(value != "NULL" for value in params[1:]):
                raise CMetadataError("nonempty zero-parameter contract")
            record.update(bindings=bindings, metadata_status="bound")
        except CMetadataError as error:
            record.update(metadata_status="blocked", reason=str(error))
        records.append(record)
    return records

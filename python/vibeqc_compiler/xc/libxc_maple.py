# Copyright (C) 2026 VibeQC contributors
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. See external/libxc-7.0.0/COPYING or https://mozilla.org/MPL/2.0/.
"""Fail-closed Libxc Maple importer for compiler-side XC experiments.

This module intentionally implements only a small expression subset. It is a
front-end into the existing scalar Graph, not a second algebra system. The
first qualified slice imports gga_x_pbe.mpl on VibeQC's audited
libxc-7.0.0/interior-v1 domain, where Libxc density/zeta screening branches
are owned by the caller's domain contract.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from vibeqc_compiler.integral.expr import Expr, Graph


class MapleImportError(ValueError):
    """The pinned Maple source uses syntax outside the qualified importer."""


_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
_RESERVED = frozenset(
    ("Pi", "X2S", "gga_exchange", "sqrt", "exp", "log", "log1p", "expm1")
)


@dataclass(frozen=True, slots=True)
class MapleFunction:
    """One parsed arrow-function definition."""

    parameters: tuple[str, ...]
    expression: str


@dataclass(frozen=True, slots=True)
class _FunctionRef:
    name: str


@dataclass(frozen=True, slots=True)
class _IntrinsicRef:
    name: str


@dataclass(frozen=True, slots=True)
class MapleModule:
    """Immutable parsed Maple definitions with a pinned source identity."""

    assignments: tuple[tuple[str, str], ...]
    functions: tuple[tuple[str, MapleFunction], ...]
    source_sha256: str

    @property
    def assignment_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.assignments)

    @property
    def function_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.functions)

    def call(
        self,
        graph: Graph,
        name: str,
        *arguments: Expr | float | Fraction,
    ) -> Expr:
        """Lower one parsed Maple function call into the canonical scalar Graph."""

        evaluator = _Evaluator(self, graph)
        return evaluator.call(name, arguments)


def _strip_comments(source: str) -> str:
    """Preserve line boundaries while removing Maple line/nested block comments."""
    output: list[str] = []
    offset = 0
    depth = 0
    quote: str | None = None
    while offset < len(source):
        char = source[offset]
        pair = source[offset : offset + 2]
        if depth:
            if pair in ("(*", "*)"):
                depth += 1 if pair == "(*" else -1
                offset += 2
                continue
            if char == "\n":
                output.append(char)
        elif quote is not None:
            output.append(char)
            if char == "\\" and offset + 1 < len(source):
                offset += 1
                output.append(source[offset])
            elif char == quote:
                quote = None
        elif pair == "(*":
            output.append(" ")
            depth = 1
            offset += 2
            continue
        elif pair == "*)":
            raise MapleImportError("unmatched block comment terminator")
        elif char == "#":
            while True:
                end = source.find("\n", offset)
                if end < 0:
                    offset = len(source)
                    break
                line = source[offset:end]
                backslashes = len(line) - len(line.rstrip("\\"))
                output.append("\n")
                offset = end + 1
                if backslashes % 2 == 0:
                    break
            continue
        else:
            output.append(char)
            if char in ("'", '"', "`"):
                quote = char
        offset += 1
    if depth:
        raise MapleImportError("unterminated block comment")
    return "".join(output)


def _preprocess(source: str, defines: frozenset[str]) -> str:
    """Select a deterministic subset of Libxc's simple conditional directives."""

    source = _strip_comments(source)
    active = True
    stack: list[tuple[bool, bool, bool]] = []
    output: list[str] = []
    for raw_line in source.splitlines():
        line = raw_line.strip()

        if not line:
            if active:
                output.append(raw_line)
            continue
        if line.startswith("$ifdef "):
            symbol = line.split(None, 1)[1].strip()
            matched = symbol in defines
            stack.append((active, matched, False))
            active = active and matched
            continue
        if line.startswith("$elif "):
            if not stack:
                raise MapleImportError("$elif without $ifdef")
            parent, matched, seen_else = stack[-1]
            if seen_else:
                raise MapleImportError("$elif after $else")
            symbol = line.split(None, 1)[1].strip()
            branch = (not matched) and symbol in defines
            stack[-1] = (parent, matched or branch, False)
            active = parent and branch
            continue
        if line == "$else":
            if not stack:
                raise MapleImportError("$else without $ifdef")
            parent, matched, seen_else = stack[-1]
            if seen_else:
                raise MapleImportError("$else after $else")
            stack[-1] = (parent, True, True)
            active = parent and not matched
            continue

        if line == "$endif":
            if not stack:
                raise MapleImportError("$endif without $ifdef")
            parent, _, _ = stack.pop()
            active = parent
            continue
        if line.startswith("$"):
            if active:
                raise MapleImportError(f"unsupported Maple directive {line!r}")
            continue
        if active:
            output.append(raw_line)
    if stack:
        raise MapleImportError("unterminated $ifdef block")
    return "\n".join(output)


def import_maple_source(source: str, *, defines: Iterable[str] = ()) -> MapleModule:
    """Parse the qualified assignment/arrow-function subset used by PBE-X."""

    selected = _preprocess(source, frozenset(defines))
    assignments: dict[str, str] = {}
    functions: dict[str, MapleFunction] = {}
    for raw_statement in re.split(r":(?![=])", selected):
        statement = " ".join(raw_statement.split())
        if not statement:
            continue

        if ":=" not in statement:
            raise MapleImportError(f"unsupported Maple statement {statement!r}")
        name, right = (part.strip() for part in statement.split(":=", 1))
        if not _IDENTIFIER.fullmatch(name):
            raise MapleImportError(f"unsupported Maple definition name {name!r}")
        if name in _RESERVED:
            raise MapleImportError(
                f"reserved Maple intrinsic cannot be redefined: {name!r}"
            )
        if name in assignments or name in functions:
            raise MapleImportError(f"duplicate Maple definition {name!r}")
        if "->" not in right:
            _parse_expression(right)
            assignments[name] = right
            continue
        parameter_text, expression = (part.strip() for part in right.split("->", 1))
        if parameter_text.startswith("(") and parameter_text.endswith(")"):
            parameter_text = parameter_text[1:-1]
        parameters = tuple(item.strip() for item in parameter_text.split(","))
        if (
            not parameters
            or len(set(parameters)) != len(parameters)
            or any(not _IDENTIFIER.fullmatch(item) for item in parameters)
        ):
            raise MapleImportError(f"unsupported parameter list {parameter_text!r}")
        _parse_expression(expression)
        functions[name] = MapleFunction(parameters, expression)
    return MapleModule(
        tuple(assignments.items()),
        tuple(functions.items()),
        hashlib.sha256(source.encode()).hexdigest(),
    )


_ALLOWED_AST = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.UAdd,
    ast.USub,
    ast.Load,
)


@cache
def _parse_expression(expression: str) -> ast.Expression:
    translated = expression.replace("^", "**")
    try:
        tree = ast.parse(translated, mode="eval")
    except SyntaxError as exc:
        raise MapleImportError(f"unsupported Maple expression {expression!r}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            raise MapleImportError(
                f"unsupported Maple expression node {type(node).__name__}"
            )
        if isinstance(node, ast.Call) and (
            node.keywords or not isinstance(node.func, ast.Name)
        ):
            raise MapleImportError(
                "only positional calls to named Maple functions are supported"
            )
    return tree


class _Evaluator:
    """Lower parsed expressions directly into the canonical scalar DAG."""

    def __init__(self, module: MapleModule, graph: Graph) -> None:
        self.module = module
        self.graph = graph
        self.assignments = dict(module.assignments)
        self.functions = dict(module.functions)
        self._assignment_cache: dict[str, Expr] = {}
        self._assignment_stack: set[str] = set()

    def call(
        self,
        name: str,
        arguments: Sequence[Expr | float | Fraction],
    ) -> Expr:
        function = self.functions.get(name)
        if function is None:
            raise MapleImportError(f"unknown Maple function {name!r}")
        if len(arguments) != len(function.parameters):
            raise MapleImportError(
                f"{name} expects {len(function.parameters)} arguments, got "
                f"{len(arguments)}"
            )
        environment = dict(zip(function.parameters, arguments, strict=True))
        value = self._eval(_parse_expression(function.expression).body, environment)
        return self._as_expr(value)

    def _as_expr(self, value: object) -> Expr:
        from vibeqc_compiler.integral.expr import Expr

        if isinstance(value, Expr):
            return value
        if isinstance(value, Fraction | int):
            return self.graph.constant(value)
        if isinstance(value, float):
            return self.graph.approximate_constant(value)
        raise MapleImportError(
            f"expected scalar expression, got {type(value).__name__}"
        )

    def _name(self, name: str, environment: Mapping[str, object]) -> object:
        if name in environment:
            return environment[name]
        if name == "Pi":
            return self.graph.approximate_constant(math.pi)
        if name == "X2S":
            return self.graph.approximate_constant(
                1.0 / (2.0 * (6.0 * math.pi**2) ** (1.0 / 3.0))
            )
        if name in self.functions:
            return _FunctionRef(name)
        if name in ("gga_exchange", "sqrt", "exp", "log", "log1p", "expm1"):
            return _IntrinsicRef(name)
        if name not in self.assignments:
            raise MapleImportError(f"unknown Maple symbol {name!r}")

        cached = self._assignment_cache.get(name)
        if cached is not None:
            return cached
        if name in self._assignment_stack:
            raise MapleImportError(f"cyclic Maple assignment involving {name!r}")
        self._assignment_stack.add(name)
        try:
            value = self._as_expr(
                self._eval(_parse_expression(self.assignments[name]).body, {})
            )
        finally:
            self._assignment_stack.remove(name)
        self._assignment_cache[name] = value
        return value

    def _eval(self, node: ast.AST, environment: Mapping[str, object]) -> object:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return Fraction(str(node.value))
        if isinstance(node, ast.Name):
            return self._name(node.id, environment)
        if isinstance(node, ast.UnaryOp):
            value = self._as_expr(self._eval(node.operand, environment))
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
            raise MapleImportError("unsupported unary operator")

        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, environment)
            right = self._eval(node.right, environment)
            if isinstance(node.op, ast.Pow):
                base = self._as_expr(left)
                if not isinstance(right, Fraction | int | float):
                    raise MapleImportError(
                        "Maple exponents must be compile-time scalars"
                    )
                return base.pow(float(right))
            left_expr = self._as_expr(left)
            right_expr = self._as_expr(right)
            if isinstance(node.op, ast.Add):
                return left_expr + right_expr
            if isinstance(node.op, ast.Sub):
                return left_expr - right_expr
            if isinstance(node.op, ast.Mult):
                return left_expr * right_expr
            if isinstance(node.op, ast.Div):
                return left_expr / right_expr
            raise MapleImportError("unsupported binary operator")
        if isinstance(node, ast.Call):
            target = self._eval(node.func, environment)
            arguments = tuple(self._eval(item, environment) for item in node.args)
            if isinstance(target, _FunctionRef):
                scalar_arguments = tuple(self._as_expr(value) for value in arguments)
                return self.call(target.name, scalar_arguments)
            if isinstance(target, _IntrinsicRef):
                return self._intrinsic(target.name, arguments)
            raise MapleImportError("Maple call target is not callable")
        raise MapleImportError(f"unsupported Maple node {type(node).__name__}")

    def _intrinsic(self, name: str, arguments: Sequence[object]) -> Expr:
        if name == "gga_exchange":
            if len(arguments) != 5 or not isinstance(arguments[0], _FunctionRef):
                raise MapleImportError(
                    "gga_exchange requires a function and four scalars"
                )
            function = arguments[0]
            if not isinstance(function, _FunctionRef):
                raise MapleImportError(
                    "gga_exchange requires a function and four scalars"
                )
            rs, z, xs0, xs1 = arguments[1:]
            rs_expr, z_expr = self._as_expr(rs), self._as_expr(z)
            term0 = self._lda_x_spin(rs_expr, z_expr) * self.call(
                function.name, (self._as_expr(xs0),)
            )
            term1 = self._lda_x_spin(rs_expr, -z_expr) * self.call(
                function.name, (self._as_expr(xs1),)
            )
            return term0 + term1
        if len(arguments) != 1:
            raise MapleImportError(f"{name} requires one scalar argument")
        value = self._as_expr(arguments[0])
        if name == "sqrt":
            return value.pow(0.5)
        if name == "exp":
            return self.graph.exponential(value)
        if name in ("log", "log1p", "expm1"):
            return self.graph.stable_unary(name, value)
        raise MapleImportError(f"unknown Maple intrinsic {name!r}")

    def _lda_x_spin(self, rs: Expr, z: Expr) -> Expr:
        """Interior-domain form of Libxc util.mpl lda_x_spin."""

        rs_factor = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        x_factor = 3.0 / 8.0 * (3.0 / math.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)
        coefficient = self.graph.approximate_constant(
            -x_factor * 2.0 ** (-4.0 / 3.0) * rs_factor
        )
        return coefficient * (1 + z).pow(4.0 / 3.0) / rs

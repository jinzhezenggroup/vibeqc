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
import json
import math
import re
import typing
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from vibeqc_compiler.integral.expr import Expr, Graph


class MapleImportError(ValueError):
    """The pinned Maple source uses syntax outside the qualified importer."""


IMPORTER_SEMANTICS = "libxc-maple-graph/v3"
_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
_RESERVED = frozenset(
    (
        "Pi",
        "X2S",
        "gga_exchange",
        "f_zeta",
        "mphi",
        "tt",
        "sqrt",
        "exp",
        "log",
        "log1p",
        "expm1",
    )
)
_DEFINE = re.compile(r"^\$define\s+([A-Za-z_]\w*)\s*$")
_INCLUDE = re.compile(r'^\$include\s+"([^"]+)"\s*$')


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
    source_hashes: tuple[tuple[str, str], ...] = ()
    include_edges: tuple[tuple[str, str], ...] = ()
    defines: tuple[str, ...] = ()
    initial_defines: tuple[str, ...] = ()

    @property
    def transitive_sha256(self) -> str:
        """Hash the selected source graph and active preprocessor symbols."""

        payload = json.dumps(
            {
                "entry_sha256": self.source_sha256,
                "defines": self.defines,
                "initial_defines": self.initial_defines,
                "importer_semantics": IMPORTER_SEMANTICS,
                "include_edges": self.include_edges,
                "sources": self.source_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

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


def _preprocess(
    source: str,
    defines: set[str],
    *,
    include_loader: typing.Callable[[str], str] | None = None,
) -> str:
    """Select qualified directives and optionally expand pinned includes."""

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
        define = _DEFINE.fullmatch(line)
        if define:
            if active:
                defines.add(define.group(1))
            continue
        include = _INCLUDE.fullmatch(line)
        if include:
            if active:
                if include_loader is None:
                    raise MapleImportError(f"unsupported Maple directive {line!r}")
                output.append(include_loader(include.group(1)))
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


def _reject_captured_redefinition(
    name: str,
    assignments: Mapping[str, str],
    functions: Mapping[str, MapleFunction],
) -> None:
    """Reject overrides that require unsupported assignment-time value capture."""

    def references(expression: str) -> set[str]:
        return {
            node.id
            for node in ast.walk(_parse_expression(expression))
            if isinstance(node, ast.Name)
        }

    for captured, expression in assignments.items():
        if captured == name:
            continue
        pending = references(expression)
        visited: set[str] = set()
        while pending:
            dependency = pending.pop()
            if dependency == name:
                raise MapleImportError(
                    f"redefinition of {name!r} would change captured assignment "
                    f"{captured!r}; assignment-time rebinding is not supported"
                )
            if dependency in visited:
                continue
            visited.add(dependency)
            if dependency in assignments:
                pending.update(references(assignments[dependency]))
            elif dependency in functions:
                function = functions[dependency]
                pending.update(
                    references(function.expression) - set(function.parameters)
                )


def _parse_selected_source(
    selected: str,
    *,
    source_sha256: str,
    source_hashes: tuple[tuple[str, str], ...],
    include_edges: tuple[tuple[str, str], ...],
    defines: tuple[str, ...],
    initial_defines: tuple[str, ...],
    allow_redefinition: bool,
) -> MapleModule:
    """Parse one selected source stream into final Maple definitions."""

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
        is_function = "->" in right
        if name in assignments or name in functions:
            same_kind = (is_function and name in functions) or (
                not is_function and name in assignments
            )
            if not allow_redefinition or not same_kind:
                raise MapleImportError(f"duplicate Maple definition {name!r}")
            _reject_captured_redefinition(name, assignments, functions)
        if not is_function:
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
        assignments=tuple(assignments.items()),
        functions=tuple(functions.items()),
        source_sha256=source_sha256,
        source_hashes=source_hashes,
        include_edges=include_edges,
        defines=defines,
        initial_defines=initial_defines,
    )


def import_maple_source(source: str, *, defines: Iterable[str] = ()) -> MapleModule:
    """Parse one in-memory source without permitting includes or redefinition."""

    initial_defines = tuple(sorted(set(defines)))
    selected_defines = set(initial_defines)
    selected = _preprocess(source, selected_defines)
    sha256 = hashlib.sha256(source.encode()).hexdigest()
    return _parse_selected_source(
        selected,
        source_sha256=sha256,
        source_hashes=(("<memory>", sha256),),
        include_edges=(),
        defines=tuple(sorted(selected_defines)),
        initial_defines=initial_defines,
        allow_redefinition=False,
    )


def import_maple_file(
    root: Path,
    entry: str,
    *,
    defines: Iterable[str] = (),
    support_files: Iterable[str] = (),
) -> MapleModule:
    """Load one bounded include graph rooted inside a pinned source directory."""

    source_root = Path(root).resolve()
    initial_defines = tuple(sorted(set(defines)))
    selected_defines = set(initial_defines)
    visiting: list[Path] = []
    loaded: set[Path] = set()
    hashes: dict[str, str] = {}
    include_edges: list[tuple[str, str]] = []

    def resolve(candidate: Path) -> tuple[Path, str]:
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(source_root).as_posix()
        except (OSError, ValueError) as error:
            raise MapleImportError(
                f"Maple source escapes or is absent from pinned root: {candidate}"
            ) from error
        if not resolved.is_file():
            raise MapleImportError(f"Maple source is not a regular file: {relative}")
        return resolved, relative

    def read_source(resolved: Path, relative: str) -> str:
        raw = resolved.read_bytes()
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MapleImportError(f"Maple source is not UTF-8: {relative}") from error
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        return source

    def load(candidate: Path, *, parent: str | None = None) -> str:
        resolved, relative = resolve(candidate)
        if resolved in visiting:
            names = [
                *(item.relative_to(source_root).as_posix() for item in visiting),
                relative,
            ]
            raise MapleImportError(f"cyclic Maple include: {' -> '.join(names)}")
        if resolved in loaded:
            raise MapleImportError(f"duplicate Maple include {relative!r}")
        if parent is not None:
            include_edges.append((parent, relative))
        visiting.append(resolved)
        loaded.add(resolved)
        source = read_source(resolved, relative)
        try:
            return _preprocess(
                source,
                selected_defines,
                include_loader=lambda name: load(
                    resolved.parent / name, parent=relative
                ),
            )
        finally:
            visiting.pop()

    entry_path, entry_relative = resolve(source_root / entry)
    selected = load(entry_path)
    for support in support_files:
        support_path, support_relative = resolve(source_root / support)
        if support_path not in loaded:
            read_source(support_path, support_relative)
    source_hashes = tuple(sorted(hashes.items()))
    return _parse_selected_source(
        selected,
        source_sha256=hashes[entry_relative],
        source_hashes=source_hashes,
        include_edges=tuple(include_edges),
        defines=tuple(sorted(selected_defines)),
        initial_defines=initial_defines,
        allow_redefinition=True,
    )


_ALLOWED_AST = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Subscript,
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
        self._assignment_cache: dict[str, object] = {}
        self._assignment_stack: set[str] = set()

    def call(
        self,
        name: str,
        arguments: Sequence[object],
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
        if name in (
            "gga_exchange",
            "f_zeta",
            "mphi",
            "tt",
            "sqrt",
            "exp",
            "log",
            "log1p",
            "expm1",
        ):
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
            value = self._eval(_parse_expression(self.assignments[name]).body, {})
        finally:
            self._assignment_stack.remove(name)
        self._assignment_cache[name] = value
        return value

    def _even_power(self, value: Expr, exponent: int) -> Expr:
        """Lower the pinned even-power forms before symbolic differentiation."""

        from vibeqc_compiler.integral.expr import Expr

        if exponent not in (2, 4):
            raise MapleImportError("qualified even-power lowering supports 2 and 4")
        if exponent == 4:
            squared = self._even_power(value, 2)
            return self._even_power(squared, 2)
        node = self.graph.node(value)
        if node.operation == "multiply":
            return self.graph.multiply_many(
                self._even_power(Expr(self.graph, identifier), 2)
                for identifier in node.arguments
            )
        if node.operation == "reciprocal":
            child = Expr(self.graph, node.arguments[0])
            return self.graph.reciprocal(self._even_power(child, 2))
        if (
            node.operation == "power"
            and isinstance(node.payload, float)
            and node.payload == 0.5
        ):
            return Expr(self.graph, node.arguments[0])
        return value.pow(2.0)

    @staticmethod
    def _literal_one(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and node.value == 1
        )

    @classmethod
    def _one_plus_operand(cls, node: ast.AST) -> ast.AST | None:
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
            return None
        if cls._literal_one(node.left):
            return node.right
        if cls._literal_one(node.right):
            return node.left
        return None

    def _eval(self, node: ast.AST, environment: Mapping[str, object]) -> object:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return Fraction(str(node.value))
        if isinstance(node, ast.List):
            return tuple(self._eval(item, environment) for item in node.elts)
        if isinstance(node, ast.Subscript):
            values = self._eval(node.value, environment)
            index = self._eval(node.slice, environment)
            if (
                not isinstance(values, tuple)
                or not isinstance(index, Fraction)
                or index.denominator != 1
            ):
                raise MapleImportError(
                    "Maple indexing requires a constant one-based list index"
                )
            offset = int(index) - 1
            if not 0 <= offset < len(values):
                raise MapleImportError("Maple list index is out of bounds")
            return values[offset]
        if isinstance(node, ast.Name):
            return self._name(node.id, environment)
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand, environment)
            if isinstance(value, Fraction):
                if isinstance(node.op, ast.UAdd):
                    return value
                if isinstance(node.op, ast.USub):
                    return -value
            scalar = self._as_expr(value)
            if isinstance(node.op, ast.UAdd):
                return scalar
            if isinstance(node.op, ast.USub):
                return -scalar
            raise MapleImportError("unsupported unary operator")

        if isinstance(node, ast.BinOp):
            if (
                isinstance(node.op, ast.Sub)
                and self._literal_one(node.right)
                and isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Name)
                and self._eval(node.left.func, environment) == _IntrinsicRef("exp")
                and len(node.left.args) == 1
                and not node.left.keywords
            ):
                argument = self._as_expr(self._eval(node.left.args[0], environment))
                return self.graph.stable_unary("expm1", argument)
            left = self._eval(node.left, environment)
            right = self._eval(node.right, environment)
            if isinstance(left, Fraction) and isinstance(right, Fraction):
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    if right.denominator == 1:
                        return left ** int(right)
                    return float(left) ** float(right)
            if isinstance(node.op, ast.Pow):
                base = self._as_expr(left)
                if not isinstance(right, Fraction | int | float):
                    raise MapleImportError(
                        "Maple exponents must be compile-time scalars"
                    )
                if (
                    isinstance(right, Fraction)
                    and right.denominator == 1
                    and int(right) in (2, 4)
                ):
                    return self._even_power(base, int(right))
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
            if (
                target == _IntrinsicRef("log")
                and len(node.args) == 1
                and not node.keywords
            ):
                operand = self._one_plus_operand(node.args[0])
                if operand is not None:
                    value = self._as_expr(self._eval(operand, environment))
                    return self.graph.stable_unary("log1p", value)
            arguments = tuple(self._eval(item, environment) for item in node.args)
            if isinstance(target, _FunctionRef):
                return self.call(target.name, arguments)
            if isinstance(target, _IntrinsicRef):
                return self._intrinsic(target.name, arguments)
            raise MapleImportError("Maple call target is not callable")
        raise MapleImportError(f"unsupported Maple node {type(node).__name__}")

    def _intrinsic(self, name: str, arguments: Sequence[object]) -> Expr:
        if name == "f_zeta":
            if len(arguments) != 1:
                raise MapleImportError("f_zeta requires one scalar argument")
            z = self._as_expr(arguments[0])
            two_four_thirds = self.graph.approximate_constant(2.0 ** (4.0 / 3.0))
            return ((1 + z).pow(4.0 / 3.0) + (1 - z).pow(4.0 / 3.0) - 2) / (
                two_four_thirds - 2
            )
        if name == "mphi":
            if len(arguments) != 1:
                raise MapleImportError("mphi requires one scalar argument")
            z = self._as_expr(arguments[0])
            return ((1 + z).pow(2.0 / 3.0) + (1 - z).pow(2.0 / 3.0)) / 2
        if name == "tt":
            if len(arguments) != 3:
                raise MapleImportError("tt requires rs, zeta and xt")
            rs = self._as_expr(arguments[0])
            z = self._as_expr(arguments[1])
            xt = self._as_expr(arguments[2])
            phi = self._intrinsic("mphi", (z,))
            two_one_third = self.graph.approximate_constant(2.0 ** (1.0 / 3.0))
            return xt / (4 * two_one_third * phi * rs.pow(0.5))
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

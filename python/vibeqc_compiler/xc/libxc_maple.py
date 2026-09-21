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


IMPORTER_SEMANTICS = "libxc-maple-graph/v7"
_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
_RESERVED = frozenset(
    (
        "Pi",
        "X2S",
        "K_FACTOR_C",
        "MU_GE",
        "DBL_EPSILON",
        "gga_exchange",
        "mgga_exchange",
        "my_piecewise3",
        "my_piecewise5",
        "m_min",
        "m_max",
        "m_abs",
        "abs",
        "t_total",
        "n_total",
        "opz_pow_n",
        "_maple_diff2",
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
class _Comparison:
    operation: str
    left: object
    right: object


def _normalize_bindings(
    bindings: Mapping[str, str | int | float | Fraction] | None,
) -> tuple[tuple[str, str], ...]:
    """Canonicalize caller-owned Libxc scalar parameters for identity/lowering."""

    normalized: list[tuple[str, str]] = []
    for name, raw_value in sorted((bindings or {}).items()):
        if not _IDENTIFIER.fullmatch(name) or name in _RESERVED:
            raise MapleImportError(f"unsupported Maple binding name {name!r}")
        try:
            value = (
                raw_value
                if isinstance(raw_value, Fraction)
                else Fraction(str(raw_value))
            )
        except (ValueError, ZeroDivisionError) as error:
            raise MapleImportError(
                f"Maple binding {name!r} must be a finite scalar"
            ) from error
        normalized.append((name, str(value)))
    return tuple(normalized)


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
    bindings: tuple[tuple[str, str], ...] = ()

    @property
    def transitive_sha256(self) -> str:
        """Hash the selected source graph and active preprocessor symbols."""

        payload = json.dumps(
            {
                "entry_sha256": self.source_sha256,
                "defines": self.defines,
                "initial_defines": self.initial_defines,
                "bindings": self.bindings,
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
    bindings: tuple[tuple[str, str], ...],
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
        if name in dict(bindings):
            raise MapleImportError(
                f"Maple source definition collides with external binding {name!r}"
            )
        is_function = "->" in right
        if name in assignments or name in functions:
            same_kind = (is_function and name in functions) or (
                not is_function and name in assignments
            )
            if not allow_redefinition or not same_kind:
                raise MapleImportError(f"duplicate Maple definition {name!r}")
            if not is_function and assignments.get(name) == right:
                continue
            if is_function and name in functions:
                parameter_text, expression = (
                    part.strip() for part in right.split("->", 1)
                )
                if parameter_text.startswith("(") and parameter_text.endswith(")"):
                    parameter_text = parameter_text[1:-1]
                parameters = tuple(item.strip() for item in parameter_text.split(","))
                if functions[name] == MapleFunction(parameters, expression):
                    continue
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
        bindings=bindings,
    )


def import_maple_source(
    source: str,
    *,
    defines: Iterable[str] = (),
    bindings: Mapping[str, str | int | float | Fraction] | None = None,
) -> MapleModule:
    """Parse one in-memory source without permitting includes or redefinition."""

    initial_defines = tuple(sorted(set(defines)))
    normalized_bindings = _normalize_bindings(bindings)
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
        bindings=normalized_bindings,
        allow_redefinition=False,
    )


def import_maple_file(
    root: Path,
    entry: str,
    *,
    defines: Iterable[str] = (),
    support_files: Iterable[str] = (),
    bindings: Mapping[str, str | int | float | Fraction] | None = None,
    allow_duplicate_includes: bool = False,
) -> MapleModule:
    """Load one bounded include graph rooted inside a pinned source directory."""

    source_root = Path(root).resolve()
    initial_defines = tuple(sorted(set(defines)))
    normalized_bindings = _normalize_bindings(bindings)
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
        if resolved in loaded and not allow_duplicate_includes:
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
        bindings=normalized_bindings,
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
    ast.Compare,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _split_top_level_once(text: str) -> tuple[str, str]:
    depth = 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            return text[:index], text[index + 1 :]
    raise MapleImportError("bounded Maple add requires an index range")


def _expand_bounded_add(expression: str) -> str:
    """Expand only source-evidenced finite add(body, i=N..M) reductions."""

    pattern = re.compile(r"\badd\s*\(")
    while True:
        match = pattern.search(expression)
        if match is None:
            return expression
        start = match.start()
        open_paren = expression.find("(", match.start())
        depth = 1
        close_paren = open_paren + 1
        while close_paren < len(expression) and depth:
            char = expression[close_paren]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            close_paren += 1
        if depth:
            raise MapleImportError("unterminated bounded Maple add")
        inside = expression[open_paren + 1 : close_paren - 1]
        body, range_text = _split_top_level_once(inside)
        range_match = re.fullmatch(
            r"\s*([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*",
            range_text,
        )
        if range_match is None:
            raise MapleImportError(
                "only finite integer Maple add(body, i=N..M) is supported"
            )
        variable, lower_text, upper_text = range_match.groups()
        lower, upper = int(lower_text), int(upper_text)
        if upper < lower or upper - lower > 32:
            raise MapleImportError("bounded Maple add range is unsupported")
        terms = [
            re.sub(rf"\b{re.escape(variable)}\b", f"({value})", body)
            for value in range(lower, upper + 1)
        ]
        expanded = "(" + " + ".join(f"({term})" for term in terms) + ")"
        expression = expression[:start] + expanded + expression[close_paren:]


def _translate_eval_diff(expression: str) -> str:
    """Translate the exact two-argument eval(diff(...), substitutions) form."""

    match = re.fullmatch(
        r"\s*eval\s*\(\s*diff\s*\(\s*([A-Za-z_]\w*)\s*"
        r"\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)"
        r"\s*,\s*([A-Za-z_]\w*)\s*\)\s*,\s*\[\s*"
        r"\2\s*=\s*([A-Za-z_]\w*)\s*,\s*\3\s*=\s*"
        r"([A-Za-z_]\w*)\s*\]\s*\)\s*",
        expression,
    )
    if match is None:
        if "eval(" in expression or "diff(" in expression:
            raise MapleImportError(
                "only two-argument eval(diff(f(x1,x2), xi), substitutions) is supported"
            )
        return expression
    function, first, second, derivative, value0, value1 = match.groups()
    if first == second:
        raise MapleImportError("Maple diff placeholders must be distinct")
    if derivative == first:
        index = 0
    elif derivative == second:
        index = 1
    else:
        raise MapleImportError("Maple diff variable must be a function argument")
    return f"_maple_diff2({function}, {index}, {value0}, {value1})"


@cache
def _parse_expression(expression: str) -> ast.Expression:
    translated = _translate_eval_diff(_expand_bounded_add(expression)).replace(
        "^", "**"
    )
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
        self.bindings = {name: Fraction(value) for name, value in module.bindings}
        self._assignment_cache: dict[str, object] = {}
        self._assignment_stack: set[str] = set()
        self._derivative_cache: dict[
            tuple[str, int], tuple[Expr, tuple[Expr, ...]]
        ] = {}

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
        result = self._as_expr(value)
        if (
            name == "scan_gx"
            and function.parameters == ("x",)
            and function.expression == "1 - exp(-scan_a1/sqrt(X2S*x))"
        ):
            # The pinned SCAN expression has the analytic x -> 0+ limit 1.
            # Make that limit explicit so value/derivatives never evaluate the
            # inactive reciprocal-sqrt branch at an exactly zero gradient.
            x = self._as_expr(arguments[0])
            return self.graph.select_le(x, 0, 1, result)
        return result

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
        if name == "K_FACTOR_C":
            return self.graph.approximate_constant(
                3.0 / 10.0 * (6.0 * math.pi**2) ** (2.0 / 3.0)
            )
        if name == "MU_GE":
            return Fraction(10, 81)
        if name == "DBL_EPSILON":
            return self.graph.approximate_constant(2.220446049250313e-16)
        if name in self.bindings:
            return self.bindings[name]
        if name in self.functions:
            return _FunctionRef(name)
        if name in (
            "gga_exchange",
            "mgga_exchange",
            "my_piecewise3",
            "my_piecewise5",
            "m_min",
            "m_max",
            "m_abs",
            "abs",
            "t_total",
            "n_total",
            "opz_pow_n",
            "_maple_diff2",
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

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise MapleImportError("chained Maple comparisons are unsupported")
            left = self._eval(node.left, environment)
            right = self._eval(node.comparators[0], environment)
            operation = node.ops[0]
            if isinstance(operation, ast.Lt):
                kind = "lt"
            elif isinstance(operation, ast.LtE):
                kind = "le"
            elif isinstance(operation, ast.Gt):
                kind = "gt"
            elif isinstance(operation, ast.GtE):
                kind = "ge"
            else:
                raise MapleImportError("unsupported Maple comparison")
            return _Comparison(kind, left, right)

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

    def _select_comparison(
        self,
        comparison: _Comparison,
        if_true: object,
        if_false: object,
    ) -> Expr:
        left, right = comparison.left, comparison.right
        if isinstance(left, Fraction) and isinstance(right, Fraction):
            selected = {
                "lt": left < right,
                "le": left <= right,
                "gt": left > right,
                "ge": left >= right,
            }[comparison.operation]
            return self._as_expr(if_true if selected else if_false)
        left_expr, right_expr = self._as_expr(left), self._as_expr(right)
        true_expr, false_expr = self._as_expr(if_true), self._as_expr(if_false)
        if comparison.operation == "le":
            return self.graph.select_le(left_expr, right_expr, true_expr, false_expr)
        if comparison.operation == "lt":
            return self.graph.select_le(right_expr, left_expr, false_expr, true_expr)
        if comparison.operation == "gt":
            return self.graph.select_le(left_expr, right_expr, false_expr, true_expr)
        if comparison.operation == "ge":
            return self.graph.select_le(right_expr, left_expr, true_expr, false_expr)
        raise MapleImportError("unsupported Maple comparison")

    def _substitute(
        self,
        root: Expr,
        replacements: Mapping[int, Expr],
    ) -> Expr:
        """Rebuild one derivative root with placeholder variables substituted."""

        from vibeqc_compiler.integral.expr import Expr

        cache: dict[int, Expr] = {}

        def visit(identifier: int) -> Expr:
            if identifier in replacements:
                return replacements[identifier]
            cached = cache.get(identifier)
            if cached is not None:
                return cached
            node = self.graph.nodes[identifier]
            if node.operation in ("constant", "variable"):
                result = Expr(self.graph, identifier)
            else:
                arguments = tuple(visit(item) for item in node.arguments)
                if node.operation == "add":
                    result = self.graph.add_many(arguments)
                elif node.operation == "multiply":
                    result = self.graph.multiply_many(arguments)
                elif node.operation == "reciprocal":
                    result = self.graph.reciprocal(arguments[0])
                elif node.operation == "exp":
                    result = self.graph.exponential(arguments[0])
                elif node.operation in ("log", "log1p", "expm1"):
                    result = self.graph.stable_unary(node.operation, arguments[0])
                elif node.operation == "select_le":
                    result = self.graph.select_le(*arguments)
                elif node.operation in ("atan", "asinh", "erf"):
                    result = self.graph.transcendental_unary(
                        node.operation, arguments[0]
                    )
                elif node.operation == "power":
                    if not isinstance(node.payload, (Fraction, float)):
                        raise MapleImportError(
                            "Maple derivative power requires a numeric exponent"
                        )
                    result = self.graph.power(arguments[0], float(node.payload))
                else:
                    raise MapleImportError(
                        f"cannot substitute Maple derivative operation {node.operation!r}"
                    )
            cache[identifier] = result
            return result

        return visit(root.identifier)

    def _intrinsic(self, name: str, arguments: Sequence[object]) -> Expr:
        if name == "_maple_diff2":
            if (
                len(arguments) != 4
                or not isinstance(arguments[0], _FunctionRef)
                or not isinstance(arguments[1], Fraction)
                or arguments[1].denominator != 1
            ):
                raise MapleImportError(
                    "_maple_diff2 requires a binary function, derivative index, "
                    "and two scalar substitutions"
                )
            function_ref = arguments[0]
            derivative_index = int(arguments[1])
            function = self.functions.get(function_ref.name)
            if function is None or len(function.parameters) != 2:
                raise MapleImportError(
                    "qualified Maple diff requires a two-argument function"
                )
            if derivative_index not in (0, 1):
                raise MapleImportError("qualified Maple diff index must be 0 or 1")
            key = (function_ref.name, derivative_index)
            cached = self._derivative_cache.get(key)
            if cached is None:
                placeholders = tuple(
                    self.graph.variable(f"__maple_diff_{function_ref.name}_{index}")
                    for index in range(2)
                )
                value = self.call(function_ref.name, placeholders)
                derivative = self.graph.differentiate(
                    value, placeholders[derivative_index]
                )
                cached = (derivative, placeholders)
                self._derivative_cache[key] = cached
            derivative, placeholders = cached
            replacements = {
                placeholders[0].identifier: self._as_expr(arguments[2]),
                placeholders[1].identifier: self._as_expr(arguments[3]),
            }
            return self._substitute(derivative, replacements)
        if name == "t_total":
            if len(arguments) != 3:
                raise MapleImportError("t_total requires zeta and two tau scalars")
            z = self._as_expr(arguments[0])
            ts0 = self._as_expr(arguments[1])
            ts1 = self._as_expr(arguments[2])
            return ts0 * ((1 + z) / 2).pow(5.0 / 3.0) + ts1 * ((1 - z) / 2).pow(
                5.0 / 3.0
            )
        if name == "n_total":
            if len(arguments) != 1:
                raise MapleImportError("n_total requires rs")
            rs = self._as_expr(arguments[0])
            rs_factor = self.graph.approximate_constant(
                (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
            )
            return (rs_factor / rs).pow(3.0)
        if name == "opz_pow_n":
            if len(arguments) != 2:
                raise MapleImportError("opz_pow_n requires zeta and exponent")
            z = self._as_expr(arguments[0])
            exponent = arguments[1]
            if not isinstance(exponent, Fraction | int | float):
                raise MapleImportError(
                    "opz_pow_n exponent must be a compile-time scalar"
                )
            return (1 + z).pow(float(exponent))
        if name == "my_piecewise3":
            if len(arguments) != 3 or not isinstance(arguments[0], _Comparison):
                raise MapleImportError(
                    "my_piecewise3 requires one comparison and two scalar branches"
                )
            return self._select_comparison(arguments[0], arguments[1], arguments[2])
        if name == "my_piecewise5":
            if (
                len(arguments) != 5
                or not isinstance(arguments[0], _Comparison)
                or not isinstance(arguments[2], _Comparison)
            ):
                raise MapleImportError(
                    "my_piecewise5 requires two comparisons and three scalar branches"
                )
            fallback = self._select_comparison(arguments[2], arguments[3], arguments[4])
            return self._select_comparison(arguments[0], arguments[1], fallback)
        if name in ("m_min", "m_max"):
            if len(arguments) != 2:
                raise MapleImportError(f"{name} requires two scalar arguments")
            left, right = self._as_expr(arguments[0]), self._as_expr(arguments[1])
            if name == "m_min":
                return self.graph.select_le(left, right, left, right)
            return self.graph.select_le(left, right, right, left)
        if name in ("m_abs", "abs"):
            if len(arguments) != 1:
                raise MapleImportError(f"{name} requires one scalar argument")
            if isinstance(arguments[0], Fraction):
                return self.graph.constant(abs(arguments[0]))
            value = self._as_expr(arguments[0])
            return self.graph.select_le(value, 0, -value, value)
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
        if name == "mgga_exchange":
            if len(arguments) != 9 or not isinstance(arguments[0], _FunctionRef):
                raise MapleImportError(
                    "mgga_exchange requires a function and eight scalars"
                )
            function = arguments[0]
            rs, z, xs0, xs1, u0, u1, t0, t1 = arguments[1:]
            rs_expr, z_expr = self._as_expr(rs), self._as_expr(z)
            term0 = self._lda_x_spin(rs_expr, z_expr) * self.call(
                function.name,
                (
                    self._as_expr(xs0),
                    self._as_expr(u0),
                    self._as_expr(t0),
                ),
            )
            term1 = self._lda_x_spin(rs_expr, -z_expr) * self.call(
                function.name,
                (
                    self._as_expr(xs1),
                    self._as_expr(u1),
                    self._as_expr(t1),
                ),
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

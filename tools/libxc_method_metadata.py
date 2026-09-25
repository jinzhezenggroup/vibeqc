"""Fail-closed extraction of Libxc hybrid composition metadata.

This module reads the pinned Libxc C *metadata owners* without compiling or
executing C.  It recognizes a deliberately small arithmetic subset sufficient
for mix coefficients, CAM parameters and VV10 constants.  Unsupported control
flow or assignments are rejected rather than guessed.
"""

from __future__ import annotations

import ast
import re
from fractions import Fraction
from typing import Any

from tools.libxc_bulk_metadata import (
    CMetadataError,
    array_values,
    brace_body,
    constant_definitions,
    constant_value,
    split_fields,
    strip_c_comments,
)

METHOD_METADATA_SEMANTICS = "libxc-hybrid-method-metadata/v1"
_IDENTIFIER = r"[A-Za-z_]\w*"
_HYBRID_FAMILIES = ("XC_FAMILY_HYB_GGA", "XC_FAMILY_HYB_MGGA")


class MethodMetadataError(CMetadataError):
    """Hybrid metadata needs semantics outside the audited extraction subset."""


def _fraction_value(
    expression: str,
    definitions: dict[str, str | None],
    env: dict[str, Fraction] | None = None,
    seen: tuple[str, ...] = (),
) -> Fraction:
    """Evaluate bounded arithmetic as exact rationals, never binary floats."""
    if len(expression) > 4096 or len(seen) > 32:
        raise MethodMetadataError(
            "hybrid constant exceeds the bounded expression limit"
        )
    expression = re.sub(
        r"(?<![\w.])((?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)[lLfF]\b",
        r"\1",
        expression.strip(),
    )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise MethodMetadataError(
            f"unsupported hybrid arithmetic: {expression}"
        ) from error
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise MethodMetadataError("hybrid arithmetic exceeds the bounded AST limit")
    values = {} if env is None else env

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            # ast.Constant.value has already rounded decimal/scientific tokens.
            # Recover the original lexeme, including digits below binary64 range.
            token = ast.get_source_segment(expression, node)
            if token is None:
                raise MethodMetadataError("missing hybrid numeric source token")
            return Fraction(token)
        if isinstance(node, ast.Name):
            if node.id in values:
                return values[node.id]
            raw = definitions.get(node.id)
            if node.id in seen or raw is None:
                raise MethodMetadataError(
                    f"unknown, ambiguous or cyclic hybrid constant: {node.id}"
                )
            return _fraction_value(raw, definitions, values, (*seen, node.id))
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
                return left / right
        raise MethodMetadataError(f"unsupported hybrid arithmetic: {expression}")

    try:
        return visit(tree.body)
    except ZeroDivisionError as error:
        raise MethodMetadataError("singular hybrid arithmetic") from error


def _function_body(text: str, name: str) -> str:
    if not re.fullmatch(_IDENTIFIER, name):
        raise MethodMetadataError("hybrid owner is not a named C function")
    matches = list(
        re.finditer(
            rf"\b(?:static\s+)?void\s+{re.escape(name)}\s*\([^)]*\)\s*\{{",
            text,
        )
    )
    if len(matches) != 1:
        raise MethodMetadataError(f"missing or ambiguous hybrid owner function: {name}")
    body, _ = brace_body(text, matches[0].end() - 1)
    # This extractor models linear metadata owners, not C control flow.
    # Ignore quoted text, but reject branches/loops/early returns before regex
    # extraction can mistake a conditional assignment for an unconditional one.
    code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', body)
    if re.search(r"\b(?:if|else|for|while|do|switch|goto|return)\b|[?#]", code):
        raise MethodMetadataError("hybrid owner requires unsupported control flow")
    return body


def _string_array(text: str, name: str) -> list[str]:
    result: list[str] = []
    for raw in array_values(text, name):
        try:
            value = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as error:
            raise MethodMetadataError(
                "hybrid parameter name is not a literal string"
            ) from error
        if not isinstance(value, str):
            raise MethodMetadataError("hybrid parameter name is not a string")
        result.append(value)
    return result


def _parameter_defaults(
    text: str,
    parameter_initializer: str,
    definitions: dict[str, str | None],
) -> tuple[list[str], list[Fraction], str]:
    if not parameter_initializer.startswith("{") or not parameter_initializer.endswith(
        "}"
    ):
        raise MethodMetadataError(
            "hybrid external parameter contract is not an initializer"
        )
    fields = split_fields(parameter_initializer[1:-1])
    if len(fields) != 5:
        raise MethodMetadataError("unsupported hybrid external parameter contract")
    count = constant_value(fields[0], definitions)
    if type(count) is not int or not 0 <= count <= 256:
        raise MethodMetadataError(
            "hybrid external parameter count exceeds supported bound"
        )
    if not count:
        if any(value != "NULL" for value in fields[1:]):
            raise MethodMetadataError("nonempty zero-parameter hybrid contract")
        return [], [], fields[4]
    names = _string_array(text, fields[1])
    values = [
        _fraction_value(value, definitions) for value in array_values(text, fields[3])
    ]
    if len(names) != count or len(values) != count:
        raise MethodMetadataError(
            "hybrid parameter array length does not match registration"
        )
    return names, values, fields[4]


def _initial_mix(
    init_body: str, definitions: dict[str, str | None]
) -> tuple[list[str], list[Fraction]] | None:
    calls = list(
        re.finditer(
            rf"\bxc_mix_init\s*\(\s*p\s*,\s*([^,]+),\s*({_IDENTIFIER})\s*,\s*({_IDENTIFIER})\s*\)\s*;",
            init_body,
        )
    )
    if not calls:
        return None
    if len(calls) != 1:
        raise MethodMetadataError(
            "multiple xc_mix_init calls require control-flow interpretation"
        )
    count = constant_value(calls[0][1], definitions)
    ids = [value.strip() for value in array_values(init_body, calls[0][2])]
    coefficients = [
        _fraction_value(value, definitions)
        for value in array_values(init_body, calls[0][3])
    ]
    if type(count) is not int or count != len(ids) or count != len(coefficients):
        raise MethodMetadataError("hybrid mix array length does not match xc_mix_init")
    if any(not re.fullmatch(r"XC_(?:LDA|GGA|MGGA)_[A-Z0-9_]+", value) for value in ids):
        raise MethodMetadataError(
            "hybrid mix contains an unsupported auxiliary functional ID"
        )
    return [value.removeprefix("XC_") for value in ids], coefficients


def _setter_state(
    text: str,
    setter: str,
    names: list[str],
    defaults: list[Fraction],
    definitions: dict[str, str | None],
    initial_mix: list[Fraction] | None,
    initial_cam: tuple[Fraction, Fraction, Fraction],
) -> tuple[list[Fraction] | None, Fraction, Fraction, Fraction]:
    """Return mix, CAM alpha/beta/omega after applying default parameters."""
    mix = None if initial_mix is None else list(initial_mix)
    alpha, beta, omega = initial_cam
    by_name = dict(zip(names, defaults, strict=True))
    if setter == "set_ext_params_cpy_cam":
        try:
            return mix, by_name["_alpha"], by_name["_beta"], by_name["_omega"]
        except KeyError as error:
            raise MethodMetadataError(
                "copy-CAM setter lacks alpha/beta/omega parameters"
            ) from error
    if setter in ("set_ext_params_exx", "set_ext_params_cpy_exx"):
        if not defaults:
            raise MethodMetadataError("exact-exchange setter lacks a coefficient")
        return mix, defaults[-1], beta, omega
    if setter == "NULL":
        return mix, alpha, beta, omega

    body = _function_body(text, setter)
    env: dict[str, Fraction] = {}
    seen_mix: set[int] = set()
    seen_cam: set[str] = set()
    auxiliary_omegas: list[Fraction] = []
    for raw in body.split(";"):
        statement = raw.strip()
        if not statement:
            continue
        if re.fullmatch(r"assert\s*\(\s*p\s*!=\s*NULL\s*\)", statement):
            continue
        declaration = re.fullmatch(r"(?:const\s+)?double\s+(.+)", statement, re.DOTALL)
        if declaration:
            for name in split_fields(declaration[1]):
                if not re.fullmatch(_IDENTIFIER, name.strip()):
                    raise MethodMetadataError("unsupported hybrid local declaration")
            continue
        assignment = re.fullmatch(rf"({_IDENTIFIER})\s*=\s*(.+)", statement, re.DOTALL)
        if assignment:
            name, expression = assignment.groups()
            parameter = re.fullmatch(
                r"get_ext_param\s*\(\s*p\s*,\s*ext_params\s*,\s*([^)]*)\)",
                expression,
            )
            if parameter:
                index = constant_value(parameter[1], definitions)
                if type(index) is not int or not 0 <= index < len(defaults):
                    raise MethodMetadataError(
                        "hybrid setter reads an invalid parameter index"
                    )
                env[name] = defaults[index]
            else:
                # Unknown reassignment must not retain a stale earlier value.
                env[name] = _fraction_value(expression, definitions, env)
            continue
        assignment = re.fullmatch(
            r"p->mix_coef\s*\[\s*(\d+)\s*\]\s*=\s*(.+)", statement, re.DOTALL
        )
        if assignment:
            index = int(assignment[1])
            if mix is None or index >= len(mix) or index in seen_mix:
                raise MethodMetadataError("invalid or duplicate hybrid mix assignment")
            mix[index] = _fraction_value(assignment[2], definitions, env)
            seen_mix.add(index)
            continue
        assignment = re.fullmatch(
            r"p->cam_(alpha|beta|omega)\s*=\s*(.+)", statement, re.DOTALL
        )
        if assignment:
            field = assignment[1]
            if field in seen_cam:
                raise MethodMetadataError("duplicate hybrid CAM assignment")
            seen_cam.add(field)
            value = _fraction_value(assignment[2], definitions, env)
            if field == "alpha":
                alpha = value
            elif field == "beta":
                beta = value
            else:
                omega = value
            continue
        auxiliary = re.fullmatch(
            r'xc_func_set_ext_params_name\s*\(\s*p->func_aux\[(\d+)\]\s*,\s*"_omega"\s*,\s*(.+)\)',
            statement,
            re.DOTALL,
        )
        if auxiliary:
            if mix is None or int(auxiliary[1]) >= len(mix):
                raise MethodMetadataError("invalid hybrid auxiliary parameter target")
            auxiliary_omegas.append(_fraction_value(auxiliary[2], definitions, env))
            continue
        if re.fullmatch(
            r"set_ext_params_cam\s*\(\s*p\s*,\s*ext_params\s*\)", statement
        ):
            try:
                alpha, beta, omega = (
                    by_name["_alpha"],
                    by_name["_beta"],
                    by_name["_omega"],
                )
            except KeyError as error:
                raise MethodMetadataError(
                    "CAM setter lacks alpha/beta/omega parameters"
                ) from error
            continue
        raise MethodMetadataError(f"unsupported hybrid setter statement: {statement}")
    if any(value != omega for value in auxiliary_omegas):
        raise MethodMetadataError(
            "auxiliary range parameter differs from the hybrid CAM omega"
        )
    return mix, alpha, beta, omega


def _validate_initializer(init_body: str) -> None:
    """Reject every initializer statement outside the modeled metadata subset.

    Searching only for known calls is insufficient: an unmodeled helper or array
    write could change the mixture after the recognized initializer. Allocation
    of the direct MGGA parameter block is the sole non-metadata operation admitted
    here; its values are populated by the separately audited default setter.
    """
    patterns = (
        (
            rf"(?:static\s+)?(?:const\s+)?(?:int|double)\s+{_IDENTIFIER}"
            r"\s*\[\s*[^\]]*\s*\]\s*=\s*\{[^{}]*\}"
        ),
        (
            rf"xc_mix_init\s*\(\s*p\s*,\s*[^,]+,\s*{_IDENTIFIER}"
            rf"\s*,\s*{_IDENTIFIER}\s*\)"
        ),
        r"xc_hyb_init_hybrid\s*\(\s*p\s*,\s*[^()]+\)",
        r"xc_hyb_init_cam\s*\(\s*p\s*,\s*[^,()]+,\s*[^,()]+,\s*[^,()]+\)",
        r"p->nlc_(?:b|C)\s*=\s*[^;]+",
        r"assert\s*\(\s*p->params\s*==\s*NULL\s*\)",
        rf"p->params\s*=\s*libxc_malloc\s*\(\s*sizeof\s*\(\s*{_IDENTIFIER}\s*\)\s*\)",
    )
    for raw in init_body.split(";"):
        statement = raw.strip()
        if statement and not any(
            re.fullmatch(pattern, statement, re.DOTALL) for pattern in patterns
        ):
            raise MethodMetadataError(
                f"unsupported hybrid initializer statement: {statement}"
            )


def _init_state(
    init_body: str, definitions: dict[str, str | None]
) -> tuple[Fraction, Fraction, Fraction, Fraction | None, Fraction | None]:
    _validate_initializer(init_body)
    alpha = beta = omega = Fraction(0)
    global_calls = list(
        re.finditer(r"xc_hyb_init_hybrid\s*\(\s*p\s*,\s*([^)]*)\)", init_body)
    )
    cam_calls = list(
        re.finditer(
            r"xc_hyb_init_cam\s*\(\s*p\s*,\s*([^,]+),\s*([^,]+),\s*([^)]*)\)",
            init_body,
        )
    )
    if len(global_calls) + len(cam_calls) > 1:
        raise MethodMetadataError(
            "multiple hybrid initialization calls require control flow"
        )
    if global_calls:
        alpha = _fraction_value(global_calls[0][1], definitions)
    elif cam_calls:
        alpha = _fraction_value(cam_calls[0][1], definitions)
        beta = _fraction_value(cam_calls[0][2], definitions)
        omega = _fraction_value(cam_calls[0][3], definitions)
    nlc: dict[str, Fraction] = {}
    for key, c_name in (("b", "b"), ("c", "C")):
        matches = list(re.finditer(rf"p->nlc_{c_name}\s*=\s*([^;]+);", init_body))
        if len(matches) > 1:
            raise MethodMetadataError(
                "multiple nonlocal parameter assignments require control flow"
            )
        if matches:
            nlc[key] = _fraction_value(matches[0][1], definitions)
    if bool(nlc) and set(nlc) != {"b", "c"}:
        raise MethodMetadataError("partial VV10 parameter pair")
    return alpha, beta, omega, nlc.get("b"), nlc.get("c")


def _method_identifier(registration: str) -> str:
    for prefix in ("HYB_GGA_XC_", "HYB_MGGA_XC_", "HYB_GGA_X_", "HYB_MGGA_X_"):
        if registration.startswith(prefix):
            return registration.removeprefix(prefix).replace("_", "-")
    raise MethodMetadataError("hybrid registration is not a supported method owner")


def extract_method_registrations(
    c_source: str, source_name: str = "<memory>"
) -> list[dict[str, Any]]:
    """Extract representational MethodIR inputs from one pinned hybrid C owner."""
    text = strip_c_comments(c_source)
    definitions = constant_definitions(text)
    records: list[dict[str, Any]] = []
    for match in re.finditer(
        rf"\bconst\s+xc_func_info_type\s+xc_func_info_({_IDENTIFIER})\s*=\s*\{{",
        text,
    ):
        registration = match[1].upper()
        record: dict[str, Any] = {
            "registration": registration,
            "metadata_semantics": METHOD_METADATA_SEMANTICS,
            "source": source_name,
        }
        try:
            body, _ = brace_body(text, match.end() - 1)
            fields = split_fields(body)
            if len(fields) != 13:
                raise MethodMetadataError("unsupported xc_func_info_type layout")
            if fields[3] not in _HYBRID_FAMILIES:
                raise MethodMetadataError(
                    "registration is not a supported hybrid XC family"
                )
            if "XC_FLAGS_3D" not in re.split(r"\s*\|\s*", fields[5]):
                raise MethodMetadataError("non-3D hybrid functional")
            kind = fields[1]
            if kind not in ("XC_EXCHANGE_CORRELATION", "XC_EXCHANGE"):
                raise MethodMetadataError(
                    "hybrid registration is not exchange-correlation or exchange"
                )
            split_exchange = kind == "XC_EXCHANGE"
            if split_exchange and not registration.startswith(
                ("HYB_GGA_X_", "HYB_MGGA_X_")
            ):
                raise MethodMetadataError(
                    "split hybrid exchange registration has unsupported naming"
                )
            libxc_id = constant_value(fields[0], definitions)
            if type(libxc_id) is not int or libxc_id <= 0:
                raise MethodMetadataError(
                    "hybrid functional ID must be a positive integer"
                )
            names, defaults, setter = _parameter_defaults(text, fields[7], definitions)
            init_name = fields[8].strip()
            if split_exchange:
                # Global split-hybrid exchange owners such as M06-2X and MN15
                # may share an initializer with unrelated variants and therefore
                # contain registration-dependent control flow. Their
                # set_ext_params_*_exx contract is sufficient here: the last
                # default is the exact-exchange fraction and the preceding
                # parameters belong to the semilocal Maple body.
                if setter not in ("set_ext_params_exx", "set_ext_params_cpy_exx"):
                    raise MethodMetadataError(
                        "split hybrid exchange requires the audited global-exchange setter"
                    )
                component_ids = None
                initial_mix = None
                init_alpha = init_beta = init_omega = Fraction(0)
                nlc_b = nlc_c = None
            else:
                init_body = _function_body(text, init_name)
                initial = _initial_mix(init_body, definitions)
                component_ids = None if initial is None else initial[0]
                initial_mix = None if initial is None else initial[1]
                init_alpha, init_beta, init_omega, nlc_b, nlc_c = _init_state(
                    init_body, definitions
                )
            mix, alpha, beta, omega = _setter_state(
                text,
                setter,
                names,
                defaults,
                definitions,
                initial_mix,
                (init_alpha, init_beta, init_omega),
            )
            if any(value < 0 for value in (alpha, omega)):
                raise MethodMetadataError(
                    "negative CAM alpha/omega is not representable"
                )
            components: list[tuple[str, Fraction]] = []
            direct_stem = None
            split_exchange_component = None
            paired_correlation_component = None
            if component_ids is not None:
                assert mix is not None
                components = [
                    (name, coefficient)
                    for name, coefficient in zip(component_ids, mix, strict=True)
                    if coefficient
                ]
            elif split_exchange:
                prefix = (
                    "HYB_MGGA_X_" if registration.startswith("HYB_MGGA_X_")
                    else "HYB_GGA_X_"
                )
                family_prefix = "MGGA" if prefix == "HYB_MGGA_X_" else "GGA"
                direct_stem = registration.removeprefix(prefix)
                split_exchange_component = registration
                paired_correlation_component = f"{family_prefix}_C_{direct_stem}"
                components = [(registration, Fraction(1))]
            elif registration.startswith("HYB_MGGA_XC_"):
                direct_stem = registration.removeprefix("HYB_MGGA_XC_")
            else:
                raise MethodMetadataError(
                    "hybrid has neither xc_mix_init nor a supported direct semilocal body"
                )

            exact = short = long = Fraction(0)
            if omega:
                short = alpha + beta
                long = alpha
                if short < 0 or long < 0:
                    raise MethodMetadataError(
                        "negative range-separated exact-exchange fraction"
                    )
            else:
                exact = alpha
            nonlocal_variant = None
            if nlc_b is not None or nlc_c is not None:
                if "XC_FLAGS_VV10" not in re.split(r"\s*\|\s*", fields[5]):
                    raise MethodMetadataError("nonlocal constants lack XC_FLAGS_VV10")
                nonlocal_variant = "vv10"
            record.update(
                status="generated",
                identifier=_method_identifier(registration),
                libxc_id=libxc_id,
                family=fields[3].removeprefix("XC_FAMILY_").lower(),
                flags=fields[5],
                components=[[name, str(value)] for name, value in components],
                direct_semilocal_stem=direct_stem,
                **(
                    {
                        "split_exchange_component": split_exchange_component,
                        "paired_correlation_component": paired_correlation_component,
                    }
                    if split_exchange
                    else {}
                ),
                exact_exchange=str(exact),
                short_range_exchange=str(short),
                long_range_exchange=str(long),
                range_omega=str(omega),
                nonlocal_variant=nonlocal_variant,
                nlc_b=None if nlc_b is None else str(nlc_b),
                nlc_c=None if nlc_c is None else str(nlc_c),
            )
        except (
            CMetadataError,
            MethodMetadataError,
            AssertionError,
            ValueError,
        ) as error:
            record.update(status="blocked", reason=str(error))
        records.append(record)
    return records

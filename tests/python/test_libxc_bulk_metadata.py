"""Adversarial checks for source-owned C parameter binding, never C execution."""

from __future__ import annotations

import pytest

from tools.libxc_bulk_metadata import (
    CMetadataError,
    constant_definitions,
    constant_value,
    extract_registrations,
    split_fields,
    strip_c_comments,
)

SOURCE = """
#define XC_GGA_X_TRIAL 777
#define N 2
typedef struct { double kappa, mu; } trial_params;
static const char *names[N] = {"_unrelated_label_1", "_unrelated_label_2"};
static const double defaults[N] = {0.804, 10.0/81.0};
const xc_func_info_type xc_func_info_gga_x_trial = {
 XC_GGA_X_TRIAL, XC_EXCHANGE, "a name, with // literal comments",
 XC_FAMILY_GGA, {NULL,NULL,NULL,NULL,NULL}, XC_FLAGS_3D | MAPLE2C_FLAGS,
 1e-15, {N, names, descriptions, defaults, set_ext_params_cpy},
 init, NULL, NULL, &work_gga, NULL
};
"""
MAPLE = "(* prefix: trial_params *params; *)\nf := x -> params_a_mu*x:\n"


def record(source: str = SOURCE) -> dict:
    rows = extract_registrations(source, MAPLE, "")
    assert len(rows) == 1
    return rows[0]


def test_struct_layout_not_display_name_owns_binding() -> None:
    result = record()
    assert result["metadata_status"] == "bound"
    assert result["bindings"]["params_a_kappa"] == "0.804"
    assert float(result["bindings"]["params_a_mu"]) == pytest.approx(10 / 81)
    assert not any("unrelated_label" in name for name in result["bindings"])


def test_bounded_double_arrays_have_both_upstream_spellings() -> None:
    result = record(SOURCE.replace("double kappa, mu;", "double coeffs[2];"))
    assert result["metadata_status"] == "bound"
    bindings = result["bindings"]
    assert bindings["params_a_coeffs"] == [
        bindings["params_a_coeffs_0_"],
        bindings["params_a_coeffs_1_"],
    ]


def test_const_double_arrays_preserve_copy_parameter_layout() -> None:
    result = record(SOURCE.replace("double kappa, mu;", "const double coeffs[2];"))
    assert result["metadata_status"] == "bound"
    assert result["bindings"]["params_a_coeffs"] == ["0.804", repr(float(10 / 81))]


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        ("set_ext_params_cpy", "lspbe_set_ext_params", "custom parameter setter"),
        ("double kappa, mu;", "int kappa; double mu;", "non-double"),
        ("double kappa, mu;", "double coeffs[3];", "partial parameter array"),
        ("{0.804, 10.0/81.0}", "{0.804}", "array length"),
        ("{0.804, 10.0/81.0}", "{0.804, undefined_macro}", "unknown"),
        ("&work_gga", "&custom_work_gga", "custom or absent"),
        ("XC_FLAGS_3D", "XC_FLAGS_2D", "non-3D"),
        ("XC_FAMILY_GGA", "XC_FAMILY_HYB_GGA", "MethodIR composition"),
        ("XC_EXCHANGE,", "XC_KINETIC,", "non-XC"),
    ],
)
def test_unsupported_metadata_never_becomes_default_binding(
    old: str, new: str, reason: str
) -> None:
    result = record(SOURCE.replace(old, new))
    assert result["metadata_status"] == "blocked"
    assert reason in result["reason"]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("7/2", 3), ("-7/2", -3), ("7.0/2", 3.5), ("0xFF", 255), ("0.25L", 0.25)],
)
def test_compile_time_constants_preserve_c_integer_semantics(
    expression: str, expected: float
) -> None:
    assert constant_value(expression, {}) == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "sqrt(2)", "1/0", "1e999", "x[0]", "True", "2**3"],
)
def test_nonliteral_or_unbounded_c_evaluation_is_rejected(expression: str) -> None:
    with pytest.raises(CMetadataError):
        constant_value(expression, {})


def test_ambiguous_and_cyclic_macro_bindings_fail_closed() -> None:
    definitions = constant_definitions(
        "#define X 1\n#define X 2\n#define A B\n#define B A\n"
    )
    for name in ("X", "A"):
        with pytest.raises(CMetadataError):
            constant_value(name, definitions)


def test_c_comment_scanner_preserves_strings_and_commas() -> None:
    source = '"a, // still a string", /* comment */ {1,2}, "/* literal */"'
    assert split_fields(strip_c_comments(source)) == [
        '"a, // still a string"',
        "{1,2}",
        '"/* literal */"',
    ]


@pytest.mark.parametrize("text", ["1,,2", "{1,2", '"unterminated', "(1]"])
def test_malformed_initializer_is_rejected(text: str) -> None:
    with pytest.raises(CMetadataError):
        split_fields(text)

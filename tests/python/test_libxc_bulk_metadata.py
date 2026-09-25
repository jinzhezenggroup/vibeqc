"""Adversarial checks for source-owned C parameter binding, never C execution."""

from __future__ import annotations

from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[2]
FULL_LIBXC = ROOT / "upstream" / "libxc-fulltree" / "7.0.0"


def _fulltree_records(
    owner: str, maple: str, *, allow_hybrid_exchange: bool = False
) -> list[dict]:
    return extract_registrations(
        (FULL_LIBXC / "src" / owner).read_text(),
        (FULL_LIBXC / "maple" / "mgga_exc" / maple).read_text(),
        (FULL_LIBXC / "src" / "util.h").read_text(),
        allow_hybrid_exchange=allow_hybrid_exchange,
    )


def record(source: str = SOURCE) -> dict:
    rows = extract_registrations(source, MAPLE, "")
    assert len(rows) == 1
    return rows[0]


def test_m06_2x_correlation_const_array_layout_binds_from_pinned_source() -> None:
    rows = _fulltree_records("mgga_c_m06l.c", "mgga_c_m06l.mpl")
    row = next(item for item in rows if item["name"] == "MGGA_C_M06_2X")
    assert row["metadata_status"] == "bound"
    assert row["family"] == "mgga"
    assert len(row["bindings"]["params_a_css"]) == 5
    assert len(row["bindings"]["params_a_dab"]) == 6


def test_m06_2x_split_exchange_binds_only_with_explicit_opt_in() -> None:
    ordinary = _fulltree_records("hyb_mgga_x_m05.c", "hyb_mgga_x_m05.mpl")
    blocked = next(item for item in ordinary if item["name"] == "HYB_MGGA_X_M06_2X")
    assert blocked["metadata_status"] == "blocked"

    rows = _fulltree_records(
        "hyb_mgga_x_m05.c",
        "hyb_mgga_x_m05.mpl",
        allow_hybrid_exchange=True,
    )
    row = next(item for item in rows if item["name"] == "HYB_MGGA_X_M06_2X")
    assert row["metadata_status"] == "bound"
    assert row["family"] == "mgga"
    assert row["exact_exchange_parameter"] == "0.54"
    assert row["bindings"]["params_a_csi_HF"] == "1.0"
    assert "params_a_cx" not in row["bindings"]


def test_mn15_split_exchange_and_correlation_bind_from_pinned_source() -> None:
    exchange = _fulltree_records(
        "mgga_x_mn12.c",
        "mgga_x_mn12.mpl",
        allow_hybrid_exchange=True,
    )
    xrow = next(item for item in exchange if item["name"] == "HYB_MGGA_X_MN15")
    assert xrow["metadata_status"] == "bound"
    assert xrow["exact_exchange_parameter"] == "0.44"

    correlation = _fulltree_records("mgga_c_m08.c", "mgga_c_m08.mpl")
    crow = next(item for item in correlation if item["name"] == "MGGA_C_MN15")
    assert crow["metadata_status"] == "bound"
    assert len(crow["bindings"]["params_a_m08_a"]) == 12
    assert len(crow["bindings"]["params_a_m08_b"]) == 12


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

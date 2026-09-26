"""Pinned Libxc hybrid metadata lowers to MethodIR data without C execution."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from vibeqc_compiler.method import METHOD_CATALOG
from vibeqc_compiler.method._generated_libxc_methods import (
    BLOCKED_LIBXC_METHODS,
    LIBXC_METHODS,
)

from tools.libxc_method_metadata import extract_method_registrations

ROOT = Path(__file__).resolve().parents[2]
LIBXC = ROOT / "upstream" / "libxc" / "7.0.0"
FULL_LIBXC = ROOT / "upstream" / "libxc-fulltree" / "7.0.0" / "src"


def _row(filename: str, registration: str) -> dict:
    rows = extract_method_registrations((LIBXC / filename).read_text(), filename)
    return next(row for row in rows if row["registration"] == registration)


def _full_row(filename: str, registration: str) -> dict:
    rows = extract_method_registrations((FULL_LIBXC / filename).read_text(), filename)
    return next(row for row in rows if row["registration"] == registration)


def test_b3lyp_composition_and_exchange_are_source_derived_exactly() -> None:
    row = _row("hyb_gga_xc_b3lyp.c", "HYB_GGA_XC_B3LYP")
    assert row["status"] == "generated"
    assert row["components"] == [
        ["LDA_X", "2/25"],
        ["GGA_X_B88", "18/25"],
        ["LDA_C_VWN_RPA", "19/100"],
        ["GGA_C_LYP", "81/100"],
    ]
    assert row["exact_exchange"] == "1/5"


def test_cam_b3lyp_uses_libxc_cam_convention_for_sr_lr_exchange() -> None:
    row = _row("hyb_gga_xc_cam_b3lyp.c", "HYB_GGA_XC_CAM_B3LYP")
    assert row["status"] == "generated"
    assert row["short_range_exchange"] == "19/100"
    assert row["long_range_exchange"] == "13/20"
    assert row["range_omega"] == "33/100"
    assert row["components"][:2] == [
        ["GGA_X_B88", "7/20"],
        ["GGA_X_ITYH", "23/50"],
    ]


def test_m062x_split_global_hybrid_exchange_is_source_derived() -> None:
    row = _full_row("hyb_mgga_x_m05.c", "HYB_MGGA_X_M06_2X")
    assert row["status"] == "generated"
    assert row["identifier"] == "M06-2X"
    assert row["components"] == [["HYB_MGGA_X_M06_2X", "1"]]
    assert row["split_exchange_component"] == "HYB_MGGA_X_M06_2X"
    assert row["paired_correlation_component"] == "MGGA_C_M06_2X"
    assert row["exact_exchange"] == "27/50"
    assert row["short_range_exchange"] == "0"
    assert row["long_range_exchange"] == "0"


def test_mn15_split_global_hybrid_exchange_is_source_derived() -> None:
    # This owner has registration-dependent initializer control flow. The
    # audited set_ext_params_cpy_exx contract deliberately avoids interpreting
    # that control flow and derives the global exact-exchange fraction from the
    # last external parameter.
    row = _full_row("mgga_x_mn12.c", "HYB_MGGA_X_MN15")
    assert row["status"] == "generated"
    assert row["identifier"] == "MN15"
    assert row["components"] == [["HYB_MGGA_X_MN15", "1"]]
    assert row["paired_correlation_component"] == "MGGA_C_MN15"
    assert row["exact_exchange"] == "11/25"


def test_split_range_hybrid_remains_fail_closed() -> None:
    row = _full_row("mgga_x_mn12.c", "HYB_MGGA_X_MN12_SX")
    assert row["status"] == "blocked"
    assert "global-exchange setter" in row["reason"]


def test_wb97mv_direct_semilocal_and_vv10_metadata_are_source_derived() -> None:
    row = _row("hyb_mgga_xc_wb97mv.c", "HYB_MGGA_XC_WB97M_V")
    assert row["status"] == "generated"
    assert row["direct_semilocal_stem"] == "WB97M_V"
    assert row["short_range_exchange"] == "3/20"
    assert row["long_range_exchange"] == "1"
    assert row["range_omega"] == "3/10"
    assert (row["nonlocal_variant"], row["nlc_b"], row["nlc_c"]) == (
        "vv10",
        "6",
        "1/100",
    )


def test_generated_inventory_drives_existing_method_catalog_entries() -> None:
    for name in (
        "B3P86",
        "B3LYP",
        "B3LYP5",
        "B3PW91",
        "B5050LYP",
        "CAM-B3LYP",
        "CAMH-B3LYP",
        "WB97M-V",
    ):
        assert name in LIBXC_METHODS
        assert name in METHOD_CATALOG
    assert METHOD_CATALOG["B3LYP"].exact_exchange.numerator == 1
    assert METHOD_CATALOG["B3LYP"].exact_exchange.denominator == 5
    assert METHOD_CATALOG["WB97M-V"].nonlocal_correlation is not None


def test_representable_generated_methods_need_no_named_catalog_edit() -> None:
    # These names were not hand-authored MethodSpec entries. Their components are
    # already representable, so pinned Libxc metadata is sufficient for MethodIR.
    for name in ("REVB3LYP", "KMLYP", "QTP17", "RCAM-B3LYP", "TUNED-CAM-B3LYP"):
        assert name in LIBXC_METHODS
        assert name in METHOD_CATALOG


def test_complex_dynamic_mix_owners_remain_explicitly_blocked() -> None:
    assert set(BLOCKED_LIBXC_METHODS) == {
        "HYB_GGA_XC_APF",
        "HYB_GGA_XC_WC04",
        "HYB_GGA_XC_WP04",
    }
    assert all("parameter array" in reason for reason in BLOCKED_LIBXC_METHODS.values())


def test_generated_libxc_method_module_is_fresh() -> None:
    from tools.generate_libxc_methods import OUTPUT, build_catalog, render

    generated, blocked = build_catalog()
    assert OUTPUT.read_text(encoding="utf-8") == render(generated, blocked)


@pytest.mark.parametrize(
    "expression", ("0.1234567890123456789012345", "1.234567890123456789e-37", "1e-400")
)
def test_fraction_parser_retains_decimal_source_digits(expression: str) -> None:
    from tools.libxc_method_metadata import _fraction_value

    assert _fraction_value(expression, {}) == Fraction(expression)


@pytest.mark.parametrize(
    "statement",
    (
        "if (p->nspin == 1) p->mix_coef[0] = 1.0 - a0 - ax;",
        "for (int i=0; i<1; ++i) p->mix_coef[0] = 1.0 - a0 - ax;",
        "return; p->mix_coef[0] = 1.0 - a0 - ax;",
        "ax = unknown_scale(ax); p->mix_coef[0] = 1.0 - a0 - ax;",
    ),
)
def test_metadata_rejects_uninterpreted_setter_semantics(statement: str) -> None:
    text = (LIBXC / "hyb_gga_xc_b3lyp.c").read_text()
    old = "p->mix_coef[0] = 1.0 - a0 - ax;"
    assert old in text
    changed = text.replace(old, statement)
    row = next(
        row
        for row in extract_method_registrations(changed)
        if row["registration"] == "HYB_GGA_XC_B3LYP"
    )
    assert row["status"] == "blocked"


def test_metadata_applies_setter_assignments_in_source_order() -> None:
    text = (LIBXC / "hyb_gga_xc_b3lyp.c").read_text()
    old = "p->mix_coef[1] = ax;"
    changed = text.replace(old, old + "\n  ax = 0.5;")
    row = next(
        row
        for row in extract_method_registrations(changed)
        if row["registration"] == "HYB_GGA_XC_B3LYP"
    )
    assert row["status"] == "generated"
    assert row["components"][:2] == [["LDA_X", "2/25"], ["GGA_X_B88", "18/25"]]


@pytest.mark.parametrize(
    "statement",
    (
        "unknown_update(p);",
        "p->cam_alpha = 0.3;",
        "p->nlc_b *= 2.0;",
        "p->params = unknown_allocator();",
    ),
)
def test_metadata_rejects_uninterpreted_initializer_semantics(statement: str) -> None:
    text = (LIBXC / "hyb_mgga_xc_wb97mv.c").read_text()
    changed = text.replace("p->nlc_b = 6.0;", "p->nlc_b = 6.0;\n" + statement)
    (row,) = extract_method_registrations(changed)
    assert row["status"] == "blocked"
    assert "initializer statement" in row["reason"]


def test_metadata_rejects_mix_array_mutation_before_initialization() -> None:
    text = (LIBXC / "hyb_gga_xc_b3lyp.c").read_text()
    old = "xc_mix_init(p, 5, funcs_id, funcs_coef);"
    changed = text.replace(old, "funcs_coef[0] = 0.9;\n" + old)
    row = next(
        row
        for row in extract_method_registrations(changed)
        if row["registration"] == "HYB_GGA_XC_B3P86_NWCHEM"
    )
    assert row["status"] == "blocked"
    assert "initializer statement" in row["reason"]


def test_generated_metadata_component_sequences_are_immutable() -> None:
    components = LIBXC_METHODS["B3LYP"]["components"]
    assert isinstance(components, tuple)
    assert all(isinstance(item, tuple) for item in components)
    with pytest.raises(TypeError):
        components[0][1] = "1/2"


@pytest.mark.parametrize("identifier", sorted(LIBXC_METHODS))
def test_generated_exchange_and_vv10_match_independent_libxc(identifier: str) -> None:
    """Compare parsed C text with the separately compiled Libxc default state."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    if libxc.__version__ != "7.0.0":
        pytest.skip("metadata oracle requires the pinned Libxc 7.0.0 semantics")
    record = LIBXC_METHODS[identifier]
    # Numeric IDs bypass PySCF aliases (notably the configurable B3LYP alias).
    code = record["libxc_id"]
    omega, alpha, beta = libxc.rsh_coeff(code)

    def number(field: str) -> float:
        return float(Fraction(record[field]))

    assert number("range_omega") == pytest.approx(omega, rel=0, abs=2e-15)
    if omega:
        assert number("short_range_exchange") == pytest.approx(
            alpha + beta, rel=0, abs=2e-15
        )
        assert number("long_range_exchange") == pytest.approx(alpha, rel=0, abs=2e-15)
    else:
        for spin in (0, 1):
            assert number("exact_exchange") == pytest.approx(
                libxc.hybrid_coeff(code, spin), rel=0, abs=2e-15
            )
    if record["nonlocal_variant"]:
        ((parameters, weight),) = libxc.nlc_coeff(code)
        assert weight == 1
        assert (number("nlc_b"), number("nlc_c")) == pytest.approx(
            parameters, rel=0, abs=2e-15
        )
    else:
        assert not libxc.nlc_coeff(code)

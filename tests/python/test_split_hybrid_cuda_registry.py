"""Generated native/public registry for split global hybrids."""

from fractions import Fraction
from pathlib import Path

import pytest
from vibeqc.ks import (
    SPLIT_HYBRID_SCF_DOMAIN,
    KsOptions,
    native_ks_options,
    native_xc_functional_code,
    resolve_ks_method,
    resolve_ks_options,
)
from vibeqc_compiler.dft.grid import GridSpec
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.xc._generated_split_hybrids import SPLIT_HYBRIDS

from tools.generate_xc_split_hybrid_registry import (
    GGA_CODE_BASE,
    MGGA_CODE_BASE,
    emit_python_registry,
    emit_registry,
    registry_entries,
)

ROOT = Path(__file__).resolve().parents[2]


def test_split_hybrid_registry_uses_stable_libxc_encoded_codes() -> None:
    entries = {entry["identifier"]: entry for entry in registry_entries()}
    assert set(entries) == {"M06-2X", "MN15"}
    for name, libxc_id, exchange, correlation, fraction in (
        ("M06-2X", 450, "HYB_MGGA_X_M06_2X", "MGGA_C_M06_2X", Fraction(27, 50)),
        ("MN15", 268, "HYB_MGGA_X_MN15", "MGGA_C_MN15", Fraction(11, 25)),
    ):
        entry = entries[name]
        assert entry["family"] == "mgga"
        assert entry["code"] == MGGA_CODE_BASE | libxc_id
        assert entry["exchange_registration"] == exchange
        assert entry["correlation_registration"] == correlation
        assert (
            Fraction(
                entry["exact_exchange_numerator"], entry["exact_exchange_denominator"]
            )
            == fraction
        )
    assert GGA_CODE_BASE != MGGA_CODE_BASE


def test_split_hybrid_registry_is_host_safe_and_device_generated() -> None:
    source = emit_registry()
    for token in (
        "kM062XFunctionalCode = 0x201c2U",
        "kMN15FunctionalCode = 0x2010cU",
        "split_hybrid_registered",
        "split_hybrid_functional_code",
        "HYB_MGGA_X_M06_2X",
        "MGGA_C_M06_2X",
        "HYB_MGGA_X_MN15",
        "MGGA_C_MN15",
        "split_hybrid_is_mgga",
        "split_hybrid_composition",
        "return {27U, 50U, true};",
        "return {11U, 25U, true};",
        "#if defined(__CUDACC__)",
        "evaluate_split_hybrid",
        "m06_2x_device",
        "mn15_device",
    ):
        assert token in source
    assert source.index("#include <cmath>") < source.index(
        "namespace vibeqc::dft::generated"
    )


def test_split_hybrid_registry_generation_is_deterministic() -> None:
    assert emit_registry() == emit_registry()


def test_split_hybrid_python_registry_is_fresh() -> None:
    committed = ROOT / "python/vibeqc_compiler/xc/_generated_split_hybrids.py"
    assert emit_python_registry() == committed.read_text(encoding="utf-8")


@pytest.mark.parametrize("identifier", ("M06-2X", "MN15"))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_split_hybrid_public_methodir_and_c_abi(identifier: str, spin: str) -> None:
    record = SPLIT_HYBRIDS[identifier]
    selector = identifier.lower() + ("-rks" if spin == "unpolarized" else "-uks")
    method_ir, functional = resolve_ks_method(selector)
    assert method_ir.identity == resolve_method(identifier, spin=spin).identity
    assert functional.spin == spin
    assert functional.ingredients == ("rho", "sigma", "tau")
    assert functional.to_payload()["qualification"] == "cuda-point-validated"
    assert functional.to_payload()["production_admitted"] is False
    assert native_xc_functional_code(selector) == record["functional_code"]
    exact = Fraction(record["exact_exchange"])
    assert method_ir.full_range_exact_exchange == exact
    options = resolve_ks_options(selector, KsOptions(grid=GridSpec()))
    assert options.scf_domain == SPLIT_HYBRID_SCF_DOMAIN
    descriptor = native_ks_options(options)
    assert descriptor.semilocal_component_count == 2
    assert descriptor.exchange_term_count == 1
    assert descriptor.spin_channels == (1 if spin == "unpolarized" else 2)
    assert descriptor.exchange_terms[0].coefficient == float(exact)
    assert descriptor.exchange_terms[0].fock_coefficient == float(
        -exact / (2 if spin == "unpolarized" else 1)
    )


@pytest.mark.parametrize("selector", ("pbe0-rks", "b3lyp-uks", "wb97m-v"))
def test_legacy_mapping_catalog_remains_importable_and_resolvable(
    selector: str,
) -> None:
    options = resolve_ks_options(selector, KsOptions(grid=GridSpec()))
    descriptor = native_ks_options(options)
    assert descriptor.semilocal_component_count > 0
    assert descriptor.exchange_term_count > 0


def test_generated_split_hybrid_records_remain_immutable_mappings() -> None:
    record = SPLIT_HYBRIDS["MN15"]
    assert tuple(dict(record)["components"]) == record["components"]
    with pytest.raises(TypeError):
        record["functional_code"] = 0

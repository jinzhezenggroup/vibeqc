"""Generated native registry for qualified split global hybrids."""

from pathlib import Path

from tools.generate_xc_split_hybrid_registry import (
    GGA_CODE_BASE,
    MGGA_CODE_BASE,
    emit_registry,
    registry_entries,
)


def test_split_hybrid_registry_uses_stable_libxc_encoded_codes() -> None:
    entries = {entry["identifier"]: entry for entry in registry_entries()}
    assert set(entries) == {"M06-2X", "MN15"}
    assert entries["M06-2X"]["family"] == "mgga"
    assert entries["MN15"]["family"] == "mgga"
    assert entries["M06-2X"]["code"] == (MGGA_CODE_BASE | 450)
    assert entries["MN15"]["code"] == (MGGA_CODE_BASE | 268)
    assert GGA_CODE_BASE != MGGA_CODE_BASE


def test_split_hybrid_registry_is_host_safe_and_device_generated() -> None:
    source = emit_registry()
    assert "kM062XFunctionalCode = 0x201c2U" in source
    assert "kMN15FunctionalCode = 0x2010cU" in source
    assert "split_hybrid_registered" in source
    assert "split_hybrid_is_mgga" in source
    assert "#if defined(__CUDACC__)" in source
    assert "evaluate_split_hybrid" in source
    assert "m06_2x_device" in source
    assert "mn15_device" in source
    assert source.index("#include <cmath>") < source.index(
        "namespace vibeqc::dft::generated"
    )


def test_split_hybrid_registry_generation_is_deterministic() -> None:
    assert emit_registry() == emit_registry()

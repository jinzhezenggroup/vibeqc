"""Every active imported family participates in a mixed functional identity."""

import typing
from copy import deepcopy
from fractions import Fraction

import pytest
from vibeqc_compiler.xc import spec as module


def records() -> tuple[tuple[str, str], ...]:
    candidates = (
        ("pbe_maple_provenance", "GGA_X_PBE"),
        ("p86_pz_maple_provenance", "GGA_C_P86"),
        ("rsh_maple_provenance", "GGA_C_LYP"),
        ("b88_vwn_maple_provenance", "LDA_C_VWN"),
        ("scan_maple_provenance", "MGGA_C_SCAN"),
    )
    return tuple(
        (name, component) for name, component in candidates if hasattr(module, name)
    )


def mixed() -> module.FunctionalSpec:
    return module.FunctionalSpec(
        "mixed-imports",
        tuple((c, Fraction(1)) for _, c in records()),
        spin="unpolarized",
    )


def components(record: dict[str, typing.Any]) -> set[str]:
    if record.get("kind") == "composite":
        return {key for child in record["sources"] for key in components(child)}
    return set(record["components"])


def test_all_active_families_survive_integration() -> None:
    assert components(mixed().to_payload()["expression_provenance"]) == {
        c for _, c in records()
    }


@pytest.mark.parametrize("name", [name for name, _ in records()])
def test_each_adapter_invalidates_combined_identity(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    before = mixed().identity
    original = getattr(module, name)

    def changed(value: typing.Any) -> typing.Any:
        result = deepcopy(original(value))
        if result is not None:
            result["adapter_sha256"] = "0" * 64
        return result

    monkeypatch.setattr(module, name, changed)
    assert mixed().identity != before

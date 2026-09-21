"""Mixed-family source identity must include every active imported formula."""

from copy import deepcopy
from fractions import Fraction

import pytest
from vibeqc_compiler.xc import spec as module
from vibeqc_compiler.xc.spec import FunctionalSpec


def mixed() -> FunctionalSpec:
    return FunctionalSpec(
        "mixed-pbe-lyp",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_LYP", Fraction(1))),
        spin="unpolarized",
    )


def test_combined_source_inventory_keeps_both_families() -> None:
    payload = mixed().to_payload()["expression_provenance"]
    assert set(payload["components"]) == {"GGA_X_PBE", "GGA_C_LYP"}


@pytest.mark.parametrize("helper", ("rsh_maple_provenance", "pbe_maple_provenance"))
def test_either_adapter_change_invalidates_mixed_identity(
    monkeypatch: pytest.MonkeyPatch, helper: str
) -> None:
    spec = mixed()
    old = spec.identity
    original = getattr(module, helper)

    def changed(components: tuple) -> dict | None:
        result = deepcopy(original(components))
        if result is not None:
            result["adapter_sha256"] = "0" * 64
        return result

    monkeypatch.setattr(module, helper, changed)
    assert spec.identity != old

"""Production cutover gates for pinned Libxc omegaB97M-V semilocal XC."""

import typing

import numpy as np
import pytest
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.xc import expression_dispatch
from vibeqc_compiler.xc.program import build_program
from vibeqc_compiler.xc.wb97mv_maple import wb97mv_maple_provenance


def _spec(spin: str) -> typing.Any:
    return resolve_method("WB97M-V", spin=spin).primitives[0].functional


def test_production_wb97mv_recovers_independent_unpolarized_oracle() -> None:
    spec = _spec("unpolarized")
    actual = build_program(spec, order=2).evaluate(
        np.array((0.5, 0.031, 0.14))[:, None]
    )[:, 0]
    np.testing.assert_allclose(
        actual[:4],
        np.array(
            (
                -0.20794318755308802,
                -0.584943261595045,
                -0.0057485048406733,
                -0.04806556699716644,
            )
        ),
        rtol=4e-12,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        actual[4:],
        np.array(
            (
                -0.4519292851770841,
                0.01463966616204928,
                -0.20181739805554566,
                -0.01737790146793691,
                0.00094734584691088,
                0.6669726284210433,
            )
        ),
        rtol=9e-11,
        atol=3e-12,
    )


def test_program_dispatch_calls_wb97mv_maple_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False
    original = expression_dispatch.wb97mv_energy_expression

    def replacement(spec: typing.Any) -> typing.Any:
        nonlocal called
        called = True
        return original(spec)

    monkeypatch.setattr(expression_dispatch, "wb97mv_energy_expression", replacement)
    build_program(_spec("unpolarized"), order=0)
    assert called


def test_wb97mv_identity_records_maple_provenance() -> None:
    spec = _spec("polarized")
    provenance = spec.to_payload()["expression_provenance"]
    direct = wb97mv_maple_provenance(spec.components, spec.range_omega)
    assert direct is not None
    assert provenance["kind"] == "libxc-maple"
    assert set(provenance["components"]) == {
        "MGGA_X_WB97M_V",
        "MGGA_C_WB97M_V",
    }
    assert provenance["components"] == direct["components"]
    assert len(provenance["adapter_sha256"]) == 64
    assert len(provenance["importer_sha256"]) == 64


def test_cutover_adapter_is_pinned_in_scientific_source_registry() -> None:
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    registry = json.loads((root / "upstream/manifest.json").read_text())
    inputs = registry["products"]["libxc-xc-admission"]["canonical_inputs"]
    relative = "python/vibeqc_compiler/xc/wb97mv_maple.py"
    assert (
        inputs[relative] == hashlib.sha256((root / relative).read_bytes()).hexdigest()
    )

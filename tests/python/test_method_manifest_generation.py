from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from vibeqc import _generated_methods, _native

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ABI_IDS = {
    "rhf": 1,
    "uhf": 2,
    "wb97m-v": 3,
    "rccsd(t)": 4,
    "mp2": 5,
    "lda-rks": 6,
    "pbe-rks": 7,
    "lda-uks": 8,
    "pbe-uks": 9,
    "r2scan-rks": 10,
    "r2scan-uks": 11,
    "rccsd": 12,
    "pbe0-rks": 13,
    "pbe0-uks": 14,
    "gfn2-xtb": 15,
    "b3lyp-rks": 16,
    "b3lyp-uks": 17,
    "pbe-d4-rks": 18,
}


def test_public_method_generated_metadata_is_fresh() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_method_manifest.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )


def test_public_method_abi_ids_are_explicit_and_stable() -> None:
    payload = json.loads(
        (ROOT / "manifests/public_methods.json").read_text(encoding="utf-8")
    )
    actual = {entry["name"]: entry["abi_id"] for entry in payload["methods"]}
    assert actual == EXPECTED_ABI_IDS
    assert dict(_generated_methods.METHOD_NAME_TO_ID) == {
        **EXPECTED_ABI_IDS,
        "ccsd(t)": 4,
        "gfn2": 15,
    }


def test_public_method_provider_sets_are_generated() -> None:
    assert _generated_methods.HF_METHOD_IDS == frozenset({1, 2})
    assert _generated_methods.NATIVE_DFT_METHOD_IDS == frozenset(
        {6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18}
    )


def test_native_binding_reexports_generated_method_constants() -> None:
    for symbol, value in _generated_methods.METHOD_CONSTANTS.items():
        assert getattr(_native, symbol) == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("abi_id", 2**31),
        ("supports_batch", "false"),
        ("supports_batch", 1),
        ("properties", {"energy": True}),
        ("properties", ["energy", "energy"]),
    ],
)
def test_manifest_rejects_lossy_abi_and_capability_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    from tools import generate_method_manifest as generator

    payload = json.loads(generator.MANIFEST.read_text())
    payload["methods"][0][field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(generator, "MANIFEST", path)
    with pytest.raises(ValueError):
        generator.load_manifest()


def test_generated_python_supports_an_empty_provider_group() -> None:
    from tools import generate_method_manifest as generator

    methods = [m for m in generator.load_manifest() if m["provider"] == "reserved"]
    generated = generator.emit_python(methods)
    compile(generated, "generated-methods", "exec")
    assert "HF_METHOD_IDS = frozenset(())" in generated
    assert "NATIVE_DFT_METHOD_IDS = frozenset(())" in generated
